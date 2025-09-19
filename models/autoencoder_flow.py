import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, reduce
from typing import List, Callable, Union, Any, TypeVar, Tuple
import numpy as np

# Import the existing BetaVAE
from models.autoencoder import BetaVAE
from models.archs.flow import build_latent_flow

from numbers import Number
from torch.autograd import Variable

Tensor = TypeVar('torch.tensor')

def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent

def standard_normal_logprob(z):
    """
    Compute log p(z) for z ~ N(0, I)
    Args:
        z: Tensor of shape [B, D]
    Returns:
        log_prob: Tensor of shape [B]
    """
    batch_size, dim = z.shape
    
    # Constant term: -D/2 * log(2π)
    const = -0.5 * dim * np.log(2 * np.pi)
    
    # Variable term: -½ ||z||²
    norm_squared = (z ** 2).sum(dim=1)  # Sum over dimensions
    
    return const - 0.5 * norm_squared  # Shape: [B]

# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
def truncated_normal(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def sample_diagonal_MultiGauss(mu, log_var, n):
		# reference :
		# http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n

		# Convert z_log_var to std
		std = torch.exp(0.5 * log_var)
		def expand(v):
			if isinstance(v, Number):
				return torch.Tensor([v]).expand(n, 1)
			else:
				return v.expand(n, *v.size())
		if n != 1 :
			mu = expand(mu)
			std = expand(std)
		eps = Variable(std.data.new(std.size()).normal_().to(std.device))
		samples =  mu + eps * std
		samples = samples.reshape((n * mu.shape[1],)+ mu.shape[2:])
		return samples

class FlowBetaVAE(BetaVAE):
    """BetaVAE enhanced with Normalizing Flows for more flexible latent distributions"""
    
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 kl_std=1.0,
                 input_resolution: int = 64,
                 beta: int = 4,
                 gamma: float = 10.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 use_flow: bool = True,
                 flow_depth: int = 4,
                 flow_hidden_dim: int = 512,
                 **kwargs) -> None:
        
        super().__init__(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            kl_std=kl_std,
            input_resolution = input_resolution,
            beta=beta,
            gamma=gamma,
            max_capacity=max_capacity,
            Capacity_max_iter=Capacity_max_iter,
            loss_type=loss_type,
            **kwargs
        )
        
        self.use_flow = use_flow
        
        if self.use_flow:
            self.flow = build_latent_flow(
                latent_dim=latent_dim,
                flow_depth=flow_depth,
                hidden_dim=flow_hidden_dim
            )
    
    def encode_with_flow(self, enc_input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encodes input and applies normalizing flow
        Returns: (z_flow, mu, log_var)
        """
        mu, log_var = self.encode(enc_input)
        z = self.reparameterize(mu, log_var)
        
        if self.use_flow:
            # Apply flow transformation
            z_flow = self.flow(z, reverse=False)
            return z_flow, mu, log_var
        else:
            return z, mu, log_var
    
    def forward(self, data: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(data)
        z = self.reparameterize(mu, log_var)
        
        if self.use_flow:
            # During training, we still decode from z (not w!)
            # The flow is only used in the loss computation
            pass
        
        recons = self.decode(z)  # Always decode from z during training
        return [recons, data, mu, log_var, z]
    
    def loss_function(self, *args, **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]  # Not used for loss, just passed through
        data = args[1]    # Not used for loss, just passed through
        mu = args[2]
        log_var = args[3]
        z = args[4]       # Original z (before flow if using flow)
        
        kld_weight = kwargs['M_N']
        
        if self.use_flow:
            # Transform z (complex) to w (simple)
            batch_size = z.shape[0]
            logpx = torch.zeros(batch_size, 1).to(z.device)
            w, delta_logp = self.flow(z, logpx=logpx, reverse=False)
            # flow loss
            # # H[Q(z|X)]
            entropy = gaussian_entropy(logvar=log_var)      # (B, )
            
            # Log probability of w under standard normal
            # This should return [B], not [B, latent_dim]
            log_pw = standard_normal_logprob(w) # [B]
            log_pz = log_pw - delta_logp # [B]
            #log_pw_per_dim = -0.5 * (w.pow(2) + np.log(2 * np.pi))  # [B, latent_dim]
            #log_pw = log_pw_per_dim.sum(dim=1)  # [B]
    		
            # Loss
            loss_entropy = -entropy.mean()
            loss_prior = -log_pz.mean()
            kl_loss = loss_prior+loss_entropy
        else:
            # Original KL computation
            if self.kl_std == 'zero_mean':
                latent = self.reparameterize(mu, log_var)
                l2_size_loss = torch.sum(torch.norm(latent, dim=-1))
                kl_loss = l2_size_loss / latent.shape[0]
            else:
                std = torch.exp(0.5 * log_var)
                gt_dist = torch.distributions.normal.Normal(
                    torch.zeros_like(mu), 
                    torch.ones_like(std) * self.kl_std
                )
                sampled_dist = torch.distributions.normal.Normal(mu, std)
                kl = torch.distributions.kl.kl_divergence(sampled_dist, gt_dist)
                kl_loss = reduce(kl, 'b ... -> b (...)', 'mean').mean()
        
        return kld_weight * kl_loss  # Only KL loss, no reconstruction loss
        
    def sample(self, num_samples: int, truncate_std: float = 1.0, **kwargs) -> Tensor:
        """Sample from the model"""
        device = next(self.parameters()).device
        
        if self.use_flow:
            # True truncated normal
            w = torch.zeros(num_samples, self.latent_dim).to(device)
            truncated_normal(w, mean=0, std=1, trunc_std=truncate_std)
            # Start from w ~ N(0,I)
            #w = torch.randn(num_samples, self.latent_dim).to(device)
            # Transform w → z using inverse flow
            z = self.flow(w, reverse=True)
        else:
            z = torch.zeros(num_samples, self.latent_dim).to(device)
            truncated_normal(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.decode(z)
        return samples
    
    def get_latent(self, x):
        """Get latent code for reconstruction"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z