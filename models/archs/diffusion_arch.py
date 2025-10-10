import math
import torch
import torch.nn.functional as F
from torch import nn, einsum 

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many, check_shape

from rotary_embedding_torch import RotaryEmbedding

from diff_utils.model_utils import * 

from random import sample

from models.archs.vecsetx import VecSetAutoEncoder
from models.archs.vecsetx_bottleneck import NormalizedBottleneck

class CausalTransformer(nn.Module):
    def __init__(
        self,
        dim, 
        depth,
        dim_in_out=None,
        cross_attn=False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True, 
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True, 
        normformer = False,
        rotary_emb = True,
        use_flash_attn = True,  # Enable flash attention
        use_checkpoint = False,  # Enable gradient checkpointing
        **kwargs
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None
        rotary_emb_cross = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])

        dim_in_out = default(dim_in_out, dim)
        self.use_same_dims = (dim_in_out is None) or (dim_in_out==dim)
        point_feature_dim = kwargs.get('point_feature_dim', dim)

        if cross_attn:
            # First layer - dimension change from dim_in_out to dim
            self.layers.append(nn.ModuleList([
                Attention(dim = dim_in_out, out_dim=dim, causal = True, 
                         dim_head = dim_head, heads = heads, rotary_emb = rotary_emb,
                         use_flash_attn=use_flash_attn),  # Add flash
                Attention(dim = dim, kv_dim=point_feature_dim, causal = True, 
                         dim_head = dim_head, heads = heads, dropout = attn_dropout, 
                         rotary_emb = rotary_emb_cross, use_flash_attn=use_flash_attn),  # Add flash
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))
            # Middle layers - all use dim
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, 
                             rotary_emb = rotary_emb, use_flash_attn=use_flash_attn),
                    Attention(dim = dim, kv_dim=point_feature_dim, causal = True, 
                             dim_head = dim_head, heads = heads, dropout = attn_dropout, 
                             rotary_emb = rotary_emb_cross, use_flash_attn=use_flash_attn),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            # Last layer - dimension change from dim to dim_in_out
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, out_dim=dim, causal = True, dim_head = dim_head, 
                         heads = heads, rotary_emb = rotary_emb, use_flash_attn=use_flash_attn),
                Attention(dim = dim, kv_dim=point_feature_dim, out_dim=dim_in_out, causal = True, 
                         dim_head = dim_head, heads = heads, dropout = attn_dropout, 
                         rotary_emb = rotary_emb_cross, use_flash_attn=use_flash_attn),
                FeedForward(dim = dim_in_out, out_dim=dim_in_out, mult = ff_mult, 
                           dropout = ff_dropout, post_activation_norm = normformer)
            ]))
            
        else:
            # Non-cross-attention version
            self.layers.append(nn.ModuleList([
                Attention(dim = dim_in_out, out_dim=dim, causal = True, dim_head = dim_head, 
                         heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb,
                         use_flash_attn=use_flash_attn),
                FeedForward(dim = dim, out_dim=dim, mult = ff_mult, dropout = ff_dropout, 
                           post_activation_norm = normformer)
            ]))
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, 
                             dropout = attn_dropout, rotary_emb = rotary_emb, use_flash_attn=use_flash_attn),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, 
                               post_activation_norm = normformer)
                ]))
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, out_dim=dim_in_out, causal = True, dim_head = dim_head, 
                         heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb,
                         use_flash_attn=use_flash_attn),
                FeedForward(dim = dim_in_out, out_dim=dim_in_out, mult = ff_mult, 
                           dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim_in_out, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim_in_out, dim_in_out, bias = False) if final_proj else nn.Identity()

        self.cross_attn = cross_attn
    
    def _forward_block(self, x, block, attn_bias, context=None):
        """Forward through a single transformer block"""
        if self.cross_attn:
            self_attn, cross_attn, ff = block
            if context is not None:
                x = self_attn(x, attn_bias=attn_bias) + x
                x = cross_attn(x, context=context) + x
            else:
                x = self_attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x
        else:
            attn, ff = block
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x
        return x
    
    def forward(self, x, time_emb=None, context=None):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)
        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        # Use gradient checkpointing if enabled
        for idx, block in enumerate(self.layers):
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._forward_block, 
                    x, 
                    block, 
                    attn_bias, 
                    context,
                    use_reentrant=False
                )
            else:
                # Handle first and last blocks specially for dimension changes
                if (idx == 0 or idx == len(self.layers) - 1) and not self.use_same_dims:
                    # These blocks change dimensions, handle separately
                    if self.cross_attn:
                        self_attn, cross_attn, ff = block
                        x = self_attn(x, attn_bias=attn_bias)
                        x = cross_attn(x, context=context) if context is not None else x
                    else:
                        attn, ff = block
                        x = attn(x, attn_bias=attn_bias)
                    x = ff(x)
                else:
                    x = self._forward_block(x, block, attn_bias, context)

        out = self.norm(x)
        return self.project_out(out)
    
    def forward_old(self, x, time_emb=None, context=None):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        if self.cross_attn:
            #assert context is not None 
            for idx, (self_attn, cross_attn, ff) in enumerate(self.layers):
                #print("x1 shape: ", x.shape)
                if (idx==0 or idx==len(self.layers)-1) and not self.use_same_dims:
                    x = self_attn(x, attn_bias = attn_bias)
                    x = cross_attn(x, context=context) # removing attn_bias for now 
                else:
                    x = self_attn(x, attn_bias = attn_bias) + x 
                    x = cross_attn(x, context=context) + x  # removing attn_bias for now 
                #print("x2 shape, context shape: ", x.shape, context.shape)
                
                #print("x3 shape, context shape: ", x.shape, context.shape)
                x = ff(x) + x
        
        else:
            for idx, (attn, ff) in enumerate(self.layers):
                #print("x1 shape: ", x.shape)
                if (idx==0 or idx==len(self.layers)-1) and not self.use_same_dims:
                    x = attn(x, attn_bias = attn_bias)
                else:
                    x = attn(x, attn_bias = attn_bias) + x
                #print("x2 shape: ", x.shape)
                x = ff(x) + x
                #print("x3 shape: ", x.shape)

        out = self.norm(x)
        return self.project_out(out)

class DiffusionNet(nn.Module):

    def __init__(
        self,
        dim,
        dim_in_out=None,
        num_timesteps = None,
        num_time_embeds = 1,
        cond = None,
        **kwargs
    ):
        super().__init__()
        self.num_time_embeds = num_time_embeds
        self.dim = dim
        self.cond = cond
        self.cross_attn = kwargs.get('cross_attn', False)
        self.cond_dropout = kwargs.get('cond_dropout', False)
        self.point_feature_dim = kwargs.get('point_feature_dim', dim)

        self.dim_in_out = default(dim_in_out, dim)
        #print("dim, in out, point feature dim: ", dim, dim_in_out, self.point_feature_dim)
        #print("cond dropout: ", self.cond_dropout)

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, self.dim_in_out * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(self.dim_in_out), MLP(self.dim_in_out, self.dim_in_out * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # last input to the transformer: "a final embedding whose output from the Transformer is used to predicted the unnoised CLIP image embedding"
        self.learned_query = nn.Parameter(torch.randn(self.dim_in_out))
        self.causal_transformer = CausalTransformer(dim = dim, dim_in_out=self.dim_in_out, **kwargs)

        if cond:
            # output dim of pointnet needs to match model dim; unless add additional linear layer
            # self.pointnet = ConvPointnet(c_dim=self.point_feature_dim) 
            
            # Get the sample_pc_size from kwargs to match what will be sampled
            sample_pc_size = kwargs.get('sample_pc_size', 512)
            
            # Use a lightweight VecSet encoder for conditioning
            # This matches main architecture's approach
            self.cond_encoder = VecSetAutoEncoder(
                depth=6,  # Shallower than main encoder (which uses 24)
                dim=self.point_feature_dim,  # 256 or 512
                output_dim=1,  # Not used for encoding only
                num_inputs=sample_pc_size,  # Sampled points
                num_latents=32,  # Fewer tokens for efficiency (vs 256 in main)
                latent_dim=16,  # Same bottleneck dimension
                query_type='learnable',  # SAME AS YOUR MAIN MODEL
                bottleneck=NormalizedBottleneck,  # Simple bottleneck (no VAE regularization)
                bottleneck_args={'dim': self.point_feature_dim, 'latent_dim': 16}
            )
            
            # Project conditioning tokens to expected dimension if needed
            self.cond_projection = nn.Linear(16, self.point_feature_dim) if self.point_feature_dim != 16 else nn.Identity()


    def forward(
        self,
        data, 
        diffusion_timesteps,
        pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):

        if self.cond:
            assert type(data) is tuple
                              #cond is point cloud [B, N, 3]
            data, cond = data # adding noise to cond_feature so doing this in diffusion.py

            #print("data, cond shape: ", data.shape, cond.shape) # B, dim_in_out; B, N, 3
            #print("pass cond: ", pass_cond)
            if self.cond_dropout:
                # classifier-free guidance: 20% unconditional 
                prob = torch.randint(low=0, high=10, size=(1,))
                percentage = 8
                if prob < percentage or pass_cond==0:
                    # Zero conditioning for classifier-free guidance
                    cond_feature = torch.zeros(
                        (cond.shape[0], 32, self.point_feature_dim),  # 32 tokens
                        device=data.device
                    )
                    #cond_feature = torch.zeros( (cond.shape[0], cond.shape[1], self.point_feature_dim), device=data.device )
                    #print("zeros shape: ", cond_feature.shape) 
                elif prob >= percentage or pass_cond==1:
                    # Encode point cloud with VecSet encoder
                    bottleneck = self.cond_encoder.encode(cond)
                    cond_tokens = bottleneck['x']  # [B, 32, 16]
                    cond_feature = self.cond_projection(cond_tokens)  # [B, 32, point_feature_dim]
                    #cond_feature = self.pointnet(cond, cond)
                    #print("cond shape: ", cond_feature.shape)
            else:
                bottleneck = self.cond_encoder.encode(cond)
                cond_tokens = bottleneck['x']
                cond_feature = self.cond_projection(cond_tokens)
                #cond_feature = self.pointnet(cond, cond)

            
        batch, dim, device, dtype = *data.shape, data.device, data.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds(diffusion_timesteps)

        data = data.unsqueeze(1)

        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch)

        model_inputs = [time_embed, data, learned_queries]

        if self.cond and not self.cross_attn:
            # For non-cross-attention, need to flatten conditioning
            cond_feature_flat = cond_feature.reshape(batch, -1)  # [B, 32*256] = [B, 8192]
            # Project to expected dimension
            if not hasattr(self, 'cond_to_latent'):
                self.cond_to_latent = nn.Linear(32 * self.point_feature_dim, self.dim_in_out)
            cond_as_latent = self.cond_to_latent(cond_feature_flat)  # [B, 4096]
            cond_as_latent = cond_as_latent.unsqueeze(1)  # [B, 1, 4096]
            model_inputs.insert(0, cond_as_latent)
            # Now: [cond, time, data, learned_query] all with shape [B, 1, 4096]
        
        tokens = torch.cat(model_inputs, dim = 1) # (b, 3/4, d); batch and d=512 same across the model_inputs 
        #print("tokens shape: ", tokens.shape)

        if self.cross_attn:
            cond_feature = None if not self.cond else cond_feature
            #print("tokens shape: ", tokens.shape, cond_feature.shape)
            tokens = self.causal_transformer(tokens, context=cond_feature)
        else:
            tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the sdf layer embedding (per DDPM timestep)
        pred = tokens[..., -1, :]

        return pred

