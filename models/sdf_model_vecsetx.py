# models/sdf_model_vecset.py
import torch.nn as nn
import torch
from models.archs.sdf_decoder import SdfDecoder
from models.archs.vecsetx import VecSetAutoEncoder
from models.archs.vecsetx_bottleneck import NormalizedBottleneck, KLBottleneck

class SdfModelVecSet(nn.Module):
    def __init__(self, specs):
        super().__init__()
        
        model_specs = specs["SdfModelSpecs"]
        self.hidden_dim = model_specs["hidden_dim"]
        self.latent_dim = model_specs["latent_dim"]
        
        # Replace ConvPointnet with VecSetX encoder
        self.encoder = VecSetAutoEncoder(
            depth=12,  # Reduced depth for just encoding
            dim=self.hidden_dim,
            num_inputs=specs.get("PCsize", 8192),
            num_latents=model_specs["num_latents"],  # Fewer tokens for efficiency
            latent_dim=self.latent_dim,
            query_type='learnable',
            bottleneck=NormalizedBottleneck,  # Or KLBottleneck for explicit VAE
            bottleneck_args={'dim': self.hidden_dim, 'latent_dim': self.latent_dim}
        )
        
        # Keep the original SDF decoder
        self.decoder = SdfDecoder(
            latent_size=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            skip_connection=model_specs.get("skip_connection", True),
            tanh_act=model_specs.get("tanh_act", False)
        )
        
    def encode_to_latent(self, pc):
        """Encode point cloud to latent representation for diffusion"""
        bottleneck = self.encoder.encode(pc)
        # Flatten tokens to single vector for diffusion
        # Shape: [B, num_latents, latent_dim] -> [B, num_latents * latent_dim]
        latent = bottleneck['x'].reshape(pc.shape[0], -1)
        return latent, bottleneck
    
    def decode_from_latent(self, latent, xyz):
        """Decode from latent to SDF values"""
        # Reshape back to tokens
        B = latent.shape[0]
        latent_tokens = latent.reshape(B, self.encoder.num_latents, -1)
        
        # Process through VecSet decoder layers
        x = self.encoder.bottleneck.post(latent_tokens)
        for self_attn, self_ff in self.encoder.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        
        # Use cross-attention to get features at query points
        queries_embeddings = self.encoder.point_embed(xyz)
        features = self.encoder.decoder_cross_attn(queries_embeddings, context=x)
        
        # Pass through SDF decoder
        combined = torch.cat([xyz, features], dim=-1)
        return self.decoder(combined)
    
    def forward(self, pc, xyz):
        latent, _ = self.encode_to_latent(pc)
        return self.decode_from_latent(latent, xyz).squeeze()