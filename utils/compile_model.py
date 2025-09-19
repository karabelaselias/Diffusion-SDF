import torch
from typing import Optional

def compile_model(model, mode='default', backend='inductor', compile_eikonal=True):
    """
    Compile model components for faster execution
    
    Args:
        model: The model to compile
        mode: 'default', 'reduce-overhead', 'max-autotune'
        backend: 'inductor' (default), 'cudagraphs', etc.
    """
    if torch.__version__ < '2.0.0':
        print("torch.compile requires PyTorch 2.0+")
        return model
    
    # Check if CUDA is available and model is on GPU
    if not torch.cuda.is_available():
        print("torch.compile works best with CUDA")
        return model
    
    # Compile different components based on model type
    if hasattr(model, 'sdf_model'):
        print("Compiling SDF model components...")
        # Compile the decoder with reduce-overhead for many small ops
        model.sdf_model.model = torch.compile(
            model.sdf_model.model, 
            mode='max-autotune',
            fullgraph=False
        )
        
        # Compile pointnet but be careful with grid_sample
        model.sdf_model.pointnet = torch.compile(
            model.sdf_model.pointnet,
            mode='default',
            fullgraph=False,  # Important: allow breaks for grid_sample
            disable=False
        )

    if hasattr(model, 'vae_model'):
        print("Compiling VAE model...")
        # These should compile well
        model.vae_model.encoder = torch.compile(
            model.vae_model.encoder,
            mode='default',
            fullgraph=False
        )
        model.vae_model.decoder = torch.compile(
            model.vae_model.decoder,
            mode='default',
            fullgraph=False
        )

    if hasattr(model, 'diffusion_model'):
        print("Compiling Diffusion model...")
        if hasattr(model.diffusion_model, 'model'):
            model.diffusion_model.model = torch.compile(
                model.diffusion_model.model,
                mode='max-autotune',  # Best for transformer models
                fullgraph=False
            )

    # Optionally compile the gradient computation for eikonal
    if compile_eikonal and hasattr(model, 'use_eikonal') and model.use_eikonal:
        print("Compiling eikonal gradient computation...")
        # Compile with specific settings for gradient computation
        model.compute_gradient = torch.compile(
            model.compute_gradient,
            mode='default',  # Don't over-optimize gradient computation
            fullgraph=False,  # Allow graph breaks
            dynamic=True  # Handle dynamic shapes better
        )
        
        # Also compile the eikonal loss computation
        model.eikonal_loss = torch.compile(
            model.eikonal_loss,
            mode='reduce-overhead',  # Good for reduction operations
            fullgraph=True  # This one can be a full graph
        )
    
    return model