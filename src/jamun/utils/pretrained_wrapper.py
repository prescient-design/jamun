"""
Pretrained model wrapper utilities for seamless integration with Hydra configs.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
import logging

from jamun.utils.pretrained import load_pretrained_model_from_checkpoint
from jamun.utils import mean_center_f, unsqueeze_trailing
from jamun.model import Denoiser
from jamun.utils import find_checkpoint


def compute_normalization_factors(
    sigma: float | torch.Tensor,
    *,
    average_squared_distance: float,
    normalization_type: str | None,
    sigma_data: float | None = None,
    D: int = 3,
    device: torch.device | None = None,
) -> tuple[float, float, float, float]:
    """Compute the normalization factors for the input, skip connection, output, and noise."""
    sigma = torch.as_tensor(sigma, device=device)

    if normalization_type is None:
        c_in = torch.as_tensor(1.0, device=device)
        c_skip = torch.as_tensor(0.0, device=device)
        c_out = torch.as_tensor(1.0, device=device)
        c_noise = torch.as_tensor(sigma, device=device)
        return c_in, c_skip, c_out, c_noise

    if normalization_type == "EDM":
        c_skip = (sigma_data**2) / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
        c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
        c_noise = torch.log(sigma / sigma_data) * 0.25
        return c_in, c_skip, c_out, c_noise

    if normalization_type == "JAMUN":
        A = torch.as_tensor(average_squared_distance, device=device)
        B = torch.as_tensor(2 * D * sigma**2, device=device)

        c_in = 1.0 / torch.sqrt(A + B)
        c_skip = A / (A + B)
        c_out = torch.sqrt((A * B) / (A + B))
        c_noise = torch.log(sigma) / 4
        return c_in, c_skip, c_out, c_noise

    raise ValueError(f"Unknown normalization type: {normalization_type}")



class DenoiserWrapper(nn.Module):
    """
    Wrapper around a denoiser model that matches the spatial module interface.
    
    This allows pretrained denoiser models to be used as spatial/temporal modules
    in the spatiotemporal architecture by replicating the full denoiser logic
    including normalization factors computed from the denoiser's own parameters.
    """
    
    def __init__(self, denoiser_model: nn.Module, c_in: float = 1.0, trainable: bool = True):
        """
        Initialize the wrapper.
        
        Args:
            denoiser_model: The pretrained denoiser model
            c_in: Rescaling factor to convert positions from overlaying model scale
            trainable: Whether to keep the model trainable (default: True)
        """
        super().__init__()
        self.denoiser = denoiser_model
        self.c_in = c_in
        
        # Set trainability
        if not trainable:
            for param in self.denoiser.parameters():
                param.requires_grad = False
                
    def forward(self, pos, topology, batch, num_graphs, c_noise, effective_radial_cutoff):
        """
        Forward pass that replicates the denoiser's xhat and xhat_normalized methods.
        
        Args:
            pos: Node positions (input to spatial module)
            topology: Graph topology information (already contains bonded edges)
            batch: Batch indices
            num_graphs: Number of graphs in batch
            c_noise: Noise conditioning parameter (already computed)
            effective_radial_cutoff: Radial cutoff
            
        Returns:
            Denoised positions from the pretrained model
        """
        # Sample sigma from the denoiser's own sigma distribution
        sigma = self.denoiser.sigma_distribution.sample().to(pos.device)
        
        # Rescale positions from overlaying model scale
        y = pos / self.c_in
        
        # Replicate xhat logic
        if self.denoiser.mean_center:
            y = mean_center_f(y, batch, num_graphs)
        
        # Replicate xhat_normalized logic
        # Compute the normalization factors for the rescaled positions
        c_in, c_skip, c_out, _ = compute_normalization_factors(
            sigma,
            average_squared_distance=self.denoiser.average_squared_distance,
            normalization_type=self.denoiser.normalization_type,
            sigma_data=self.denoiser.sigma_data,
            D=y.shape[-1],
            device=y.device,
        )

        # Adjust dimensions
        c_in = unsqueeze_trailing(c_in, y.ndim - 1)
        c_skip = unsqueeze_trailing(c_skip, y.ndim - 1)
        c_out = unsqueeze_trailing(c_out, y.ndim - 1)
        c_noise = c_noise.unsqueeze(0) if c_noise.dim() == 0 else c_noise
        
        # Ensure c_noise is float type (fix for dtype mismatch)
        c_noise = c_noise.float()

        # Scale input positions by c_in
        y_scaled = y * c_in

        # # Call the denoiser's architecture (topology already has edges)
        # # Add this right before line 129 in the pretrained wrapper call
        # print("=== Debugging pretrained denoiser input types ===")
        # print(f"y_scaled dtype: {y_scaled.dtype}, shape: {y_scaled.shape}")
        # print(f"topology.edge_index dtype: {topology.edge_index.dtype if hasattr(topology, 'edge_index') else 'N/A'}")
        # print(f"c_noise dtype: {c_noise.dtype}, shape: {c_noise.shape}")
        # print(f"batch dtype: {batch.dtype}, shape: {batch.shape}")
        # print(f"effective_radial_cutoff dtype: {type(effective_radial_cutoff)}")

        # # Check if topology has any Long tensors
        # for attr_name in dir(topology):
        #     if not attr_name.startswith('_'):
        #         attr = getattr(topology, attr_name)
        #         if isinstance(attr, torch.Tensor):
        #             print(f"topology.{attr_name} dtype: {attr.dtype}")
        g_pred = self.denoiser.g(
            pos=y_scaled,
            topology=topology,
            c_noise=c_noise,
            effective_radial_cutoff=effective_radial_cutoff,
            batch=batch,
            num_graphs=num_graphs,
        )

        # Compute final prediction with skip connection
        xhat = c_skip * y + c_out * g_pred

        # Mean center the prediction if needed
        if self.denoiser.mean_center:
            xhat = mean_center_f(xhat, batch, num_graphs)

        return xhat


def return_wrapped_denoiser(
    wandb_run_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None, 
    checkpoint_type: str = "best_so_far",
    c_in: float = 1.0,
    trainable: bool = True
) -> DenoiserWrapper:
    """
    Load a pretrained denoiser model and return it wrapped for use in spatiotemporal architecture.
    
    This function is designed to be used directly as a _target_ in Hydra configs.
    The wrapper replicates the full denoiser logic including normalization factors
    computed from the denoiser's own training parameters.
    
    Args:
        wandb_run_path: Path to wandb run (e.g., "entity/project/run_id")
        checkpoint_path: Direct path to checkpoint file
        checkpoint_type: Type of checkpoint to load ("best_so_far", "latest", etc.)
        c_in: Rescaling factor to convert positions from overlaying model scale
        trainable: Whether to keep the loaded model trainable
        
    Returns:
        DenoiserWrapper containing the pretrained model
        
    Example usage in config:
        spatial_module:
          _target_: jamun.utils.pretrained_wrapper.return_wrapped_denoiser
          wandb_run_path: "your_entity/your_project/run_id"
          c_in: 1.0
          trainable: false
    """
    py_logger = logging.getLogger("jamun")
    
    if not wandb_run_path and not checkpoint_dir:
        raise ValueError("Either wandb_run_path or checkpoint_path must be provided")
    
    # Load the pretrained model
    py_logger.info(f"Loading pretrained denoiser from: {wandb_run_path or checkpoint_dir}")
    
    # pretrained_model = load_pretrained_model_from_checkpoint(
    #     checkpoint_path=checkpoint_path,
    #     wandb_run_path=wandb_run_path,
    #     checkpoint_type=checkpoint_type
    # )
    checkpoint_path = find_checkpoint(wandb_train_run_path=wandb_run_path, checkpoint_dir=checkpoint_dir, checkpoint_type=checkpoint_type)
    pretrained_model = Denoiser.load_from_checkpoint(checkpoint_path)
    
    if pretrained_model is None:
        raise RuntimeError(f"Failed to load pretrained model from {wandb_run_path or checkpoint_path}")
    
    py_logger.info("✓ Successfully loaded pretrained denoiser")
    
    # Wrap the model
    wrapped_model = DenoiserWrapper(pretrained_model, c_in=c_in, trainable=trainable)
    
    py_logger.info(f"✓ Using c_in rescaling factor: {c_in}")
    py_logger.info(f"✓ Using denoiser's own normalization parameters:")
    py_logger.info(f"  - normalization_type: {pretrained_model.normalization_type}")
    py_logger.info(f"  - average_squared_distance: {pretrained_model.average_squared_distance}")
    if hasattr(pretrained_model, 'sigma_data') and pretrained_model.sigma_data is not None:
        py_logger.info(f"  - sigma_data: {pretrained_model.sigma_data}")
    py_logger.info(f"  - mean_center: {pretrained_model.mean_center}")
    
    if not trainable:
        py_logger.info("✓ Frozen pretrained denoiser (not trainable)")
    else:
        py_logger.info("✓ Pretrained denoiser is trainable")
        
    return wrapped_model