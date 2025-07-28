import logging
from typing import Callable, Dict, Tuple, Union

import lightning.pytorch as pl
import numpy as np
import torch
import torch_geometric
import e3nn
from e3tools import radius_graph
from jamun.model.denoiser_conditional import Denoiser
# Fix e3nn optimization for avoiding script issues
e3nn.set_optimization_defaults(jit_script_fx=False)

from jamun.utils import align_A_to_B_batched, mean_center, unsqueeze_trailing
from jamun.utils.align import kabsch_algorithm
from jamun.utils.checkpoint import find_checkpoint

class Conditioner(pl.LightningModule):
    """
    Base class for conditioners.
    """
    def __init__(self, N_structures: int, **kwargs):
        super().__init__()
        self.N_structures = N_structures

class PositionConditioner(pl.LightningModule):
    """
    Condition the hidden state on the position of the structure.
    """
    def __init__(self, N_structures: int, align_hidden_states: bool = True, **kwargs):
        super().__init__()
        self.N_structures = N_structures
        self.align_hidden_states = align_hidden_states
    def forward(self, y: torch_geometric.data.Batch) -> list[torch.Tensor]:
        conditioned_structures = [y.pos]  # Start with current position
        for positions in y.hidden_state: 
            if self.align_hidden_states:
                aligned_positions = kabsch_algorithm(positions, y.pos, y.batch, y.num_graphs)
                conditioned_structures.append(aligned_positions)
            else:
                conditioned_structures.append(positions)
        return conditioned_structures

class SelfConditioner(pl.LightningModule):
    """
    No conditioning, but add the position of the structure to itself to make it compatible with the denoiser.
    """
    def __init__(self, N_structures: int, **kwargs):
        super().__init__()
        self.N_structures = N_structures
    def forward(self, y: torch_geometric.data.Batch) -> list[torch.Tensor]:
        conditioned_structures = [y.pos for _ in range(self.N_structures)]  # Include current position
        return conditioned_structures

class MeanConditioner(pl.LightningModule):
    """
    Condition on the mean across time steps of positions and hidden states.
    For each atom and coordinate, averages across all T+1 structures (current + hidden states).
    """
    def __init__(self, N_structures: int, **kwargs):
        super().__init__()
        self.N_structures = N_structures
        
    def forward(self, y: torch_geometric.data.Batch) -> list[torch.Tensor]:
        # Start with current position
        all_positions = [y.pos]
        
        # Add all hidden states if they exist
        if hasattr(y, "hidden_state") and y.hidden_state is not None:
            all_positions.extend(y.hidden_state)
        
        # Stack all positions along a new dimension and compute mean across time steps
        # Shape: (T+1, N, 3) -> (N, 3) where T is number of hidden states
        stacked_positions = torch.stack(all_positions, dim=0)  # (T+1, N, 3)
        mean_positions = torch.mean(stacked_positions, dim=0)  # (N, 3)
        # mean center the mean positions
        dummy_graph = y.clone()
        dummy_graph.pos = mean_positions
        # mean center the mean positions
        mean_positions = mean_center(dummy_graph).pos
        # align the mean positions to the current positions
        aligned_mean_positions = kabsch_algorithm(mean_positions, y.pos, y.batch, y.num_graphs)

        # Return the mean repeated N_structures times
        conditioned_structures = [aligned_mean_positions for _ in range(self.N_structures)]
        
        return conditioned_structures

class DenoisedConditioner(pl.LightningModule):
    """
    Conditioner that uses a pretrained denoiser to denoise hidden states.
    
    Takes hidden states, unscales them using c_in, denoises each structure,
    then recenters and aligns them to the current noisy positions.
    """
    def __init__(self, N_structures: int, pretrained_model_path: str, c_in: float, **kwargs):
        super().__init__()
        self.N_structures = N_structures
        self.c_in = c_in
        self.pretrained_model_path = pretrained_model_path

        # Load the pretrained denoiser
        py_logger = logging.getLogger("jamun")
        py_logger.info(f"Loading pretrained denoiser from wandb run: {pretrained_model_path}")
        
        # Find the checkpoint for the wandb run
        checkpoint_path = find_checkpoint(
            wandb_train_run_path=pretrained_model_path,
            checkpoint_type="best_so_far"
        )
        
        # Load the denoiser from checkpoint
        self.pretrained_denoiser = Denoiser.load_from_checkpoint(checkpoint_path, strict=False)
        self.pretrained_denoiser.eval()  # Set to evaluation mode
        
        # Freeze the pretrained model parameters
        for param in self.pretrained_denoiser.parameters():
            param.requires_grad = False
        
        # Extract sigma from the pretrained denoiser
        self.denoiser_sigma = self._extract_sigma_from_denoiser()
        py_logger.info(f"Extracted sigma from pretrained denoiser: {self.denoiser_sigma}")
        py_logger.info(f"Successfully loaded pretrained denoiser with c_in={c_in}")
        
    def _extract_sigma_from_denoiser(self) -> float:
        """Extract sigma value from the pretrained denoiser's sigma distribution."""
        sigma_distribution = self.pretrained_denoiser.sigma_distribution
        
        # Handle different types of sigma distributions
        if hasattr(sigma_distribution, 'sigma'):
            # For ConstantSigma distribution
            return float(sigma_distribution.sigma)
        elif hasattr(sigma_distribution, 'mean'):
            # For other distributions that might have a mean
            return float(sigma_distribution.mean)
        else:
            # Fallback - sample from the distribution
            sample = sigma_distribution.sample()
            return float(sample)
            
    def forward(self, y: torch_geometric.data.Batch) -> list[torch.Tensor]:
        """
        Forward pass that denoises hidden states and returns conditioned structures.
        
        Args:
            y: Batch containing current positions and hidden states
            
        Returns:
            List of tensors: [y.pos, *denoised_hidden_states]
        """
        # Use the sigma from the pretrained denoiser
        sigma_to_use = self.denoiser_sigma
        
        conditioned_structures = [y.pos]  # Start with current position
        
        # Check if we have hidden states to process
        if not hasattr(y, "hidden_state") or y.hidden_state is None:
            # If no hidden states, just repeat current position
            conditioned_structures.extend([y.pos for _ in range(self.N_structures - 1)])
            return conditioned_structures
            
        # # Move pretrained denoiser to same device as input
        # device = y.pos.device
        # self.pretrained_denoiser = self.pretrained_denoiser.to(device)
        
        # Process each hidden state
        for i, hidden_positions in enumerate(y.hidden_state):
            # Unscale the hidden state positions
            unscaled_positions = hidden_positions / self.c_in
            
            # Create a batch for denoising
            denoising_batch = y.clone()
            denoising_batch.pos = unscaled_positions
            
            # Remove hidden states from the denoising batch to avoid recursion
            if hasattr(denoising_batch, "hidden_state"):
                delattr(denoising_batch, "hidden_state")
            
            # Denoise the unscaled positions using the denoiser's sigma
            with torch.no_grad():
                denoised_batch = self.pretrained_denoiser.xhat(denoising_batch, sigma_to_use)
                denoised_positions = denoised_batch.pos
            
            # Align the denoised positions to the current noisy positions
            aligned_positions = kabsch_algorithm(denoised_positions, y.pos, y.batch, y.num_graphs)
            
            conditioned_structures.append(aligned_positions)
            
            # Break if we've processed enough structures
            if len(conditioned_structures) >= self.N_structures:
                break
                
        # If we don't have enough hidden states, pad with the last denoised structure
        while len(conditioned_structures) < self.N_structures:
            conditioned_structures.append(conditioned_structures[-1])
            
        return conditioned_structures


class ConditionerSpiked(Conditioner):
    """
    A conditioner that concatenates hidden states with the clean structure.
    
    The conditioning order is:
    1. Hidden states (y.hidden_state) - if present
    2. Clean structure positions (x_clean.pos) - if provided at the end
    """
    
    def __init__(self, N_structures: int, **kwargs):
        super().__init__(N_structures, **kwargs)
    
    def forward(self, y: torch_geometric.data.Batch, x_clean: torch_geometric.data.Batch = None) -> list[torch.Tensor]:
        """
        Create conditioning structures by concatenating hidden states with clean structure.
        
        Args:
            y: The noisy sample batch containing positions and hidden states
            x_clean: The clean sample batch containing ground truth positions
            
        Returns:
            List of tensors to be concatenated for conditioning
        """
        conditioned_structures = [y.pos]
        
        # Add hidden states if they exist
        if hasattr(y, "hidden_state") and y.hidden_state is not None:
            for hidden_pos in y.hidden_state:
                conditioned_structures.append(hidden_pos)
        
        # Add clean structure positions at the end if provided
        if x_clean is not None:
            conditioned_structures.pop(-1)
            conditioned_structures.append(x_clean.pos)
        
        return conditioned_structures


class SpatioTemporalConditioner(pl.LightningModule):
    """
    Conditioner that uses a spatio-temporal model to process hidden states.
    
    This conditioner takes the current positions and hidden states, processes them
    through a spatio-temporal model, and returns a single conditioned structure.
    Always returns exactly one structure regardless of N_structures parameter.
    
    By default, the spatiotemporal model is trainable. Set freeze_spatiotemporal_model=True
    to freeze the parameters (e.g., when using a pretrained model).
    """
    
    def __init__(
        self, 
        N_structures: int,
        spatiotemporal_model: torch.nn.Module,
        c_noise: float = 0.0,
        freeze_spatiotemporal_model: bool = False,
        **kwargs
    ):
        """
        Initialize the SpatioTemporalConditioner.
        
        Args:
            N_structures: Number of structures parameter (ignored - this conditioner always returns 1 structure)
            spatiotemporal_model: The E3SpatioTemporal model to use for processing
            c_noise: Noise conditioning parameter
            freeze_spatiotemporal_model: Whether to freeze spatiotemporal model parameters
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__()
        self.N_structures = N_structures
        self.spatiotemporal_model = spatiotemporal_model
        self.c_noise = c_noise
        self.freeze_spatiotemporal_model = freeze_spatiotemporal_model
        
        # Only freeze parameters if explicitly requested
        if self.freeze_spatiotemporal_model:
            self.freeze_spatiotemporal_parameters()
            # Set to evaluation mode when frozen
            self.spatiotemporal_model.eval()
    
    def freeze_spatiotemporal_parameters(self):
        """Freeze the spatiotemporal model parameters."""
        for param in self.spatiotemporal_model.parameters():
            param.requires_grad = False
            
    def unfreeze_spatiotemporal_parameters(self):
        """Unfreeze the spatiotemporal model parameters."""
        for param in self.spatiotemporal_model.parameters():
            param.requires_grad = True
    
    def configure_for_inference(self):
        """Configure the conditioner for inference (freeze parameters and set eval mode)."""
        self.freeze_spatiotemporal_model = True
        self.freeze_spatiotemporal_parameters()
        self.spatiotemporal_model.eval()
    
    def configure_for_training(self):
        """Configure the conditioner for training (unfreeze parameters and set train mode)."""
        self.freeze_spatiotemporal_model = False
        self.unfreeze_spatiotemporal_parameters()
        self.spatiotemporal_model.train()
        
    def forward(self, y: torch_geometric.data.Batch) -> list[torch.Tensor]:
        """
        Forward pass that processes the batch through the spatio-temporal model.
        
        Args:
            y: Batch containing current positions and hidden states
            
        Returns:
            List containing a single conditioned structure tensor (always length 1)
        """
        # Prepare noise conditioning
        device = y.pos.device
        sigma = torch.tensor(self.c_noise, device=device)
        sigma = unsqueeze_trailing(sigma, 1)
        
        # Process through spatio-temporal model
        # Only disable gradients if the model is frozen
        if self.freeze_spatiotemporal_model:
            with torch.no_grad():
                spatial_features = self.spatiotemporal_model(y, sigma)
        else:
            # Allow gradients to flow when training
            spatial_features = self.spatiotemporal_model(y, sigma)
        
        # The spatiotemporal model returns spatial features, not positions
        # We need to use these features to condition the structure
        # For now, we'll use the current position as the conditioned structure
        # In a more sophisticated implementation, we might use the features
        # to modify the positions or use them in other ways
        conditioned_position = y.pos
            
        # Return list with single conditioned structure (always length 1)
        return [conditioned_position]