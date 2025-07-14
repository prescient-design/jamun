import logging
from typing import Callable, Dict, Optional, Tuple, Union

import lightning.pytorch as pl
import numpy as np
import torch
import torch_geometric
from e3tools import radius_graph, scatter

from jamun.utils import align_A_to_B_batched, mean_center, unsqueeze_trailing
from jamun.utils.align import kabsch_algorithm

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
        
        # Return the mean repeated N_structures times
        conditioned_structures = [mean_positions for _ in range(self.N_structures)]
        
        return conditioned_structures 