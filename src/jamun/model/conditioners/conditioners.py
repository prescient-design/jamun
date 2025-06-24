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
    def __init__(self, N_structures: int, **kwargs):
        super().__init__()
        self.N_structures = N_structures
    def forward(self, y: torch_geometric.data.Batch) -> list[torch.Tensor]:
        conditioned_structures = []
        for positions in y.hidden_state: 
            aligned_positions = kabsch_algorithm(positions, y.pos, y.batch, y.num_graphs)
            conditioned_structures.append(aligned_positions)
        return conditioned_structures

class SelfConditioner(pl.LightningModule):
    """
    No conditioning, but add the position of the structure to itself to make it compatible with the denoiser.
    """
    def __init__(self, N_structures: int, **kwargs):
        super().__init__()
        self.N_structures = N_structures
    def forward(self, y: torch_geometric.data.Batch) -> list[torch.Tensor]:
        conditioned_structures = [y.pos for _ in range(self.N_structures-1)]
        return conditioned_structures 