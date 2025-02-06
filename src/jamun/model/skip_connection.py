import e3nn.o3
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableSkipConnection(nn.Module):
    """Learnable skip connection for tensors."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.as_tensor(0.01))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = (1 - self.weight) * x1 + self.weight * x2
        return x1


class ResidueLearnableSkipConnection(nn.Module):
    """Learnable skip connection for residue."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.as_tensor(0.01))

    def forward(self, residue1: torch.Tensor, residue2: torch.Tensor) -> torch.Tensor:
        residue1["features"] = (1 - self.weight) * residue1["features"] + self.weight * residue2["features"]
        return residue1
