import torch
import torch.nn as nn
import torch.nn.functional as F


import e3nn.o3

from jamun.model.noise_conditioning import NoiseConditionalScaling


class NoiseConditionalSkipConnection(nn.Module):
    """Noise-conditional skip connection for tensors."""

    def __init__(self, irreps_in: e3nn.o3.Irreps, noise_input_dims: int = 1):
        super().__init__()
        self.weights = NoiseConditionalScaling(irreps_in, noise_input_dims=noise_input_dims)
        self.irreps_in = irreps_in
        self.irreps_out = irreps_in

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        weights = self.weights.compute_scales(c_noise)
        weights = F.sigmoid(weights)
        x1 = (1 - weights) * x1 + weights * x2
        return x1


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
