import e3nn.o3
import torch
import torch.nn as nn
import torch.nn.functional as F

from jamun.utils import unsqueeze_trailing


class NoiseEmbedding(nn.Module):
    """Noise embedding for tensors."""

    def __init__(self, output_dims: int, num_layers: int = 2):
        super().__init__()
        self.noise_embedder = nn.Sequential()
        self.noise_embedder.append(nn.Linear(1, output_dims))
        for _ in range(num_layers):
            self.noise_embedder.append(nn.SELU())
            self.noise_embedder.append(nn.Linear(output_dims, output_dims))

        self.irreps_in = e3nn.o3.Irreps("1x0e")
        self.irreps_out = e3nn.o3.Irreps(f"{output_dims}x0e")

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        return self.noise_embedder(c_noise.reshape(-1, 1))


class NoiseConditionalScaling(nn.Module):
    """Noise-conditional scaling for tensors."""

    def __init__(self, irreps_in: e3nn.o3.Irreps, noise_input_dims: int = 1, num_layers: int = 1):
        super().__init__()

        self.scale_predictor = nn.Sequential()
        self.scale_predictor.append(nn.Linear(noise_input_dims, irreps_in.num_irreps))
        for _ in range(num_layers):
            self.scale_predictor.append(nn.SELU())
            self.scale_predictor.append(nn.Linear(irreps_in.num_irreps, irreps_in.num_irreps))

        # Initialize such that the scaling is all ones.
        with torch.no_grad():
            self.scale_predictor[-1].weight.fill_(0.0)
            self.scale_predictor[-1].bias.fill_(1.0)

        self.irreps_in = irreps_in
        self.irreps_out = irreps_in
        self.tp = e3nn.o3.ElementwiseTensorProduct(
            self.irreps_in, f"{self.irreps_in.num_irreps}x0e",
        )

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        c_noise = unsqueeze_trailing(c_noise, x.ndim - c_noise.ndim)
        scales = self.scale_predictor(c_noise)
        x = self.tp(x, scales)
        return x


class NoiseConditionalSkipConnection(nn.Module):
    """Noise-conditional skip connection for tensors."""

    def __init__(self, irreps_in: e3nn.o3.Irreps, noise_input_dims: int = 1):
        super().__init__()
        self.weights = NoiseConditionalScaling(irreps_in, noise_input_dims=noise_input_dims)
        self.irreps_in = irreps_in
        self.irreps_out = irreps_in
        self.tp = e3nn.o3.ElementwiseTensorProduct(
            self.irreps_in, f"{self.irreps_in.num_irreps}x0e",
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        weights = self.weights.scale_predictor(c_noise)
        weights = F.sigmoid(weights)
        x1 = self.tp(x1, weights) + self.tp(x2, 1 - weights)
        return x1