import e3nn.o3
import torch
import torch.nn as nn

from jamun.utils import unsqueeze_trailing

# TODO: fix
torch._dynamo.config.capture_dynamic_output_shape_ops = True

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
        self.repeats = torch.concatenate([torch.as_tensor(ir.dim).repeat(mul) for mul, ir in self.irreps_out])

    def compute_scales(self, c_noise: torch.Tensor) -> torch.Tensor:
        scales = self.scale_predictor(c_noise)
        scales = scales.repeat_interleave(self.repeats.to(scales.device), dim=-1)
        return scales

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        c_noise = unsqueeze_trailing(c_noise, x.ndim - c_noise.ndim)
        scales = self.compute_scales(c_noise)
        x = x * scales
        return x
