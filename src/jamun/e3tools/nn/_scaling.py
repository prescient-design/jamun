import torch
import e3nn


class ScaleIrreps(torch.nn.Module):
    """Scales each irrep by a weight."""

    def __init__(self, irreps_in: torch.Tensor):
        super().__init__()

        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.irreps_out = e3nn.o3.Irreps(irreps_in)
        self.repeats = torch.concatenate([torch.as_tensor(ir.dim).repeat(mul) for mul, ir in self.irreps_out])

    def forward(self, data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights_repeated = weights.repeat_interleave(self.repeats, dim=-1)
        return data * weights_repeated