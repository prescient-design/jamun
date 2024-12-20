import e3nn.o3
import torch
from e3nn.util.jit import compile_mode


@compile_mode("script")
class LayerNorm(torch.nn.Module):
    """LayerNorm as in Equiformer."""

    def __init__(self, irreps: e3nn.o3.Irreps):
        super().__init__()
        self.irreps = irreps
        self.gamma_s = torch.nn.Parameter(torch.ones(self.irreps[0]))
        self.beta_s = torch.nn.Parameter(torch.zeros(self.irreps[0]))
        self.gamma_v = torch.nn.Parameter(torch.ones(self.irreps[1]))

    def forward(self, s, v):
        # -- scalar
        x = s
        # subtact mean
        mu = x.mean(dim=1, keepdim=True)
        x = x - mu
        # normalize and rms
        square_norm_x = x.square()
        rms = (square_norm_x.mean(dim=1) + 1e-6).sqrt()  # [N]
        # apply params
        s = self.gamma_s[None, :] * x / rms[:, None] + self.beta_s[None, :]

        # -- vector
        x = v
        # normalize and rms
        square_norm_x = x.square()
        rms = (square_norm_x.sum(dim=[1, 2]) / self.irreps[1] + 1e-6).sqrt()  # [N]
        # apply params
        v = self.gamma_v[None, :, None] * x / rms[:, None, None]

        return s, v
