import torch
import torch.nn as nn
import e3nn


class GateActivation(nn.Module):
    """A useful wrapper for the e3nn gate activation function."""

    def __init__(self, irreps_in: e3nn.o3.Irreps, irreps_out: e3nn.o3.Irreps, irreps_gate: e3nn.o3.Irreps):
        super().__init__()
        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.irreps_out = e3nn.o3.Irreps(irreps_out)
        self.irreps_gate = e3nn.o3.Irreps(irreps_gate)

        act = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sigmoid,
            -1: torch.tanh,
        }
        irreps_scalars = e3nn.o3.Irreps([(mul, ir) for mul, ir in self.irreps_gate if ir.l == 0])
        irreps_gated = e3nn.o3.Irreps([(mul, ir) for mul, ir in self.irreps_gate if ir.l > 0])
        irreps_gates = e3nn.o3.Irreps([(mul, "0e") for mul, _ in irreps_gated])

        self.gate = e3nn.nn.Gate(
            irreps_scalars,
            [act[ir.p] for _, ir in irreps_scalars],  # scalar
            irreps_gates,
            [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
            irreps_gated,
        )
        self.pre_gate = e3nn.o3.Linear(self.irreps_in, self.gate.irreps_in)
        self.post_gate = e3nn.o3.Linear(self.gate.irreps_out, self.irreps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_gate(x)
        x = self.gate(x)
        x = self.post_gate(x)
        return x
