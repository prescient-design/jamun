import math

import torch
from e3nn.util.jit import compile_mode
from torch.nn import functional as F


@compile_mode("script")
class MLP(torch.nn.Module):
    def __init__(self, num_neurons, activation, apply_layer_norm=False):
        super(MLP, self).__init__()
        self.activation = activation
        self.layers = torch.nn.ModuleList()
        self.apply_layer_norm = apply_layer_norm
        if apply_layer_norm:
            self.layer_norms = torch.nn.ModuleList()
            idx = 0
        for nin, nout in zip(num_neurons[:-1], num_neurons[1:]):
            self.layers.append(torch.nn.Linear(nin, nout, bias=True))
            if self.apply_layer_norm:
                if idx < len(num_neurons) - 1:
                    self.layer_norms.append(torch.nn.LayerNorm(nout))
                idx += 1

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
            if self.apply_layer_norm:
                x = self.layer_norms[i](x)
        x = self.layers[-1](x)

        return x


@compile_mode("script")
class BesselBasis(torch.nn.Module):
    def __init__(self, rc, radial_num_basis=16) -> None:
        super().__init__()
        self.rc = rc
        self.radial_num_basis = radial_num_basis
        self.prefactor = 2.0 / self.rc

        bessel_weights = torch.linspace(start=1.0, end=self.radial_num_basis, steps=self.radial_num_basis) * math.pi
        self.bessel_weights = torch.nn.Parameter(bessel_weights)

    def forward(self, x):
        if len(x.size()) == 2:
            arg = self.bessel_weights[None, None, :] * x[:, :, None]
        elif len(x.size()) == 1:
            arg = self.bessel_weights[None, :] * x[:, None]
        else:
            raise ValueError

        # in the preprint, no div by input
        return self.prefactor * torch.sin(arg / self.rc)  # / x.unsqueeze(-1)


class SinuisoidalBasis(torch.nn.Module):
    """supply sinuisoidal basis at float value x in [0, xmax]"""

    def __init__(self, xmax, d=32) -> None:  # number of basis
        super().__init__()
        self.xmax = xmax
        assert (d % 2) == 0
        self.d = d
        self.prefactor = 2.0 / self.xmax
        self.register_buffer("weights", torch.linspace(start=1.0, end=d // 2, steps=d // 2) * math.pi)

    def forward(self, x):
        if len(x.size()) == 2:
            arg = self.weights[None, None, :] * x[:, :, None]
        elif len(x.size()) == 1:
            arg = self.weights[None, :] * x[:, None]
        else:
            raise ValueError

        arg = arg / self.xmax

        return self.prefactor * torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


@compile_mode("script")
class RadialNN(torch.nn.Module):
    def __init__(
        self,
        num_out_features,
        rc,
        radial_num_basis=16,
        radial_num_hidden=16,
        radial_num_layers=2,
        include_edge_features=False,
        include_time_features=False,
        num_edge_features=None,
        num_time_features=None,
        basis_type="bessel",
    ) -> None:
        super().__init__()
        self.num_out_features = num_out_features
        self.rc = rc
        self.radial_num_basis = radial_num_basis
        self.radial_num_hidden = radial_num_hidden
        self.radial_num_layers = radial_num_layers
        self.include_edge_features = include_edge_features
        self.include_time_features = include_time_features
        self.num_edge_features = num_edge_features
        self.num_time_features = num_time_features
        if self.include_edge_features:
            assert isinstance(num_edge_features, int)
        else:
            self.num_edge_features = 0

        if self.include_time_features:
            assert isinstance(num_time_features, int)
        else:
            self.num_time_features = 0

        # ---- MLP
        num_input_features = self.radial_num_basis + self.num_edge_features + self.num_time_features
        self.mlp = MLP(
            [num_input_features] + [self.radial_num_hidden] * self.radial_num_layers + [self.num_out_features],
            F.silu,
        )
        # ---- bassel basis
        self.basis_type = basis_type
        assert basis_type in ["bessel", "sinusoidal"]
        if basis_type == "bessel":
            self.basis = BesselBasis(self.rc, self.radial_num_basis)
        elif basis_type == "sinusoidal":
            self.basis = SinuisoidalBasis(self.rc, self.radial_num_basis)

    def forward(self, r_ij, edges_ij, ts=None):
        # compute basis
        inputs = [self.basis(r_ij)]

        # combine edge and time features, as needed
        if self.include_edge_features:
            inputs.append(edges_ij)

        if self.include_time_features:
            inputs.append(ts)

        inputs = torch.cat(inputs, dim=-1)
        weight = self.mlp(inputs)

        return weight
