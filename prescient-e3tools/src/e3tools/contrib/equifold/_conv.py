import torch
from e3nn.util.jit import compile_mode
from einops import rearrange
from torch.nn import functional as F
from torch_scatter import scatter

from ._linear import Linear


@compile_mode("script")
class Convnet(torch.nn.Module):
    def __init__(
        self,
        irreps_in,  # (nc scalar, nc vector)
        irreps_out,  # (nc scalar, nc vector)
        radial_nn,
        div_factor=0.0,
        apply_resnet=True,  # dummy # FIXME
    ) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.div_factor = div_factor

        self.nc_s_in = nc_s = self.irreps_in[0]
        self.nc_v_in = nc_v = self.irreps_in[1]
        self.nc_s = nc_s
        assert nc_s == nc_v
        self.nc_s_out = self.irreps_out[0]
        self.nc_v_out = self.irreps_out[1]

        # could hypers-opt biases
        self.linear1 = Linear(2 * nc_s, 2 * nc_s, 2 * nc_s, nc_s, add_bias=True)
        self.linear2 = Linear(2 * nc_s, 2 * nc_s, 2 * nc_s, nc_s, add_bias=True)
        self.linear3 = Linear(nc_s, self.nc_s_out, nc_v, self.nc_v_out, add_bias=True)
        self.linear_self = Linear(nc_s, self.nc_s_out, nc_v, self.nc_v_out, add_bias=False)

        self.radial_nn1, self.radial_nn2 = [radial_nn(num_out_features=4 * nc_s) for _ in range(2)]

    def forward(self, s, v, edges_ij, r_ij, r_ij_vec, src, dst, weight_cutoff=None, ts=None):
        """
        args:
            N = len(src)
            edges [N]: precomputed residue num diff embedding
            r_ij [N]
            r_ij_vec [N, 3]
            s, v [N_CG]
        """
        s0, v0 = s, v  # for skip

        # ---- tp among tensors
        s1, v1 = s[dst], v[dst]
        s2, v2 = s[src], v[src]
        # could have more linears here
        ss = s1 * s2
        vv = (v1 * v2).sum(dim=-1)
        sv = s1.unsqueeze(-1) * v2
        vs = v1 * s2.unsqueeze(-1)
        s = torch.cat([ss, vv], dim=1)
        v = torch.cat([sv, vs], dim=1)
        # multiply by weight
        weights = self.radial_nn1(r_ij, edges_ij, ts)
        w_s, w_v = rearrange(weights, "r (c m) -> c r m", c=2)
        s = w_s * s
        v = w_v.unsqueeze(-1) * v
        s, v = self.linear1(s, v)
        s, s_gate = s[:, : self.nc_s], s[:, self.nc_s :]
        s = F.silu(s)
        v = torch.sigmoid(s_gate).unsqueeze(-1) * v

        # ---- tp with sh
        ss = s
        vv = (v * r_ij_vec.unsqueeze(-2)).sum(dim=-1)
        sv = s.unsqueeze(-1) * r_ij_vec.unsqueeze(-2)
        vs = v
        s = torch.cat([ss, vv], dim=1)
        v = torch.cat([sv, vs], dim=1)
        # multiply by weight
        weights = self.radial_nn2(r_ij, edges_ij, ts)
        w_s, w_v = rearrange(weights, "r (c m) -> c r m", c=2)
        s = w_s * s
        v = w_v.unsqueeze(-1) * v
        s, v = self.linear2(s, v)
        s, s_gate = s[:, : self.nc_s], s[:, self.nc_s :]
        s = F.silu(s)
        v = torch.sigmoid(s_gate).unsqueeze(-1) * v

        # ---- reduction
        s = scatter(s, dst, dim=0, dim_size=len(s0)) / self.div_factor
        v = scatter(v, dst, dim=0, dim_size=len(s0)) / self.div_factor
        s, v = self.linear3(s, v)

        # --- self-interaction and resnet
        s0, v0 = self.linear_self(s0, v0)
        s = s0 + s if s0 is not None else None
        v = v0 + v if v0 is not None else None

        return s, v
