import torch
from e3nn.util.jit import compile_mode
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch_scatter import composite, scatter

from ._layer_norm import LayerNorm
from ._linear import Linear


@compile_mode("script")
class DTPByHead(torch.nn.Module):
    def __init__(self, nc_s_in, nc_v_in, nc_s_out, nc_v_out, num_heads) -> None:
        """
        depth-wise tensor product with a sh

        performs
        - DTP w/ provided weights or internal (uvu)
        - apply linear w/ bias
        """
        super().__init__()
        assert nc_s_in == nc_v_in
        self.nc_s_in = nc_s_in
        self.nc_v_in = nc_v_in
        self.num_heads = num_heads
        self.dim_post_dtp = 2 * nc_s_in
        self.weight_numel = 4 * nc_s_in * num_heads

        # determine tp out shapes
        self.nc_s_out = nc_s_out
        self.nc_v_out = nc_v_out

        # weights for linear
        # scalar
        w_s = torch.empty((num_heads, nc_s_out, self.dim_post_dtp))
        nn.init.xavier_uniform_(w_s, gain=1)
        self.w_s = torch.nn.Parameter(w_s)
        self.b_s = torch.nn.Parameter(torch.zeros((num_heads, nc_s_out)))
        # vector
        w_v = torch.empty((num_heads, nc_v_out, self.dim_post_dtp))
        nn.init.xavier_uniform_(w_v, gain=1)
        self.w_v = torch.nn.Parameter(w_v)

    def forward(self, s, v, r_ij_vec, weights):
        """reshaping is done at the input"""
        w_ss, w_sv, w_vs, w_vv = rearrange(weights, "ij (c h m) -> c h ij m", c=4, h=self.num_heads)

        # tp
        ss = w_ss * s
        sv = w_sv.unsqueeze(-1) * s.unsqueeze(-1) * r_ij_vec.unsqueeze(-2)
        vs = w_vs.unsqueeze(-1) * v
        vv = w_vv * (v * r_ij_vec.unsqueeze(-2)).sum(-1)
        s = rearrange([ss, vv], "c h ij m -> h ij (c m)")
        v = rearrange([sv, vs], "c h ij m k -> h ij (c m) k")

        # apply linear
        # z = ij
        s = torch.einsum("h m n, h z n -> h z m", self.w_s, s) + self.b_s[:, None, :]
        v = torch.einsum("h m n, h z n k -> h z m k", self.w_v, v)

        return s, v


@compile_mode("script")
class Equiformer(torch.nn.Module):
    """Implements Fig.1b of Equiformer"""

    def __init__(
        self,
        irreps_in,  # (nc scalar, nc vector)
        irreps_out,  # (nc scalar, nc vector)
        radial_nn,
        num_heads=1,
        apply_layer_norm=True,  # for both attn and ff
        apply_resnet=True,  # only concerns ff block
        ff_mul=3,
    ) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.num_heads = num_heads
        self.apply_layer_norm = apply_layer_norm
        if apply_layer_norm:
            self.layer_norm_attn = LayerNorm(self.irreps_in)
            self.layer_norm_ff = LayerNorm(self.irreps_in)
        self.apply_resnet = apply_resnet

        self.nc_s_in = nc_s_in = self.irreps_in[0]
        self.nc_v_in = nc_v_in = self.irreps_in[1]

        # ---- initial mixing
        # linear, group by head, and all vs all tensor prod
        self.linear_src = Linear(nc_s_in, nc_s_in, nc_v_in, nc_v_in, add_bias=True)
        self.linear_dst = Linear(nc_s_in, nc_s_in, nc_v_in, nc_v_in, add_bias=True)
        assert nc_v_in == nc_s_in
        # -- linear after tp
        self.nc_by_head = nc_s_in // num_heads  # ex: 8 = 32 / 4
        nc_middle = 2 * self.nc_by_head  # ex: 16
        nc_s_in_by_head = nc_v_in_by_head = 2 * (  # noqa: F841
            self.nc_by_head**2
        )  # combins all pairwise (and single); ex: 512 scalar
        w_s = torch.empty((num_heads, nc_middle, nc_s_in_by_head))
        nn.init.xavier_uniform_(w_s, gain=1)
        self.w_s_init = torch.nn.Parameter(w_s)
        self.b_s_init = torch.nn.Parameter(torch.zeros((num_heads, nc_middle)))
        # vector
        w_v = torch.empty((num_heads, nc_middle, nc_s_in_by_head))
        nn.init.xavier_uniform_(w_v, gain=1)
        self.w_v_init = torch.nn.Parameter(w_v)

        # ---- pre-attn dtp with sh
        nc_s_out_by_head = 3 * self.nc_by_head  # ex: 14
        nc_v_out_by_head = self.nc_by_head  # ex: 8
        self.pre_attn_dtp_linear = DTPByHead(nc_middle, nc_middle, nc_s_out_by_head, nc_v_out_by_head, num_heads)
        self.radialnn = radial_nn(num_out_features=self.pre_attn_dtp_linear.weight_numel)

        # ---- attn linear
        w_s = torch.empty((num_heads, self.nc_by_head, 2 * self.nc_by_head))
        nn.init.xavier_uniform_(w_s, gain=1)
        self.attn_msg_w_s = torch.nn.Parameter(w_s)
        self.attn_msg_b_s = torch.nn.Parameter(torch.zeros((num_heads, self.nc_by_head)))
        # vector
        w_v = torch.empty((num_heads, self.nc_by_head, 2 * self.nc_by_head))
        nn.init.xavier_uniform_(w_v, gain=1)
        self.attn_msg_w_v = torch.nn.Parameter(w_v)

        # ---- attn weight
        self.attn_weight_relu = torch.nn.LeakyReLU(0.1)
        w = torch.empty((num_heads, self.nc_by_head))
        nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("leaky_relu", 0.1))
        self.attn_w = torch.nn.Parameter(w)

        # ---- attn final linear
        self.linear_attn_final = Linear(nc_s_in, nc_s_in, nc_v_in, nc_v_in, add_bias=True)

        # ---- feed-forward
        # ff1 -> gate -> ff2
        self.ff_mul = ff_mul
        self.nc_s_out = nc_s_out = self.irreps_out[0]
        self.nc_v_out = nc_v_out = self.irreps_out[1]
        assert nc_v_out > 0, "assume there will always be at least one vector output"
        # -- comput v norms: (nc_s, nc_v) -> (nc_s + nc_v, nc_v)
        # -- ff1: (nc_s + nc_v, nc_v) -> (m * nc_s out + m * nc_v out, m * nc_v out)
        self.ff1 = Linear(
            nc_s_in,  #  + nc_v_in,
            ff_mul * nc_s_out + ff_mul * nc_v_out,
            nc_v_in,
            ff_mul * nc_v_out,
            add_bias=True,
        )
        # -- gate: (m * nc_s out + m * nc_v out, m * nc_v out) -> (m * nc_s out, m * nc_v out)
        # -- ff2: (m * nc_s out, m * nc_v out) -> (nc_s out, nc_v out)
        self.ff2 = Linear(ff_mul * nc_s_out, nc_s_out, ff_mul * nc_v_out, nc_v_out, add_bias=True)

    def forward(self, s, v, edges_ij, r_ij, r_ij_vec, src, dst, weight_cutoff=None, ts=None):
        """
        args:
            N = len(src)
            edges [N]: precomputed residue num diff embedding
            r_ij [N]
            r_ij_vec [N, 3]
            s, v [N_CG]
        """
        # ---- attn module
        s0, v0 = s, v  # for skip

        if self.apply_layer_norm:
            s, v = self.layer_norm_attn(s, v)

        # ---- initial mixing
        # i is the dst/query, which gets first dim
        s_i, v_i = self.linear_dst(s, v)
        s_i, v_i = s_i[dst], v_i[dst]
        s_j, v_j = self.linear_src(s, v)
        s_j, v_j = s_j[src], v_j[src]
        s_i = rearrange(s_i, "i (h m)   -> h  i  m ()", h=self.num_heads)
        v_i = rearrange(v_i, "i (h m) k -> h  i  m () k", h=self.num_heads)
        s_j = rearrange(s_j, "j (h n)   -> h  j ()  n", h=self.num_heads)
        v_j = rearrange(v_j, "j (h n) k -> h  j ()  n k", h=self.num_heads)

        # all vs all tensor prod
        ss = rearrange(s_i * s_j, "h ij m n -> h ij (m n)")
        sv = rearrange(s_i.unsqueeze(-1) * v_j, "h ij m n k -> h ij (m n) k")
        vs = rearrange(v_i * s_j.unsqueeze(-1), "h ij m n k -> h ij (m n) k")
        vv = rearrange((v_i * v_j).sum(-1), "h ij m n -> h ij (m n)")
        s_ij = torch.cat([ss, vv], dim=-1)
        v_ij = torch.cat([sv, vs], dim=-2)
        # ss_sum = rearrange(s_i + s_j, "h ij m n -> h ij (m n)")
        # vv_sum = rearrange(v_i + v_j, "h ij m n k -> h ij (m n) k")
        # s_ij = torch.cat([ss, vv, ss_sum], dim=-1)
        # v_ij = torch.cat([sv, vs, vv_sum], dim=-2)

        # linear
        # z = ij
        s_ij = torch.einsum("h m n, h z n -> h z m", self.w_s_init, s_ij) + self.b_s_init[:, None, :]
        v_ij = torch.einsum("h m n, h z n k -> h z m k", self.w_v_init, v_ij)

        # ---- pre attn dtp with sh
        weights = self.radialnn(r_ij, edges_ij, ts)
        s_ij, v_ij = self.pre_attn_dtp_linear(s_ij, v_ij, r_ij_vec, weights)
        # s_ij h ij m
        # v_ij h ij m k

        # split (grouped by head)
        s_ij0, gate_v, s_ij = rearrange(s_ij, "h ij (c m) -> c h ij m", c=3)

        # -- compute messages
        # gate
        s_ij = F.silu(s_ij)
        v_ij = torch.sigmoid(gate_v).unsqueeze(-1) * v_ij
        # tp; r_ij_vec (ij k)
        ss = s_ij
        sv = s_ij.unsqueeze(-1) * r_ij_vec[None, :, None, :]
        vs = v_ij
        vv = torch.einsum("h z m k, z k -> h z m", [v_ij, r_ij_vec])
        s = rearrange([ss, vv], "c h ij m -> h ij (c m)")
        v = rearrange([sv, vs], "c h ij m k -> h ij (c m) k")
        # apply linear
        s_ij = torch.einsum("h m n, h z n -> h z m", self.attn_msg_w_s, s) + self.attn_msg_b_s[:, None, :]
        v_ij = torch.einsum("h m n, h z n k -> h z m k", self.attn_msg_w_v, v)

        # -- compute attn score
        z_ij = torch.einsum("h n, h z n -> h z", self.attn_w, s_ij0)
        z_ij = weight_cutoff * F.softplus(z_ij)  # zero attention on far away
        a_ij = composite.scatter_softmax(z_ij, dst, dim=1, dim_size=len(s0))
        # print(scatter(a_ij, dst, dim=1, dim_size=len(s0)))
        # assert False

        # -- combine
        # print(s_ij.size(), a_ij.size())
        s = scatter(a_ij[:, :, None] * s_ij, dst, dim=1, dim_size=len(s0))
        s = rearrange(s, "h i m -> i (h m)")
        v = scatter(a_ij[:, :, None, None] * v_ij, dst, dim=1, dim_size=len(s0))
        v = rearrange(v, "h i m k -> i (h m) k")
        s, v = self.linear_attn_final(s, v)

        # skip
        s = s0 + s
        v = v0 + v

        # ---- ff module
        if self.apply_resnet:
            s0, v0 = s, v  # for skip

        if self.apply_layer_norm:
            s, v = self.layer_norm_ff(s, v)

        # -- norm
        # todo: eliminate this?
        # v_norm = (nodes["v"].square().sum(-1) + 1e-6).sqrt() # [N, nc_v]

        # -- ff1
        # s = torch.cat([s, v_norm], dim=1)
        s, v = self.ff1(s, v)

        # -- gate
        if self.nc_s_out > 0:
            offset = self.ff_mul * self.nc_s_out  # for scalar
            gate_v = s[:, offset:]
            s = F.silu(s[:, :offset])
        else:
            gate_v = s
            s = None
        v = torch.sigmoid(gate_v).unsqueeze(-1) * v

        # -- ff2
        s, v = self.ff2(s, v)

        if self.apply_resnet:
            s = s0 + s
            v = v0 + v

        return s, v
