import torch
from e3nn.util.jit import compile_mode
from torch import nn


@compile_mode("script")
class Linear(torch.nn.Module):
    def __init__(self, nc_s_in, nc_s_out, nc_v_in, nc_v_out, add_bias=False) -> None:
        super().__init__()
        self.nc_s_in = nc_s_in
        self.nc_s_out = nc_s_out
        self.nc_v_in = nc_v_in
        self.nc_v_out = nc_v_out

        assert (nc_v_out > 0) or (nc_s_out > 0)

        if nc_s_out > 0:
            w_s = torch.empty((nc_s_out, nc_s_in))
            nn.init.xavier_uniform_(w_s, gain=1)
            self.w_s = torch.nn.Parameter(w_s)
            self.add_bias = add_bias
            if self.add_bias:
                self.b_s = torch.nn.Parameter(torch.zeros(nc_s_out))

        if nc_v_out > 0:
            w_v = torch.empty((nc_v_out, nc_v_in))
            nn.init.xavier_uniform_(w_v, gain=1)
            self.w_v = torch.nn.Parameter(w_v)

    def forward(self, s, v):
        if self.nc_s_out > 0:
            s = torch.einsum("ij,...j->...i", self.w_s, s)
            if self.add_bias:
                if len(s.size()) == 2:
                    s = s + self.b_s[None, :]
                elif len(s.size()) == 3:
                    s = s + self.b_s[None, None, :]
                else:
                    raise NotImplementedError
        else:
            s = None
        v = torch.einsum("ij,...jk->...ik", self.w_v, v) if self.nc_v_out > 0 else None

        return s, v
