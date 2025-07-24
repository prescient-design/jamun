import torch
import torch_geometric
from e3tools import scatter


def mean_center(x: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
    """Mean centers the positions."""
    x = x.clone("pos")
    mean_pos = scatter(x.pos, x.batch, dim=0, dim_size=x.num_graphs, reduce="mean")
    x.pos = x.pos - mean_pos[x.batch]
    return x


def mean_center_f(pos: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Mean centers the positions."""
    mean_pos = scatter(pos, batch, dim=0, dim_size=num_graphs, reduce="mean")
    pos = pos - mean_pos[batch]
    return pos
