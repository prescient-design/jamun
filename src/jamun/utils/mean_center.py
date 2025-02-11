import functools
import torch_geometric
import torch_scatter
import torch


@functools.partial(torch.compile, dynamic=True)
def mean_center(y: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
    """Mean centers the positions."""
    mean_pos = torch_scatter.scatter_mean(y.pos, y.batch, dim=0, dim_size=y.num_graphs)
    y.pos -= mean_pos[y.batch]
    return y
