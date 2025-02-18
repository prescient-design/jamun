import functools
import torch_geometric
import torch_scatter
import torch


def mean_center(x: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
    """Mean centers the positions."""
    x = x.clone("pos")
    mean_pos = torch_scatter.scatter_mean(x.pos, x.batch, dim=0, dim_size=x.num_graphs)
    x.pos -= mean_pos[x.batch]
    return x
