import torch_geometric
import torch_scatter


def mean_center(y: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
    """Mean centers the positions."""
    mean_pos = torch_scatter.scatter_mean(y.pos, y.batch, dim=0)
    y.pos -= mean_pos[y.batch]
    return y
