import torch_geometric
from e3tools import scatter


def mean_center(x: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
    """Mean centers the positions."""
    x = x.clone("pos")
    mean_pos = scatter(x.pos, x.batch, dim=0, dim_size=x.num_graphs, reduce="mean")
    x.pos -= mean_pos[x.batch]
    return x
