import torch_geometric.transforms
import torch_geometric.data
import torch


class RadiusGraph(torch_geometric.transforms.BaseTransform):
    """Transform that adds edges to a graph based on a radius cutoff."""

    def __init__(self, **kwargs):
        self.transform_kwargs = kwargs

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        pos = data.pos
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=pos.device)
        radial_edge_index = torch_geometric.nn.radius_graph(pos, batch=batch, **self.transform_kwargs)

        bonded_edge_index = data.edge_index
        edge_index = torch.cat((radial_edge_index, bonded_edge_index), dim=-1)
        bond_mask = torch.cat(
            (
                torch.zeros(radial_edge_index.shape[1], dtype=torch.long, device=pos.device),
                torch.ones(bonded_edge_index.shape[1], dtype=torch.long, device=pos.device),
            ),
            dim=0,
        )
        data.edge_index = edge_index
        data.bond_mask = bond_mask
        return data


