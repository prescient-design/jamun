from functools import cache

import e3tools
import numpy as np
import torch
import torch_geometric

from jamun.utils import align_A_to_B_batched_looped_f, mean_center_f


@cache
def compute_normalization_factors(
    sigma: float | torch.Tensor,
    *,
    average_squared_distance: float,
    normalization_type: str | None,
    sigma_data: float | None = None,
    D: int = 3,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the normalization factors for the input, skip connection, output, and noise."""
    sigma = torch.as_tensor(sigma, device=device)

    if normalization_type is None:
        c_in = torch.as_tensor(1.0, device=device)
        c_skip = torch.as_tensor(0.0, device=device)
        c_out = torch.as_tensor(1.0, device=device)
        c_noise = torch.as_tensor(sigma, device=device)
        return c_in, c_skip, c_out, c_noise

    if normalization_type == "EDM":
        c_skip = (sigma_data**2) / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
        c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
        c_noise = torch.log(sigma / sigma_data) * 0.25
        return c_in, c_skip, c_out, c_noise

    if normalization_type == "JAMUN":
        A = torch.as_tensor(average_squared_distance, device=device)
        B = torch.as_tensor(2 * D * sigma**2, device=device)

        c_in = 1.0 / torch.sqrt(A + B)
        c_skip = A / (A + B)
        c_out = torch.sqrt((A * B) / (A + B))
        c_noise = torch.log(sigma) / 4
        return c_in, c_skip, c_out, c_noise

    raise ValueError(f"Unknown normalization type: {normalization_type}")


def add_edges(
    y: torch.Tensor,
    topology: torch_geometric.data.Batch,
    batch: torch.Tensor,
    radial_cutoff: float,
) -> torch_geometric.data.Batch:
    """Add edges to the graph based on the effective radial cutoff."""
    if topology.get("edge_index") is not None:
        return topology

    with torch.cuda.nvtx.range("clone_topology"):
        topology = topology.clone()

    with torch.cuda.nvtx.range("radial_graph"):
        radial_edge_index = e3tools.radius_graph(y, radial_cutoff, batch)

    # print(f"Number of radial edges: {radial_edge_index.shape[1]}")
    # print(f"Number of bonded edges: {topology.bonded_edge_index.shape[1]}")

    with torch.cuda.nvtx.range("concatenate_edges"):
        edge_index = torch.cat((radial_edge_index, topology.bonded_edge_index), dim=-1)
        if topology.bonded_edge_index.numel() == 0:
            bond_mask = torch.zeros(radial_edge_index.shape[1], dtype=torch.long, device=y.device)
        else:
            bond_mask = torch.cat(
                (
                    torch.zeros(radial_edge_index.shape[1], dtype=torch.long, device=y.device),
                    torch.ones(topology.bonded_edge_index.shape[1], dtype=torch.long, device=y.device),
                ),
                dim=0,
            )

    topology.edge_index = edge_index
    topology.bond_mask = bond_mask
    return topology


def compute_rmsd_metrics(
    *,
    x: torch.Tensor,
    xhat: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    sigma: float | torch.Tensor,
    mean_center: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute the loss."""
    if mean_center:
        with torch.cuda.nvtx.range("mean_center_x"):
            x = mean_center_f(x, batch, num_graphs)

    with torch.cuda.nvtx.range("align_xhat"):
        xhat_aligned = align_A_to_B_batched_looped_f(
            xhat,
            x,
            batch,
            num_graphs,
        )

    # Compute the raw loss.
    with torch.cuda.nvtx.range("raw_coordinate_loss"):
        raw_coordinate_loss = (xhat - x).pow(2).sum(dim=-1)
        raw_coordinate_loss_aligned = (xhat_aligned - x).pow(2).sum(dim=-1)

    # Take the mean over each graph.
    with torch.cuda.nvtx.range("mean_over_graphs"):
        mse = e3tools.scatter(raw_coordinate_loss, batch, dim=0, dim_size=num_graphs, reduce="mean")
        mse_aligned = e3tools.scatter(raw_coordinate_loss_aligned, batch, dim=0, dim_size=num_graphs, reduce="mean")

    # Compute the scaled RMSD.
    with torch.cuda.nvtx.range("scaled_rmsd"):
        rmsd = torch.sqrt(mse)
        rmsd_aligned = torch.sqrt(mse_aligned)

        D = xhat.shape[-1]
        scaled_rmsd = rmsd / (sigma * np.sqrt(D))

    return {
        "mse": mse,
        "mse_aligned": mse_aligned,
        "rmsd": rmsd,
        "rmsd_aligned": rmsd_aligned,
        "scaled_rmsd": scaled_rmsd,
    }
