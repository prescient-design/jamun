#!/usr/bin/env python3
"""
Helper functions for creating network architectures used in transformer development.
"""

import functools

import e3tools
import numpy as np
import torch
import torch_geometric
from convert_spatiotemporal import spatial_to_temporal_graphs

from jamun.model.arch.e3conv import E3Conv
from jamun.utils.average_squared_distance import compute_average_squared_distance


def compute_temporal_average_squared_distance_from_dataset(
    dataset, num_samples: int = 100, verbose: bool = False
) -> float:
    """
    Compute average squared distance between neighboring vertices in temporal graphs.

    Args:
        dataset: Dataset containing spatial graphs with hidden states
        num_samples: Number of samples to use for estimation
        verbose: Whether to print verbose output

    Returns:
        float: Average squared distance between temporal neighbors
    """

    avg_sq_dists = []
    num_graphs = 0

    # Follow pattern from average_squared_distance.py
    for item in dataset:
        if num_graphs >= num_samples:
            break
        for graph in item:
            if num_graphs >= num_samples:
                break
            # Convert to temporal graphs here
            temporal_batch = spatial_to_temporal_graphs(graph)
            temporal_graphs = torch_geometric.data.Batch.to_data_list(temporal_batch)
            graph_mean = 0.0
            num_nodes = graph.pos.shape[0]
            for temporal_graph in temporal_graphs:
                avg_sq_dist = compute_average_squared_distance(temporal_graph.pos, cutoff=None)
                graph_mean += avg_sq_dist / num_nodes
            avg_sq_dists.append(graph_mean)
            num_graphs += 1
        mean_avg_sq_dist = sum(avg_sq_dists) / num_graphs

    if verbose:
        print(f"Total graphs processed: {num_graphs}")
        print(f"Total temporal graphs processed: {len(avg_sq_dists)}")
        print(f"Mean average squared distance between temporal nodes: {mean_avg_sq_dist:.6f}")
        print(f"Standard deviation: {np.std(avg_sq_dists):.6f}")

    return float(mean_avg_sq_dist)


def add_edges(
    y: torch.Tensor,
    topology: torch_geometric.data.Batch,
    batch: torch.Tensor,
    radial_cutoff: float,
) -> torch_geometric.data.Batch:
    """Add edges to the graph based on the effective radial cutoff."""
    if topology.get("edge_index") is not None:
        return topology

    topology = topology.clone()
    with torch.cuda.nvtx.range("radial_graph"):
        radial_edge_index = e3tools.radius_graph(y, radial_cutoff, batch)

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


def apply_e3conv_to_positions(e3conv_model, pos, topology, batch, effective_radial_cutoff=5.0):
    """
    Apply E3Conv model to a set of positions using existing graph topology.

    Args:
        e3conv_model: E3Conv model instance
        pos (torch.Tensor): Positions [N, 3]
        topology (torch_geometric.data.Batch): Existing graph topology from dataloader
        batch (torch.Tensor): Batch tensor from the graph
        effective_radial_cutoff (float): Radial cutoff for edges

    Returns:
        torch.Tensor: Node features [N, feature_dim]
    """
    # Clone topology to avoid modifying original
    topology_with_edges = topology.clone()

    # Add edges using the local add_edges function
    topology_with_edges = add_edges(pos, topology_with_edges, batch, effective_radial_cutoff)

    # Use noise conditioning of 0.0 (no noise)
    c_noise = torch.zeros(pos.shape[0], dtype=pos.dtype, device=pos.device)

    # Apply E3Conv
    num_graphs = batch.max().item() + 1  # Number of graphs in the batch
    node_features = e3conv_model(
        pos=pos,
        topology=topology_with_edges,
        batch=batch,
        num_graphs=num_graphs,
        c_noise=c_noise,
        effective_radial_cutoff=effective_radial_cutoff,
    )

    return node_features


def create_e3conv_network():
    """
    Create an E3Conv network with parameters matching the yaml configuration.

    Returns:
        E3Conv: Configured E3Conv network
    """

    # Hidden layer factory as specified in yaml
    hidden_layer_factory = functools.partial(e3tools.nn.ConvBlock, conv=functools.partial(e3tools.nn.Conv))

    # Output head factory as specified in yaml
    output_head_factory = functools.partial(
        e3tools.nn.EquivariantMLP,
        irreps_hidden_list=["120x0e + 32x1e"],  # Using irreps_hidden from yaml
    )

    # Create E3Conv with exact parameters from yaml
    e3conv = E3Conv(
        irreps_out="1x1e",  # 3D vector output
        irreps_hidden="120x0e + 32x1e",  # Hidden representations
        irreps_sh="1x0e + 1x1e",  # Spherical harmonics
        hidden_layer_factory=hidden_layer_factory,
        output_head_factory=output_head_factory,
        use_residue_information=True,  # Assuming True, matches yaml ${data.use_residue_information}
        n_layers=1,  # Number of layers
        edge_attr_dim=64,  # Edge attribute dimension
        atom_type_embedding_dim=8,  # Atom type embedding
        atom_code_embedding_dim=8,  # Atom code embedding
        residue_code_embedding_dim=32,  # Residue code embedding
        residue_index_embedding_dim=8,  # Residue index embedding
        use_residue_sequence_index=False,  # As specified in yaml
        num_atom_types=20,  # Number of atom types
        max_sequence_length=10,  # Max sequence length
        num_atom_codes=10,  # Number of atom codes
        num_residue_types=25,  # Number of residue types
        test_equivariance=False,  # Disable for production
        reduce=None,  # No reduction
    )

    return e3conv


def get_e3conv_output_irreps():
    """
    Get the output irreps of the E3Conv network.

    Returns:
        str: Output irreps string
    """
    return "1x1e"  # 3D vector output as specified in yaml
