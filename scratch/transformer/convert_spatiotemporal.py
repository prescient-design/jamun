#!/usr/bin/env python3
"""
Functions for converting between spatial and temporal graph representations.
"""

import torch
import torch_geometric


def calculate_temporal_positions(temporal_length, device=None):
    """
    Calculate normalized temporal positions for nodes in a temporal graph.

    Args:
        temporal_length: Total number of nodes in the temporal sequence
        device: Device to create tensors on

    Returns:
        torch.Tensor: Normalized positions [0, 1/T, 2/T, ..., (T-1)/T]
    """
    if temporal_length <= 1:
        return torch.tensor([0.0], device=device)

    # Create positions [0, 1, 2, ..., T-1] and normalize by T
    positions = torch.arange(temporal_length, dtype=torch.float32, device=device)
    normalized_positions = positions / temporal_length

    return normalized_positions


def spatial_to_temporal_graphs(batch, graph_type="fan"):
    """
    Convert a batch of spatial graphs to temporal graphs with configurable connectivity.

    For each spatial node with position + hidden states, create a temporal graph where:
    - Node 0: current position
    - Nodes 1-T: hidden state positions
    - Connectivity depends on graph_type parameter

    Args:
        batch: Input spatial graph batch
        graph_type: Type of connectivity to use
            - "fan": Hub connects to all + sequential connections (0->all, i->(i+1))
            - "hub_n_spoke": Only hub-spoke connections (0->all, no sequential)
            - "complete": Complete graph with self-loops (all-to-all including self)
            - "complete_no_self": Complete graph without self-loops (all-to-all excluding self)
    """

    # Validate graph_type
    valid_types = ["fan", "hub_n_spoke", "complete", "complete_no_self"]
    if graph_type not in valid_types:
        raise ValueError(f"graph_type must be one of {valid_types}, got {graph_type}")

    # Get device from input batch
    device = batch.pos.device

    # Get dimensions
    num_spatial_nodes = batch.pos.shape[0]

    # Check if we have hidden states
    if hasattr(batch, "hidden_state") and batch.hidden_state is not None and len(batch.hidden_state) > 0:
        num_hidden_states = len(batch.hidden_state)
        temporal_length = 1 + num_hidden_states  # current + hidden
    else:
        # If no hidden states, just use current position
        num_hidden_states = 0
        temporal_length = 1

    # print(f"Creating {graph_type} temporal graphs: {num_spatial_nodes} spatial nodes -> {num_spatial_nodes} temporal graphs of length {temporal_length}")

    # Store reference to spatial graph
    spatial_graph = batch.clone()

    # Set connectivity type code for tracking
    connectivity_type_map = {"fan": 0, "hub_n_spoke": 1, "complete": 2, "complete_no_self": 3}

    temporal_graphs = []

    for node_idx in range(num_spatial_nodes):
        # Build temporal positions: [current_pos, hidden_1, hidden_2, ...]
        temporal_positions = [batch.pos[node_idx]]  # Start with current position

        # Add hidden state positions
        if num_hidden_states > 0:
            for hidden_pos in batch.hidden_state:
                temporal_positions.append(hidden_pos[node_idx])

        temporal_pos = torch.stack(temporal_positions)  # Shape: [T, 3]

        # Calculate temporal positions for this sequence
        temporal_position = calculate_temporal_positions(temporal_length, device=device)

        # Create edge connectivity based on graph_type
        if temporal_length > 1:
            if graph_type == "fan":
                # Original fan system: hub-spoke + sequential
                # Hub connections: 0->1, 0->2, 0->3, ..., 0->T-1
                hub_src = [0] * (temporal_length - 1)
                hub_dst = list(range(1, temporal_length))

                # Sequential connections: 1->2, 2->3, ..., (T-2)->(T-1)
                seq_src = list(range(1, temporal_length - 1))
                seq_dst = list(range(2, temporal_length))

                # Combine all edges
                all_src = hub_src + seq_src
                all_dst = hub_dst + seq_dst

                edge_index = torch.tensor([all_src, all_dst], dtype=torch.long, device=device)

            elif graph_type == "hub_n_spoke":
                # Hub-and-spoke only: 0 connects to all others, no sequential
                hub_src = [0] * (temporal_length - 1)
                hub_dst = list(range(1, temporal_length))

                edge_index = torch.tensor([hub_src, hub_dst], dtype=torch.long, device=device)

            elif graph_type == "complete":
                # Complete graph without self-loops: all-to-all excluding self
                src_nodes = []
                dst_nodes = []

                for i in range(temporal_length):
                    for j in range(temporal_length):
                        if i != j:  # Exclude self-loops
                            src_nodes.append(i)
                            dst_nodes.append(j)

                edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long, device=device)

        else:
            # Single node case
            if graph_type == "complete":
                # Single node with self-loop
                edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
            else:
                # Single node, no edges for other types
                edge_index = torch.tensor([[], []], dtype=torch.long, device=device)

        # Create temporal graph for this spatial node
        temporal_graph = torch_geometric.data.Data(
            pos=temporal_pos,
            edge_index=edge_index,
            spatial_node_idx=torch.tensor([node_idx], device=device),  # Track which spatial node this came from
            temporal_length=torch.tensor([temporal_length], device=device),
            temporal_position=temporal_position,  # Normalized position in sequence [0, 1/T, 2/T, ...]
            connectivity_type=torch.tensor([connectivity_type_map[graph_type]], device=device),
            graph_type=graph_type,  # Store graph type as string for debugging
        )
        temporal_graphs.append(temporal_graph)

    # Batch all temporal graphs
    temporal_batch = torch_geometric.data.Batch.from_data_list(temporal_graphs)

    # Store spatial graph reference and graph type
    temporal_batch.spatial_graph = spatial_graph
    temporal_batch.graph_type = graph_type

    return temporal_batch


def temporal_to_spatial_graphs(temporal_batch):
    """
    Convert temporal graphs back to spatial graphs.
    Take the 0th node position from each temporal graph as the updated spatial position.
    """
    # Get the spatial graph template
    spatial_graph = temporal_batch.spatial_graph.clone()

    # Extract 0th node positions from each temporal graph
    num_temporal_graphs = temporal_batch.num_graphs
    updated_positions = []

    # Iterate through each temporal graph in the batch
    for graph_idx in range(num_temporal_graphs):
        # Get the node range for this temporal graph
        start_idx = temporal_batch.ptr[graph_idx]

        # The 0th node of each temporal graph is at the start of its range
        updated_positions.append(temporal_batch.pos[start_idx])

    # Stack to create new position tensor
    updated_positions = torch.stack(updated_positions)

    # Update spatial graph with new positions
    spatial_graph.pos = updated_positions

    return spatial_graph
