"""
E(3)-equivariant spatio-temporal models and conversion functions.

This module contains:
- E3Transformer: E(3)-equivariant transformer for temporal graph processing
- E3SpatioTemporal: Unified spatio-temporal processing model
- Spatial-temporal graph conversion utilities
"""

from typing import Dict, Union

import e3nn
import torch
import torch.nn as nn
from e3nn import o3
import torch_geometric
import torch_geometric.data
import e3tools
import e3tools.nn
import logging
from jamun.model.arch.e3conv import E3Conv


def calculate_temporal_positions(temporal_length, mode="linear", device=None):
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
    
    if mode == "linear":
        # Create positions [0, 1, 2, ..., T-1] and normalize by T
        positions = torch.arange(temporal_length, dtype=torch.float32, device=device)
        normalized_positions = positions / temporal_length
    elif mode == "zeros":
        # Create positions [0, 1, 2, ..., T-1] and normalize by T
        positions = torch.arange(temporal_length, dtype=torch.float32, device=device)
        positions = torch.zeros_like(positions)
    
    return normalized_positions


def spatial_to_temporal_graphs(batch, temporal_position_mode="linear"):
    """
    Convert a batch of spatial graphs to temporal graphs.
    
    For each spatial node with position + hidden states, create a temporal graph where:
    - Node 0: current position
    - Nodes 1-T: hidden state positions  
    - Connectivity: Node 0 connects to all others, sequential connections 1->2->3->...
    """
    # Get device from input batch
    device = batch.pos.device
    
    # Get dimensions
    num_spatial_nodes = batch.pos.shape[0]
    
    # Check if we have hidden states
    if hasattr(batch, 'hidden_state') and batch.hidden_state is not None and len(batch.hidden_state) > 0:
        num_hidden_states = len(batch.hidden_state)
        temporal_length = 1 + num_hidden_states  # current + hidden
    else:
        # If no hidden states, just use current position
        num_hidden_states = 0
        temporal_length = 1
    
    # Store reference to spatial graph
    # spatial_graph = batch.clone()
    
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
        temporal_position = calculate_temporal_positions(temporal_length, mode=temporal_position_mode, device=device)
        
        # Create edge connectivity
        if temporal_length > 1:
            # Node 0 connects to all others: 0->1, 0->2, 0->3, ..., 0->T-1
            hub_src = [0] * (temporal_length - 1)
            hub_dst = list(range(1, temporal_length))
            
            # Sequential connections: 1->2, 2->3, ..., (T-2)->(T-1)
            seq_src = list(range(1, temporal_length - 1))
            seq_dst = list(range(2, temporal_length))
            
            # Combine all edges
            all_src = hub_src + seq_src
            all_dst = hub_dst + seq_dst
            
            edge_index = torch.tensor([all_src, all_dst], dtype=torch.long, device=device)
        else:
            # Single node, no edges
            edge_index = torch.tensor([[], []], dtype=torch.long, device=device)
        
        # Create temporal graph for this spatial node
        temporal_graph = torch_geometric.data.Data(
            pos=temporal_pos,
            edge_index=edge_index,
            spatial_node_idx=torch.tensor([node_idx], dtype=torch.long, device=device),  # Track which spatial node this came from
            temporal_length=torch.tensor([temporal_length], dtype=torch.long, device=device),
            temporal_position=temporal_position  # Normalized position in sequence [0, 1/T, 2/T, ...]
        )
        temporal_graphs.append(temporal_graph)
    
    # Batch all temporal graphs
    temporal_batch = torch_geometric.data.Batch.from_data_list(temporal_graphs)
    
    # Store spatial graph reference
    # temporal_batch.spatial_graph = spatial_graph
    
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


class E3Transformer(nn.Module):
    """E(3)-equivariant transformer with temporal graph support."""

    def __init__(
        self,
        irreps_out: Union[str, e3nn.o3.Irreps],
        irreps_hidden: Union[str, e3nn.o3.Irreps],
        irreps_sh: Union[str, e3nn.o3.Irreps],
        irreps_node_attr: Union[str, e3nn.o3.Irreps],
        num_layers: int,
        edge_attr_dim: int,
        num_attention_heads: int,
        reduce: str | None = None,
        irreps_node_attr_temporal: Union[str, e3nn.o3.Irreps] = "1x1e",
        radial_edge_attr_encoding_function: str = "gaussian",
        node_attr_temporal_encoding_function: str = "gaussian",
        edge_attr_temporal_encoding_function: str = "gaussian",
    ):
        super().__init__()

        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr) # input irreps
        self.irreps_node_attr_temporal = o3.Irreps(irreps_node_attr_temporal)
        self.num_layers = num_layers
        self.edge_attr_dim = edge_attr_dim
        self.num_attention_heads = num_attention_heads
        self.reduce = reduce
        self.sh = o3.SphericalHarmonics(
            irreps_out=self.irreps_sh, normalize=True, normalization="component"
        )
        # Split edge attribute dimensions: radial and temporal (bondedness is optional)
        self.radial_edge_attr_dim = self.edge_attr_dim // 2
        self.temporal_edge_attr_dim = self.edge_attr_dim - self.radial_edge_attr_dim
        self.temporal_node_attr_dim = self.irreps_node_attr_temporal.dim
        # Optional bondedness embedding (only used if bond_mask exists in graph)
        self.embed_bondedness = nn.Embedding(2, self.edge_attr_dim // 3)
        self.edge_attr_temporal_encoding_function = edge_attr_temporal_encoding_function
        self.node_attr_temporal_encoding_function = node_attr_temporal_encoding_function
        self.radial_edge_attr_encoding_function = radial_edge_attr_encoding_function
        # Gate for combining node attributes with temporal position
        # Input: node_attr (from data) + temporal_position (1x0e scalar)
        # irreps_with_temporal = self.irreps_node_attr + o3.Irreps("1x0e")
        irreps_with_temporal = self.irreps_node_attr + self.irreps_node_attr_temporal
        self.temporal_gate = e3tools.nn.GateWrapper(irreps_in=irreps_with_temporal, \
                                                    irreps_out=self.irreps_hidden, \
                                                    irreps_gate=irreps_with_temporal,)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                e3tools.nn.TransformerBlock(
                    irreps_in=self.irreps_hidden,
                    irreps_out=self.irreps_hidden,
                    irreps_sh=self.irreps_sh,
                    edge_attr_dim=self.edge_attr_dim,
                    num_heads=self.num_attention_heads,
                )
            )
        self.output_head = e3tools.nn.EquivariantMLP(
            irreps_in=self.irreps_hidden,
            irreps_out=self.irreps_out,
            irreps_hidden_list=[self.irreps_hidden],
        )

    def forward(
        self,
        node_attr: torch.Tensor,
        temporal_graph: torch_geometric.data.Batch,
        effective_radial_cutoff: float,
        temporal_cutoff: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass of the E3Transformer model."""
        # Extract graph data
        pos = temporal_graph.pos
        edge_index = temporal_graph.edge_index
        temporal_position = temporal_graph.temporal_position
        batch = temporal_graph.batch
        num_graphs = temporal_graph.num_graphs

        src, dst = edge_index
        edge_vec = pos[src] - pos[dst]
        edge_sh = self.sh(edge_vec)

        # Compute edge attributes: radial and temporal
        if self.radial_edge_attr_encoding_function != "ones":
            radial_edge_attr = e3nn.math.soft_one_hot_linspace(
                edge_vec.norm(dim=1),
                0.0,
                temporal_cutoff,
                self.radial_edge_attr_dim,
                basis=self.radial_edge_attr_encoding_function,
                cutoff=True,
            )
        else:
            radial_edge_attr = e3nn.math.soft_one_hot_linspace(
                edge_vec.norm(dim=1),
                0.0,
                temporal_cutoff,
                self.radial_edge_attr_dim,
                basis="gaussian",
                cutoff=True,
            )
            radial_edge_attr = torch.ones_like(radial_edge_attr)
        
        # Temporal edge attributes from temporal_position differences
        temporal_edge_vec = temporal_position[src] - temporal_position[dst]
        if self.edge_attr_temporal_encoding_function != "ones":
            temporal_edge_attr = e3nn.math.soft_one_hot_linspace(
                temporal_edge_vec.abs(),  # Use absolute difference
                0.0,
                2.0,
                self.temporal_edge_attr_dim,
                basis=self.edge_attr_temporal_encoding_function,
                cutoff=True,
            )
        else:
            temporal_edge_attr = e3nn.math.soft_one_hot_linspace(
                temporal_edge_vec.abs(),  # Use absolute difference
                0.0,
                2.0,
                self.temporal_edge_attr_dim,
                basis="gaussian",
                cutoff=True,
            )
            temporal_edge_attr = torch.ones_like(temporal_edge_attr)

        # temporal_edge_attr = torch.ones_like(temporal_edge_attr) # TODO: remove this, this is hacking. 

        # Optional bondedness (if bond_mask exists in the temporal graph)
        if hasattr(temporal_graph, 'bond_mask') and temporal_graph.bond_mask is not None:
            bonded_edge_attr = self.embed_bondedness(temporal_graph.bond_mask)
            edge_attr = torch.cat((bonded_edge_attr, radial_edge_attr, temporal_edge_attr), dim=-1)
        else:
            edge_attr = torch.cat((radial_edge_attr, temporal_edge_attr), dim=-1)

        # Process node attributes with temporal gating

        # Concatenate node_attr with temporal_position (scalar)
        if self.node_attr_temporal_encoding_function != "ones":
            temporal_position = e3nn.math.soft_one_hot_linspace(
                temporal_position,  # Use absolute difference
                0.0, # time always starts at 0
                1.0, # time always ends at 1
                self.temporal_node_attr_dim,
                basis=self.node_attr_temporal_encoding_function,
                cutoff=True,
            )
        else:
            temporal_position = e3nn.math.soft_one_hot_linspace(
                temporal_position,  # Use absolute difference
                0.0, # time always starts at 0
                1.0, # time always ends at 1
                self.temporal_node_attr_dim,
                basis="gaussian",
                cutoff=True,
            )
            temporal_position = torch.ones_like(temporal_position)
        temporal_position_expanded = temporal_position  # [N, 1] for concatenation
        node_attr_with_temporal = torch.cat([node_attr, temporal_position_expanded], dim=-1)
        
        # Apply temporal gate
        node_attr_processed = self.temporal_gate(node_attr_with_temporal)

        # Perform message passing with gated node attributes
        for layer in self.layers:
            node_attr_processed = layer(node_attr_processed, edge_index, edge_attr, edge_sh)
        node_attr_processed = self.output_head(node_attr_processed)

        # Pool over nodes.
        if self.reduce is not None:
            node_attr_processed = e3tools.scatter(
                node_attr_processed,
                index=batch,
                dim=0,
                dim_size=num_graphs,
                reduce=self.reduce,
            )
        
        return node_attr_processed


class E3SpatioTemporal(nn.Module):
    """
    E(3)-equivariant spatio-temporal model that combines spatial and temporal processing.
    
    This model implements the complete workflow:
    1. Process input spatial graph and hidden states through spatial module
    2. Pool spatial features to temporal graph representation
    3. Process temporal graph through temporal module  
    4. Pool temporal features back to spatial representation
    5. Convert temporal graph back to spatial graph
    """
    
    def __init__(
        self,
        spatial_module: nn.Module,
        temporal_module: nn.Module,
        spatial_to_temporal_pooler: nn.Module,
        temporal_to_spatial_pooler: nn.Module,
        radial_cutoff: float,
        temporal_cutoff: float = 1.0,
    ):
        """
        Initialize the E3SpatioTemporal model.
        
        Args:
            spatial_module: Module for processing spatial positions (e.g., E3Conv)
            temporal_module: Module for processing temporal graphs (e.g., E3Transformer)
            spatial_to_temporal_pooler: Module to convert spatial-temporal features to temporal node attributes
            temporal_to_spatial_pooler: Module to convert temporal features back to spatial features
            radial_cutoff: Cutoff for spatial radial edge weights
            temporal_cutoff: Cutoff for temporal edge weights
        """
        super().__init__()
        
        self.spatial_module = spatial_module
        self.temporal_module = temporal_module
        self.spatial_to_temporal_pooler = spatial_to_temporal_pooler
        self.temporal_to_spatial_pooler = temporal_to_spatial_pooler
        self.radial_cutoff = radial_cutoff
        self.temporal_cutoff = temporal_cutoff
    
    
    def forward(
        self,
        batch: torch_geometric.data.Batch,
        c_noise: torch.Tensor,
        return_temporal_features: bool = False,
        return_temporal_graph: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass implementing the complete spatio-temporal workflow.
        
        Args:
            batch: Input spatial graph batch with pos, batch, num_graphs, and optionally hidden_state
            c_noise: Noise conditioning tensor
            return_temporal_features: Whether to return intermediate temporal features
            return_temporal_graph: Whether to return the temporal graph
            
        Returns:
            If return_temporal_features or return_temporal_graph is True, returns dict with:
                - 'spatial_features': Final spatial features
                - 'spatial_graph': Output spatial graph
                - 'temporal_features': Temporal features (if requested)
                - 'temporal_graph': Temporal graph (if requested)
            Otherwise returns just the final spatial features tensor
        """
        # Store original device
        device = batch.pos.device
        
        # Step 1: Convert spatial graph to temporal graphs
        temporal_batch = spatial_to_temporal_graphs(batch)
        
        # Step 2: Process all positions (current + hidden states) with spatial module
        # Create topology for spatial processing (without positions)
        topology = batch.clone()
        # Remove position-dependent attributes but keep graph structure
        if hasattr(topology, 'pos'):
            del topology.pos
        if hasattr(topology, 'batch'):
            del topology.batch  
        if hasattr(topology, 'num_graphs'):
            del topology.num_graphs
            
        node_attr_list = []
        
        # Process current positions
        node_attr_current = self.spatial_module(
            pos=batch.pos, 
            topology=topology, 
            batch=batch.batch,
            num_graphs=batch.num_graphs,
            c_noise=c_noise,
            effective_radial_cutoff=self.radial_cutoff
        ).unsqueeze(1)  # [N, 1, features]
        node_attr_list.append(node_attr_current)
        
        # Process hidden state positions if they exist
        if hasattr(batch, 'hidden_state') and batch.hidden_state is not None and len(batch.hidden_state) > 0:
            for hidden_pos in batch.hidden_state:
                node_attr_hidden = self.spatial_module(
                    pos=hidden_pos,
                    topology=topology,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    c_noise=c_noise,
                    effective_radial_cutoff=self.radial_cutoff
                ).unsqueeze(1)  # [N, 1, features]
                node_attr_list.append(node_attr_hidden)
        
        # Step 3: Stack spatial-temporal features
        node_attr_spatial_temporal = torch.cat(node_attr_list, dim=1)  # [N, T, features]
        
        # Step 4: Convert spatial-temporal features to temporal node attributes
        temporal_node_attr = self.spatial_to_temporal_pooler(node_attr_spatial_temporal, temporal_batch)
        
        # Step 5: Process temporal graph through temporal module
        temporal_output = self.temporal_module(
            temporal_node_attr,
            temporal_batch,
            self.radial_cutoff,
            self.temporal_cutoff
        )
        
        # Step 6: Pool temporal features back to spatial features
        spatial_features = self.temporal_to_spatial_pooler(temporal_output, temporal_batch)
        
        # Step 7: Convert temporal graph back to spatial graph
        # output_spatial_graph = temporal_to_spatial_graphs(temporal_batch)
        output_spatial_graph = batch 

        # Prepare return values
        if return_temporal_features or return_temporal_graph:
            result = {
                'spatial_features': spatial_features,
                'spatial_graph': output_spatial_graph,
            }
            if return_temporal_features:
                result['temporal_features'] = temporal_output
            if return_temporal_graph:
                result['temporal_graph'] = temporal_batch
            return result
        else:
            return spatial_features
            
    def get_spatial_output_irreps(self):
        """Get the irreps of the spatial module output."""
        if hasattr(self.spatial_module, 'irreps_out'):
            return self.spatial_module.irreps_out
        else:
            raise AttributeError("Spatial module does not have irreps_out attribute")
            
    def get_temporal_output_irreps(self):
        """Get the irreps of the temporal module output.""" 
        if hasattr(self.temporal_module, 'irreps_out'):
            return self.temporal_module.irreps_out
        else:
            raise AttributeError("Temporal module does not have irreps_out attribute") 