#!/usr/bin/env python3
"""
Lightning modules for converting node attributes between spatial and temporal representations.
"""

import torch
import torch_geometric
import pytorch_lightning as pl


class SpatialToTemporalNodeAttr(pl.LightningModule):
    """
    Lightning module to transfer node attributes from spatial nodes to temporal nodes 
    by repeating first temporal feature.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, spatial_node_attr_temporal, temporal_batch):
        """
        Transfer node attributes from spatial nodes to temporal nodes by repeating first temporal feature.
        Takes the first temporal feature (t=0) and repeats it T times for each spatial node.
        
        Args:
            spatial_node_attr_temporal (torch.Tensor): Node attributes [N_spatial, T, attr_dim]
            temporal_batch (torch_geometric.data.Batch): Batch of temporal graphs
            
        Returns:
            torch.Tensor: Node attributes for temporal nodes [N_temporal, attr_dim]
        """
        num_spatial_nodes, temporal_length, attr_dim = spatial_node_attr_temporal.shape
        num_temporal_graphs = temporal_batch.num_graphs
        
        # Verify consistency
        assert num_spatial_nodes == num_temporal_graphs, \
            f"Mismatch: {num_spatial_nodes} spatial nodes vs {num_temporal_graphs} temporal graphs"
        
        # Verify temporal length consistency
        expected_temporal_nodes = temporal_batch.pos.shape[0]
        expected_total_nodes = num_spatial_nodes * temporal_length
        assert expected_total_nodes == expected_temporal_nodes, \
            f"Temporal length mismatch: {expected_total_nodes} vs {expected_temporal_nodes}"
        
        # Extract first temporal feature (t=0) and repeat it T times for each spatial node
        first_temporal_features = spatial_node_attr_temporal[:, 0, :]  # [N, attr_dim]
        
        # Repeat each spatial node's first temporal feature T times
        temporal_node_attr = first_temporal_features.repeat_interleave(temporal_length, dim=0)  # [N*T, attr_dim]
        
        # Verify the output shape matches the temporal batch
        assert temporal_node_attr.shape[0] == expected_temporal_nodes, \
            f"Output shape mismatch: {temporal_node_attr.shape[0]} vs expected {expected_temporal_nodes}"
        
        return temporal_node_attr


class TemporalToSpatialNodeAttr(pl.LightningModule):
    """
    Lightning module to convert temporal node attributes back to spatial node attributes.
    Takes the first temporal node attribute from each temporal graph.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, temporal_node_attr, temporal_batch):
        """
        Convert temporal node attributes back to spatial node attributes.
        Takes the first temporal node attribute from each temporal graph.
        
        Args:
            temporal_node_attr (torch.Tensor): Node attributes for temporal nodes [N_temporal, attr_dim]
            temporal_batch (torch_geometric.data.Batch): Batch of temporal graphs
            
        Returns:
            torch.Tensor: Node attributes for spatial nodes [N_spatial, attr_dim]
        """
        num_temporal_graphs = temporal_batch.num_graphs
        attr_dim = temporal_node_attr.shape[1]
        
        # Extract the first node attribute from each temporal graph
        spatial_node_attr = []
        
        for graph_idx in range(num_temporal_graphs):
            # Get the node range for this temporal graph
            start_idx = temporal_batch.ptr[graph_idx]
            
            # The 0th node of each temporal graph is at the start of its range
            first_node_attr = temporal_node_attr[start_idx]
            spatial_node_attr.append(first_node_attr)
        
        # Stack to create spatial node attribute tensor
        spatial_node_attr = torch.stack(spatial_node_attr)
        
        # Verify output shape
        assert spatial_node_attr.shape == (num_temporal_graphs, attr_dim), \
            f"Output shape mismatch: {spatial_node_attr.shape} vs expected ({num_temporal_graphs}, {attr_dim})"
        
        return spatial_node_attr


class TemporalToSpatialNodeAttrMean(pl.LightningModule):
    """
    Lightning module to convert temporal node attributes back to spatial node attributes by averaging.
    Takes the mean of all temporal node attributes for each temporal graph.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, temporal_node_attr, temporal_batch):
        """
        Convert temporal node attributes back to spatial node attributes by averaging.
        Takes the mean of all temporal node attributes for each temporal graph.
        
        Args:
            temporal_node_attr (torch.Tensor): Node attributes for temporal nodes [N_temporal, attr_dim]
            temporal_batch (torch_geometric.data.Batch): Batch of temporal graphs
            
        Returns:
            torch.Tensor: Node attributes for spatial nodes [N_spatial, attr_dim]
        """
        num_temporal_graphs = temporal_batch.num_graphs
        attr_dim = temporal_node_attr.shape[1]
        
        # Extract the mean node attributes from each temporal graph
        spatial_node_attr = []
        
        for graph_idx in range(num_temporal_graphs):
            # Get the node range for this temporal graph
            start_idx = temporal_batch.ptr[graph_idx]
            end_idx = temporal_batch.ptr[graph_idx + 1] if graph_idx + 1 < len(temporal_batch.ptr) else len(temporal_node_attr)
            
            # Take the mean of all temporal nodes for this spatial node
            temporal_nodes_attr = temporal_node_attr[start_idx:end_idx]  # [temporal_length, attr_dim]
            mean_node_attr = temporal_nodes_attr.mean(dim=0)  # [attr_dim]
            spatial_node_attr.append(mean_node_attr)
        
        # Stack to create spatial node attribute tensor
        spatial_node_attr = torch.stack(spatial_node_attr)
        
        # Verify output shape
        assert spatial_node_attr.shape == (num_temporal_graphs, attr_dim), \
            f"Output shape mismatch: {spatial_node_attr.shape} vs expected ({num_temporal_graphs}, {attr_dim})"
        
        return spatial_node_attr


class SpatialTemporalToTemporalNodeAttr(pl.LightningModule):
    """
    Lightning module to convert spatial node attributes arranged temporally to temporal node attributes.
    Converts from [N, T, features] to [NT, features] with correct temporal graph ordering.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, spatial_node_attr_temporal, temporal_batch):
        """
        Convert spatial node attributes arranged temporally to temporal node attributes.
        Converts from [N, T, features] to [NT, features] with correct temporal graph ordering.
        
        Args:
            spatial_node_attr_temporal (torch.Tensor): Node attributes [N_spatial, T, attr_dim]
            temporal_batch (torch_geometric.data.Batch): Batch of temporal graphs for validation
            
        Returns:
            torch.Tensor: Node attributes for temporal nodes [N_temporal, attr_dim]
        """
        num_spatial_nodes, temporal_length, attr_dim = spatial_node_attr_temporal.shape
        num_temporal_graphs = temporal_batch.num_graphs
        
        # Verify consistency with temporal batch
        assert num_spatial_nodes == num_temporal_graphs, \
            f"Mismatch: {num_spatial_nodes} spatial nodes vs {num_temporal_graphs} temporal graphs"
        
        # Verify temporal length consistency
        expected_temporal_nodes = temporal_batch.pos.shape[0]
        expected_total_nodes = num_spatial_nodes * temporal_length
        assert expected_total_nodes == expected_temporal_nodes, \
            f"Temporal length mismatch: {expected_total_nodes} vs {expected_temporal_nodes}"
        
        # Reshape to match temporal graph ordering: [N, T, features] -> [N*T, features]
        # Temporal graph arranges nodes as: [node0_t0, node0_t1, ..., node0_tT-1, node1_t0, ...]
        temporal_node_attr = spatial_node_attr_temporal.reshape(num_spatial_nodes * temporal_length, attr_dim)
        
        # Verify the output shape matches the temporal batch
        assert temporal_node_attr.shape[0] == expected_temporal_nodes, \
            f"Output shape mismatch: {temporal_node_attr.shape[0]} vs expected {expected_temporal_nodes}"
        
        return temporal_node_attr


# Legacy function interfaces for backward compatibility
def spatial_to_temporal_node_attr(spatial_node_attr_temporal, temporal_batch):
    """Legacy function interface for backward compatibility."""
    module = SpatialToTemporalNodeAttr()
    return module(spatial_node_attr_temporal, temporal_batch)


def temporal_to_spatial_node_attr(temporal_node_attr, temporal_batch):
    """Legacy function interface for backward compatibility."""
    module = TemporalToSpatialNodeAttr()
    return module(temporal_node_attr, temporal_batch)


def temporal_to_spatial_node_attr_mean(temporal_node_attr, temporal_batch):
    """Legacy function interface for backward compatibility."""
    module = TemporalToSpatialNodeAttrMean()
    return module(temporal_node_attr, temporal_batch)


def spatial_temporal_to_temporal_node_attr(spatial_node_attr_temporal, temporal_batch):
    """Legacy function interface for backward compatibility."""
    module = SpatialTemporalToTemporalNodeAttr()
    return module(spatial_node_attr_temporal, temporal_batch) 