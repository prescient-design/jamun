#!/usr/bin/env python3
"""
Simple test script for the debugged denoiser_conditional using default hydra config.
Tests with sigma = 0.0 and sigma = 0.1.

Device Handling:
- This script manually handles CUDA device placement for standalone testing
- PyTorch Lightning DOES handle device placement automatically when using the Trainer
- In Lightning, you typically don't need to call .to(device) manually on models or data
- Lightning moves models to the specified device and handles data loading automatically
- For standalone scripts like this one, manual device handling is required
"""

import e3nn

e3nn.set_optimization_defaults(jit_script_fx=False)
import torch
import torch_geometric

# Import spatial-temporal conversion functions
from convert_spatiotemporal import spatial_to_temporal_graphs, temporal_to_spatial_graphs

# Import node attribute conversion functions
from pooling import SpatialTemporalToTemporalNodeAttr, TemporalToSpatialNodeAttrMean

from jamun.data import parse_datasets_from_directory
from jamun.utils import unsqueeze_trailing

# Setup device - use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")


def to_device(obj, device):
    """Helper function to move objects to device, handling various types."""
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list | tuple):
        return type(obj)(to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {key: to_device(value, device) for key, value in obj.items()}
    else:
        return obj


def move_graph_to_device(graph, device):
    """Move a PyTorch Geometric graph and all its tensor attributes to device."""
    # Move the graph using standard .to() method
    graph = graph.to(device)

    # Manually move any custom tensor attributes that might not be handled
    for attr_name in dir(graph):
        if not attr_name.startswith("_"):  # Skip private attributes
            attr_value = getattr(graph, attr_name, None)
            if isinstance(attr_value, torch.Tensor):
                setattr(graph, attr_name, attr_value.to(device))

    return graph


dataset = parse_datasets_from_directory(
    root="/data/bucket/kleinhej/capped_diamines/timewarp_splits/train",
    traj_pattern="^(.*).xtc",
    pdb_pattern="^(.*).pdb",
    filter_codes=["ALA_ALA"],
    as_iterable=False,
    subsample=80,
    total_lag_time=8,
    lag_subsample_rate=10,
    start_frame=800000,
    num_frames=200000,
)
# temporal_distance_cutoff = compute_temporal_average_squared_distance_from_datasets(dataset)
breakpoint()
# convert to dataloader, then pull data

graph = dataset[0].__getitem__(0)
batch = torch_geometric.data.Batch.from_data_list([graph])
from helpers import add_edges

batch = add_edges(batch.pos, batch, batch.batch, 0.05)

# Move batch data to device
batch = move_graph_to_device(batch, device)
print(f"Moved batch to device: {batch.pos.device}")

# Use E3Conv architecture for spatial feature processing
from e3nn import o3
from helpers import create_e3conv_network, get_e3conv_output_irreps

# Create E3Conv network with yaml configuration parameters
spatial_e3conv = create_e3conv_network()

# Move E3Conv model to device
spatial_e3conv = spatial_e3conv.to(device)
print(f"Moved E3Conv model to device: {next(spatial_e3conv.parameters()).device}")

print(f"E3Conv output irreps: {get_e3conv_output_irreps()}")
output_irreps = o3.Irreps(get_e3conv_output_irreps())
print(f"E3Conv output dimension: {output_irreps.dim}")

print("\n" + "=" * 50)
print("TEMPORAL GRAPH CONVERSION")
print("=" * 50)


def test_temporal_conversion(batch, graph_type="fan"):
    """Test the conversion functions with example output."""
    print("=== Testing Temporal Graph Conversion ===")

    print("Original spatial batch:")
    print(f"  - pos shape: {batch.pos.shape}")
    print(
        f"  - hidden_state length: {len(batch.hidden_state) if hasattr(batch, 'hidden_state') and batch.hidden_state else 0}"
    )
    if hasattr(batch, "hidden_state") and batch.hidden_state:
        print(f"  - hidden_state[0] shape: {batch.hidden_state[0].shape}")

    # Convert to temporal
    temporal_batch = spatial_to_temporal_graphs(batch)

    print("\nTemporal batch:")
    print(f"  - pos shape: {temporal_batch.pos.shape}")
    print(f"  - edge_index shape: {temporal_batch.edge_index.shape}")
    print(f"  - num_graphs: {temporal_batch.num_graphs}")
    print("  - example edge_index for first temporal graph:")

    # Show first temporal graph structure
    first_graph_end = temporal_batch.ptr[1] if temporal_batch.num_graphs > 1 else len(temporal_batch.pos)
    first_graph_edges = temporal_batch.edge_index[:, temporal_batch.edge_index[0] < first_graph_end]
    print(f"    {first_graph_edges}")

    print("\n  - COMPLETE edge_index for entire temporal batch:")
    print(f"    Shape: {temporal_batch.edge_index.shape}")
    print(f"    {temporal_batch.edge_index}")

    print(f"\n  - Temporal graph boundaries (ptr): {temporal_batch.ptr}")
    print("  - Graph node ranges:")
    for i in range(temporal_batch.num_graphs):
        start = temporal_batch.ptr[i]
        end = temporal_batch.ptr[i + 1] if i + 1 < len(temporal_batch.ptr) else len(temporal_batch.pos)
        print(f"    Graph {i}: nodes {start}-{end - 1} ({end - start} nodes)")

    print("\n  - Temporal positions for each graph:")
    print(f"    Shape: {temporal_batch.temporal_position.shape}")
    print(f"    First graph temporal_position: {temporal_batch.temporal_position[:5]}")  # Show first 5 positions
    print(f"    All temporal_position values: {temporal_batch.temporal_position}")

    # Convert back to spatial
    reconstructed_spatial = temporal_to_spatial_graphs(temporal_batch)

    print("\nReconstructed spatial:")
    print(f"  - pos shape: {reconstructed_spatial.pos.shape}")
    print(f"  - position difference from original: {torch.norm(reconstructed_spatial.pos - batch.pos)}")

    return temporal_batch, reconstructed_spatial


# Test the temporal conversion
temporal_batch, reconstructed_spatial = test_temporal_conversion(batch)

# Move temporal batch to device (should already be on correct device now)
temporal_batch = move_graph_to_device(temporal_batch, device)
reconstructed_spatial = move_graph_to_device(reconstructed_spatial, device)
print("Temporal batch device verification:")
print(f"  - pos device: {temporal_batch.pos.device}")
print(f"  - edge_index device: {temporal_batch.edge_index.device}")
print(f"  - batch device: {temporal_batch.batch.device}")
print(f"  - ptr device: {temporal_batch.ptr.device}")
if hasattr(temporal_batch, "temporal_position"):
    print(f"  - temporal_position device: {temporal_batch.temporal_position.device}")
if hasattr(temporal_batch, "spatial_node_idx"):
    print(f"  - spatial_node_idx device: {temporal_batch.spatial_node_idx.device}")

print("\n" + "=" * 50)
print("PROCESSING ALL TEMPORAL POSITIONS WITH E3CONV")
print("=" * 50)
breakpoint()
# Process all temporal positions with E3Conv
with torch.no_grad():
    # Create topology without positions for E3Conv processing
    # add edges to the topology
    sigma = torch.tensor(0.0, device=device)
    from jamun.utils import unsqueeze_trailing

    sigma = unsqueeze_trailing(sigma, 1)
    topology = batch.clone()
    topology = move_graph_to_device(topology, device)
    del topology.pos, topology.batch, topology.num_graphs

    # Process current positions: [N, 3] -> [N, 1, num_features]
    node_attr_current = spatial_e3conv(
        batch.pos, topology, batch.batch, num_graphs=batch.num_graphs, c_noise=sigma, effective_radial_cutoff=0.05
    ).unsqueeze(1)

    # Process hidden state positions and collect all temporal features
    node_attr_list = [node_attr_current]
    breakpoint()
    if hasattr(batch, "hidden_state") and batch.hidden_state:
        for hidden_pos in batch.hidden_state:
            node_attr_hidden = node_attr_current = spatial_e3conv(
                hidden_pos,
                topology,
                batch.batch,
                num_graphs=batch.num_graphs,
                c_noise=sigma,
                effective_radial_cutoff=0.05,
            ).unsqueeze(1)
            node_attr_list.append(node_attr_hidden)

    # Stack along temporal dimension: [N, T, num_features]
    breakpoint()
    node_attr_spatial_temporal = torch.cat(node_attr_list, dim=1)

    breakpoint()
    # Convert spatial-temporal features to temporal node attributes with proper ordering
    spatial_temporal_pooler = SpatialTemporalToTemporalNodeAttr()
    spatial_node_attr_all_temporal = spatial_temporal_pooler(node_attr_spatial_temporal, temporal_batch)

breakpoint()
print(f"Node attributes for all temporal positions: {spatial_node_attr_all_temporal.shape}")
print(f"First spatial node temporal features: {node_attr_spatial_temporal[0].shape}")
print(f"Total norm (should be nonzero): {torch.norm(spatial_node_attr_all_temporal):.6f}")
print(f"Spatial node attributes device: {spatial_node_attr_all_temporal.device}")

print("\n" + "=" * 50)
print("E3TRANSFORMER TEST")
print("=" * 50)

breakpoint()


def test_e3_transformer(batch, temporal_batch, spatial_node_attr_all_temporal, device):
    """Test the E3Transformer with temporal graphs."""
    from temporal_transformer import E3Transformer

    print("=== Testing E3Transformer ===")

    # Use the precomputed temporal node attributes (processed by E3Conv)
    print(f"Using temporal node attributes (processed by E3Conv): {spatial_node_attr_all_temporal.shape}")
    print(f"Sample temporal node attr: {spatial_node_attr_all_temporal[0]}")

    # The node attributes are already arranged to match temporal graph ordering
    temporal_node_attr = spatial_node_attr_all_temporal

    print("Input shapes:")
    print(f"  - temporal_node_attr: {temporal_node_attr.shape}")
    print(f"  - temporal_graph.pos: {temporal_batch.pos.shape}")
    print(f"  - temporal_graph.edge_index: {temporal_batch.edge_index.shape}")
    print(f"  - temporal_graph.temporal_position: {temporal_batch.temporal_position.shape}")
    print(f"  - temporal_graph.batch: {temporal_batch.batch.shape}")
    print(f"  - temporal_graph.num_graphs: {temporal_batch.num_graphs}")

    # Create E3Transformer model that takes 1x1e node attributes (E3Conv output)
    transformer = E3Transformer(
        irreps_out="3x1e",  # 3D output (like positions)
        irreps_hidden="8x0e + 4x1e",  # Hidden representations
        irreps_sh="1x0e + 1x1e",  # Spherical harmonics
        irreps_node_attr="1x1e",  # Input node attributes match E3Conv output
        num_layers=2,
        edge_attr_dim=24,  # Split into 2 parts: 12+12 (radial+temporal)
        num_attention_heads=1,  # Single attention head for simpler test
    )

    # Move transformer to device
    transformer = transformer.to(device)
    print(f"Moved transformer to device: {next(transformer.parameters()).device}")

    print("\nTransformer parameters:")
    print(f"  - irreps_out: {transformer.irreps_out}")
    print(f"  - irreps_hidden: {transformer.irreps_hidden}")
    print(f"  - irreps_node_attr: {transformer.irreps_node_attr}")
    print(f"  - temporal_gate.irreps_out: {transformer.temporal_gate.irreps_out}")
    print(f"  - radial_edge_attr_dim: {transformer.radial_edge_attr_dim}")
    print(f"  - temporal_edge_attr_dim: {transformer.temporal_edge_attr_dim}")

    # Forward pass with tensor and graph (like E3Conv)
    effective_radial_cutoff = 5.0  # Define the cutoff in forward pass
    temporal_cutoff = 1.0  # Default temporal cutoff (no cutoff for temporal contributions)
    with torch.no_grad():
        try:
            transformer_output = transformer(
                temporal_node_attr, temporal_batch, effective_radial_cutoff, temporal_cutoff
            )
            print("\n✅ Transformer forward pass successful!")
            print(f"Transformer output shape: {transformer_output.shape}")
            print(f"Transformer output sample: {transformer_output[0]}")
            print(f"Transformer output norm: {torch.norm(transformer_output):.6f}")
            print(f"Used effective_radial_cutoff: {effective_radial_cutoff}")
            print(f"Used temporal_cutoff: {temporal_cutoff}")
            return True
        except Exception as e:
            print(f"\n❌ Transformer forward pass failed: {e}")
            import traceback

            traceback.print_exc()
            return False


# Test the complete workflow: E3Conv -> Transformer
success = test_e3_transformer(batch, temporal_batch, spatial_node_attr_all_temporal, device)

print("\n" + "=" * 50)
print("TEMPORAL TO SPATIAL MEAN POOLING")
print("=" * 50)
breakpoint()
# Demonstrate mean pooling from temporal features back to spatial features
print("=== Testing Mean Pooling ===")

# Use the transformer output or the original temporal features for pooling demonstration
print(f"Input temporal features shape: {spatial_node_attr_all_temporal.shape}")

# Create mean pooling module and apply it
temporal_to_spatial_pooler = TemporalToSpatialNodeAttrMean()
spatial_features_pooled = temporal_to_spatial_pooler(spatial_node_attr_all_temporal, temporal_batch)

print(f"Output spatial features shape: {spatial_features_pooled.shape}")
print(f"Number of spatial nodes recovered: {spatial_features_pooled.shape[0]}")
print(f"Original spatial nodes: {batch.pos.shape[0]}")
print(f"Feature dimension: {spatial_features_pooled.shape[1]}")
print(f"Sample pooled features (first node): {spatial_features_pooled[0]}")
print(f"Pooled features norm: {torch.norm(spatial_features_pooled):.6f}")

# Verify that we correctly recovered the spatial dimension
assert spatial_features_pooled.shape[0] == batch.pos.shape[0], (
    f"Spatial node count mismatch: {spatial_features_pooled.shape[0]} vs {batch.pos.shape[0]}"
)
print("✅ Mean pooling successfully converted temporal features back to spatial!")

print("\n" + "=" * 50)
print("TESTS COMPLETED")
print("=" * 50)

print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)

# Device summary
print("Device Summary:")
print(f"  - Used device: {device}")
if torch.cuda.is_available():
    print(f"  - CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print(f"  - CUDA memory cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

print("\nTest Results:")
print(f"  - Manual workflow tests: {'✅ PASSED' if success else '❌ FAILED'}")

if success:
    print("\n🎉 ALL TESTS PASSED!")
    print("The manual spatio-temporal workflow is working correctly.")
    print("To test the unified E3SpatioTemporal model, run: python3 test_e3_spatiotemporal.py")
else:
    print("\n⚠️  Some tests failed. Check the output above for details.")
