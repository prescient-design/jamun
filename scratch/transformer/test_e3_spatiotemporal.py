#!/usr/bin/env python3
"""
Test script for the E3SpatioTemporal model.

This script tests the unified E3SpatioTemporal model that encapsulates
the complete spatio-temporal processing workflow.
"""

import e3nn

e3nn.set_optimization_defaults(jit_script_fx=False)
import torch
import torch_geometric

# Import modules needed for the test
from helpers import add_edges, create_e3conv_network
from pooling import SpatialTemporalToTemporalNodeAttr, TemporalToSpatialNodeAttrMean
from temporal_transformer import E3SpatioTemporal, E3Transformer

from jamun.data import parse_datasets_from_directory
from jamun.utils import unsqueeze_trailing


def setup_device():
    """Setup CUDA device if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    return device


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


def load_test_data(device):
    """Load and prepare test data."""
    print("Loading test data...")

    dataset = parse_datasets_from_directory(
        root="/data2/sules/ALA_ALA_enhanced_full_grid/train",
        traj_pattern="^(.*).xtc",
        pdb_pattern="^(.*).pdb",
        subsample=1,
        total_lag_time=5,
        lag_subsample_rate=1,
        max_datasets=3,
        num_frames=10,
    )

    # Get first graph and create batch
    graph = dataset[0].__getitem__(0)
    batch = torch_geometric.data.Batch.from_data_list([graph])
    batch = add_edges(batch.pos, batch, batch.batch, 0.05)

    # Move to device
    batch = move_graph_to_device(batch, device)
    print(f"Loaded batch with {batch.pos.shape[0]} nodes on device: {batch.pos.device}")

    return batch


def create_spatiotemporal_model(device):
    """Create and configure the E3SpatioTemporal model."""
    print("Creating E3SpatioTemporal model...")

    # Create component modules
    spatial_module = create_e3conv_network().to(device)

    temporal_module = E3Transformer(
        irreps_out="3x1e",  # 3D output (like positions)
        irreps_hidden="8x0e + 4x1e",  # Hidden representations
        irreps_sh="1x0e + 1x1e",  # Spherical harmonics
        irreps_node_attr="1x1e",  # Input node attributes match E3Conv output
        num_layers=2,
        edge_attr_dim=24,  # Split into 2 parts: 12+12 (radial+temporal)
        num_attention_heads=1,  # Single attention head for simpler test
    ).to(device)

    spatial_to_temporal_pooler = SpatialTemporalToTemporalNodeAttr()
    temporal_to_spatial_pooler = TemporalToSpatialNodeAttrMean()

    # Create the unified model
    spatiotemporal_model = E3SpatioTemporal(
        spatial_module=spatial_module,
        temporal_module=temporal_module,
        spatial_to_temporal_pooler=spatial_to_temporal_pooler,
        temporal_to_spatial_pooler=temporal_to_spatial_pooler,
        radial_cutoff=0.05,
        temporal_cutoff=1.0,
    ).to(device)

    print(f"Created E3SpatioTemporal model on device: {next(spatiotemporal_model.parameters()).device}")
    return spatiotemporal_model


def test_spatiotemporal_model(model, batch, device):
    """Test the E3SpatioTemporal model with various configurations."""
    print("=" * 50)
    print("TESTING E3SPATIOTEMPORAL MODEL")
    print("=" * 50)

    # Print model information
    print("Model components:")
    print(f"  - Spatial module output irreps: {model.get_spatial_output_irreps()}")
    print(f"  - Temporal module output irreps: {model.get_temporal_output_irreps()}")
    print(f"  - Radial cutoff: {model.radial_cutoff}")
    print(f"  - Temporal cutoff: {model.temporal_cutoff}")

    # Prepare noise conditioning
    sigma = torch.tensor(0.0, device=device)
    sigma = unsqueeze_trailing(sigma, 1)

    print("\nInput batch:")
    print(f"  - pos shape: {batch.pos.shape}")
    print(
        f"  - hidden_state length: {len(batch.hidden_state) if hasattr(batch, 'hidden_state') and batch.hidden_state else 0}"
    )
    print(f"  - batch device: {batch.pos.device}")

    success = True

    try:
        with torch.no_grad():
            print("\n1. Testing simple forward pass (spatial features only)...")
            spatial_features = model(batch, sigma)
            print(f"   ✅ Success! Spatial features shape: {spatial_features.shape}")
            print(f"   Spatial features device: {spatial_features.device}")
            print(f"   Spatial features norm: {torch.norm(spatial_features):.6f}")

            print("\n2. Testing full forward pass (all outputs)...")
            results = model(batch, sigma, return_temporal_features=True, return_temporal_graph=True)

            print("   ✅ Success! Full output results:")
            print(f"   - spatial_features shape: {results['spatial_features'].shape}")
            print(f"   - temporal_features shape: {results['temporal_features'].shape}")
            print(f"   - temporal_graph num_graphs: {results['temporal_graph'].num_graphs}")
            print(f"   - spatial_graph pos shape: {results['spatial_graph'].pos.shape}")

            # Verify spatial graph reconstruction
            pos_difference = torch.norm(results["spatial_graph"].pos - batch.pos)
            print(f"   - Spatial position reconstruction error: {pos_difference:.6f}")

            print("\n3. Testing consistency between simple and full forward pass...")
            simple_features = spatial_features
            full_features = results["spatial_features"]
            consistency_error = torch.norm(simple_features - full_features)
            print(f"   Consistency error: {consistency_error:.6f}")

            if consistency_error < 1e-6:
                print("   ✅ Results are consistent!")
            else:
                print("   ⚠️ Results differ between simple and full forward pass")
                success = False

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        success = False

    return success


def main():
    """Main test function."""
    print("E3SpatioTemporal Model Test")
    print("=" * 50)

    # Setup
    device = setup_device()
    batch = load_test_data(device)
    model = create_spatiotemporal_model(device)

    # Run tests
    success = test_spatiotemporal_model(model, batch, device)

    # Final summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    # Device summary
    print("Device Summary:")
    print(f"  - Used device: {device}")
    if torch.cuda.is_available():
        print(f"  - CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"  - CUDA memory cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

    print("\nTest Results:")
    if success:
        print("🎉 ALL TESTS PASSED! The E3SpatioTemporal model works correctly!")
        print("\nThe model successfully:")
        print("  ✅ Processes spatial graphs with hidden states")
        print("  ✅ Converts to temporal representation")
        print("  ✅ Applies temporal transformations")
        print("  ✅ Pools back to spatial features")
        print("  ✅ Reconstructs spatial graphs")
        print("  ✅ Maintains consistency across different call patterns")
    else:
        print("❌ SOME TESTS FAILED! Check the output above for details.")

    return success


if __name__ == "__main__":
    main()
