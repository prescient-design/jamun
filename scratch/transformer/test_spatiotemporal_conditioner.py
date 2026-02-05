#!/usr/bin/env python3
"""
Test script for loading and testing a conditional denoiser with spatiotemporal conditioner.
Uses the new approach where SpatioTemporalConditioner outputs [y.pos, spatial_features]
and E3ConvConditionalSpatioTemporal handles concatenated inputs.
"""

import e3nn

e3nn.set_optimization_defaults(jit_script_fx=False)

import sys
from typing import Any

import torch
import torch_geometric

# Add the src directory to path to import jamun modules
sys.path.insert(0, "src")

from jamun.data import parse_datasets_from_directory
from jamun.distributions._distributions import ConstantSigma
from jamun.model.arch.e3conv import E3Conv
from jamun.model.arch.e3conv_conditional import (
    E3ConvConditionalSpatioTemporal,  # Changed from E3ConvConditionalWithInputAttr
)
from jamun.model.arch.spatiotemporal import E3SpatioTemporal, E3Transformer
from jamun.model.conditioners.conditioners import SpatioTemporalConditioner
from jamun.model.denoiser_conditional import Denoiser  # Changed from DenoiserWithInputAttr
from jamun.model.pooling import SpatialTemporalToTemporalNodeAttr, TemporalToSpatialNodeAttrMean
from jamun.utils.average_squared_distance import (
    compute_temporal_average_squared_distance_from_datasets,  # Import temporal function
)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def create_spatial_module() -> E3Conv:
    """Create E3Conv spatial module with reasonable parameters."""
    import functools

    import e3tools

    # Create factory functions
    hidden_layer_factory = functools.partial(e3tools.nn.ConvBlock, conv=functools.partial(e3tools.nn.Conv))

    output_head_factory = functools.partial(e3tools.nn.EquivariantMLP, irreps_hidden_list=["120x0e + 32x1e"])

    return E3Conv(
        irreps_out="3x1e",  # Changed to match temporal module input
        irreps_hidden="120x0e + 32x1e",
        irreps_sh="1x0e + 1x1e",
        hidden_layer_factory=hidden_layer_factory,
        output_head_factory=output_head_factory,
        n_layers=1,
        edge_attr_dim=64,
        use_residue_information=True,
        atom_type_embedding_dim=8,
        atom_code_embedding_dim=8,
        residue_code_embedding_dim=32,
        residue_index_embedding_dim=8,
        use_residue_sequence_index=False,
        num_atom_types=20,
        max_sequence_length=10,
        num_atom_codes=10,
        num_residue_types=25,
        test_equivariance=False,
        reduce=None,
    )


def create_temporal_module() -> E3Transformer:
    """Create E3Transformer temporal module."""
    return E3Transformer(
        irreps_out="3x1e",  # Final spatial features output
        irreps_hidden="8x0e + 4x1e",
        irreps_sh="1x0e + 1x1e",
        irreps_node_attr="3x1e",  # Match spatial module output
        num_layers=2,
        edge_attr_dim=24,
        num_attention_heads=1,
        reduce=None,
    )


def create_spatiotemporal_model() -> E3SpatioTemporal:
    """Create the complete E3SpatioTemporal model."""
    spatial_module = create_spatial_module()
    temporal_module = create_temporal_module()

    # Create pooling modules
    spatial_to_temporal_pooler = SpatialTemporalToTemporalNodeAttr(irreps_out="3x1e")  # Match spatial module output
    temporal_to_spatial_pooler = TemporalToSpatialNodeAttrMean(irreps_out="3x1e")  # Match temporal module output

    # Compute radial cutoff using temporal average squared distance
    print("Computing radial cutoff from temporal dataset...")
    try:
        # Load dataset to compute temporal average squared distance
        dataset = parse_datasets_from_directory(
            root="/data2/sules/ALA_ALA_enhanced_full_grid/train",
            traj_pattern="^(.*).xtc",
            pdb_pattern="^(.*).pdb",
            subsample=1,
            total_lag_time=5,
            lag_subsample_rate=1,
            max_datasets=2,  # Keep small for testing
            num_frames=5,  # Small number of frames
        )

        # Compute temporal average squared distance
        temporal_avg_sq_dist = compute_temporal_average_squared_distance_from_datasets(
            [dataset],  # Pass as list since function expects multiple datasets
            num_samples=50,  # Use fewer samples for testing
            verbose=True,
        )

        # Use a multiple of the temporal average squared distance as the radial cutoff
        # Typically we might use sqrt(temporal_avg_sq_dist) * some_factor
        import math

        radial_cutoff = math.sqrt(temporal_avg_sq_dist) * 2.0  # Scale factor of 2.0
        print(f"Computed radial cutoff: {radial_cutoff:.6f} nm")

    except Exception as e:
        print(f"Warning: Failed to compute temporal cutoff ({e}), using default value 0.05")
        radial_cutoff = 0.05

    return E3SpatioTemporal(
        spatial_module=spatial_module,
        temporal_module=temporal_module,
        spatial_to_temporal_pooler=spatial_to_temporal_pooler,
        temporal_to_spatial_pooler=temporal_to_spatial_pooler,
        radial_cutoff=radial_cutoff,
        temporal_cutoff=1.0,
    )


def create_spatiotemporal_conditioner() -> SpatioTemporalConditioner:
    """Create SpatioTemporalConditioner with E3SpatioTemporal model."""
    spatiotemporal_model = create_spatiotemporal_model()

    return SpatioTemporalConditioner(
        N_structures=1,  # Changed to 2 for [y.pos, spatial_features]
        spatiotemporal_model=spatiotemporal_model,
        c_noise=0.0,
        freeze_spatiotemporal_model=False,  # Keep trainable
    )


def create_conditional_denoiser_config() -> dict[str, Any]:
    """Create configuration for Denoiser with spatiotemporal conditioner."""
    import functools

    import e3tools.nn

    def create_arch():
        """Create the E3ConvConditionalSpatioTemporal architecture module."""
        # Hidden layer factory
        hidden_layer_factory = functools.partial(e3tools.nn.ConvBlock, conv=functools.partial(e3tools.nn.Conv))

        # Output head factory
        output_head_factory = functools.partial(e3tools.nn.EquivariantMLP, irreps_hidden_list=["16x0e + 8x1e"])

        return E3ConvConditionalSpatioTemporal(
            irreps_out="1x1e",  # Output should be 3 components (1x1e) to match position
            irreps_hidden="16x0e + 8x1e",
            irreps_sh="1x0e + 1x1e",
            hidden_layer_factory=hidden_layer_factory,
            output_head_factory=output_head_factory,
            n_layers=2,
            edge_attr_dim=32,
            use_residue_information=True,
            atom_type_embedding_dim=8,
            atom_code_embedding_dim=8,
            residue_code_embedding_dim=16,
            residue_index_embedding_dim=8,
            use_residue_sequence_index=False,
            num_atom_types=20,
            max_sequence_length=10,
            num_atom_codes=10,
            num_residue_types=25,
            test_equivariance=False,
            reduce=None,
            N_structures=1,  # Changed to 2 for [y.pos, spatial_features]
            input_attr_irreps="3x1e",  # spatial_features only (9 components = 3x1e)
        )

    def create_optim(params):
        """Create the optimizer."""
        return torch.optim.Adam(params, lr=0.001)

    return {
        # Required Denoiser parameters (changed from DenoiserWithInputAttr)
        "arch": create_arch,
        "optim": create_optim,
        "sigma_distribution": ConstantSigma(sigma=0.1),
        "max_radius": 1000.0,
        "average_squared_distance": 10.0,  # Dummy value for testing
        "add_fixed_noise": False,
        "add_fixed_ones": False,
        "align_noisy_input_during_training": True,
        "align_noisy_input_during_evaluation": True,
        "mean_center": True,
        "mirror_augmentation_rate": 0.0,
        "bond_loss_coefficient": 1.0,
        "normalization_type": "JAMUN",
        "sigma_data": None,
        "lr_scheduler_config": None,
        "use_torch_compile": False,  # Disable for testing
        "torch_compile_kwargs": None,
        "conditioner": create_spatiotemporal_conditioner(),
    }


def add_edges_to_batch(batch: torch_geometric.data.Batch, cutoff: float = 0.05) -> torch_geometric.data.Batch:
    """Add edges to batch using existing utility from denoiser."""
    # Use e3tools radius_graph directly since we don't need the full denoiser add_edges logic
    import e3tools

    if hasattr(batch, "edge_index") and batch.edge_index is not None:
        return batch

    # Add radius-based edges
    edge_index = e3tools.radius_graph(batch.pos, cutoff, batch.batch)
    batch.edge_index = edge_index

    # Add bonded edges if they exist
    if hasattr(batch, "bonded_edge_index") and batch.bonded_edge_index is not None:
        bond_mask = torch.cat(
            [
                torch.zeros(edge_index.shape[1], dtype=torch.long, device=batch.pos.device),
                torch.ones(batch.bonded_edge_index.shape[1], dtype=torch.long, device=batch.pos.device),
            ]
        )
        batch.edge_index = torch.cat([edge_index, batch.bonded_edge_index], dim=1)
        batch.bond_mask = bond_mask
    else:
        batch.bond_mask = torch.zeros(edge_index.shape[1], dtype=torch.long, device=batch.pos.device)

    return batch


def load_test_data():
    """Load ALA_ALA test dataset."""
    print("Loading ALA_ALA dataset...")

    dataset = parse_datasets_from_directory(
        root="/data2/sules/ALA_ALA_enhanced_full_grid/train",
        traj_pattern="^(.*).xtc",
        pdb_pattern="^(.*).pdb",
        subsample=1,
        total_lag_time=5,
        lag_subsample_rate=1,
        max_datasets=2,  # Keep small for testing
        num_frames=5,  # Small number of frames
    )

    print(f"Loaded dataset with {len(dataset)} samples")

    # Get a sample and create batch
    graph = dataset[0].__getitem__(0)
    batch = torch_geometric.data.Batch.from_data_list([graph])

    # Add edges
    batch = add_edges_to_batch(batch, cutoff=0.05)

    # Move to device
    batch = batch.to(device)

    print("Batch info:")
    print(f"  - pos shape: {batch.pos.shape}")
    print(f"  - edge_index shape: {batch.edge_index.shape}")
    print(
        f"  - hidden_state length: {len(batch.hidden_state) if hasattr(batch, 'hidden_state') and batch.hidden_state else 0}"
    )
    if hasattr(batch, "hidden_state") and batch.hidden_state:
        print(f"  - hidden_state[0] shape: {batch.hidden_state[0].shape}")

    return batch


def test_spatiotemporal_conditioner(conditioner: SpatioTemporalConditioner, batch: torch_geometric.data.Batch):
    """Test the spatiotemporal conditioner."""
    print("\n" + "=" * 50)
    print("TESTING SPATIOTEMPORAL CONDITIONER")
    print("=" * 50)

    try:
        # Test forward pass
        conditioned_structures = conditioner(batch)

        print("✅ Conditioner forward pass successful!")
        print(f"Number of conditioned structures: {len(conditioned_structures)} (expected: 2)")
        print(f"First structure (y.pos) shape: {conditioned_structures[0].shape}")
        print(f"Second structure (spatial_features) shape: {conditioned_structures[1].shape}")
        print(f"Original position shape: {batch.pos.shape}")
        print(f"Position difference norm: {torch.norm(conditioned_structures[0] - batch.pos):.6f}")

        # Verify we got exactly two structures
        assert len(conditioned_structures) == 2, f"Expected 2 structures, got {len(conditioned_structures)}"

        return True, conditioned_structures

    except Exception as e:
        print(f"❌ Conditioner test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_conditional_denoiser_creation():
    """Test creating Denoiser with spatiotemporal conditioner."""
    print("\n" + "=" * 50)
    print("TESTING DENOISER WITH SPATIOTEMPORAL CONDITIONER CREATION")
    print("=" * 50)

    try:
        # Create configuration
        config = create_conditional_denoiser_config()

        # Create denoiser (this will instantiate all components)
        denoiser = Denoiser(**config)
        denoiser = denoiser.to(device)

        print("✅ Denoiser created successfully!")
        print(f"Denoiser device: {next(denoiser.parameters()).device}")
        print(f"Has conditioner: {hasattr(denoiser, 'conditioning_module')}")
        print(f"Architecture type: {type(denoiser.g).__name__}")
        print(f"Conditioner type: {type(denoiser.conditioning_module).__name__}")

        # Check if spatiotemporal model is properly set up
        if hasattr(denoiser.conditioning_module, "spatiotemporal_model"):
            st_model = denoiser.conditioning_module.spatiotemporal_model
            print(f"SpatioTemporal model type: {type(st_model).__name__}")
            print(f"Spatial module type: {type(st_model.spatial_module).__name__}")
            print(f"Temporal module type: {type(st_model.temporal_module).__name__}")

        return True, denoiser

    except Exception as e:
        print(f"❌ Denoiser creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_denoiser_forward_pass(denoiser: Denoiser, batch: torch_geometric.data.Batch):
    """Test the complete denoiser forward pass."""
    print("\n" + "=" * 50)
    print("TESTING DENOISER WITH SPATIOTEMPORAL CONDITIONER FORWARD PASS")
    print("=" * 50)

    try:
        # Test with sigma = 0.1
        sigma = 0.1

        # Debug: check conditioned structures shapes
        conditioned_structures = denoiser.conditioning_module(batch)
        print("DEBUG: Conditioned structures shapes:")
        for i, struct in enumerate(conditioned_structures):
            print(f"  Structure {i}: {struct.shape}")

        concatenated = torch.cat([*conditioned_structures], dim=-1)
        print(f"DEBUG: Concatenated shape: {concatenated.shape}")
        print("DEBUG: Expected irreps: 4x1e = 12 components")

        with torch.no_grad():
            xhat_batch = denoiser.xhat(batch, sigma)

        print("✅ Denoiser forward pass successful!")
        print(f"Input shape: {batch.pos.shape}")
        print(f"Output shape: {xhat_batch.pos.shape}")
        print(f"Output norm: {torch.norm(xhat_batch.pos):.6f}")
        print(f"Used sigma: {sigma}")

        # Verify output shapes match input
        assert xhat_batch.pos.shape == batch.pos.shape, f"Shape mismatch: {xhat_batch.pos.shape} vs {batch.pos.shape}"

        return True

    except Exception as e:
        print(f"❌ Denoiser forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("CONDITIONAL DENOISER WITH SPATIOTEMPORAL CONDITIONER TEST")
    print("=" * 60)

    # Load test data
    batch = load_test_data()

    # Test conditioner creation and forward pass
    conditioner = create_spatiotemporal_conditioner()
    conditioner = conditioner.to(device)

    conditioner_success, conditioned_structures = test_spatiotemporal_conditioner(conditioner, batch)

    if not conditioner_success:
        print("❌ Conditioner test failed, stopping here.")
        return

    # Test complete denoiser creation
    denoiser_success, denoiser = test_conditional_denoiser_creation()

    if not denoiser_success:
        print("❌ Denoiser creation failed, stopping here.")
        return

    # Test complete forward pass
    forward_success = test_denoiser_forward_pass(denoiser, batch)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("Test Results:")
    print(f"  - Conditioner test: {'✅ PASSED' if conditioner_success else '❌ FAILED'}")
    print(f"  - Denoiser creation: {'✅ PASSED' if denoiser_success else '❌ FAILED'}")
    print(f"  - Forward pass test: {'✅ PASSED' if forward_success else '❌ FAILED'}")

    if conditioner_success and denoiser_success and forward_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("The conditional denoiser with spatiotemporal conditioner is working correctly!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

    # Device memory summary
    if torch.cuda.is_available():
        print("\nCUDA Memory:")
        print(f"  - Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        print(f"  - Cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
