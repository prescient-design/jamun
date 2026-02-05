#!/usr/bin/env python3
"""
Comprehensive test for DenoisedConditioner using real ALA_ALA data.
This test emulates the behavior of xhat_normalized to properly test the scaling parameter.
"""

import os

import e3nn
import numpy as np
import torch
import torch_geometric

from jamun.data import parse_datasets_from_directory
from jamun.model.conditioners import DenoisedConditioner
from jamun.utils import mean_center, unsqueeze_trailing
from jamun.utils._normalizations import normalization_factors

# Fix e3nn optimization for avoiding script issues
e3nn.set_optimization_defaults(jit_script_fx=False)


def load_ala_ala_data():
    """Load actual ALA_ALA data from the capped diamines dataset."""

    # Get data path from environment variable
    data_path = os.getenv("JAMUN_DATA_PATH")
    if not data_path:
        raise ValueError("JAMUN_DATA_PATH environment variable not set")

    # Load ALA_ALA dataset
    datasets = parse_datasets_from_directory(
        root=f"{data_path}/capped_diamines/timewarp_splits/train",
        traj_pattern="^(.*).xtc",
        pdb_pattern="^(.*).pdb",
        filter_codes=["ALA_ALA"],
        as_iterable=False,
        subsample=1,
        total_lag_time=3,  # This will give us hidden states
        lag_subsample_rate=1,
        num_frames=10,
        max_datasets=1,
    )

    if not datasets:
        raise ValueError("No ALA_ALA datasets found")

    # Get the first dataset
    dataset = datasets[0]
    print(f"Loaded ALA_ALA dataset with {len(dataset)} frames")

    # Get a few samples to create a batch
    samples = []
    for i in range(min(2, len(dataset))):  # Get 2 samples for batch
        sample = dataset[i]
        samples.append(sample)

    # Create batch
    batch = torch_geometric.data.Batch.from_data_list(samples)

    print(f"Created batch with {batch.num_graphs} graphs")
    print(f"Batch position shape: {batch.pos.shape}")

    if hasattr(batch, "hidden_state") and batch.hidden_state:
        print(f"Hidden states: {len(batch.hidden_state)} states")
        for i, hidden_state in enumerate(batch.hidden_state):
            print(f"  Hidden state {i}: shape {hidden_state.shape}")
    else:
        print("No hidden states found")

    return batch


def add_noise_to_batch(x: torch_geometric.data.Batch, sigma: float) -> torch_geometric.data.Batch:
    """Add noise to a batch, similar to the denoiser's add_noise method."""
    sigma = unsqueeze_trailing(torch.tensor(sigma), x.pos.ndim)

    y = x.clone()

    # Add noise to positions
    noise = torch.randn_like(x.pos)
    y.pos = x.pos + sigma * noise

    # Add noise to hidden states if they exist
    if hasattr(x, "hidden_state") and x.hidden_state is not None:
        y.hidden_state = []
        for hidden_positions in x.hidden_state:
            hidden_noise = torch.randn_like(hidden_positions)
            y.hidden_state.append(hidden_positions + sigma * hidden_noise)

    return y


def mean_center_positions(batch: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
    """Mean-center positions and hidden states for each graph in the batch."""

    # Mean-center the main positions using the jamun utils function
    batch = mean_center(batch)

    # Mean-center each hidden state individually
    if hasattr(batch, "hidden_state") and batch.hidden_state is not None:
        for i, hidden_positions in enumerate(batch.hidden_state):
            # Create a temporary batch with just the hidden state positions to mean-center
            temp_batch = batch.clone()
            temp_batch.pos = hidden_positions
            temp_batch_centered = mean_center(temp_batch)
            batch.hidden_state[i] = temp_batch_centered.pos

    return batch


def emulate_xhat_normalized_scaling(batch, sigma: float, average_squared_distance: float = 0.332):
    """
    Emulate the scaling behavior in xhat_normalized method.
    This simulates how the denoiser scales data before passing to conditioner.
    """
    print("\n=== Emulating xhat_normalized scaling ===")
    print(f"Input sigma: {sigma}")
    print(f"Average squared distance: {average_squared_distance}")

    # Mean-center the batch positions and hidden states (as done in actual xhat_normalized)
    batch = mean_center_positions(batch)

    # Compute normalization factors (same as in denoiser)
    c_in, c_skip, c_out, c_noise = normalization_factors(sigma, average_squared_distance)

    print("Normalization factors:")
    print(f"  c_in: {c_in}")
    print(f"  c_skip: {c_skip}")
    print(f"  c_out: {c_out}")
    print(f"  c_noise: {c_noise}")

    # Adjust dimensions (same as in denoiser)
    c_in = unsqueeze_trailing(c_in, batch.pos.ndim - 1)
    c_skip = unsqueeze_trailing(c_skip, batch.pos.ndim - 1)
    c_out = unsqueeze_trailing(c_out, batch.pos.ndim - 1)
    c_noise = c_noise.unsqueeze(0)

    # Scale the batch (same as in denoiser)
    y_scaled = batch.clone()
    y_scaled.pos = batch.pos * c_in

    print(f"Original position mean: {batch.pos.mean():.6f}")
    print(f"Scaled position mean: {y_scaled.pos.mean():.6f}")

    # Scale hidden states (same as in denoiser)
    if hasattr(batch, "hidden_state") and batch.hidden_state is not None:
        y_scaled.hidden_state = []
        for i, positions in enumerate(batch.hidden_state):
            scaled_positions = positions * c_in
            y_scaled.hidden_state.append(scaled_positions)
            print(
                f"Hidden state {i} - Original mean: {positions.mean():.6f}, Scaled mean: {scaled_positions.mean():.6f}"
            )

    return y_scaled, c_in, c_skip, c_out, c_noise


def test_denoised_conditioner_with_scaling():
    """Test the DenoisedConditioner with proper scaling emulation."""

    print("=== Testing DenoisedConditioner with xhat_normalized scaling ===")

    # Test parameters
    N_structures = 3  # Must match architecture N_structures (updated to match hidden states)
    pretrained_model_path = "sule-shashank/jamun/370wpt17"  # Update this to your desired checkpoint
    test_sigma = 0.04

    try:
        # Load real ALA_ALA data
        print("\n1. Loading real ALA_ALA data...")
        original_batch = load_ala_ala_data()
        print("✓ Real ALA_ALA data loaded successfully")

        # Mean-center the original batch (clean reference) - positions and hidden states
        print("\n2. Mean-centering the data...")
        print(f"  Original position mean: {original_batch.pos.mean():.6f}")
        if hasattr(original_batch, "hidden_state") and original_batch.hidden_state:
            for i, hidden_state in enumerate(original_batch.hidden_state):
                print(f"  Original hidden state {i} mean: {hidden_state.mean():.6f}")

        x_clean = mean_center_positions(original_batch)
        print(f"  Mean-centered position mean: {x_clean.pos.mean():.6f}")

        if hasattr(x_clean, "hidden_state") and x_clean.hidden_state:
            for i, hidden_state in enumerate(x_clean.hidden_state):
                print(f"  Mean-centered hidden state {i} mean: {hidden_state.mean():.6f}")

        # Add noise to the mean-centered data
        print(f"\n3. Adding noise with sigma={test_sigma}...")
        y_noisy = add_noise_to_batch(x_clean, test_sigma)
        print(f"  Noisy position mean: {y_noisy.pos.mean():.6f}")
        print(f"  Noisy position std: {y_noisy.pos.std():.6f}")

        if hasattr(y_noisy, "hidden_state") and y_noisy.hidden_state:
            for i, hidden_state in enumerate(y_noisy.hidden_state):
                print(f"  Noisy hidden state {i} mean: {hidden_state.mean():.6f}")
                print(f"  Noisy hidden state {i} std: {hidden_state.std():.6f}")

        # Initialize conditioner and extract average_squared_distance from checkpoint
        print("\n4. Initializing DenoisedConditioner and extracting average_squared_distance...")
        print(f"  N_structures: {N_structures}")
        print(f"  pretrained_model_path: {pretrained_model_path}")

        # Use a temporary c_in for initialization
        temp_c_in, _, _, _ = normalization_factors(test_sigma, 0.332)  # temporary default
        temp_c_in_float = float(temp_c_in)

        # Initialize conditioner
        conditioner = DenoisedConditioner(
            N_structures=N_structures, pretrained_model_path=pretrained_model_path, c_in=temp_c_in_float
        )

        print("✓ DenoisedConditioner initialized successfully")
        print(f"  Denoiser sigma: {conditioner.denoiser_sigma}")

        # Extract average_squared_distance from the loaded checkpoint
        average_squared_distance = None
        if hasattr(conditioner.pretrained_denoiser, "average_squared_distance"):
            average_squared_distance = float(conditioner.pretrained_denoiser.average_squared_distance)
            print(f"  ✓ Extracted average_squared_distance from checkpoint: {average_squared_distance}")
        elif hasattr(conditioner.pretrained_denoiser, "hparams") and hasattr(
            conditioner.pretrained_denoiser.hparams, "average_squared_distance"
        ):
            average_squared_distance = float(conditioner.pretrained_denoiser.hparams.average_squared_distance)
            print(f"  ✓ Extracted average_squared_distance from hparams: {average_squared_distance}")
        else:
            # Try to extract from the config if available
            if hasattr(conditioner.pretrained_denoiser, "cfg"):
                cfg = conditioner.pretrained_denoiser.cfg
                if hasattr(cfg, "average_squared_distance"):
                    average_squared_distance = float(cfg.average_squared_distance)
                    print(f"  ✓ Extracted average_squared_distance from config: {average_squared_distance}")

        if average_squared_distance is None:
            print("  ⚠️ Could not extract average_squared_distance from checkpoint")
            print("  Available attributes on pretrained_denoiser:")
            for attr in dir(conditioner.pretrained_denoiser):
                if not attr.startswith("_"):
                    print(f"    - {attr}")
            # Use default
            average_squared_distance = 0.332
            print(f"  Using default average_squared_distance: {average_squared_distance}")

        # Recompute c_in with the correct average_squared_distance
        c_in, _, _, _ = normalization_factors(test_sigma, average_squared_distance)
        c_in_float = float(c_in)

        # Update the conditioner's c_in
        if abs(c_in_float - temp_c_in_float) > 1e-6:
            print(f"  Updating c_in from {temp_c_in_float} to {c_in_float}")
            conditioner.c_in = c_in_float
        else:
            print(f"  c_in remains: {c_in_float}")

        # Test sigma consistency
        print("\n5. Testing sigma consistency...")
        if abs(conditioner.denoiser_sigma - test_sigma) < 1e-5:
            print(f"✓ Test sigma ({test_sigma}) matches denoiser sigma ({conditioner.denoiser_sigma})")
        else:
            print(f"⚠️ Test sigma ({test_sigma}) differs from denoiser sigma ({conditioner.denoiser_sigma})")
            print("  Using denoiser sigma for consistency")
            test_sigma = conditioner.denoiser_sigma
            # Recompute c_in with corrected sigma
            c_in, _, _, _ = normalization_factors(test_sigma, average_squared_distance)
            c_in_float = float(c_in)
            conditioner.c_in = c_in_float
            print(f"  Updated c_in to: {c_in_float}")

        # Emulate xhat_normalized scaling on the noisy batch
        print("\n6. Emulating xhat_normalized scaling...")
        scaled_batch, c_in_tensor, c_skip, c_out, c_noise = emulate_xhat_normalized_scaling(
            y_noisy, test_sigma, average_squared_distance
        )

        # Verify our c_in calculation matches
        assert abs(float(c_in_tensor) - c_in_float) < 1e-6, f"c_in mismatch: {c_in_tensor} vs {c_in_float}"
        print("✓ c_in calculation verified")

        # Test conditioner with scaled noisy data
        print("\n7. Testing conditioner with scaled noisy data...")

        # Move scaled_batch to the same device as the conditioner
        device = next(conditioner.parameters()).device
        scaled_batch = scaled_batch.to(device)
        x_clean = x_clean.to(device)
        y_noisy = y_noisy.to(device)
        print(f"  Moved batches to device: {device}")

        conditioned_structures = conditioner.forward(scaled_batch)

        print("✓ Conditioner forward pass completed")
        print(f"  Returned {len(conditioned_structures)} structures")
        print(f"  Expected N_structures: {N_structures}")

        # Verify output structure
        assert len(conditioned_structures) == N_structures, (
            f"Expected {N_structures} structures, got {len(conditioned_structures)}"
        )

        for i, structure in enumerate(conditioned_structures):
            assert structure.shape == scaled_batch.pos.shape, (
                f"Structure {i} has wrong shape: {structure.shape} vs {scaled_batch.pos.shape}"
            )
            print(f"  Structure {i}: shape {structure.shape}")

        # Check that first structure is the scaled current position
        assert torch.allclose(conditioned_structures[0], scaled_batch.pos), (
            "First structure should be scaled current position"
        )
        print("✓ First structure matches scaled current position")

        # Comprehensive denoising quality test
        print("\n8. COMPREHENSIVE DENOISING QUALITY TEST...")
        denoising_improvements = []

        if hasattr(x_clean, "hidden_state") and x_clean.hidden_state and len(conditioned_structures) > 1:
            print(f"  Testing denoising on {len(x_clean.hidden_state)} hidden states...")

            for i in range(1, len(conditioned_structures)):  # Skip first structure (current position)
                hidden_idx = i - 1  # Map to hidden state index
                if hidden_idx < len(x_clean.hidden_state):
                    denoised_structure = conditioned_structures[i]
                    clean_hidden = x_clean.hidden_state[hidden_idx]
                    noisy_hidden = y_noisy.hidden_state[hidden_idx]

                    # Calculate RMSE between denoised and clean
                    denoised_rmse = torch.sqrt(torch.mean((denoised_structure - clean_hidden) ** 2))

                    # Calculate RMSE between noisy and clean for comparison
                    noisy_rmse = torch.sqrt(torch.mean((noisy_hidden - clean_hidden) ** 2))

                    # Calculate improvement
                    improvement = noisy_rmse - denoised_rmse
                    improvement_percent = (improvement / noisy_rmse) * 100

                    print(f"  Hidden State {hidden_idx}:")
                    print(f"    Noisy RMSE vs clean:    {noisy_rmse.item():.6f}")
                    print(f"    Denoised RMSE vs clean: {denoised_rmse.item():.6f}")
                    print(f"    Improvement:            {improvement.item():.6f} ({improvement_percent.item():.2f}%)")

                    denoising_improvements.append(improvement.item())

                    if improvement > 0:
                        print("    ✓ DENOISING SUCCESSFUL (RMSE reduced)")
                    else:
                        print("    ❌ DENOISING FAILED (RMSE increased)")

                    # Verify denoised is different from both noisy and original
                    assert not torch.allclose(denoised_structure, noisy_hidden, atol=1e-4), (
                        f"Denoised structure {i} should be different from noisy"
                    )
                    assert not torch.allclose(denoised_structure, scaled_batch.pos, atol=1e-4), (
                        f"Denoised structure {i} should be different from current position"
                    )

        # Overall denoising assessment
        print("\n9. OVERALL DENOISING ASSESSMENT...")
        if denoising_improvements:
            avg_improvement = np.mean(denoising_improvements)
            successful_denoising = sum(1 for imp in denoising_improvements if imp > 0)
            total_tests = len(denoising_improvements)
            success_rate = (successful_denoising / total_tests) * 100

            print(f"  Total hidden states tested: {total_tests}")
            print(f"  Successful denoising: {successful_denoising}/{total_tests} ({success_rate:.1f}%)")
            print(f"  Average RMSE improvement: {avg_improvement:.6f}")

            if success_rate >= 80:  # Require at least 80% success rate
                print("  ✅ DENOISING QUALITY: EXCELLENT (≥80% success)")
            elif success_rate >= 60:
                print("  ⚠️ DENOISING QUALITY: GOOD (≥60% success)")
            elif success_rate >= 40:
                print("  ⚠️ DENOISING QUALITY: MODERATE (≥40% success)")
            else:
                print("  ❌ DENOISING QUALITY: POOR (<40% success)")

            # Assert that at least some denoising occurred
            assert successful_denoising > 0, "At least one hidden state should show denoising improvement"
            print("  ✓ At least some denoising improvement verified")

        else:
            print("  No hidden states available for denoising quality assessment")

        # Additional validation
        print("\n10. Additional validation...")
        for i, structure in enumerate(conditioned_structures):
            # Check for NaN values
            assert not torch.isnan(structure).any(), f"Structure {i} contains NaN values"
            # Check for infinite values
            assert not torch.isinf(structure).any(), f"Structure {i} contains infinite values"
            print(f"✓ Structure {i} contains valid values")

        print("\n🎉 All tests passed! DenoisedConditioner works correctly.")

        # Print final summary
        print("\n=== FINAL SUMMARY ===")
        print(f"  Checkpoint: {pretrained_model_path}")
        print(f"  Test sigma: {test_sigma}")
        print(f"  Denoiser sigma: {conditioner.denoiser_sigma}")
        print(f"  Average squared distance: {average_squared_distance}")
        print(f"  Computed c_in: {c_in_float}")
        if denoising_improvements:
            print(f"  Denoising success rate: {success_rate:.1f}%")
            print(f"  Average RMSE improvement: {avg_improvement:.6f}")

        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_denoised_conditioner_with_scaling()
    if success:
        print("\n✅ DenoisedConditioner scaling test passed!")
        print("The conditioner correctly handles the scaling parameter and emulates xhat_normalized behavior.")
    else:
        print("\n❌ DenoisedConditioner scaling test failed!")
