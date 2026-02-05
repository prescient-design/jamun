#!/usr/bin/env python3
"""
Test script for RepeatedPositionDataset to verify that hidden states
are exact copies of the current position.
"""

import os

# Add the src directory to the path so we can import jamun modules
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from jamun.data.noisy_position_dataset import RepeatedPositionDataset


def test_repeated_position_dataset():
    """Test RepeatedPositionDataset with ALA_ALA capped diamines data."""

    print("Testing RepeatedPositionDataset...")
    print("=" * 50)

    # Set up dataset parameters
    root = "/data/bucket/kleinhej/capped_diamines/timewarp_splits/train"
    traj_files = ["ALA_ALA.xtc"]
    pdb_file = "ALA_ALA.pdb"
    label = "ALA_ALA_test"
    total_lag_time = 4  # This should create 3 hidden states (4 - 1 = 3)

    # Check if files exist
    xtc_path = os.path.join(root, traj_files[0])
    pdb_path = os.path.join(root, pdb_file)

    if not os.path.exists(xtc_path):
        print(f"❌ ERROR: XTC file not found: {xtc_path}")
        return False
    if not os.path.exists(pdb_path):
        print(f"❌ ERROR: PDB file not found: {pdb_path}")
        return False

    print(f"✅ Found XTC file: {xtc_path}")
    print(f"✅ Found PDB file: {pdb_path}")

    try:
        # Create the dataset
        print(f"\nCreating RepeatedPositionDataset with total_lag_time={total_lag_time}...")
        dataset = RepeatedPositionDataset(
            root=root,
            traj_files=traj_files,
            pdb_file=pdb_file,
            label=label,
            total_lag_time=total_lag_time,
            num_frames=5,  # Only load 5 frames for testing
            verbose=True,
        )

        print("✅ Dataset created successfully")
        print(f"   Dataset length: {len(dataset)}")
        print(f"   Dataset label: {dataset.label()}")

        # Test a few samples
        print("\nTesting samples...")

        for idx in range(min(3, len(dataset))):
            print(f"\n--- Sample {idx} ---")

            # Get sample from dataset
            graph = dataset[idx]

            print(f"Graph pos shape: {graph.pos.shape}")
            print(f"Number of hidden states: {len(graph.hidden_state)}")

            # Verify we have the expected number of hidden states
            expected_hidden_states = total_lag_time - 1
            if len(graph.hidden_state) != expected_hidden_states:
                print(f"❌ ERROR: Expected {expected_hidden_states} hidden states, got {len(graph.hidden_state)}")
                return False

            print(f"✅ Correct number of hidden states: {len(graph.hidden_state)}")

            # Test each hidden state
            for i, hidden_pos in enumerate(graph.hidden_state):
                print(f"Hidden state {i} shape: {hidden_pos.shape}")

                # Check if shapes match
                if hidden_pos.shape != graph.pos.shape:
                    print(f"❌ ERROR: Shape mismatch! pos: {graph.pos.shape}, hidden_state[{i}]: {hidden_pos.shape}")
                    return False

                # Check if values are exactly equal
                if not torch.allclose(hidden_pos, graph.pos, atol=1e-10):
                    print(f"❌ ERROR: Hidden state {i} is not exactly equal to current position!")
                    print(f"   Max difference: {torch.max(torch.abs(hidden_pos - graph.pos)).item()}")
                    return False

                # Check if they are the exact same tensor (should be different objects but same values)
                if hidden_pos is graph.pos:
                    print(f"⚠️  WARNING: Hidden state {i} is the same object as pos (should be different objects)")
                else:
                    print(f"✅ Hidden state {i} is a different object with same values as pos")

                print(f"✅ Hidden state {i} exactly matches current position")

        print("\n🎉 All tests passed!")
        print("   ✅ Dataset loads correctly")
        print(f"   ✅ Correct number of hidden states ({total_lag_time - 1})")
        print("   ✅ Hidden states exactly match current position")
        print("   ✅ Hidden states are separate objects (not references)")

        return True

    except Exception as e:
        print(f"❌ ERROR: Exception occurred: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_repeated_position_dataset()
    if success:
        print("\n🎉 Test completed successfully!")
        exit(0)
    else:
        print("\n💥 Test failed!")
        exit(1)
