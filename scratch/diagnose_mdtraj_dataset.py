#!/usr/bin/env python3
"""
Diagnostic script to understand MDtrajDataset subsampling and lag behavior.
"""

import os
import sys

import mdtraj as md

# Add the project root to the path
sys.path.insert(0, "/homefs/home/sules/jamun")

from jamun.data._mdtraj import MDtrajDataset, get_subsampled_indices


def test_trajectory_loading():
    """Test basic trajectory loading without any subsampling."""
    print("=" * 60)
    print("TESTING BASIC TRAJECTORY LOADING")
    print("=" * 60)

    # Use one of your actual trajectory files
    traj_file = "/data2/sules/ALA_ALA_enhanced_full_grid/train/swarm_1ps_000_001.xtc"
    pdb_file = "/data2/sules/ALA_ALA_enhanced_full_grid/train/swarm_1ps_000_001.pdb"

    print(f"Loading trajectory: {traj_file}")
    print(f"Loading topology: {pdb_file}")

    # Test direct mdtraj loading
    direct_traj = md.load(traj_file, top=pdb_file)
    print(f"Direct mdtraj load: {direct_traj.n_frames} frames, {direct_traj.n_atoms} atoms")

    # Test MDtrajDataset without any subsampling parameters
    print("\n--- Testing MDtrajDataset with default parameters ---")
    dataset = MDtrajDataset(
        root="/data2/sules/ALA_ALA_enhanced_full_grid/train",
        traj_files=["swarm_1ps_000_001.xtc"],
        pdb_file="swarm_1ps_000_001.pdb",
        label="test_basic",
        verbose=True,
    )

    print(f"MDtrajDataset length: {len(dataset)}")
    print(f"Dataset trajectory frames: {dataset.traj.n_frames}")
    print(f"Dataset trajectory atoms: {dataset.traj.n_atoms}")

    # Check if there are any default parameters being set
    print(f"Dataset num_frames param: {getattr(dataset, 'num_frames', 'Not set')}")
    print(f"Dataset start_frame param: {getattr(dataset, 'start_frame', 'Not set')}")
    print(f"Dataset subsample param: {getattr(dataset, 'subsample', 'Not set')}")


def test_subsampling_behavior():
    """Test different subsampling scenarios."""
    print("\n" + "=" * 60)
    print("TESTING SUBSAMPLING BEHAVIOR")
    print("=" * 60)

    base_params = {
        "root": "/data2/sules/ALA_ALA_enhanced_full_grid/train",
        "traj_files": ["swarm_1ps_000_001.xtc"],
        "pdb_file": "swarm_1ps_000_001.pdb",
        "label": "test_subsample",
        "verbose": True,
    }

    # Test 1: Explicit num_frames
    print("\n--- Test 1: Explicit num_frames ---")
    dataset1 = MDtrajDataset(**base_params, num_frames=100)
    print(f"With num_frames=100: {len(dataset1)} frames")

    # Test 2: Explicit num_frames = -1 (should load all)
    print("\n--- Test 2: num_frames=-1 (load all) ---")
    dataset2 = MDtrajDataset(**base_params, num_frames=-1)
    print(f"With num_frames=-1: {len(dataset2)} frames")

    # Test 3: No num_frames specified
    print("\n--- Test 3: No num_frames specified ---")
    dataset3 = MDtrajDataset(**base_params)
    print(f"With default num_frames: {len(dataset3)} frames")

    # Test 4: Explicit subsample
    print("\n--- Test 4: With subsample=2 ---")
    dataset4 = MDtrajDataset(**base_params, num_frames=-1, subsample=2)
    print(f"With subsample=2: {len(dataset4)} frames")


def test_lag_subsampling():
    """Test lag-based subsampling behavior."""
    print("\n" + "=" * 60)
    print("TESTING LAG SUBSAMPLING")
    print("=" * 60)

    # First get the actual trajectory length
    traj_file = "/data2/sules/ALA_ALA_enhanced_full_grid/train/swarm_1ps_000_001.xtc"
    pdb_file = "/data2/sules/ALA_ALA_enhanced_full_grid/train/swarm_1ps_000_001.pdb"
    direct_traj = md.load(traj_file, top=pdb_file)
    print(f"Actual trajectory length: {direct_traj.n_frames} frames")

    # Test the get_subsampled_indices function directly
    print("\n--- Testing get_subsampled_indices function ---")

    test_cases = [
        {"N": 50, "subsample": 1, "total_lag_time": 5, "lag_subsample_rate": 1},
        {"N": 250, "subsample": 1, "total_lag_time": 5, "lag_subsample_rate": 1},
        {"N": 50, "subsample": 1, "total_lag_time": 2, "lag_subsample_rate": 1},
    ]

    for i, params in enumerate(test_cases):
        print(f"\nTest case {i + 1}: {params}")
        try:
            indices = get_subsampled_indices(**params)
            print(f"  Result: {len(indices)} valid starting points")
            if len(indices) <= 5:
                print(f"  Indices: {indices}")
            else:
                print(f"  First 3 indices: {indices[:3]}")
                print(f"  Last 3 indices: {indices[-3:]}")
        except Exception as e:
            print(f"  Error: {e}")

    # Test actual MDtrajDataset with lag parameters
    print("\n--- Testing MDtrajDataset with lag parameters ---")

    base_params = {
        "root": "/data2/sules/ALA_ALA_enhanced_full_grid/train",
        "traj_files": ["swarm_1ps_000_001.xtc"],
        "pdb_file": "swarm_1ps_000_001.pdb",
        "label": "test_lag",
        "verbose": True,
    }

    # Test with different configurations
    lag_configs = [
        {"total_lag_time": 5, "lag_subsample_rate": 1},
        {"total_lag_time": 5, "lag_subsample_rate": 1, "num_frames": -1},
        {"total_lag_time": 2, "lag_subsample_rate": 1},
        {"total_lag_time": 5, "lag_subsample_rate": 1, "subsample": 1},
    ]

    for i, config in enumerate(lag_configs):
        print(f"\nLag config {i + 1}: {config}")
        try:
            dataset = MDtrajDataset(**base_params, **config)
            print(f"  Dataset length: {len(dataset)}")
            print(f"  Trajectory frames: {dataset.traj.n_frames}")
            if hasattr(dataset, "hidden_state") and dataset.hidden_state:
                print(f"  Hidden states: {len(dataset.hidden_state)} sets")
                if len(dataset.hidden_state) > 0:
                    print(f"  Hidden state 0 length: {len(dataset.hidden_state[0])}")
            print(f"  Lagged indices available: {dataset.lagged_indices is not None}")
        except Exception as e:
            print(f"  Error: {e}")


def test_configuration_parsing():
    """Test how the configuration parameters are being processed."""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION PARAMETER PROCESSING")
    print("=" * 60)

    # Simulate the exact configuration from your experiment
    print("Simulating experiment configuration:")
    config = {
        "root": "/data2/sules/ALA_ALA_enhanced_full_grid/train",
        "traj_pattern": "^(.*).xtc",
        "pdb_pattern": "^(.*).pdb",
        "subsample": 1,
        "total_lag_time": 5,
        "lag_subsample_rate": 1,
        "max_datasets": 1,  # This limits to 1 dataset
    }

    print(f"Config: {config}")

    # This should be what parse_datasets_from_directory creates
    from jamun.data._utils import parse_datasets_from_directory

    print("\nCreating datasets with parse_datasets_from_directory...")
    try:
        datasets = parse_datasets_from_directory(**config)
        print(f"Number of datasets created: {len(datasets)}")

        for i, dataset in enumerate(datasets[:3]):  # Show first 3
            print(f"\nDataset {i}: {dataset.label()}")
            print(f"  Length: {len(dataset)}")
            print(f"  Trajectory frames: {dataset.traj.n_frames}")
            print(f"  Has hidden states: {dataset.hidden_state is not None}")
            if dataset.hidden_state:
                print(f"  Hidden states count: {len(dataset.hidden_state)}")

    except Exception as e:
        print(f"Error creating datasets: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all diagnostic tests."""
    print("MDTRAJ DATASET DIAGNOSTIC SCRIPT")
    print("=" * 60)

    # Check if files exist
    test_files = [
        "/data2/sules/ALA_ALA_enhanced_full_grid/train/swarm_1ps_000_001.xtc",
        "/data2/sules/ALA_ALA_enhanced_full_grid/train/swarm_1ps_000_001.pdb",
    ]

    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            return

    try:
        test_trajectory_loading()
        test_subsampling_behavior()
        test_lag_subsampling()
        test_configuration_parsing()

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
