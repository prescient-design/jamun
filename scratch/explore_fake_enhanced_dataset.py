#!/usr/bin/env python3
"""
Script to explore the fake enhanced dataset using parse_datasets_from_directory.

Specifically explores how many trajectories we get when:
- subsample_rate = 10 (called 'subsample' in the function)
- total_lag_time = 5
- lag_subsample_rate = 1
"""

import logging
import os
import sys

import dotenv

# Set up logging
logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("fake_enhanced_exploration")

# Load environment variables
dotenv.load_dotenv(".env", verbose=True)
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")

# Add jamun to path if needed
project_root = os.path.abspath(".")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jamun
import jamun.data


def explore_dataset_parameters():
    """Explore the fake enhanced dataset with specified parameters."""

    # Dataset parameters as requested
    dataset_root = "/data2/sules/fake_enhanced_data/ALA_ALA_organized"
    pdb_file = "/data/bucket/kleinhej/capped_diamines/timewarp_splits/train/ALA_ALA.pdb"
    traj_pattern = "^(.*).xtc"

    # Subsampling parameters as specified by user
    subsample_rate = 10  # Called 'subsample' in the function
    total_lag_time = 5
    lag_subsample_rate = 1

    py_logger.info("=" * 60)
    py_logger.info("EXPLORING FAKE ENHANCED DATASET")
    py_logger.info("=" * 60)
    py_logger.info(f"Dataset root: {dataset_root}")
    py_logger.info(f"PDB file: {pdb_file}")
    py_logger.info(f"Trajectory pattern: {traj_pattern}")
    py_logger.info(f"Subsample rate: {subsample_rate}")
    py_logger.info(f"Total lag time: {total_lag_time}")
    py_logger.info(f"Lag subsample rate: {lag_subsample_rate}")
    py_logger.info("=" * 60)

    # Parse datasets for each split
    for split in ["train", "val", "test"]:
        py_logger.info(f"\n--- Exploring {split.upper()} split ---")

        try:
            datasets = jamun.data.parse_datasets_from_directory(
                root=f"{dataset_root}/{split}",
                traj_pattern=traj_pattern,
                pdb_file=pdb_file,
                as_iterable=False,
                subsample=subsample_rate,
                total_lag_time=total_lag_time,
                lag_subsample_rate=lag_subsample_rate,
                max_datasets=None,  # Load all datasets to get full count
                verbose=True,
            )

            py_logger.info(f"Number of datasets found: {len(datasets)}")

            if datasets:
                # Analyze first dataset in detail
                first_dataset = datasets[0]
                py_logger.info(f"First dataset label: {first_dataset.label()}")
                py_logger.info(f"Number of frames in first dataset: {len(first_dataset)}")

                # Check hidden state structure
                sample_data = first_dataset[0]
                if hasattr(sample_data, "hidden_state") and sample_data.hidden_state:
                    py_logger.info(f"Hidden state length: {len(sample_data.hidden_state)}")
                    py_logger.info(f"Shape of first hidden state: {sample_data.hidden_state[0].shape}")
                else:
                    py_logger.info("No hidden state found (expected for regular subsampling)")

                # Calculate total trajectories across all datasets
                total_frames = sum(len(dataset) for dataset in datasets)
                py_logger.info(f"Total frames across all datasets: {total_frames}")

                # Estimate original frames before subsampling
                original_frames_estimate = total_frames * subsample_rate
                py_logger.info(f"Estimated original frames (before subsampling): {original_frames_estimate}")

                # Show some dataset labels
                py_logger.info(f"First 5 dataset labels: {[ds.label() for ds in datasets[:5]]}")
                if len(datasets) > 5:
                    py_logger.info(f"... and {len(datasets) - 5} more datasets")

        except Exception as e:
            py_logger.error(f"Error processing {split} split: {e}")
            continue

    py_logger.info("\n" + "=" * 60)
    py_logger.info("EXPLORATION COMPLETE")
    py_logger.info("=" * 60)


def compare_with_different_parameters():
    """Compare trajectory counts with different subsampling parameters."""

    py_logger.info("\n" + "=" * 60)
    py_logger.info("PARAMETER COMPARISON")
    py_logger.info("=" * 60)

    dataset_root = "/data2/sules/fake_enhanced_data/ALA_ALA_organized/train"  # Just use train split
    pdb_file = "/data/bucket/kleinhej/capped_diamines/timewarp_splits/train/ALA_ALA.pdb"
    traj_pattern = "^(.*).xtc"

    # Test different parameter combinations
    test_cases = [
        {"subsample": 1, "total_lag_time": None, "lag_subsample_rate": None, "desc": "No subsampling"},
        {"subsample": 10, "total_lag_time": None, "lag_subsample_rate": None, "desc": "Subsample 10, no lag"},
        {"subsample": 10, "total_lag_time": 5, "lag_subsample_rate": 1, "desc": "User's requested parameters"},
        {"subsample": 10, "total_lag_time": 3, "lag_subsample_rate": 1, "desc": "Different lag time"},
        {"subsample": 5, "total_lag_time": 5, "lag_subsample_rate": 1, "desc": "Different subsample rate"},
    ]

    for i, params in enumerate(test_cases):
        py_logger.info(f"\nTest case {i + 1}: {params['desc']}")
        py_logger.info(
            f"Parameters: subsample={params['subsample']}, total_lag_time={params['total_lag_time']}, lag_subsample_rate={params['lag_subsample_rate']}"
        )

        try:
            # Limit to first few datasets for speed
            datasets = jamun.data.parse_datasets_from_directory(
                root=dataset_root,
                traj_pattern=traj_pattern,
                pdb_file=pdb_file,
                as_iterable=False,
                subsample=params["subsample"],
                total_lag_time=params["total_lag_time"],
                lag_subsample_rate=params["lag_subsample_rate"],
                max_datasets=3,  # Limit for speed
                verbose=False,
            )

            if datasets:
                frames_per_dataset = [len(ds) for ds in datasets]
                total_frames = sum(frames_per_dataset)
                py_logger.info(f"  -> {len(datasets)} datasets, {total_frames} total frames")
                py_logger.info(f"  -> Frames per dataset: {frames_per_dataset}")

                # Check if lagged data exists
                sample = datasets[0][0]
                if hasattr(sample, "hidden_state") and sample.hidden_state:
                    py_logger.info(f"  -> Hidden state length: {len(sample.hidden_state)}")
                else:
                    py_logger.info("  -> No hidden state")
            else:
                py_logger.warning("  -> No datasets found")

        except Exception as e:
            py_logger.error(f"  -> Error: {e}")


if __name__ == "__main__":
    # First explore with user's specific parameters
    explore_dataset_parameters()

    # Then compare with different parameters
    compare_with_different_parameters()
