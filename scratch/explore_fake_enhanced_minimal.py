#!/usr/bin/env python3
"""
Minimal script to explore fake enhanced dataset trajectory counts.

Answers: How many trajectories do we get when subsample=10, total_lag_time=5, lag_subsample_rate=1?
"""

import logging
import os
import sys

import dotenv

# Set up logging
logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("minimal_exploration")

# Load environment variables
dotenv.load_dotenv(".env", verbose=True)

# Add jamun to path
project_root = os.path.abspath(".")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jamun
import jamun.data

def quick_exploration():
    """Quick exploration of dataset with limited scope."""
    
    # Dataset parameters
    dataset_root = "/data2/sules/fake_enhanced_data/ALA_ALA_organized/train"  # Just train for speed
    pdb_file = "/data/bucket/kleinhej/capped_diamines/timewarp_splits/train/ALA_ALA.pdb"
    traj_pattern = "^(.*).xtc"
    
    # User's parameters
    subsample_rate = 10
    total_lag_time = 5
    lag_subsample_rate = 1
    
    py_logger.info("MINIMAL EXPLORATION OF FAKE ENHANCED DATASET")
    py_logger.info("=" * 60)
    py_logger.info(f"Parameters: subsample={subsample_rate}, total_lag_time={total_lag_time}, lag_subsample_rate={lag_subsample_rate}")
    py_logger.info("=" * 60)
    
    try:
        # Limit to first 5 datasets for speed
        py_logger.info("Loading first 5 datasets from train split...")
        datasets = jamun.data.parse_datasets_from_directory(
            root=dataset_root,
            traj_pattern=traj_pattern,
            pdb_file=pdb_file,
            as_iterable=False,
            subsample=subsample_rate,
            total_lag_time=total_lag_time,
            lag_subsample_rate=lag_subsample_rate,
            max_datasets=5,  # LIMIT for speed
            verbose=True
        )
        
        py_logger.info(f"Successfully loaded {len(datasets)} datasets")
        
        if datasets:
            # Analyze each dataset
            total_frames = 0
            for i, dataset in enumerate(datasets):
                frames = len(dataset)
                total_frames += frames
                py_logger.info(f"Dataset {i+1} ('{dataset.label()}'): {frames} frames")
                
                # Check first dataset in detail
                if i == 0:
                    sample = dataset[0]
                    py_logger.info(f"  Sample position shape: {sample.pos.shape}")
                    if hasattr(sample, 'hidden_state') and sample.hidden_state:
                        py_logger.info(f"  Hidden state: {len(sample.hidden_state)} lag frames")
                        py_logger.info(f"  First hidden state shape: {sample.hidden_state[0].shape}")
                    else:
                        py_logger.info(f"  No hidden state found")
            
            py_logger.info("-" * 40)
            py_logger.info(f"TOTAL FRAMES across {len(datasets)} datasets: {total_frames}")
            py_logger.info(f"Average frames per dataset: {total_frames / len(datasets):.1f}")
            
            # Extrapolate to estimate full dataset
            py_logger.info("\nESTIMATING FULL DATASET:")
            py_logger.info("Assuming all datasets have similar sizes...")
            
            # We could try to count total datasets but that might be slow
            # Instead, let's just report what we found
            py_logger.info(f"With subsample={subsample_rate}, each dataset gives ~{total_frames/len(datasets):.0f} trajectories")
            py_logger.info(f"Total lag time {total_lag_time} creates hidden states for conditional training")
            
        else:
            py_logger.warning("No datasets found!")
            
    except Exception as e:
        py_logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_different_subsample_rates():
    """Test how trajectory count changes with different subsample rates."""
    
    py_logger.info("\n" + "=" * 60)
    py_logger.info("TESTING DIFFERENT SUBSAMPLE RATES")
    py_logger.info("=" * 60)
    
    dataset_root = "/data2/sules/fake_enhanced_data/ALA_ALA_organized/train"
    pdb_file = "/data/bucket/kleinhej/capped_diamines/timewarp_splits/train/ALA_ALA.pdb"
    traj_pattern = "^(.*).xtc"
    
    # Test different subsample rates (keeping lag parameters constant)
    test_params = [
        {"subsample": 1, "total_lag_time": 5, "lag_subsample_rate": 1, "desc": "No subsampling"},
        {"subsample": 5, "total_lag_time": 5, "lag_subsample_rate": 1, "desc": "Subsample 5"},
        {"subsample": 10, "total_lag_time": 5, "lag_subsample_rate": 1, "desc": "User's parameters (subsample 10)"},
        {"subsample": 20, "total_lag_time": 5, "lag_subsample_rate": 1, "desc": "Subsample 20"},
    ]
    
    for params in test_params:
        py_logger.info(f"\nTesting: {params['desc']}")
        py_logger.info(f"  subsample={params['subsample']}, total_lag_time={params['total_lag_time']}, lag_subsample_rate={params['lag_subsample_rate']}")
        
        try:
            # Load just one dataset for comparison
            datasets = jamun.data.parse_datasets_from_directory(
                root=dataset_root,
                traj_pattern=traj_pattern,
                pdb_file=pdb_file,
                as_iterable=False,
                subsample=params['subsample'],
                total_lag_time=params['total_lag_time'],
                lag_subsample_rate=params['lag_subsample_rate'],
                max_datasets=1,  # Just one dataset for speed
                verbose=False
            )
            
            if datasets:
                frames = len(datasets[0])
                py_logger.info(f"  -> {frames} frames in first dataset")
            else:
                py_logger.info(f"  -> No datasets found")
                
        except Exception as e:
            py_logger.info(f"  -> Error: {e}")

if __name__ == "__main__":
    quick_exploration()
    test_different_subsample_rates() 