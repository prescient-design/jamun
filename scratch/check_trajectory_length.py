#!/usr/bin/env python3
"""
Check the raw trajectory length in xtc files.
"""

import logging
import os
import sys
import mdtraj as md

# Set up logging
logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("traj_length_check")

def check_trajectory_lengths():
    """Check the length of trajectories in raw xtc files."""
    
    dataset_root = "/data2/sules/fake_enhanced_data/ALA_ALA_organized/train"
    pdb_file = "/data/bucket/kleinhej/capped_diamines/timewarp_splits/train/ALA_ALA.pdb"
    
    py_logger.info("CHECKING RAW TRAJECTORY LENGTHS")
    py_logger.info("=" * 50)
    
    # Get a few trajectory files to sample
    import glob
    xtc_files = glob.glob(os.path.join(dataset_root, "*.xtc"))
    
    py_logger.info(f"Found {len(xtc_files)} total xtc files")
    py_logger.info("Checking first 10 files...")
    
    lengths = []
    
    for i, xtc_file in enumerate(xtc_files[:10]):
        try:
            # Load trajectory with topology
            traj = md.load(xtc_file, top=pdb_file)
            length = traj.n_frames
            lengths.append(length)
            
            filename = os.path.basename(xtc_file)
            py_logger.info(f"{i+1:2d}. {filename}: {length} frames")
            
        except Exception as e:
            py_logger.error(f"Error loading {xtc_file}: {e}")
    
    if lengths:
        py_logger.info("-" * 50)
        py_logger.info(f"Statistics from {len(lengths)} files:")
        py_logger.info(f"  Minimum length: {min(lengths)} frames")
        py_logger.info(f"  Maximum length: {max(lengths)} frames")
        py_logger.info(f"  Average length: {sum(lengths)/len(lengths):.1f} frames")
        py_logger.info(f"  All lengths: {sorted(set(lengths))}")
        
        # Show how subsampling affects this
        py_logger.info("\nEffect of subsampling (with lag requirements):")
        original_length = sum(lengths) / len(lengths)
        
        # The lag requirements mean we need at least total_lag_time * lag_subsample_rate frames
        # to get any output, and then we lose some frames at the beginning
        test_cases = [
            {"subsample": 1, "total_lag_time": 5, "lag_subsample_rate": 1},
            {"subsample": 5, "total_lag_time": 5, "lag_subsample_rate": 1}, 
            {"subsample": 10, "total_lag_time": 5, "lag_subsample_rate": 1},
            {"subsample": 20, "total_lag_time": 5, "lag_subsample_rate": 1},
        ]
        
        for params in test_cases:
            # Estimate how many frames we'd get after subsampling and lag filtering
            # The algorithm starts from frames that have enough history
            min_start_frame = (params["total_lag_time"] - 1) * params["lag_subsample_rate"]
            available_frames = max(0, original_length - min_start_frame)
            subsampled_frames = available_frames // params["subsample"]
            
            py_logger.info(f"  subsample={params['subsample']:2d}: ~{subsampled_frames:.0f} frames "
                          f"(from {original_length:.0f} original)")

if __name__ == "__main__":
    check_trajectory_lengths() 