#!/usr/bin/env python3
"""
Script to reorganize ALA_ALA swarm results data.

This script takes data from /data/bucket/vanib/ALA_ALA/swarm_results/ and organizes it 
into /data2/sules/ALA_ALA_enhanced/ with train/val splits.

Input structure:
- /data/bucket/vanib/ALA_ALA/swarm_results/AA_{grid_code}/
  - swarm_1ps_{traj_code}.xtc (where traj_code is 001-005)
- /data/bucket/vanib/ALA_ALA/ALA_ALA.pdb (single PDB file to use for all)

Output structure:
- /data2/sules/ALA_ALA_enhanced/train/
  - swarm_1ps_{grid_code}_{traj_code}.xtc
  - swarm_1ps_{grid_code}_{traj_code}.pdb
- /data2/sules/ALA_ALA_enhanced/val/
  - swarm_1ps_{grid_code}_{traj_code}.xtc  
  - swarm_1ps_{grid_code}_{traj_code}.pdb

The train folder contains 172 randomly sampled grid codes, val contains the remaining 12.
Each grid code has 5 swarms (001-005), so:
- Train: 172 × 5 = 860 .xtc files + 860 .pdb files
- Val: 12 × 5 = 60 .xtc files + 60 .pdb files
"""

import os
import shutil
import random
import logging
from pathlib import Path
from typing import List, Tuple

try:
    import mdtraj as md
    import numpy as np
    MDTRAJ_AVAILABLE = True
except ImportError:
    MDTRAJ_AVAILABLE = False
    logging.warning("mdtraj not available. Trajectory validation will be skipped.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logging.warning("tqdm not available. Progress bars will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SOURCE_DIR = "/data/bucket/vanib/ALA_ALA/swarms/swarm_results"
SINGLE_PDB_FILE = "/data/bucket/vanib/ALA_ALA/swarms/ALA_ALA.pdb"
TRAJECTORY_CODES = ['001', '002', '003', '004', '005']
LONG_TRAJECTORY_CODES = ['001', '003']  # For 2000ps trajectories

# Splitting strategies
SPLITTING_STRATEGIES = {
    'grid_split': {
        'target_dir': "/data2/sules/ALA_ALA_enhanced_full_swarm",
        'train_size': 172,  # Number of grid codes for training
        'description': "Random grid codes split: 172 grids for train, 12 grids for val, all trajectories"
    },
    'trajectory_split': {
        'target_dir': "/data2/sules/ALA_ALA_enhanced_full_grid", 
        'train_trajectories': ['001', '002', '003', '004'],  # First 4 trajectories for train
        'val_trajectories': ['005'],  # Last trajectory for val
        'description': "All grids split by trajectory: trajectories 001-004 for train, 005 for val"
    },
    'long_grid_split': {
        'target_dir': "/data2/sules/ALA_ALA_enhanced_long",
        'trajectory_codes': ['001', '003'],  # Only 2000ps trajectories
        'train_size': 172,  # Number of grid codes for training
        'description': "Random grid codes split for 2000ps trajectories: 172 grids for train, 12 grids for val"
    },
    'state_split': {
        'target_dir': "/data2/sules/ALA_ALA_enhanced_long_state_split",
        'trajectory_codes': ['001', '003'],  # Only 2000ps trajectories
        'phi_range': (0, 100),  # First residue phi range for validation set
        'psi_range': (-50, 100),  # First residue psi range for validation set
        'description': "Split by conformational state: trajectories with first residue phi,psi in (0,100)x(-50,100) go to val, others to train"
    }
}

def get_all_grid_codes(source_dir: str) -> List[str]:
    """
    Get all grid codes from the source directory.
    
    Args:
        source_dir: Path to the swarm results directory
        
    Returns:
        List of grid codes (e.g., ['000', '001', '002', ...])
    """
    grid_codes = []
    for item in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, item)) and item.startswith('AA_'):
            grid_code = item[3:]  # Remove 'AA_' prefix
            grid_codes.append(grid_code)
    
    grid_codes.sort()
    logger.info(f"Found {len(grid_codes)} grid codes")
    return grid_codes

def split_train_val(grid_codes: List[str], train_size: int, random_seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Randomly split grid codes into train and validation sets.
    
    Args:
        grid_codes: List of all grid codes
        train_size: Number of grid codes for training
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_grid_codes, val_grid_codes)
    """
    random.seed(random_seed)
    shuffled_codes = grid_codes.copy()
    random.shuffle(shuffled_codes)
    
    train_codes = shuffled_codes[:train_size]
    val_codes = shuffled_codes[train_size:]
    
    logger.info(f"Train set: {len(train_codes)} grid codes")
    logger.info(f"Val set: {len(val_codes)} grid codes")
    
    return train_codes, val_codes

def create_target_directories(target_dir: str):
    """Create target directory structure."""
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    logger.info(f"Created directories: {train_dir}, {val_dir}")

def copy_files_for_grid_split(
    source_dir: str,
    target_dir: str,
    grid_codes: List[str],
    trajectory_codes: List[str],
    single_pdb_file: str,
    split_name: str,
    use_2000ps: bool = False
):
    """
    Copy and rename files for a specific split (train or val).
    
    Args:
        source_dir: Source swarm results directory
        target_dir: Target directory for this split
        grid_codes: List of grid codes for this split
        trajectory_codes: List of trajectory codes (001-005)
        single_pdb_file: Path to the single PDB file to copy
        split_name: Name of the split for logging
        use_2000ps: If True, use swarm_2000ps_*.xtc files instead of swarm_1ps_*.xtc
    """
    total_operations = len(grid_codes) * len(trajectory_codes) * 2  # ×2 for .xtc and .pdb
    
    # Create progress bar
    if TQDM_AVAILABLE:
        pbar = tqdm(
            total=total_operations,
            desc=f"Copying {split_name} files",
            unit="files",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    
    copied_files = 0
    missing_files = 0
    
    if use_2000ps:
        traj_prefix = "swarm_2000ps"
    else:
        traj_prefix = "swarm_1ps"
    for grid_code in grid_codes:
        source_grid_dir = os.path.join(source_dir, f"AA_{grid_code}")
        
        if not os.path.exists(source_grid_dir):
            logger.warning(f"Source directory does not exist: {source_grid_dir}")
            # Skip all files for this grid code
            if TQDM_AVAILABLE:
                pbar.update(len(trajectory_codes) * 2)
            continue
            
        for traj_code in trajectory_codes:
            # Handle .xtc file
            source_xtc = os.path.join(source_grid_dir, f"{traj_prefix}_{traj_code}.xtc")
            target_xtc = os.path.join(target_dir, f"{traj_prefix}_{grid_code}_{traj_code}.xtc")
            
            if os.path.exists(source_xtc):
                shutil.copy2(source_xtc, target_xtc)
                copied_files += 1
            else:
                logger.warning(f"Source file does not exist: {source_xtc}")
                missing_files += 1
            
            if TQDM_AVAILABLE:
                pbar.update(1)
            
            # Handle .pdb file (copy the single PDB file)
            target_pdb = os.path.join(target_dir, f"{traj_prefix}_{grid_code}_{traj_code}.pdb")
            if os.path.exists(single_pdb_file):
                shutil.copy2(single_pdb_file, target_pdb)
                copied_files += 1
            else:
                logger.error(f"Single PDB file does not exist: {single_pdb_file}")
                missing_files += 1
            
            if TQDM_AVAILABLE:
                pbar.update(1)
    
    if TQDM_AVAILABLE:
        pbar.close()
    
    logger.info(f"{split_name}: Completed copying {copied_files} files ({missing_files} missing/failed)")

def copy_files_for_trajectory_split(
    source_dir: str,
    target_dir: str,
    all_grid_codes: List[str],
    trajectory_codes: List[str],
    single_pdb_file: str,
    split_name: str
):
    """
    Copy and rename files for trajectory-based split (all grids, specific trajectories).
    
    Args:
        source_dir: Source swarm results directory
        target_dir: Target directory for this split
        all_grid_codes: List of all grid codes to include
        trajectory_codes: List of trajectory codes for this split
        single_pdb_file: Path to the single PDB file to copy
        split_name: Name of the split for logging
    """
    total_operations = len(all_grid_codes) * len(trajectory_codes) * 2  # ×2 for .xtc and .pdb
    
    # Create progress bar
    if TQDM_AVAILABLE:
        pbar = tqdm(
            total=total_operations,
            desc=f"Copying {split_name} files",
            unit="files",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    
    copied_files = 0
    missing_files = 0
    
    for grid_code in all_grid_codes:
        source_grid_dir = os.path.join(source_dir, f"AA_{grid_code}")
        
        if not os.path.exists(source_grid_dir):
            logger.warning(f"Source directory does not exist: {source_grid_dir}")
            # Skip all files for this grid code
            if TQDM_AVAILABLE:
                pbar.update(len(trajectory_codes) * 2)
            continue
            
        for traj_code in trajectory_codes:
            # Handle .xtc file
            source_xtc = os.path.join(source_grid_dir, f"swarm_1ps_{traj_code}.xtc")
            target_xtc = os.path.join(target_dir, f"swarm_1ps_{grid_code}_{traj_code}.xtc")
            
            if os.path.exists(source_xtc):
                shutil.copy2(source_xtc, target_xtc)
                copied_files += 1
            else:
                logger.warning(f"Source file does not exist: {source_xtc}")
                missing_files += 1
            
            if TQDM_AVAILABLE:
                pbar.update(1)
            
            # Handle .pdb file (copy the single PDB file)
            target_pdb = os.path.join(target_dir, f"swarm_1ps_{grid_code}_{traj_code}.pdb")
            if os.path.exists(single_pdb_file):
                shutil.copy2(single_pdb_file, target_pdb)
                copied_files += 1
            else:
                logger.error(f"Single PDB file does not exist: {single_pdb_file}")
                missing_files += 1
            
            if TQDM_AVAILABLE:
                pbar.update(1)
    
    if TQDM_AVAILABLE:
        pbar.close()
    
    logger.info(f"{split_name}: Completed copying {copied_files} files ({missing_files} missing/failed)")

def analyze_trajectory_state(xtc_path: str, pdb_path: str, phi_range: tuple, psi_range: tuple) -> bool:
    """
    Analyze a trajectory to determine if any point has first residue phi,psi in the specified ranges.
    
    Args:
        xtc_path: Path to trajectory file
        pdb_path: Path to topology file
        phi_range: Tuple of (min, max) for phi angles in degrees
        psi_range: Tuple of (min, max) for psi angles in degrees
        
    Returns:
        True if any point in trajectory has first residue phi,psi in the specified ranges
    """
    if not MDTRAJ_AVAILABLE:
        logger.error("mdtraj not available, cannot analyze trajectory states")
        return False
    
    try:
        # Load trajectory
        traj = md.load(xtc_path, top=pdb_path)
        traj = traj[:1000] # only use first 1000 frames to avoid memory issues
        # Compute phi and psi angles
        _, phi_angles = md.compute_phi(traj)
        _, psi_angles = md.compute_psi(traj)
        
        # Convert to degrees
        phi_deg = np.degrees(phi_angles)
        psi_deg = np.degrees(psi_angles)
        
        # Check first residue (index 0) for points in specified ranges
        first_phi_in_range = (phi_deg[:, 0] > phi_range[0]) & (phi_deg[:, 0] < phi_range[1])
        first_psi_in_range = (psi_deg[:, 0] > psi_range[0]) & (psi_deg[:, 0] < psi_range[1])
        first_residue_in_range = first_phi_in_range & first_psi_in_range
        
        # Return True if any point is in range
        has_points_in_range = np.any(first_residue_in_range)
        n_points_in_range = np.sum(first_residue_in_range)
        
        logger.debug(f"Trajectory {xtc_path}: {n_points_in_range}/{len(phi_deg)} points in target range")
        
        return has_points_in_range
        
    except Exception as e:
        logger.error(f"Failed to analyze trajectory {xtc_path}: {str(e)}")
        return False

def test_mdtraj_compatibility(target_dir: str, num_samples: int = 3):
    """
    Test that mdtraj can successfully load swarm + PDB combinations.
    
    Args:
        target_dir: Target directory containing train/val splits
        num_samples: Number of random samples to test from each split
    """
    if not MDTRAJ_AVAILABLE:
        logger.warning("⚠️  mdtraj not available, skipping trajectory compatibility tests")
        return True
    
    logger.info("=== TESTING MDTRAJ COMPATIBILITY ===")
    
    for split in ['train', 'val']:
        split_dir = os.path.join(target_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        # Get all .xtc files
        xtc_files = [f for f in os.listdir(split_dir) if f.endswith('.xtc')]
        
        if not xtc_files:
            logger.warning(f"No .xtc files found in {split} directory")
            continue
            
        # Sample a few files to test
        test_files = random.sample(xtc_files, min(num_samples, len(xtc_files)))
        
        success_count = 0
        for xtc_file in test_files:
            # Get corresponding PDB file
            base_name = xtc_file.replace('.xtc', '')
            pdb_file = f"{base_name}.pdb"
            
            xtc_path = os.path.join(split_dir, xtc_file)
            pdb_path = os.path.join(split_dir, pdb_file)
            
            if not os.path.exists(pdb_path):
                logger.error(f"Missing PDB file: {pdb_path}")
                continue
                
            try:
                # Try to load trajectory with mdtraj
                traj = md.load(xtc_path, top=pdb_path)
                logger.info(f"✅ {split}: Successfully loaded {xtc_file} + {pdb_file} "
                           f"({traj.n_frames} frames, {traj.n_atoms} atoms)")
                success_count += 1
                
                # Clean up memory
                del traj
                
            except Exception as e:
                logger.error(f"❌ {split}: Failed to load {xtc_file} + {pdb_file}: {str(e)}")
        
        logger.info(f"{split}: {success_count}/{len(test_files)} trajectory tests passed")
    
    logger.info("mdtraj compatibility testing completed")
    return True

def verify_output(target_dir: str, expected_train_files: int, expected_val_files: int):
    """
    Verify the output directory structure and file counts.
    
    Args:
        target_dir: Target directory path
        expected_train_files: Expected number of files in train directory
        expected_val_files: Expected number of files in val directory
    """
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    
    train_files = len([f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))])
    val_files = len([f for f in os.listdir(val_dir) if os.path.isfile(os.path.join(val_dir, f))])
    
    train_xtc = len([f for f in os.listdir(train_dir) if f.endswith('.xtc')])
    train_pdb = len([f for f in os.listdir(train_dir) if f.endswith('.pdb')])
    val_xtc = len([f for f in os.listdir(val_dir) if f.endswith('.xtc')])
    val_pdb = len([f for f in os.listdir(val_dir) if f.endswith('.pdb')])
    
    logger.info("=== VERIFICATION RESULTS ===")
    logger.info(f"Train directory: {train_files} total files ({train_xtc} .xtc, {train_pdb} .pdb)")
    logger.info(f"Val directory: {val_files} total files ({val_xtc} .xtc, {val_pdb} .pdb)")
    logger.info(f"Expected train files: {expected_train_files}")
    logger.info(f"Expected val files: {expected_val_files}")
    
    if train_files == expected_train_files and val_files == expected_val_files:
        logger.info("✅ File counts match expectations!")
        
        # Test mdtraj compatibility
        test_mdtraj_compatibility(target_dir)
        
    else:
        logger.warning("❌ File counts do not match expectations!")

def reorganize_with_long_grid_split(grid_codes: List[str], strategy_config: dict):
    """Reorganize 2000ps data using grid-based splitting strategy."""
    target_dir = strategy_config['target_dir']
    train_size = strategy_config['train_size']
    trajectory_codes = strategy_config['trajectory_codes']
    
    logger.info(f"Using long grid split strategy: {strategy_config['description']}")
    
    if len(grid_codes) < train_size:
        logger.error(f"Not enough grid codes found. Expected at least {train_size}, found {len(grid_codes)}")
        return
    
    # Split into train and validation
    train_codes, val_codes = split_train_val(grid_codes, train_size)
    
    # Create target directories
    create_target_directories(target_dir)
    
    # Copy files for train split (using 2000ps trajectories)
    logger.info("Copying 2000ps files for train split...")
    copy_files_for_grid_split(
        SOURCE_DIR,
        os.path.join(target_dir, 'train'),
        train_codes,
        trajectory_codes,
        SINGLE_PDB_FILE,
        'TRAIN',
        use_2000ps=True
    )
    
    # Copy files for val split (using 2000ps trajectories)
    logger.info("Copying 2000ps files for val split...")
    copy_files_for_grid_split(
        SOURCE_DIR,
        os.path.join(target_dir, 'val'),
        val_codes,
        trajectory_codes,
        SINGLE_PDB_FILE,
        'VAL',
        use_2000ps=True
    )
    
    # Verify output
    expected_train_files = len(train_codes) * len(trajectory_codes) * 2  # ×2 for .xtc and .pdb
    expected_val_files = len(val_codes) * len(trajectory_codes) * 2
    
    verify_output(target_dir, expected_train_files, expected_val_files)

def reorganize_with_grid_split(grid_codes: List[str], strategy_config: dict):
    """Reorganize data using grid-based splitting strategy."""
    target_dir = strategy_config['target_dir']
    train_size = strategy_config['train_size']
    
    logger.info(f"Using grid split strategy: {strategy_config['description']}")
    
    if len(grid_codes) < train_size:
        logger.error(f"Not enough grid codes found. Expected at least {train_size}, found {len(grid_codes)}")
        return
    
    # Split into train and validation
    train_codes, val_codes = split_train_val(grid_codes, train_size)
    
    # Create target directories
    create_target_directories(target_dir)
    
    # Copy files for train split
    logger.info("Copying files for train split...")
    copy_files_for_grid_split(
        SOURCE_DIR,
        os.path.join(target_dir, 'train'),
        train_codes,
        TRAJECTORY_CODES,
        SINGLE_PDB_FILE,
        'TRAIN'
    )
    
    # Copy files for val split
    logger.info("Copying files for val split...")
    copy_files_for_grid_split(
        SOURCE_DIR,
        os.path.join(target_dir, 'val'),
        val_codes,
        TRAJECTORY_CODES,
        SINGLE_PDB_FILE,
        'VAL'
    )
    
    # Verify output
    expected_train_files = len(train_codes) * len(TRAJECTORY_CODES) * 2  # ×2 for .xtc and .pdb
    expected_val_files = len(val_codes) * len(TRAJECTORY_CODES) * 2
    
    verify_output(target_dir, expected_train_files, expected_val_files)

def reorganize_with_trajectory_split(grid_codes: List[str], strategy_config: dict):
    """Reorganize data using trajectory-based splitting strategy."""
    target_dir = strategy_config['target_dir']
    train_trajectories = strategy_config['train_trajectories']
    val_trajectories = strategy_config['val_trajectories']
    
    logger.info(f"Using trajectory split strategy: {strategy_config['description']}")
    
    # Create target directories
    create_target_directories(target_dir)
    
    # Copy files for train split (all grids, first 4 trajectories)
    logger.info("Copying files for train split...")
    copy_files_for_trajectory_split(
        SOURCE_DIR,
        os.path.join(target_dir, 'train'),
        grid_codes,
        train_trajectories,
        SINGLE_PDB_FILE,
        'TRAIN'
    )
    
    # Copy files for val split (all grids, last trajectory)
    logger.info("Copying files for val split...")
    copy_files_for_trajectory_split(
        SOURCE_DIR,
        os.path.join(target_dir, 'val'),
        grid_codes,
        val_trajectories,
        SINGLE_PDB_FILE,
        'VAL'
    )
    
    # Verify output
    expected_train_files = len(grid_codes) * len(train_trajectories) * 2  # ×2 for .xtc and .pdb
    expected_val_files = len(grid_codes) * len(val_trajectories) * 2
    
    verify_output(target_dir, expected_train_files, expected_val_files)

def copy_files_for_state_split(
    source_dir: str,
    target_dir: str,
    all_grid_codes: List[str],
    trajectory_codes: List[str],
    single_pdb_file: str,
    phi_range: tuple,
    psi_range: tuple
):
    """
    Copy and organize files based on conformational state analysis.
    
    Args:
        source_dir: Source swarm results directory
        target_dir: Target directory with train/val subdirectories
        all_grid_codes: List of all grid codes to process
        trajectory_codes: List of trajectory codes to include
        single_pdb_file: Path to the single PDB file to copy
        phi_range: Tuple of (min, max) for phi angles in degrees
        psi_range: Tuple of (min, max) for psi angles in degrees
    """
    if not MDTRAJ_AVAILABLE:
        logger.error("mdtraj not available, cannot perform state-based splitting")
        return
    
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    
    total_operations = len(all_grid_codes) * len(trajectory_codes) * 2  # ×2 for .xtc and .pdb
    
    # Create progress bar
    if TQDM_AVAILABLE:
        pbar = tqdm(
            total=total_operations,
            desc="Analyzing and copying files",
            unit="files",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    
    copied_train = 0
    copied_val = 0
    missing_files = 0
    analysis_errors = 0
    
    for grid_code in all_grid_codes:
        source_grid_dir = os.path.join(source_dir, f"AA_{grid_code}")
        
        if not os.path.exists(source_grid_dir):
            logger.warning(f"Source directory does not exist: {source_grid_dir}")
            # Skip all files for this grid code
            if TQDM_AVAILABLE:
                pbar.update(len(trajectory_codes) * 2)
            continue
            
        for traj_code in trajectory_codes:
            # Handle .xtc file - need to analyze it first
            source_xtc = os.path.join(source_grid_dir, f"swarm_2000ps_{traj_code}.xtc")
            
            if not os.path.exists(source_xtc):
                logger.warning(f"Source file does not exist: {source_xtc}")
                missing_files += 1
                if TQDM_AVAILABLE:
                    pbar.update(2)  # Skip both .xtc and .pdb
                continue
            
            # Analyze trajectory to determine train/val split
            try:
                goes_to_val = analyze_trajectory_state(source_xtc, single_pdb_file, phi_range, psi_range)
                
                if goes_to_val:
                    target_xtc = os.path.join(val_dir, f"swarm_2000ps_{grid_code}_{traj_code}.xtc")
                    target_pdb = os.path.join(val_dir, f"swarm_2000ps_{grid_code}_{traj_code}.pdb")
                    split_name = "VAL"
                    copied_val += 1
                else:
                    target_xtc = os.path.join(train_dir, f"swarm_2000ps_{grid_code}_{traj_code}.xtc")
                    target_pdb = os.path.join(train_dir, f"swarm_2000ps_{grid_code}_{traj_code}.pdb")
                    split_name = "TRAIN"
                    copied_train += 1
                
                # Copy .xtc file
                shutil.copy2(source_xtc, target_xtc)
                logger.debug(f"Copied {source_xtc} to {split_name}")
                
            except Exception as e:
                logger.error(f"Failed to analyze trajectory {source_xtc}: {str(e)}")
                analysis_errors += 1
                if TQDM_AVAILABLE:
                    pbar.update(2)  # Skip both .xtc and .pdb
                continue
            
            if TQDM_AVAILABLE:
                pbar.update(1)
            
            # Handle .pdb file (copy the single PDB file)
            if os.path.exists(single_pdb_file):
                shutil.copy2(single_pdb_file, target_pdb)
            else:
                logger.error(f"Single PDB file does not exist: {single_pdb_file}")
                missing_files += 1
            
            if TQDM_AVAILABLE:
                pbar.update(1)
    
    if TQDM_AVAILABLE:
        pbar.close()
    
    logger.info(f"State split completed:")
    logger.info(f"  TRAIN: {copied_train} trajectories")
    logger.info(f"  VAL: {copied_val} trajectories")
    logger.info(f"  Missing files: {missing_files}")
    logger.info(f"  Analysis errors: {analysis_errors}")

def reorganize_with_state_split(grid_codes: List[str], strategy_config: dict):
    """Reorganize data using conformational state-based splitting strategy."""
    target_dir = strategy_config['target_dir']
    trajectory_codes = strategy_config['trajectory_codes']
    phi_range = strategy_config['phi_range']
    psi_range = strategy_config['psi_range']
    
    logger.info(f"Using state split strategy: {strategy_config['description']}")
    logger.info(f"Target ranges: phi {phi_range}, psi {psi_range}")
    logger.info(f"Using trajectory codes: {trajectory_codes}")
    
    if not MDTRAJ_AVAILABLE:
        logger.error("mdtraj not available, cannot perform state-based splitting")
        return
    
    # Create target directories
    create_target_directories(target_dir)
    
    # Copy and split files based on conformational state
    logger.info("Analyzing trajectories and copying files...")
    copy_files_for_state_split(
        SOURCE_DIR,
        target_dir,
        grid_codes,
        trajectory_codes,
        SINGLE_PDB_FILE,
        phi_range,
        psi_range
    )
    
    # Note: We can't predict exact file counts since they depend on trajectory analysis
    logger.info("State-based reorganization completed!")

def main(strategy: str = 'trajectory_split'):
    """
    Main function to reorganize the swarm data.
    
    Args:
        strategy: Either 'grid_split' or 'trajectory_split'
    """
    logger.info("Starting swarm data reorganization...")
    
    # Validate input paths
    if not os.path.exists(SOURCE_DIR):
        logger.error(f"Source directory does not exist: {SOURCE_DIR}")
        return
    
    if not os.path.exists(SINGLE_PDB_FILE):
        logger.error(f"Single PDB file does not exist: {SINGLE_PDB_FILE}")
        return
    
    if strategy not in SPLITTING_STRATEGIES:
        logger.error(f"Invalid strategy: {strategy}. Choose from {list(SPLITTING_STRATEGIES.keys())}")
        return
    
    # Get all grid codes
    grid_codes = get_all_grid_codes(SOURCE_DIR)
    strategy_config = SPLITTING_STRATEGIES[strategy]
    
    # Execute the appropriate strategy
    if strategy == 'grid_split':
        reorganize_with_grid_split(grid_codes, strategy_config)
    elif strategy == 'trajectory_split':
        reorganize_with_trajectory_split(grid_codes, strategy_config)
    elif strategy == 'long_grid_split':
        reorganize_with_long_grid_split(grid_codes, strategy_config)
    elif strategy == 'state_split':
        reorganize_with_state_split(grid_codes, strategy_config)
    
    logger.info("Swarm data reorganization completed!")

if __name__ == "__main__":
    import sys
    
    # Default to trajectory_split for the new requirement
    strategy = 'trajectory_split'
    
    # Allow command line argument to choose strategy
    if len(sys.argv) > 1:
        strategy = sys.argv[1]
        
    print(f"Running reorganization with strategy: {strategy}")
    print(f"Description: {SPLITTING_STRATEGIES[strategy]['description']}")
    
    main(strategy) 