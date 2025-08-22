#!/usr/bin/env python3
"""
Script to organize files from ALA_ALA dataset into train/val/test directories
with random 70/10/20 split.
"""

import os
import shutil
import random
from pathlib import Path
from typing import List


def get_files_from_directory(source_dir: str) -> List[str]:
    """Get all files from the source directory."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist")
    
    files = [f for f in source_path.iterdir() if f.is_file()]
    return files


def create_target_directories(base_dir: str) -> dict:
    """Create train/val/test directories and return their paths."""
    base_path = Path(base_dir)
    
    directories = {
        'train': base_path / 'train',
        'val': base_path / 'val', 
        'test': base_path / 'test'
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return directories


def split_files(files: List[Path], train_ratio: float = 0.7, val_ratio: float = 0.1, test_ratio: float = 0.2):
    """Split files randomly into train/val/test sets."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Shuffle files randomly
    files_copy = files.copy()
    random.shuffle(files_copy)
    
    total_files = len(files_copy)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    # Split the files
    train_files = files_copy[:train_count]
    val_files = files_copy[train_count:train_count + val_count]
    test_files = files_copy[train_count + val_count:]
    
    return {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }


def copy_files(file_splits: dict, target_dirs: dict, copy_mode: str = 'copy'):
    """Copy or move files to their respective directories."""
    for split_name, files in file_splits.items():
        target_dir = target_dirs[split_name]
        
        print(f"\n{copy_mode.capitalize()}ing {len(files)} files to {split_name} directory...")
        
        for file_path in files:
            target_path = target_dir / file_path.name
            
            if copy_mode == 'copy':
                shutil.copy2(file_path, target_path)
            elif copy_mode == 'move':
                shutil.move(str(file_path), str(target_path))
            else:
                raise ValueError("copy_mode must be either 'copy' or 'move'")
        
        print(f"Completed {split_name}: {len(files)} files")


def main():
    # Configuration
    source_directory = "/data2/sules/fake_enhanced_data/ALA_ALA"
    target_base_directory = "/data2/sules/fake_enhanced_data/ALA_ALA_organized"
    
    # Split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    print(f"Organizing files from: {source_directory}")
    print(f"Target directory: {target_base_directory}")
    print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
    
    try:
        # Get all files from source directory
        print("\nGetting files from source directory...")
        files = get_files_from_directory(source_directory)
        print(f"Found {len(files)} files")
        
        if len(files) == 0:
            print("No files found in source directory. Exiting.")
            return
        
        # Create target directories
        print("\nCreating target directories...")
        target_dirs = create_target_directories(target_base_directory)
        
        # Split files randomly
        print("\nSplitting files randomly...")
        file_splits = split_files(files, train_ratio, val_ratio, test_ratio)
        
        # Print split statistics
        print(f"\nSplit statistics:")
        for split_name, files_in_split in file_splits.items():
            percentage = (len(files_in_split) / len(files)) * 100
            print(f"  {split_name}: {len(files_in_split)} files ({percentage:.1f}%)")
        
        # Copy files to target directories
        print("\nCopying files...")
        copy_files(file_splits, target_dirs, copy_mode='copy')
        
        print(f"\n✅ Successfully organized {len(files)} files!")
        print(f"Files copied to: {target_base_directory}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 