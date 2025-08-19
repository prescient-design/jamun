#!/usr/bin/env python3
"""
Script to scrape grid point information from trajectory filenames in the enhanced_full_swarm dataset.

This script analyzes the filenames in /data2/sules/ALA_ALA_enhanced_full_swarm/train to extract
grid codes and provide statistics about the dataset structure.
"""

import os
import re
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import argparse

def extract_grid_code_from_filename(filename):
    """
    Extract grid code from trajectory filename.
    
    Expected format: swarm_1ps_{grid_code}_{traj_code}.xtc
    Example: swarm_1ps_042_003.xtc -> grid_code = 042
    
    Args:
        filename: Name of the trajectory file
        
    Returns:
        Grid code as string, or None if pattern doesn't match
    """
    # Pattern to match swarm trajectory files
    pattern = r'swarm_1ps_(\d{3})_(\d{3})\.xtc'
    match = re.match(pattern, filename)
    
    if match:
        grid_code = match.group(1)
        traj_code = match.group(2)
        return grid_code, traj_code
    else:
        return None, None

def scrape_grid_points(data_dir, output_file=None):
    """
    Scrape grid point information from trajectory directory.
    
    Args:
        data_dir: Path to directory containing trajectory files
        output_file: Optional path to save results as CSV
        
    Returns:
        Dictionary with grid point statistics
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")
    
    print(f"Scraping grid points from: {data_dir}")
    print("=" * 60)
    
    # Collect grid point information
    grid_data = []
    grid_codes = set()
    traj_codes = set()
    grid_traj_counts = defaultdict(list)
    
    # Scan all files in directory
    all_files = list(data_path.iterdir())
    xtc_files = [f for f in all_files if f.suffix == '.xtc']
    
    print(f"Total files in directory: {len(all_files)}")
    print(f"XTC trajectory files: {len(xtc_files)}")
    print()
    
    # Process each XTC file
    for file_path in xtc_files:
        filename = file_path.name
        grid_code, traj_code = extract_grid_code_from_filename(filename)
        
        if grid_code is not None and traj_code is not None:
            grid_data.append({
                'filename': filename,
                'grid_code': grid_code,
                'traj_code': traj_code,
                'grid_point': int(grid_code),
                'trajectory': int(traj_code)
            })
            
            grid_codes.add(grid_code)
            traj_codes.add(traj_code)
            grid_traj_counts[grid_code].append(traj_code)
        else:
            print(f"Warning: Could not parse filename: {filename}")
    
    # Create DataFrame for analysis
    df = pd.DataFrame(grid_data)
    
    # Print statistics
    print("GRID POINT STATISTICS")
    print("=" * 60)
    print(f"Total valid trajectory files: {len(grid_data)}")
    print(f"Unique grid codes: {len(grid_codes)}")
    print(f"Unique trajectory codes: {len(traj_codes)}")
    print()
    
    print("Grid code range:")
    if grid_codes:
        grid_nums = sorted([int(gc) for gc in grid_codes])
        print(f"  Min: {min(grid_nums):03d}")
        print(f"  Max: {max(grid_nums):03d}")
        print(f"  Grid codes: {', '.join(sorted(grid_codes))}")
    print()
    
    print("Trajectory code range:")
    if traj_codes:
        traj_nums = sorted([int(tc) for tc in traj_codes])
        print(f"  Min: {min(traj_nums):03d}")
        print(f"  Max: {max(traj_nums):03d}")
        print(f"  Trajectory codes: {', '.join(sorted(traj_codes))}")
    print()
    
    # Trajectories per grid point
    trajs_per_grid = [len(trajs) for trajs in grid_traj_counts.values()]
    if trajs_per_grid:
        print("Trajectories per grid point:")
        print(f"  Min: {min(trajs_per_grid)}")
        print(f"  Max: {max(trajs_per_grid)}")
        print(f"  Mean: {sum(trajs_per_grid) / len(trajs_per_grid):.1f}")
        print()
        
        # Count distribution
        traj_count_dist = Counter(trajs_per_grid)
        print("Distribution of trajectories per grid:")
        for count, freq in sorted(traj_count_dist.items()):
            print(f"  {count} trajectories: {freq} grid points")
    print()
    
    # Check for missing trajectories
    if grid_codes and traj_codes:
        expected_total = len(grid_codes) * len(traj_codes)
        actual_total = len(grid_data)
        print(f"Expected files (grid × traj): {expected_total}")
        print(f"Actual files found: {actual_total}")
        if actual_total != expected_total:
            print(f"Missing files: {expected_total - actual_total}")
            
            # Find missing combinations
            missing = []
            for gc in grid_codes:
                for tc in traj_codes:
                    if tc not in grid_traj_counts[gc]:
                        missing.append(f"swarm_1ps_{gc}_{tc}.xtc")
            
            if missing and len(missing) <= 20:  # Only print if not too many
                print("Missing files:")
                for mf in missing:
                    print(f"  {mf}")
            elif missing:
                print(f"  (too many to list - {len(missing)} missing files)")
    print()
    
    # Sample of grid points
    if len(grid_data) > 0:
        print("Sample trajectories:")
        sample_size = min(10, len(grid_data))
        sample_df = df.sample(n=sample_size, random_state=42)
        for _, row in sample_df.iterrows():
            print(f"  {row['filename']} -> Grid: {row['grid_code']}, Traj: {row['traj_code']}")
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Also save summary statistics
        summary_file = output_path.with_suffix('.summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Grid Point Analysis Summary\n")
            f.write(f"Directory: {data_dir}\n")
            f.write(f"Total trajectory files: {len(grid_data)}\n")
            f.write(f"Unique grid codes: {len(grid_codes)}\n")
            f.write(f"Unique trajectory codes: {len(traj_codes)}\n")
            f.write(f"Grid codes: {', '.join(sorted(grid_codes))}\n")
            f.write(f"Trajectory codes: {', '.join(sorted(traj_codes))}\n")
        
        print(f"Summary saved to: {summary_file}")
    
    return {
        'data': df,
        'grid_codes': sorted(grid_codes),
        'traj_codes': sorted(traj_codes),
        'grid_traj_counts': dict(grid_traj_counts),
        'total_files': len(grid_data),
        'unique_grids': len(grid_codes),
        'unique_trajs': len(traj_codes)
    }

def main():
    parser = argparse.ArgumentParser(description='Scrape grid point information from trajectory files')
    parser.add_argument('--data-dir', '-d', 
                       default='/data2/sules/ALA_ALA_enhanced_full_swarm/train',
                       help='Directory containing trajectory files')
    parser.add_argument('--output', '-o',
                       help='Output CSV file for results')
    parser.add_argument('--also-val', action='store_true',
                       help='Also analyze validation set')
    
    args = parser.parse_args()
    
    # Analyze training set
    print("ANALYZING TRAINING SET")
    print("=" * 80)
    train_results = scrape_grid_points(args.data_dir, args.output)
    
    # Optionally analyze validation set
    if args.also_val:
        val_dir = args.data_dir.replace('/train', '/val')
        if os.path.exists(val_dir):
            print("\n" + "=" * 80)
            print("ANALYZING VALIDATION SET")
            print("=" * 80)
            val_output = args.output.replace('.csv', '_val.csv') if args.output else None
            val_results = scrape_grid_points(val_dir, val_output)
            
            # Compare train vs val
            print("\n" + "=" * 80)
            print("TRAIN vs VAL COMPARISON")
            print("=" * 80)
            train_grids = set(train_results['grid_codes'])
            val_grids = set(val_results['grid_codes'])
            
            print(f"Train grid codes: {len(train_grids)}")
            print(f"Val grid codes: {len(val_grids)}")
            print(f"Overlap: {len(train_grids & val_grids)}")
            
            if train_grids & val_grids:
                print("Warning: Train and validation sets have overlapping grid codes!")
                print(f"Overlapping codes: {sorted(train_grids & val_grids)}")
            else:
                print("Good: No overlap between train and validation grid codes")
        else:
            print(f"\nValidation directory not found: {val_dir}")

if __name__ == "__main__":
    main()
