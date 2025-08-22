#!/usr/bin/env python3
"""
Script to analyze the distribution of trajectories by grid codes.
Creates a histogram showing how many trajectory codes exist for each grid code.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import argparse

def parse_trajectory_files(data_dir):
    """Parse trajectory files and extract grid codes and traj codes."""
    # Pattern to match traj_{grid_code}_{traj_code}
    pattern = re.compile(r'^traj_(\d+)_(\d+)')
    
    grid_traj_mapping = defaultdict(list)
    
    # Scan directory for trajectory files
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory {data_dir} does not exist")
    
    files = os.listdir(data_dir)
    trajectory_files = []
    
    for filename in files:
        match = pattern.match(filename)
        if match:
            grid_code = int(match.group(1))
            traj_code = int(match.group(2))
            grid_traj_mapping[grid_code].append(traj_code)
            trajectory_files.append(filename)
    
    print(f"Found {len(trajectory_files)} trajectory files")
    print(f"Found {len(grid_traj_mapping)} unique grid codes")
    
    return grid_traj_mapping

def create_histogram(grid_traj_mapping, output_path=None):
    """Create histogram of trajectory counts per grid code."""
    
    if not grid_traj_mapping:
        print("No trajectory files found!")
        return
    
    # Get the full range of grid codes (min to max)
    all_grid_codes = list(grid_traj_mapping.keys())
    min_grid = min(all_grid_codes)
    max_grid = max(all_grid_codes)
    
    print(f"Grid code range: {min_grid} to {max_grid}")
    
    # Create array for all grid codes in range
    full_range = list(range(min_grid, max_grid + 1))
    traj_counts = []
    
    for grid_code in full_range:
        count = len(grid_traj_mapping.get(grid_code, []))
        traj_counts.append(count)
    
    # Print some statistics
    total_trajs = sum(traj_counts)
    non_zero_grids = sum(1 for count in traj_counts if count > 0)
    zero_grids = len(full_range) - non_zero_grids
    
    print(f"Total trajectories: {total_trajs}")
    print(f"Grid codes with trajectories: {non_zero_grids}")
    print(f"Grid codes with no trajectories: {zero_grids}")
    print(f"Average trajectories per grid code: {total_trajs / len(full_range):.2f}")
    print(f"Max trajectories for single grid code: {max(traj_counts)}")
    print(f"Min trajectories for single grid code: {min(traj_counts)}")
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    plt.bar(full_range, traj_counts, width=0.8, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.xlabel('Grid Code')
    plt.ylabel('Number of Trajectories')
    plt.title('Distribution of Trajectories by Grid Code')
    plt.grid(True, alpha=0.3)
    
    # Add some formatting
    if len(full_range) > 50:
        # If too many grid codes, adjust x-axis ticks
        step = max(1, len(full_range) // 20)
        plt.xticks(full_range[::step], rotation=45)
    else:
        plt.xticks(full_range)
    
    plt.tight_layout()
    
    # Save plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {output_path}")
    
    plt.show()
    
    return full_range, traj_counts

def print_detailed_stats(grid_traj_mapping):
    """Print detailed statistics about the distribution."""
    
    counts = [len(trajs) for trajs in grid_traj_mapping.values()]
    
    print("\n" + "="*50)
    print("DETAILED STATISTICS")
    print("="*50)
    
    print(f"Total unique grid codes found: {len(grid_traj_mapping)}")
    print(f"Total trajectories: {sum(counts)}")
    
    if counts:
        print(f"Mean trajectories per grid code: {np.mean(counts):.2f}")
        print(f"Median trajectories per grid code: {np.median(counts):.2f}")
        print(f"Std dev trajectories per grid code: {np.std(counts):.2f}")
        
        # Show distribution of counts
        count_dist = Counter(counts)
        print(f"\nDistribution of trajectory counts:")
        for count, frequency in sorted(count_dist.items()):
            print(f"  {count} trajectories: {frequency} grid codes")
    
    # Show some examples
    print(f"\nExample grid codes and their trajectory counts:")
    for i, (grid_code, trajs) in enumerate(sorted(grid_traj_mapping.items())[:10]):
        print(f"  Grid {grid_code}: {len(trajs)} trajectories")
    
    if len(grid_traj_mapping) > 10:
        print(f"  ... and {len(grid_traj_mapping) - 10} more")

def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory distribution by grid codes")
    parser.add_argument("--data-dir", 
                       default="/data2/sules/fake_enhanced_data/ALA_ALA",
                       help="Directory containing trajectory files")
    parser.add_argument("--output", 
                       default="scratch/grid_code_histogram.png",
                       help="Output path for histogram plot")
    
    args = parser.parse_args()
    
    print(f"Analyzing trajectories in: {args.data_dir}")
    
    # Parse trajectory files
    grid_traj_mapping = parse_trajectory_files(args.data_dir)
    
    if not grid_traj_mapping:
        print("No trajectory files found matching pattern traj_{grid_code}_{traj_code}")
        return
    
    # Print detailed statistics
    print_detailed_stats(grid_traj_mapping)
    
    # Create histogram
    print(f"\nCreating histogram...")
    full_range, traj_counts = create_histogram(grid_traj_mapping, args.output)
    
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main() 