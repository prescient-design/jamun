#!/usr/bin/env python3
"""
Trajectory noise analysis script.
Loads .xtc trajectory files, adds noise, computes norms between successive points,
and creates histograms with flexible time point filtering.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
from pathlib import Path
from typing import List, Tuple, Optional
import argparse


def load_trajectories(traj_dir: str, topology_file: str, max_files: Optional[int] = None) -> List[md.Trajectory]:
    """
    Load trajectory files from directory using MDTraj.
    
    Args:
        traj_dir: Directory containing .xtc files
        topology_file: Path to PDB topology file
        max_files: Maximum number of files to load (None for all)
    
    Returns:
        List of MDTraj trajectory objects
    """
    xtc_files = glob.glob(os.path.join(traj_dir, "*.xtc"))
    xtc_files.sort()
    
    if max_files is not None:
        xtc_files = xtc_files[:max_files]
    
    print(f"Loading {len(xtc_files)} trajectory files...")
    
    trajectories = []
    for i, xtc_file in enumerate(xtc_files):
        try:
            traj = md.load(xtc_file, top=topology_file)
            trajectories.append(traj)
            if (i + 1) % 50 == 0:
                print(f"Loaded {i + 1}/{len(xtc_files)} trajectories")
        except Exception as e:
            print(f"Warning: Failed to load {xtc_file}: {e}")
    
    print(f"Successfully loaded {len(trajectories)} trajectories")
    return trajectories


def add_noise_to_trajectory(traj: md.Trajectory, noise_magnitude: float = 0.04) -> md.Trajectory:
    """
    Add Gaussian noise to trajectory coordinates.
    
    Args:
        traj: MDTraj trajectory object
        noise_magnitude: Standard deviation of Gaussian noise to add (in nm)
    
    Returns:
        New trajectory with added noise
    """
    # Copy the trajectory to avoid modifying the original
    noisy_traj = traj.slice(range(traj.n_frames))
    
    # Add Gaussian noise to xyz coordinates
    noise = np.random.normal(0, noise_magnitude, noisy_traj.xyz.shape)
    noisy_traj.xyz += noise
    
    return noisy_traj


def compute_successive_norms(traj: md.Trajectory) -> np.ndarray:
    """
    Compute norms between successive trajectory points.
    
    Args:
        traj: MDTraj trajectory object
    
    Returns:
        Array of norms between successive points for each atom
    """
    if traj.n_frames < 2:
        return np.array([])
    
    # Calculate differences between successive frames
    diff = traj.xyz[1:] - traj.xyz[:-1]  # Shape: (n_frames-1, n_atoms, 3)
    
    # Compute norms for each atom at each time step
    norms = np.linalg.norm(diff, axis=2)  # Shape: (n_frames-1, n_atoms)
    
    return norms


def compute_norms_for_time_points(traj: md.Trajectory, time_points: List[Tuple[int, int]]) -> np.ndarray:
    """
    Compute norms between specific time points.
    
    Args:
        traj: MDTraj trajectory object
        time_points: List of (start_frame, end_frame) tuples
    
    Returns:
        Array of norms for specified time point pairs
    """
    norms = []
    
    for start_frame, end_frame in time_points:
        if start_frame < traj.n_frames and end_frame < traj.n_frames:
            diff = traj.xyz[end_frame] - traj.xyz[start_frame]  # Shape: (n_atoms, 3)
            frame_norms = np.linalg.norm(diff, axis=1)  # Shape: (n_atoms,)
            norms.extend(frame_norms)
    
    return np.array(norms)


def analyze_trajectories(trajectories: List[md.Trajectory], 
                        noise_magnitude: float = 0.04,
                        time_point_filter: Optional[List[Tuple[int, int]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze trajectories by adding noise and computing norms.
    
    Args:
        trajectories: List of MDTraj trajectory objects
        noise_magnitude: Standard deviation of Gaussian noise
        time_point_filter: Optional list of (start, end) frame pairs to analyze
    
    Returns:
        Tuple of (original_norms, noisy_norms)
    """
    original_norms = []
    noisy_norms = []
    
    print(f"Analyzing {len(trajectories)} trajectories...")
    
    for i, traj in enumerate(trajectories):
        if time_point_filter is not None:
            # Compute norms for specific time points
            orig_norm = compute_norms_for_time_points(traj, time_point_filter)
            
            # Add noise and compute norms for same time points
            noisy_traj = add_noise_to_trajectory(traj, noise_magnitude)
            noisy_norm = compute_norms_for_time_points(noisy_traj, time_point_filter)
        else:
            # Compute successive norms for all time points
            orig_norm = compute_successive_norms(traj)
            
            # Add noise and compute successive norms
            noisy_traj = add_noise_to_trajectory(traj, noise_magnitude)
            noisy_norm = compute_successive_norms(noisy_traj)
        
        # Flatten and collect norms
        original_norms.extend(orig_norm.flatten())
        noisy_norms.extend(noisy_norm.flatten())
        
        if (i + 1) % 10 == 0:
            print(f"Analyzed {i + 1}/{len(trajectories)} trajectories")
    
    return np.array(original_norms), np.array(noisy_norms)


def create_histogram(original_norms: np.ndarray, 
                    noisy_norms: np.ndarray,
                    title: str = "Norm Differences Between Successive Trajectory Points",
                    bins: int = 50,
                    save_path: Optional[str] = None):
    """
    Create histogram comparing original and noisy trajectory norms.
    
    Args:
        original_norms: Array of norms from original trajectories
        noisy_norms: Array of norms from noisy trajectories
        title: Plot title
        bins: Number of histogram bins
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    plt.hist(original_norms, bins=bins, alpha=0.7, label='Original', density=True, color='blue')
    plt.hist(noisy_norms, bins=bins, alpha=0.7, label='With Noise (σ=0.04)', density=True, color='red')
    
    plt.xlabel('Norm (nm)')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    orig_mean, orig_std = np.mean(original_norms), np.std(original_norms)
    noisy_mean, noisy_std = np.mean(noisy_norms), np.std(noisy_norms)
    
    stats_text = f'Original: μ={orig_mean:.4f}, σ={orig_std:.4f}\n'
    stats_text += f'Noisy: μ={noisy_mean:.4f}, σ={noisy_std:.4f}'
    
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze trajectory noise effects')
    parser.add_argument('--traj_dir', type=str, 
                       default='/data2/sules/fake_enhanced_data/ALA_ALA_organized/train',
                       help='Directory containing trajectory files')
    parser.add_argument('--topology', type=str,
                       default='/data/bucket/kleinhej/capped_diamines/timewarp_splits/train/ALA_ALA.pdb',
                       help='PDB topology file')
    parser.add_argument('--noise_magnitude', type=float, default=0.04,
                       help='Noise magnitude (standard deviation)')
    parser.add_argument('--max_files', type=int, default=50,
                       help='Maximum number of trajectory files to load')
    parser.add_argument('--time_filter', type=str, default=None,
                       help='Time point filter as "start1,end1;start2,end2" (e.g., "0,1" for initial->next)')
    parser.add_argument('--output', type=str, default='trajectory_noise_analysis.png',
                       help='Output plot filename')
    parser.add_argument('--bins', type=int, default=50,
                       help='Number of histogram bins')
    
    args = parser.parse_args()
    
    # Parse time filter if provided
    time_point_filter = None
    if args.time_filter:
        try:
            pairs = args.time_filter.split(';')
            time_point_filter = []
            for pair in pairs:
                start, end = map(int, pair.split(','))
                time_point_filter.append((start, end))
            print(f"Using time point filter: {time_point_filter}")
        except:
            print("Warning: Invalid time filter format. Using all successive points.")
    
    # Load trajectories
    trajectories = load_trajectories(args.traj_dir, args.topology, args.max_files)
    
    if not trajectories:
        print("No trajectories loaded. Exiting.")
        return
    
    # Analyze trajectories
    original_norms, noisy_norms = analyze_trajectories(
        trajectories, args.noise_magnitude, time_point_filter
    )
    
    # Create title based on analysis type
    if time_point_filter:
        title = f"Norm Differences for Time Points {time_point_filter}"
    else:
        title = "Norm Differences Between Successive Trajectory Points"
    title += f" (Noise σ={args.noise_magnitude})"
    
    # Create histogram
    create_histogram(original_norms, noisy_norms, title, args.bins, args.output)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Original trajectories: {len(original_norms)} data points")
    print(f"  Mean norm: {np.mean(original_norms):.6f} nm")
    print(f"  Std norm:  {np.std(original_norms):.6f} nm")
    print(f"Noisy trajectories: {len(noisy_norms)} data points")
    print(f"  Mean norm: {np.mean(noisy_norms):.6f} nm")
    print(f"  Std norm:  {np.std(noisy_norms):.6f} nm")


if __name__ == "__main__":
    main() 