import glob
import os
import itertools
import re
from collections import defaultdict

import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.colors as colors

def parse_grid_code_from_filename(filename):
    """
    Parse grid code from trajectory filename of format traj_{grid_code}_{traj_code}.xtc
    """
    basename = os.path.basename(filename)
    match = re.match(r'^traj_(\d+)_(\d+)\.xtc$', basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def select_trajectories_with_max_per_grid(traj_files, max_traj_per_grid):
    """
    Select trajectories ensuring no grid code has more than max_traj_per_grid trajectories.
    """
    grid_trajectories = defaultdict(list)
    
    # Group trajectories by grid code
    for traj_file in traj_files:
        grid_code, traj_code = parse_grid_code_from_filename(traj_file)
        if grid_code is not None:
            grid_trajectories[grid_code].append((traj_file, traj_code))
    
    print(f"Found {len(grid_trajectories)} unique grid codes")
    
    # Limit trajectories per grid code
    selected_files = []
    grid_stats = {}
    
    for grid_code, traj_list in grid_trajectories.items():
        # Sort by trajectory code for deterministic selection
        traj_list.sort(key=lambda x: x[1])
        
        # Select up to max_traj_per_grid trajectories
        selected_count = min(len(traj_list), max_traj_per_grid)
        selected_for_grid = traj_list[:selected_count]
        
        grid_stats[grid_code] = {
            'total': len(traj_list),
            'selected': selected_count
        }
        
        for traj_file, _ in selected_for_grid:
            selected_files.append(traj_file)
    
    # Print statistics
    print(f"\nGrid code statistics:")
    print(f"Total grid codes: {len(grid_stats)}")
    total_original = sum(stats['total'] for stats in grid_stats.values())
    total_selected = sum(stats['selected'] for stats in grid_stats.values())
    print(f"Total trajectories: {total_original} -> {total_selected}")
    print(f"Max trajectories per grid: {max_traj_per_grid}")
    
    # Show distribution
    selected_counts = [stats['selected'] for stats in grid_stats.values()]
    print(f"Distribution of selected trajectories per grid:")
    for count in sorted(set(selected_counts)):
        num_grids = sum(1 for c in selected_counts if c == count)
        print(f"  {count} trajectories: {num_grids} grid codes")
    
    return sorted(selected_files)

def create_ramachandran_plot(traj_path, topology, output_dir):
    """
    Loads a trajectory, computes phi and psi angles, and saves a Ramachandran plot.
    """
    # Load trajectory
    try:
        traj = md.load(traj_path, top=topology)
    except Exception as e:
        print(f"Could not load trajectory {traj_path}. Error: {e}")
        return

    # Compute dihedral angles
    phi_indices, phi_angles = md.compute_phi(traj)
    psi_indices, psi_angles = md.compute_psi(traj)

    # Convert radians to degrees
    phi_degrees = np.rad2deg(phi_angles.flatten())
    psi_degrees = np.rad2deg(psi_angles.flatten())

    # Create plot
    plt.figure(figsize=(8, 8))
    # Use hexbin for a nicer look
    plt.hexbin(phi_degrees, psi_degrees, gridsize=180, cmap='viridis', mincnt=1)
    plt.colorbar(label='Count in bin')
    plt.title(f'Ramachandran Plot for {os.path.basename(traj_path)}')
    plt.xlabel('Phi (degrees)')
    plt.ylabel('Psi (degrees)')
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='k', linestyle='--', linewidth=0.5)

    # Save plot
    output_filename = f"ramachandran_{os.path.basename(traj_path).replace('.xtc', '.png')}"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_histogram_plot(dihedrals, name1, name2, output_dir, name_string):
    """
    Creates a 2D histogram with density for a pair of dihedrals.
    """
    # Flatten all data for the pair of dihedrals
    all_x_data = np.concatenate(dihedrals[name1])
    all_y_data = np.concatenate(dihedrals[name2])
    
    plt.figure(figsize=(10, 10))
    
    # Create 2D histogram with density
    plt.hist2d(all_x_data, all_y_data, range=((-np.pi, np.pi), (-np.pi, np.pi)),bins=100, cmap='viridis', alpha=0.8, norm=colors.LogNorm())
    plt.colorbar(label='Density')
    
    plt.title(f'Histogram (Density): {name1} vs {name2}')
    plt.xlabel(f'{name1} (radians)')
    plt.ylabel(f'{name2} (radians)')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='k', linestyle='--', linewidth=0.5)
    
    output_filename = f"histogram_density_{name1}_vs_{name2}_{name_string}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(max_traj_per_grid=10):
    """
    Loads trajectories, computes dihedral angles, and creates pairwise scatter plots and histograms.
    Only keeps up to max_traj_per_grid trajectories per grid code.
    """
    data_dir = "/data2/sules/fake_enhanced_data/ALA_ALA"
    pdb_path = "/data/bucket/kleinhej/capped_diamines/timewarp_splits/train/ALA_ALA.pdb"
    output_dir = f"/data2/sules/ramachandran_plots_ala_ala_fake_enhanced_data_max{max_traj_per_grid}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        topology = md.load_pdb(pdb_path)
    except Exception as e:
        print(f"Could not load topology file {pdb_path}. Error: {e}")
        return

    # Get all trajectory files
    all_traj_files = sorted(glob.glob(os.path.join(data_dir, "traj_*.xtc")))
    print(f"Found {len(all_traj_files)} total trajectory files")

    if not all_traj_files:
        print(f"No trajectory files found in {data_dir}")
        return

    # Select trajectories with max per grid code
    traj_files = select_trajectories_with_max_per_grid(all_traj_files, max_traj_per_grid)
    print(f"Selected {len(traj_files)} trajectory files after filtering")

    all_phi_angles = []
    all_psi_angles = []
    num_phi, num_psi = None, None

    for traj_file in tqdm(traj_files, desc="Loading trajectories"):
        try:
            traj = md.load(traj_file, top=topology)
            _, phi_angles = md.compute_phi(traj)
            _, psi_angles = md.compute_psi(traj)
            
            if num_phi is None:
                num_phi = phi_angles.shape[1]
                num_psi = psi_angles.shape[1]

            all_phi_angles.append(phi_angles[:100,:])
            all_psi_angles.append(psi_angles[:100,:])
        except Exception as e:
            print(f"Could not load or process trajectory {traj_file}. Error: {e}")
            continue

    if not all_phi_angles or not all_psi_angles:
        print("No valid trajectories were processed.")
        return

    # Dynamically create dihedral dictionary
    dihedrals = {}
    for i in range(num_phi):
        dihedrals[f'phi_{i+1}'] = [angles[:, i] for angles in all_phi_angles]
    for i in range(num_psi):
        dihedrals[f'psi_{i+1}'] = [angles[:, i] for angles in all_psi_angles]
    
    dihedral_names = list(dihedrals.keys())

    # Create line plots (existing functionality)
    create_line_plots = False
    if create_line_plots:
        print("Creating line plots...")
        for name1, name2 in itertools.combinations(dihedral_names, 2):
            plt.figure(figsize=(10, 10))
            
            for i in tqdm(range(len(traj_files)), desc=f"Plotting {name1} vs {name2}"):
                x_angles = dihedrals[name1][i]
                y_angles = dihedrals[name2][i]
                plt.plot(x_angles, y_angles, linestyle='-', alpha=0.5)
                plt.scatter(x_angles[0], y_angles[0], c='white', marker='o', edgecolor='black', s=50, zorder=5)

            plt.title(f'Ramachandran Plot: {name1} vs {name2}')
            plt.xlabel(f'{name1} (degrees)')
            plt.ylabel(f'{name2} (degrees)')
            plt.xlim(-180, 180)
            plt.ylim(-180, 180)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
            plt.axvline(0, color='k', linestyle='--', linewidth=0.5)
            
            output_filename = f"ramachandran_{name1}_vs_{name2}.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

    create_histogram_plots = True
    if create_histogram_plots:
        # Create histogram plots (new functionality)
        print("Creating histogram plots with density...")
        for name1, name2 in tqdm(itertools.combinations(dihedral_names, 2), desc="Creating histograms"):
            create_histogram_plot(dihedrals, name1, name2, output_dir, f"100_frames_max{max_traj_per_grid}")

    print(f"\nDone. Histogram plots are saved in {output_dir}")
    print(f"Used max {max_traj_per_grid} trajectories per grid code")


if __name__ == "__main__":
    # Set max trajectories per grid code to 10
    max_traj = 10
    main(max_traj_per_grid=max_traj) 