import itertools
import os

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

# Set file paths for ALA_ALA in capped diamines
xtc_file = "/data/bucket/kleinhej/capped_diamines/timewarp_splits/train/ALA_ALA.xtc"
pdb_file = "/data/bucket/kleinhej/capped_diamines/timewarp_splits/train/ALA_ALA.pdb"

output_dir = "/data2/sules/ramachandran_plots_ala_ala_fake_enhanced_data"
os.makedirs(output_dir, exist_ok=True)

print(f"XTC file exists: {os.path.exists(xtc_file)}")
print(f"PDB file exists: {os.path.exists(pdb_file)}")

# Load the trajectory (subsample=1 means load all frames)
traj = md.load(xtc_file, top=pdb_file)
print(f"Loaded trajectory with {traj.n_frames} frames and {traj.n_atoms} atoms.")

# Compute backbone dihedrals (phi and psi)
phi_indices, phi_angles = md.compute_phi(traj)
psi_indices, psi_angles = md.compute_psi(traj)

num_phi = phi_angles.shape[1]
num_psi = psi_angles.shape[1]

# Collect all dihedral arrays in a dict for easy access
# Each entry is (n_frames,)
dihedrals = {}
for i in range(num_phi):
    dihedrals[f"phi_{i + 1}"] = phi_angles[:, i]
for i in range(num_psi):
    dihedrals[f"psi_{i + 1}"] = psi_angles[:, i]

dihedral_names = list(dihedrals.keys())

# Make 2D histograms for all pairs
for name1, name2 in itertools.combinations(dihedral_names, 2):
    x = dihedrals[name1]
    y = dihedrals[name2]
    plt.figure(figsize=(8, 8))
    plt.hist2d(x, y, bins=100, range=((-np.pi, np.pi), (-np.pi, np.pi)), cmap="viridis", norm=colors.LogNorm())
    plt.colorbar(label="Density")
    plt.title(f"2D Histogram: {name1} vs {name2}")
    plt.xlabel(f"{name1} (radians)")
    plt.ylabel(f"{name2} (radians)")
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axhline(0, color="k", linestyle="--", linewidth=0.5)
    plt.axvline(0, color="k", linestyle="--", linewidth=0.5)
    output_filename = f"hist2d_{name1}_vs_{name2}_true_distribution.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

print(f"All 2D histograms saved in {output_dir}")
