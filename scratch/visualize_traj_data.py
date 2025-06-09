# xplore generated trajectories 
import mdtraj as md
import os 
# --- Option 2: Loading your own DCD and Topology file ---
print("\n--- Loading Your Own DCD and Topology File (Example) ---")
# Replace these with the actual paths to your files
dcd_file_path = f"{JAMUN_ROOT_PATH}/outputs/sample/dev/runs/2025-06-04_22-46-33/sampler/AA/predicted_samples/dcd/joined.dcd"  # Your DCD trajectory file
topology_file_path = f"{JAMUN_ROOT_PATH}/outputs/sample/dev/runs/2025-06-04_22-46-33/sampler/AA/topology.pdb" # Your topology file (e.g., .pdb, .prmtop, .psf)

# Create dummy files for this example to run without error if you don't have them
# In a real scenario, you would have your actual DCD and PDB files.
print(f'DCD file path exists: {os.path.exists(dcd_file_path)}')
print(f'Topology file path exists: {os.path.exists(topology_file_path)}')
try:
    print(f"Attempting to load trajectory: {dcd_file_path}")
    print(f"Using topology: {topology_file_path}")

    # The 'top' argument is crucial for DCD files
    traj_custom = md.load_dcd(dcd_file_path, top=topology_file_path)

    print(f"Successfully loaded custom trajectory!")
    print(f"Number of frames: {traj_custom.n_frames}")
    print(f"Number of atoms: {traj_custom.n_atoms}")
    # You can now perform analysis on traj_custom
    # For example, calculate RMSD, distances, angles, etc.

except FileNotFoundError:
    print(f"Error: One or both files not found: {dcd_file_path}, {topology_file_path}")
except Exception as e:
    print(f"An error occurred while loading your files: {e}")
print("-" * 30)

# %%
from jamun.metrics._ramachandran import plot_ramachandran

phi = md.compute_phi(traj_custom)
psi = md.compute_psi(traj_custom)

import matplotlib.pyplot as plt
import numpy as np 
fig = plt.figure()
ax = fig.add_subplot()
s = ax.scatter(phi[1], psi[1], cmap='hot', alpha=1.0)
ax.set_xlim((-np.pi, np.pi))
ax.set_ylim((-np.pi, np.pi))
c = fig.colorbar(s)

print("hello world")