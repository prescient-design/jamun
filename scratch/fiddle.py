# %%
import functools
import logging
import os

import dotenv
import tqdm

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("jamun")

import torch

torch.set_float32_matmul_precision("high")

import e3nn
import e3tools.nn

e3nn.set_optimization_defaults(jit_script_fx=False)

import jamun
import jamun.data
import jamun.distributions
import jamun.model
import jamun.model.arch

# %% 
# dataset
dotenv.load_dotenv("../.env", verbose=True)
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")
JAMUN_ROOT_PATH = os.getenv("JAMUN_ROOT_PATH")
datasets = {
    "test": jamun.data.parse_datasets_from_directory(
        root=f"{JAMUN_DATA_PATH}/timewarp/2AA-1-large/train/",
        traj_pattern="^(.*)-traj-arrays.npz",
        pdb_file="AA-traj-state0.pdb",
        filter_codes=['AA'],
        as_iterable=False,
        subsample=100,
        max_datasets=1,
    )
}

datamodule = jamun.data.MDtrajDataModule(
    datasets=datasets,
    batch_size=32,
    num_workers=2,
)
datamodule.setup('test')


# %%
arch = functools.partial(
    jamun.model.arch.E3Conv,
    irreps_out="1x1e",
    irreps_hidden="120x0e + 32x1e",
    irreps_sh="1x0e + 1x1e",
    n_layers=5,
    edge_attr_dim=64,
    atom_type_embedding_dim=8,
    atom_code_embedding_dim=8,
    residue_code_embedding_dim=32,
    residue_index_embedding_dim=8,
    use_residue_information=True,
    use_residue_sequence_index=False,
    hidden_layer_factory=functools.partial(
        e3tools.nn.ConvBlock,
        conv=e3tools.nn.Conv,
    ),
    output_head_factory=functools.partial(e3tools.nn.EquivariantMLP, irreps_hidden_list=["120x0e + 32x1e"]),
)


# %%
# Device.
device = torch.device("cuda:0")

# xplore generated trajectories 
import mdtraj as md
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

# load a jamun trained model 

# %%
