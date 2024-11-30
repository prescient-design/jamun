import logging
from typing import Optional, Union, List, Tuple, Sequence, Dict
import tempfile
import os

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import torch_geometric
import mdtraj as md
import py3Dmol
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import rdBase

import torch_scatter
from jamun.utils import align_A_to_B


def coordinates_to_trajectories(coords: torch.Tensor, structure: md.Trajectory) -> List[md.Trajectory]:
    """Converts a tensor of coordinates to MDtraj trajectories."""
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().detach().numpy()

    if coords.ndim == 3:
        coords = coords[None, ...]

    coords = einops.rearrange(
        coords,
        "batch_size atoms num_sampling_steps coords -> batch_size num_sampling_steps atoms coords",
        atoms=structure.n_atoms,
    )

    return [md.Trajectory(traj_coords, structure.topology) for traj_coords in coords]


def save_pdb(traj: md.Trajectory, path: str) -> None:
    """Saves a trajectory to a PDB file, fixing bugs in mdtraj.save_pdb."""
    topology = traj.topology
    unique_bonds = set()
    for bond in traj.topology.bonds:
        unique_bonds.add((bond.atom1.index, bond.atom2.index))

    with open(path, "w") as f:
        for frame_index, frame in enumerate(traj.xyz):
            f.write(f"MODEL        {frame_index}\n")

            for atom_index, positions in enumerate(frame):
                atom = topology.atom(atom_index)
                x, y, z = positions * 10
                f.write(
                    f"ATOM  {atom_index+1:5d} {atom.name:<4s} {atom.residue.name:3s} {atom.residue.chain.index:1d}{atom.residue.index+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.element.symbol:>2s}\n"
                )

            num_atoms = topology.n_atoms
            final_atom = topology.atom(num_atoms - 1)
            f.write(
                f"TER   {num_atoms+1:5d}      {final_atom.residue.name:3s} {final_atom.residue.chain.index:1d}{final_atom.residue.index+1:4d}\n"
            )

            # Add bonds.
            bonds = [[i + 1] for i in range(topology.n_atoms)]
            for bond in unique_bonds:
                bonds[bond[0]].append(bond[1] + 1)
                bonds[bond[1]].append(bond[0] + 1)
            for bond in bonds:
                s = "".join([f"{atom:5d}" for atom in bond])
                f.write(f"CONECT{s}\n")

            f.write("ENDMDL\n")
        f.write("END\n")


def to_rdkit_mols(traj: md.Trajectory) -> List[Chem.Mol]:
    """Converts an MDTraj trajectory to a list of RDKit molecules."""

    # Suppress RDKit warnings.
    blocker = rdBase.BlockLogs()

    # Write to a PDB.
    temp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb").name
    save_pdb(traj, temp_pdb)
    traj_mol = Chem.MolFromPDBFile(temp_pdb, removeHs=False, sanitize=False)

    if traj_mol is None:
        py_logger = logging.getLogger("jamun")
        py_logger.warning("Could not convert the trajectory to RDKit mols.")
        return []

    # Check if the input molecule has multiple conformers.
    if traj_mol.GetNumConformers() <= 1:
        return [traj_mol]

    # Create separate molecules for each conformer.
    molecules = []
    for conf_id in range(traj_mol.GetNumConformers()):
        new_mol = Chem.Mol(traj_mol)
        new_mol.RemoveAllConformers()
        conf = traj_mol.GetConformer(conf_id)
        new_conf = Chem.Conformer(conf)
        new_mol.AddConformer(new_conf, assignId=True)
        molecules.append(new_mol)
    return molecules


def animate_trajectory_with_py3Dmol(
    traj: md.Trajectory, alignment_frame: Optional[md.Trajectory] = None
) -> py3Dmol.view:
    """Create an animation of this trajectory using py3Dmol."""
    if alignment_frame:
        traj = traj.superpose(alignment_frame)

    temp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb").name
    save_pdb(traj, temp_pdb)
    with open(temp_pdb) as f:
        pdb = f.read()

    view = py3Dmol.view(width=1500, height=500)
    view.addModelsAsFrames(pdb, "pdb")
    view.setStyle(
        {
            "stick": {"radius": 0.1, "colorscheme": "grayCarbon"},
            "sphere": {"scale": 0.2},
            "colorscheme": {"prop": "index", "gradient": "roygb", "min": 0, "max": 14},
        }
    )
    view.animate(
        {
            "reps": 0,
            "interval": 100,
        }
    )
    view.zoomTo()
    return view


class ModelSamplingWrapper:
    """Wrapper to sample positions from a model."""

    def __init__(self, model: nn.Module, init_graphs: torch_geometric.data.Data, sigma: float):
        self._model = model
        self.init_graphs = init_graphs
        self.sigma = sigma

    @property
    def device(self) -> torch.device:
        return self._model.device

    def sample_initial_noisy_positions(self) -> torch.Tensor:
        pos = self.init_graphs.pos
        pos = pos + torch.randn_like(pos) * self.sigma
        return pos

    def __getattr__(self, name):
        return getattr(self._model, name)

    def score(self, y, sigma, *args, **kwargs):
        return self._model.score(self.positions_to_graph(y), sigma)

    def xhat(self, y, sigma, *args, **kwargs):
        xhat_graph = self._model.xhat(self.positions_to_graph(y), sigma)
        return xhat_graph.pos

    def positions_to_graph(self, positions: torch.Tensor) -> torch_geometric.data.Data:
        """Wraps a tensor of positions to a graph with these positions as an attribute."""
        # Check input validity
        assert len(positions) == self.init_graphs.num_nodes, "The number of positions and nodes should be the same"
        assert positions.shape[1] == 3, "Positions tensor should have a shape of (n, 3)"

        input_graphs = self.init_graphs.clone()
        input_graphs.pos = positions
        # Save for debugging.
        self.input_graphs = input_graphs
        return input_graphs.to(positions.device)

    def unbatch_samples(self, samples: Dict[str, torch.Tensor]) -> List[torch_geometric.data.Data]:
        """Unbatch samples."""
        if "batch" not in self.init_graphs:
            raise ValueError("The initial graph does not have a batch attribute.")

        output_graphs = self.init_graphs.clone()
        output_graphs = torch_geometric.data.Batch.to_data_list(output_graphs)

        py_logger = logging.getLogger("jamun")
        for key, value in samples.items():
            if value.ndim not in [2, 3]:
                py_logger.info(f"Skipping unbatching of key {key} with shape {value.shape} as it is not 2D or 3D.")
                continue

            if value.ndim == 3:
                value = einops.rearrange(
                    value,
                    "num_frames atoms coords -> atoms num_frames coords",
                )

            unbatched_values = torch_geometric.utils.unbatch(value, self.init_graphs.batch)
            for output_graph, unbatched_value in zip(output_graphs, unbatched_values, strict=True):
                if key in output_graph:
                    raise ValueError(f"Key {key} already exists in the output graph.")

                if unbatched_value.shape[0] != output_graph.num_nodes:
                    raise ValueError(
                        f"Number of nodes in unbatched value ({unbatched_value.shape[0]}) for key {key} does not match "
                        f"number of nodes in output graph ({output_graph.num_nodes})."
                    )

                output_graph[key] = unbatched_value

        return output_graphs


def plot_molecules_with_py3Dmol(
    molecules: Dict[str, Chem.Mol],
    show_atom_types: bool = False,
    show_keys: bool = True,
) -> py3Dmol.view:
    """Visualize a dictionary of molecules with py3Dmol."""
    num_keys = len(molecules)
    max_molecules_per_row = max([len(molecules[key]) for key in molecules])
    view = py3Dmol.view(
        viewergrid=(num_keys, max_molecules_per_row),
        linked=True,
        width=max_molecules_per_row * 180,
        height=num_keys * 180,
    )
    for i, key in enumerate(molecules):
        for j, mol in enumerate(molecules[key]):
            if j >= max_molecules_per_row:
                break

            try:
                view.addModel(
                    Chem.MolToMolBlock(mol), "mol", {"keepH": "true"}, viewer=(i, j)
                )
            except ValueError:
                # Sometimes RDKit errors out on weird molecules.
                continue

            if show_atom_types:
                for atom in mol.GetAtoms():
                    position = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                    view.addLabel(
                        str(atom.GetSymbol()),
                        {
                            "fontSize": 6,
                            "fontColor": "white",
                            "backgroundOpacity": 0.2,
                            "backgroundColor": "black",
                            "position": {
                                "x": position.x,
                                "y": position.y,
                                "z": position.z,
                            },
                        },
                        viewer=(i, j),
                    )

    # Add text labels for each row.
    if show_keys:
        for i, key in enumerate(molecules):
            for j in range(len(molecules[key])):
                view.addLabel(
                    key,
                    {
                        "fontSize": 12,
                        "fontColor": "black",
                        "backgroundColor": "white",
                        "backgroundOpacity": 0.8,
                        "position": {"x": -10, "y": 0, "z": 0},
                    },
                    viewer=(i, j),
                )

    view.setStyle(
        {"stick": {"color": "spectrum", "radius": 0.2}, "sphere": {"scale": 0.3}}
    )
    view.zoomTo()
    return view


def plot_pmf(trajs, amino_acid_list=None):
    """Not used. See jamun.metrics._ramachandran.plot_ramachandran instead."""
    traj = md.join(trajs)

    phis, psis = md.compute_phi(traj)[1], md.compute_psi(traj)[1]
    num_amino_acids = phis.shape[1]

    fig, axs = plt.subplots(1, num_amino_acids, figsize=(4 * num_amino_acids, 4), sharey=True)

    # defining the range
    range_val = [[-np.pi, np.pi], [-np.pi, np.pi]]

    # computing histogram and plotting for the first subplot
    for i in num_amino_acids:
        ax1 = axs[i]
        H1, x1, y1 = np.histogram2d(phis.T[0], psis.T[0], bins=40, range=range_val)
        pmf1 = -np.log(H1.T) + np.max(np.log(H1.T))
        im1 = ax1.contourf(x1[:-1], y1[:-1], pmf1, cmap="viridis")
        # cont1 = ax1.contour(x1[:-1], y1[:-1], pmf1, colors='white', linestyles='solid')
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_xlabel("$\phi$")
        ax1.set_ylabel("$\psi$")
        if amino_acid_list is not None:
            ax1.set_title(amino_acid_list[i])
        # fig.colorbar(im1, ax=ax1, label='-log(H)')

    plt.show()


def mean_center(y: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
    """Mean centers the positions."""
    mean_pos = torch_scatter.scatter_mean(y.pos, y.batch, dim=0)
    y.pos -= mean_pos[y.batch]
    return y


def align_A_to_B_batched(
    A: torch_geometric.data.Batch,
    B: torch_geometric.data.Batch
) -> torch_geometric.data.Batch:
    """Aligns each batch of A to corresponding batch in B."""
    num_batches = A.batch.max().item() + 1
    for i in range(num_batches):
        mask = A.batch == i
        A.pos[mask] = align_A_to_B(A.pos[mask], B.pos[mask])
    return A


def scaled_rmsd(x: torch.Tensor, xhat: torch.Tensor, sigma: float) -> torch.Tensor:
    """Computes the scaled RMSD between x and xhat, both assumed mean-centered."""
    assert x.shape == xhat.shape
    raw_loss = F.mse_loss(xhat, x, reduction="none")
    raw_loss = einops.rearrange(raw_loss, "... D -> (...) D")
    D = torch.as_tensor(x.shape[-1])
    raw_loss = F.mse_loss(xhat, x, reduction="none")
    raw_loss = raw_loss.sum(dim=-1)
    scaled_rmsd = torch.sqrt(raw_loss) / (sigma * torch.sqrt(D))
    scaled_rmsd = scaled_rmsd.mean()
    return scaled_rmsd
