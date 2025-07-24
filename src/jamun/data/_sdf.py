import functools
import os
import tempfile
from collections.abc import Callable

import mdtraj as md
import numpy as np
import torch
import torch.utils.data
import torch_geometric
from rdkit import Chem

from jamun import utils


def load_preprocessed_data(
    sdf_file: str, traj_file: str
) -> tuple[torch_geometric.data.Data, Chem.Mol, Chem.Mol, list[Chem.Mol]]:
    """
    Loads preprocessed data from an SDF file and a trajectory file, returning a PyTorch Geometric Data object and the corresponding RDKit molecules.

    Args:
        sdf_file (str): The input SDF file path.

    Returns:
        Tuple[torch_geometric.data.Data, Chem.Mol, Chem.Mol, np.ndarray]: A tuple containing:
            - A PyTorch Geometric Data object.
            - The molecule with heavy atoms only (excluding hydrogen atoms).
            - The molecule with hydrogenated protein.
            - A numpy array of positions for the atoms in the molecule.
    """
    # Load molecules from the SDF file
    suppl = Chem.SDMolSupplier(sdf_file)
    mols = [mol for mol in suppl if mol is not None]

    if not mols:
        raise ValueError(f"No valid molecules found in the SDF file: {sdf_file}")
    assert len(mols) == 1

    rdkit_mol_withH = mols[0]
    rdkit_mol = Chem.RemoveHs(rdkit_mol_withH)

    with np.load(traj_file) as preprocessed:
        preprocessed = dict(preprocessed)
        for key in preprocessed:
            preprocessed[key] = torch.as_tensor(preprocessed[key])

        positions = preprocessed["positions"]
        edge_index = preprocessed["edge_index"]
        atom_type_index = preprocessed["atom_type_index"]
        residue_code_index = preprocessed["residue_code_index"]
        residue_sequence_index = preprocessed["residue_sequence_index"]

        # Create a PyTorch Geometric Data object
        data = utils.DataWithResidueInformation(
            atom_type_index=atom_type_index,
            residue_code_index=residue_code_index,
            residue_sequence_index=residue_sequence_index,
            atom_code_index=atom_type_index,
            residue_index=residue_sequence_index,
            num_residues=residue_sequence_index.max().item() + 1,
            edge_index=edge_index,
        )
        return data, rdkit_mol, rdkit_mol_withH, positions


@utils.singleton
class MDtrajSDFDataset(torch.utils.data.Dataset):
    """PyTorch dataset for MDtraj trajectories from SDF files."""

    def __init__(
        self,
        root: str,
        sdf_file: str,
        traj_files: list[str],
        label: str,
        num_frames: int | None = None,
        start_frame: int | None = None,
        transform: Callable | None = None,
        subsample: int | None = None,
        loss_weight: float = 1.0,
    ):
        self.root = root
        self._label = label
        self.transform = transform
        self.loss_weight = loss_weight
        self.num_frames = num_frames
        self.preprocessed_topology = False
        self.sdf_file = os.path.join(self.root, sdf_file)

        if len(traj_files) > 1:
            raise NotImplementedError(
                f"Multiple trajectory files found: {traj_files}. Please provide a single trajectory file."
            )
        self.traj_file = os.path.join(self.root, traj_files[0])

        self.start_frame = 0 if start_frame is None else start_frame
        self.subsample = 1 if subsample is None or subsample == 0 else subsample

        self.data, self.rdkit_mol, self.rdkit_mol_withH, self.positions = load_preprocessed_data(
            self.sdf_file, self.traj_file
        )
        self.data.loss_weight = torch.tensor([self.loss_weight], dtype=torch.float32)
        self.data.dataset_label = self.label()

        # Subsample the trajectory.
        if self.num_frames is None:
            self.num_frames = self.positions.shape[0]
        self.positions = self.positions[self.start_frame : self.start_frame + self.num_frames : self.subsample]

        num_frames = self.positions.shape[0]
        num_atoms = self.positions.shape[1]
        assert self.positions.shape == (num_frames, num_atoms, 3), (
            f"Positions shape mismatch: {self.positions.shape} != ({num_frames}, {num_atoms}, 3)"
        )

    def preprocess_topology(self):
        if self.preprocessed_topology:
            return

        # Create topology from the SDF file
        pdb = Chem.MolToPDBBlock(self.rdkit_mol)
        tmp_pdb = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb").name
        with open(tmp_pdb, "w") as f:
            f.write(pdb)
        self.top = md.load_topology(tmp_pdb)

        self.traj = md.load_pdb(tmp_pdb)
        self.traj.xyz = self.positions
        assert len(self.traj) == len(self.positions), (
            f"Number of frames in trajectory {len(self.traj)} does not match positions {len(self.positions)}."
        )

        os.remove(tmp_pdb)
        self.preprocessed_topology = True

    def __getitem__(self, idx):
        graph = self.data.clone()
        graph.pos = self.positions[idx]
        if self.transform:
            graph = self.transform(graph)
        return graph

    def __len__(self):
        return len(self.positions)

    @functools.cached_property
    def topology(self) -> md.Topology:
        if not self.preprocessed_topology:
            self.preprocess_topology()
        return self.top

    @functools.cached_property
    def trajectory(self) -> md.Trajectory:
        if not self.preprocessed_topology:
            self.preprocess_topology()
        return self.traj

    def label(self) -> str:
        """Returns the dataset label."""
        return self._label
