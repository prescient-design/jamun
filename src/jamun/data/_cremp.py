import functools
import tempfile
import os
import threading
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, List
import line_profiler
profile = line_profiler.LineProfiler()

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch_geometric
from torch_geometric.data import Data
import mdtraj as md
from rdkit import Chem

from jamun import utils
from jamun.data._random_chain_dataset import StreamingRandomChainDataset
from jamun.utils.featurize_macrocycles import get_macrocycle_idxs, featurize_macrocycle_atoms, get_residues

def singleton(cls):
    """
    Decorator that implements singleton pattern by modifying __init__.
    """
    _instances = {}
    _lock = threading.Lock()

    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        # Convert args and kwargs to hashable types
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, list):
                args[i] = tuple(arg)
            if isinstance(arg, dict):
                args[i] = frozenset(arg.items())
        for key, value in kwargs.items():
            if isinstance(value, list):
                kwargs[key] = tuple(value)
            if isinstance(value, dict):
                kwargs[key] = frozenset(value.items())

        obj_key = (tuple(args), frozenset(kwargs.items()))

        if obj_key not in _instances:
            with _lock:
                if obj_key not in _instances:
                    _instances[obj_key] = self
                    original_init(self, *args, **kwargs)
                    return

        # Copy state from singleton instance
        self.__dict__.update(_instances[obj_key].__dict__)

    cls.__init__ = __init__
    return cls

def preprocess_sdf(sdf_file: str) -> Tuple[torch_geometric.data.Data, Chem.Mol, Chem.Mol, List[Chem.Mol]]:
    """
    Preprocess the SDF topology.
    
    Args:
        sdf_file (str): The input SDF file path.
    
    Returns:
        Tuple[torch_geometric.data.Data, Chem.Mol, Chem.Mol, List[str]]: A tuple containing:
            - A PyTorch Geometric Data object.
            - The molecule with heavy atoms only (excluding hydrogen atoms).
            - The molecule with hydrogenated protein.
            - List of molecules from the SDF file.
    """
    import time
    start_time = time.time()
    # Load molecules from the SDF file
    suppl = Chem.SDMolSupplier(sdf_file)
    mols = [mol for mol in suppl if mol is not None]
    # mols = [mol for mol in suppl if mol is not None]
    end_time = time.time()
    # print(f"Loading SDF took {end_time - start_time:.2f} seconds for {sdf_file}: {len(mols)} molecules found.")

    if not mols:
        raise ValueError(f"No valid molecules found in the SDF file: {sdf_file}")

    # Use the first conformer as rdkit_mol and rdkit_mol_withH
    rdkit_mol_withH = mols[0]
    rdkit_mol = Chem.RemoveHs(rdkit_mol_withH)

    start_time = time.time()
    bonds = torch.tensor([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in rdkit_mol.GetBonds()], dtype=torch.long).T
    residues = get_residues(rdkit_mol, residues_in_mol=None, macrocycle_idxs=None)
    for atom_set, residue in residues.items():
        if residue.startswith("Me"):
            residues[atom_set] = residue.replace("Me", "Me+")
    residue_sequence = [v for k, v in residues.items()]
    residue_to_sequence_index = {residue: index for index, residue in enumerate(residue_sequence)}

    atom_to_residue = {atom_idx: symbol for atom_idxs, symbol in residues.items() for atom_idx in atom_idxs}
    atom_to_residue = dict(sorted(atom_to_residue.items(), key=lambda x: x[0]))
    # print(atom_to_residue)
    atom_to_residue_sequence_index = {atom_idx: residue_to_sequence_index[symbol] for atom_idx, symbol in atom_to_residue.items()}
    atom_to_3_letter = {atom_idx: utils.convert_to_three_letter_code(symbol) for atom_idx, symbol in atom_to_residue.items()}
    atom_to_residue_index = {atom_idx: utils.encode_residue(residue) for atom_idx, residue in atom_to_3_letter.items()}
    atom_types = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]

    residue_sequence_index = torch.tensor([atom_to_residue_sequence_index[atom_idx] for atom_idx in range(len(atom_to_residue))], dtype=torch.long)
    residue_code_index = torch.tensor([v for k, v in atom_to_residue_index.items()], dtype=torch.long)
    atom_type_index = torch.tensor([utils.encode_atom_type(atom_type) for atom_type in atom_types], dtype=torch.long)

    # Create a PyTorch Geometric Data object
    data = utils.DataWithResidueInformation(
        atom_type_index=atom_type_index,
        residue_code_index=residue_code_index,
        residue_sequence_index=residue_sequence_index,
        atom_code_index=atom_type_index,
        residue_index=residue_sequence_index,
        num_residues=residue_sequence_index.max().item() + 1,
        edge_index=bonds,
    )
    end_time = time.time()
    # print(f"Tensor took {end_time - start_time:.2f} seconds for {sdf_file}")
    return data, rdkit_mol, rdkit_mol_withH, mols

@singleton
class MDtrajSDFDataset(torch.utils.data.Dataset):
    """PyTorch dataset for MDtraj trajectories from SDF files."""

    def __init__(
        self,
        root: str, 
        sdf_files: List[str],
        label: str,
        num_frames: Optional[int] = None,
        start_frame: Optional[int] = None, 
        transform: Optional[Callable] = None, 
        subsample: Optional[int] = None,
        loss_weight: float = 1.0, 
    ):

        self.root = root
        self._label = label
        self.transform = transform
        self.loss_weight = loss_weight
        self.num_frames = num_frames
        self.preprocessed = False

        # Set default values for parameters
        self.start_frame = 0 if start_frame is None else start_frame
        self.subsample = 1 if subsample is None or subsample == 0 else subsample
        self.sdf_files = [os.path.join(self.root, sdf_file) for sdf_file in sdf_files]

        import time
        start_time = time.time()
        self.data, self.rdkit_mol, self.rdkit_mol_withH, mols = preprocess_sdf(self.sdf_files[0])
        end_time = time.time()
        for sdf_file in self.sdf_files[1:]:
            suppl = Chem.SDMolSupplier(sdf_file)
            frame_mols = [mol for mol in suppl if mol is not None]
            mols.extend(frame_mols)
        self.mols = mols
        # print(f"Preprocessing SDF took {end_time - start_time:.2f} seconds for {self.sdf_files[0]}")

        self.data.loss_weight = torch.tensor([self.loss_weight], dtype=torch.float32)
        self.data.dataset_label = self.label()

        # Ensure the trajectory data is in the correct shape
        self.positions = np.stack([mol.GetConformer().GetPositions() for mol in self.mols], axis=0)
        self.positions = self.positions.astype(np.float32)  # Ensure positions are float32
        self.positions /= 10.0  # Convert to nanometers

        # Subsample the trajectory.
        if self.num_frames is None:
            self.num_frames = self.positions.shape[0]
        self.positions = self.positions[self.start_frame: self.start_frame + self.num_frames: self.subsample]

        num_frames = self.positions.shape[0]
        num_atoms = self.positions.shape[1]
        assert self.positions.shape == (num_frames, num_atoms, 3), f"Positions shape mismatch: {self.positions.shape} != ({num_frames}, {num_atoms}, 3)"


    def preprocess(self):
        if self.preprocessed:
            return

        # Create topology from the SDF file
        pdb = Chem.MolToPDBBlock(self.rdkit_mol)
        tmp_pdb = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb").name
        with open(tmp_pdb, "w") as f:
            f.write(pdb)
        self.top = md.load_topology(tmp_pdb)

        self.traj = md.load_pdb(tmp_pdb)
        self.traj.xyz = self.positions
        assert len(self.traj) == len(self.positions), f"Number of frames in trajectory {len(self.traj)} does not match positions {len(self.positions)}."
        
        os.remove(tmp_pdb)
        self.preprocessed = True

    def __getitem__(self, idx):
        graph = self.data.clone()
        graph.pos = torch.tensor(self.positions[idx])
        if self.transform:
            graph = self.transform(graph)
        return graph

    def __len__(self):
        return len(self.positions)
    
    @functools.cached_property
    def topology(self) -> md.Topology:
        if not self.preprocessed:
            self.preprocess()
        return self.top

    @functools.cached_property
    def trajectory(self) -> md.Trajectory:
        if not self.preprocessed:
            self.preprocess()
        return self.traj

    def label(self) -> str:
        """Returns the dataset label."""
        return self._label
