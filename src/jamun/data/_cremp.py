import functools
import tempfile
import os
import threading
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, List

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
    # Load molecules from the SDF file
    suppl = Chem.SDMolSupplier(sdf_file)
    mols = [mol for mol in suppl if mol is not None]
    
    if not mols:
        raise ValueError(f"No valid molecules found in the SDF file: {sdf_file}")

    # Use the first conformer as rdkit_mol and rdkit_mol_withH
    rdkit_mol_withH = mols[0]
    rdkit_mol = Chem.RemoveHs(rdkit_mol_withH)

    # Get macrocycle indices
    try:
        macrocycle_idxs = get_macrocycle_idxs(rdkit_mol)
    except ValueError as e:
        raise ValueError(f"Error in extracting macrocycle indices: {e}")

    if macrocycle_idxs is None:
        raise ValueError(f"No macrocycle detected in the protein topology.")

    bonds = torch.tensor([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in rdkit_mol.GetBonds()], dtype=torch.long).T

    residues = get_residues(rdkit_mol, residues_in_mol=None, macrocycle_idxs=None)
    residue_sequence = [v for k, v in residues.items()]
    residue_to_sequence_index = {residue: index for index, residue in enumerate(residue_sequence)}

    atom_to_residue = {atom_idx: symbol for atom_idxs, symbol in residues.items() for atom_idx in atom_idxs}
    atom_to_residue = dict(sorted(atom_to_residue.items(), key=lambda x: x[0]))
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

    return data, rdkit_mol, rdkit_mol_withH, mols

@singleton
class MDtrajSDFDataset(torch.utils.data.Dataset):
    """PyTorch dataset for MDtraj trajectories from SDF files."""

    def __init__(
        self,
        root: str, 
        sdf_file: str,
        label: str,
        num_frames: Optional[int] = None,
        start_frame: Optional[int] = None, 
        transform: Optional[Callable] = None, 
        subsample: Optional[int] = None,
        loss_weight: float = 1.0, 
        verbose: bool = False,
    ):

        self.root = root
        self._label = label
        self.transform = transform
        self.loss_weight = loss_weight

        sdf_file = os.path.join(self.root, sdf_file)

        # Preprocess the SDF file
        self.data, self.rdkit_mol, self.rdkit_mol_withH, self.mols = preprocess_sdf(sdf_file)
        self.data.loss_weight = torch.tensor([loss_weight], dtype=torch.float32)
        self.data.dataset_label = self.label()

        # Ensure the trajectory data is in the correct shape
        self.positions = np.vstack([mol.GetConformer().GetPositions() for mol in self.mols])
        self.positions /= 10.0  # Convert to nanometers

        # Assuming you know the number of frames and atoms per frame
        n_atoms = self.rdkit_mol.GetNumAtoms()
        n_frames = len(self.positions) // n_atoms

        # Reshape the trajectory data
        self.positions = self.positions.reshape((n_frames, n_atoms, 3))

        # # Print the positionsectory data and its shape
        # print("Trajectory data shape:", self.positions.shape)
        # print("Trajectory data:", self.positions)
        assert self.positions.shape == (n_frames, n_atoms, 3)

        if start_frame is None:
            start_frame = 0

        if num_frames == -1 or num_frames is None:
            num_frames = len(self.positions) - start_frame
        
        if subsample is None or subsample == 0:
            subsample = 1

        # Subsample the trajectory.
        self.positions = self.positions[start_frame: start_frame + num_frames: subsample]

        # Create topology from the SDF file
        pdb = Chem.MolToPDBBlock(self.rdkit_mol)
        tmp_pdb = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb").name
        with open(tmp_pdb, "w") as f:
            f.write(pdb)
        self.top = md.load_topology(tmp_pdb)

        self.traj = md.load_pdb(tmp_pdb)
        self.traj.xyz = self.positions
        self.traj.time = np.arange(start_frame, start_frame + num_frames, subsample)
        assert len(self.traj) == len(self.positions), f"Number of frames in trajectory {len(self.traj)} does not match positions {len(self.positions)}."
        
        os.remove(tmp_pdb)

        if verbose:
            utils.dist_log(f"Dataset {self.label()}: Loaded SDF file {sdf_file} and trajectory files: {self.trajfiles}.")
            utils.dist_log(f"Dataset {self.label()}: Loaded {len(self.trajfiles)} frames starting from index {start_frame} with subsample {subsample}.")
    
    @functools.cached_property
    def topology(self) -> md.Topology:
        return self.top

    @functools.cached_property
    def trajectory(self) -> md.Trajectory:
        return self.traj

    def label(self) -> str:
        """Returns the dataset label."""
        return self._label

    def __getitem__(self, idx):
        graph = self.data.clone()
        graph.pos = torch.tensor(self.traj.xyz[idx])
        if self.transform:
            graph = self.transform(graph)
        return graph

    def __len__(self):
        return len(self.positions)

# @singleton
# class MDtrajIterableSDFDataset(torch.utils.data.IterableDataset):
#     """PyTorch iterable dataset for MDtraj trajectories from SDF files."""

#     def __init__(self, sdf_file: str, label: str, transform: Optional[Callable] = None,
#                  subsample: Optional[int] = None, loss_weight: float = 1.0, chunk_size: int = 100,
#                  start_at_random_frame: bool = True, verbose: bool = False):
        
#         self.label = lambda: label
#         self.transform = transform
#         self.loss_weight = loss_weight
#         self.chunk_size = chunk_size
#         self.start_at_random_frame = start_at_random_frame

#         if subsample is None or subsample == 0:
#             subsample = 1
#         self.subsample = subsample

#         # Preprocess the SDF file
#         self.data, self.rdkit_mol, self.rdkit_mol_withH, self.trajfiles = preprocess_sdf(sdf_file)
#         self.data.dataset_label = self.label()
#         self.data.loss_weight = torch.tensor([loss_weight], dtype=torch.float32)

#         if verbose:
#             utils.dist_log(f"Dataset {self.label()}: Iteratively loading trajectory files and SDF file {sdf_file}.")

#     def __iter__(self):
#         trajfiles = self.trajfiles
#         if self.start_at_random_frame:
#             trajfiles = np.random.permutation(trajfiles)

#         for traj_frame in trajfiles:
#             graph = self.data.clone()
#             graph.pos = torch.tensor(traj_frame.GetConformer().GetPositions(), dtype=torch.float32)
#             if self.transform:
#                 graph = self.transform(graph)
#             yield graph


class sdfMDtrajDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for MDtraj datasets."""

    def __init__(
        self, 
        datasets: Dict[str, Sequence[MDtrajSDFDataset]], 
        batch_size: int, 
        num_workers: int = 8):
        
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.datasets = datasets
        self.concatenated_datasets = {}

        self.shuffle = True

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        print(f"Setup stage: {stage}")
        for split, datasets in self.datasets.items():
            print(f"Processing split: {split}, dataset: {datasets}")
            if datasets is None:
                print(f"No dataset found for split: {split}")
                continue

            if isinstance(datasets[0], MDtrajSDFDataset):
                self.concatenated_datasets[split] = torch.utils.data.ConcatDataset(datasets)
                self.shuffle = False

                utils.dist_log(f"Split {split}: Loaded {len(self.concatenated_datasets[split])} frames in total from {len(datasets)} datasets: {[dataset.label() for dataset in datasets]}.")

        print(f"Datasets loaded for: {self.datasets.keys()}")

    def train_dataloader(self):
        if "train" not in self.datasets:
            raise KeyError(f"No 'train' dataset found in datasets.")
        print("Creating train dataloader...")
        return torch_geometric.loader.DataLoader(
            self.concatenated_datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            prefetch_factor=10,
        )

    def val_dataloader(self):
        if "val" not in self.datasets:
            raise KeyError(f"No 'val' dataset found in datasets.")
        print("Creating val dataloader...")
        return torch_geometric.loader.DataLoader(
            self.concatenated_datasets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if "test" not in self.datasets:
            raise KeyError(f"No 'test' dataset found in datasets.")
        print("Creating test dataloader...")
        return torch_geometric.loader.DataLoader(
            self.concatenated_datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )