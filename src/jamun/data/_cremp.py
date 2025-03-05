import collections
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

from rdkit import Chem

from jamun import utils
from jamun.data._random_chain_dataset import StreamingRandomChainDataset
from jamun.utils.featurize_macrocycles import get_macrocycle_idxs, featurize_macrocycle_atoms

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

def preprocess_sdf(sdf_file: str) -> Tuple[torch_geometric.data.Data, Chem.Mol, Chem.Mol, List[str]]:
    """
    Preprocess the SDF topology.
    
    Args:
        sdf_file (str): The input SDF file path.
    
    Returns:
        Tuple[torch_geometric.data.Data, Chem.Mol, Chem.Mol, List[str]]: A tuple containing:
            - A PyTorch Geometric Data object.
            - The molecule with heavy atoms only (excluding hydrogen atoms).
            - The molecule with hydrogenated protein.
            - List of trajectory files (conformers).
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

    # Featurize macrocycle atoms
    atom_features = featurize_macrocycle_atoms(
        rdkit_mol, macrocycle_idxs=macrocycle_idxs, use_peptide_stereo=False)

    # Convert DataFrame to PyTorch tensor
    atom_features_tensors = torch.tensor(atom_features.values, dtype=torch.float)

    bonds = torch.tensor([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in rdkit_mol.GetBonds()], dtype=torch.long).T

    # Create a PyTorch Geometric Data object
    data = torch_geometric.data.Data(
        x=atom_features_tensors,
        edge_index=bonds,
    )
    

    # Use the rest of the conformers as trajfiles
    trajfiles = mols[1:]

    return data, rdkit_mol, rdkit_mol_withH, trajfiles

@singleton
class MDtrajSDFDataset(torch.utils.data.Dataset):
    """PyTorch dataset for MDtraj trajectories from SDF files."""

    def __init__(
            self,
            root: str, 
            sdf_file: str,
            trajfiles: Sequence[str], 
            label: str,
            num_frames: Optional[int] = None,
            start_frame: Optional[int] = None, 
            transform: Optional[Callable] = None, 
            subsample: Optional[int] = None,
            loss_weight: float = 1.0, 
            verbose: bool = False,
            ):
        
        self.root = root
        self.label = lambda: label
        self.transform = transform
        self.loss_weight = loss_weight

        sdf_file = os.path.join(self.root, sdf_file)
        trajfiles = [os.path.join(self.root, filename) for filename in trajfiles]

        # Preprocess the SDF file
        self.data, self.rdkit_mol, self.rdkit_mol_withH, self.trajfiles = preprocess_sdf(sdf_file)
        self.data.loss_weight = torch.tensor([loss_weight], dtype=torch.float32)
        self.data.dataset_label = self.label()
        
        # Ensure the trajectory data is in the correct shape
        self.traj = np.vstack([mol.GetConformer().GetPositions() for mol in self.trajfiles])
        
        # Assuming you know the number of frames and atoms per frame
        n_atoms = self.rdkit_mol.GetNumAtoms()
        n_frames = len(self.traj) // n_atoms

        # Reshape the trajectory data
        self.traj = self.traj.reshape((n_frames, n_atoms, 3))

        # # Print the trajectory data and its shape
        # print("Trajectory data shape:", self.traj.shape)
        # print("Trajectory data:", self.traj)
        assert self.traj.shape == (n_frames, n_atoms, 3)

        if start_frame is None:
            start_frame = 0

        if num_frames == -1 or num_frames is None:
            num_frames = len(self.trajfiles) - start_frame
        
        if subsample is None or subsample == 0:
            subsample = 1

        # Subsample the trajectory.
        self.trajfiles = self.trajfiles[start_frame: start_frame + num_frames: subsample]

        if verbose:
            utils.dist_log(f"Dataset {self.label()}: Loaded SDF file {sdf_file} and trajectory files: {self.trajfiles}.")
            utils.dist_log(f"Dataset {self.label()}: Loaded {len(self.trajfiles)} frames starting from index {start_frame} with subsample {subsample}.")
    
    def __getitem__(self, idx):
        graph = self.data.clone()
        if self.trajfiles:
            traj_frame = self.trajfiles[idx]
            graph.pos = torch.tensor(traj_frame.GetConformer().GetPositions(), dtype=torch.float32)
        if self.transform:
            graph = self.transform(graph)
        return graph

    def __len__(self):
        return len(self.trajfiles) if self.trajfiles else 1
    

@singleton
class MDtrajIterableSDFDataset(torch.utils.data.IterableDataset):
    """PyTorch iterable dataset for MDtraj trajectories from SDF files."""

    def __init__(self, sdf_file: str, label: str, transform: Optional[Callable] = None,
                 subsample: Optional[int] = None, loss_weight: float = 1.0, chunk_size: int = 100,
                 start_at_random_frame: bool = True, verbose: bool = False):
        
        self.label = lambda: label
        self.transform = transform
        self.loss_weight = loss_weight
        self.chunk_size = chunk_size
        self.start_at_random_frame = start_at_random_frame

        if subsample is None or subsample == 0:
            subsample = 1
        self.subsample = subsample

        # Preprocess the SDF file
        self.data, self.rdkit_mol, self.rdkit_mol_withH, self.trajfiles = preprocess_sdf(sdf_file)
        self.data.dataset_label = self.label()
        self.data.loss_weight = torch.tensor([loss_weight], dtype=torch.float32)

        if verbose:
            utils.dist_log(f"Dataset {self.label()}: Iteratively loading trajectory files and SDF file {sdf_file}.")

    def __iter__(self):
        trajfiles = self.trajfiles
        if self.start_at_random_frame:
            trajfiles = np.random.permutation(trajfiles)

        for traj_frame in trajfiles:
            graph = self.data.clone()
            graph.pos = torch.tensor(traj_frame.GetConformer().GetPositions(), dtype=torch.float32)
            if self.transform:
                graph = self.transform(graph)
            yield graph

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