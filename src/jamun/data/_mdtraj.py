import functools
import os
import threading
from typing import Callable, Dict, Optional, Sequence, Tuple

import lightning.pytorch as pl
import mdtraj as md
import numpy as np
import torch
import torch.utils.data
import torch_geometric

from jamun import utils


def singleton(cls):
    """Decorator to create a singleton instance of a class, based on the arguments to __init__."""

    _instances = {}
    _lock = threading.Lock()

    def get_instance(*args, **kwargs):
        # Convert args and kwargs to hashable types.
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

        obj_key = (args, frozenset(kwargs.items()))
        if obj_key not in _instances:
            with _lock:
                if obj_key not in _instances:
                    _instances[obj_key] = cls(*args, **kwargs)
        return _instances[obj_key]

    return get_instance


def preprocess_topology(topology: md.Topology) -> Tuple[torch_geometric.data.Data, md.Topology, md.Topology]:
    """Preprocess the MDtraj topology, returning a PyTorch Geometric graph, the topology with protein only, and the topology with hydrogenated protein."""
    # Select all heavy atoms in the protein.
    # This also removes all waters.
    select = topology.select("protein and not type H")
    top = topology.subset(select)

    # Select all atoms in the protein.
    select_withH = topology.select("protein")
    top_withH = topology.subset(select_withH)

    # Encode the atom types, residue codes, and residue sequence indices.
    atom_type_index = torch.tensor([utils.encode_atom_type(x.element.symbol) for x in top.atoms], dtype=torch.int32)
    residue_code_index = torch.tensor([utils.encode_residue(x.residue.name) for x in top.atoms], dtype=torch.int32)
    residue_sequence_index = torch.tensor([x.residue.index for x in top.atoms], dtype=torch.int32)
    atom_code_index = torch.tensor([utils.encode_atom_code(x.name) for x in top.atoms], dtype=torch.int32)

    bonds = torch.tensor([[bond[0].index, bond[1].index] for bond in top.bonds], dtype=torch.long).T

    # Create the graph.
    # Positions will be updated later.
    graph = utils.DataWithResidueInformation(
        atom_type_index=atom_type_index,
        residue_code_index=residue_code_index,
        residue_sequence_index=residue_sequence_index,
        atom_code_index=atom_code_index,
        residue_index=residue_sequence_index,
        num_residues=residue_sequence_index.max().item() + 1,
        edge_index=bonds,
        pos=None,
    )
    graph.residues = [x.residue.name for x in top.atoms]
    graph.atom_names = [x.name for x in top.atoms]
    return graph, top, top_withH


@singleton
class MDtrajIterableDataset(torch.utils.data.IterableDataset):
    """PyTorch iterable dataset for MDtraj trajectories."""

    def __init__(
        self,
        root: str,
        trajfiles: Sequence[str],
        pdbfile: str,
        label: str,
        transform: Optional[Callable] = None,
        loss_weight: float = 1.0,
        chunk_size: int = 100,
    ):
        self.root = root
        self.label = lambda: label
        self.transform = transform
        self.loss_weight = loss_weight
        self.chunk_size = chunk_size
        self.all_files = []

        self.trajfiles = [os.path.join(self.root, filename) for filename in trajfiles]

        pdbfile = os.path.join(self.root, pdbfile)
        topology = md.load_topology(pdbfile)

        self.graph, self.top, self.top_withH = preprocess_topology(topology)
        self.graph.dataset_label = self.label()
        self.graph.loss_weight = torch.tensor([loss_weight], dtype=torch.float32)

        utils.dist_log(f"Dataset {self.label()}: Iteratively loading trajectory files {trajfiles} and PDB file {pdbfile}.")
        self.save_topology_pdb()

    def save_topology_pdb(self):
        os.makedirs("dataset_pdbs", exist_ok=True)
        filename = f"dataset_pdbs/{self.label()}.pdb"
        traj = next(md.iterload(self.trajfiles[0], top=self.top, chunk=self.chunk_size))
        utils.save_pdb(traj[0], filename)

    def __iter__(self):
        for trajfile in self.trajfiles:
            for traj in md.iterload(trajfile, top=self.top, chunk=self.chunk_size):
                for frame in traj:
                    graph = self.graph.clone()
                    graph.pos = torch.tensor(frame.xyz[0])
                    if self.transform:
                        graph = self.transform(graph)
                    yield graph

    @functools.cached_property
    def structure(self) -> md.Trajectory:
        return self.top

    @functools.cached_property
    def trajectory(self) -> md.Trajectory:
        return md.load(self.trajfiles, top=self.top)


@singleton
class MDtrajDataset(torch.utils.data.Dataset):
    """PyTorch dataset for MDtraj trajectories."""

    def __init__(
        self,
        root: str,
        trajfiles: Sequence[str],
        pdbfile: str,
        label: str,
        num_frames: Optional[int] = None,
        start_frame: Optional[int] = None,
        transform: Optional[Callable] = None,
        subsample: Optional[int] = None,
        loss_weight: float = 1.0,
    ):
        self.root = root
        self.label = lambda: label
        self.transform = transform
        self.loss_weight = loss_weight
        self.all_files = []

        pdbfile = os.path.join(self.root, pdbfile)
        trajfiles = [os.path.join(self.root, filename) for filename in trajfiles]

        if trajfiles[0].endswith(".npz") or trajfiles[0].endswith(".npy"):
            self.traj = md.load(pdbfile)
            self.traj.xyz = np.vstack(
                [np.load(os.path.join(self.root, filename))["positions"] for filename in trajfiles]
            )

            assert self.traj.xyz.shape[1] == self.traj.n_atoms
            assert self.traj.xyz.shape[2] == 3

            self.traj.time = np.arange(self.traj.n_frames)
        else:
            self.traj = md.load(trajfiles, top=pdbfile)

        if start_frame is None:
            start_frame = 0

        if num_frames == -1 or num_frames is None:
            num_frames = self.traj.n_frames - start_frame

        if subsample is None or subsample == 0:
            subsample = 1

        # Subsample the trajectory.
        self.traj = self.traj[start_frame : start_frame + num_frames : subsample]

        topology = self.traj.topology
        self.graph, self.top, self.top_withH = preprocess_topology(topology)
        self.traj = self.traj.atom_slice(self.top.select("all"))

        self.graph.pos = torch.tensor(self.traj.xyz[0], dtype=torch.float32)
        self.graph.loss_weight = torch.tensor([loss_weight], dtype=torch.float32)
        self.graph.dataset_label = self.label()

        utils.dist_log(f"Dataset {self.label()}: Loading trajectory files {trajfiles} and PDB file {pdbfile}.")
        utils.dist_log(
            f"Dataset {self.label()}: Loaded {self.traj.n_frames} frames starting from index {start_frame} with subsample {subsample}."
        )
        self.save_topology_pdb()

    def save_topology_pdb(self):
        os.makedirs("dataset_pdbs", exist_ok=True)
        filename = f"dataset_pdbs/{self.label()}.pdb"
        utils.save_pdb(self.traj[0], filename)

    def __getitem__(self, idx):
        graph = self.graph.clone()
        graph.pos = torch.tensor(self.traj.xyz[idx])
        if self.transform:
            graph = self.transform(graph)
        return graph

    def __len__(self):
        return self.traj.n_frames

    @functools.cached_property
    def topology(self) -> md.Topology:
        return self.traj.topology

    @functools.cached_property
    def trajectory(self) -> md.Trajectory:
        return self.traj


class MDtrajDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for MDtraj datasets."""

    def __init__(
        self,
        datasets: Dict[str, Sequence[MDtrajDataset]],
        batch_size: int,
        num_workers: int = 2,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.datasets = datasets
        self.concatenated_datasets = {}

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        for split, datasets in self.datasets.items():
            if datasets is None:
                continue

            self.concatenated_datasets[split] = torch.utils.data.ConcatDataset(datasets)
            utils.dist_log(
                f"Split {split}: Loaded {len(self.concatenated_datasets[split])} frames in total from {len(datasets)} datasets: {[dataset.label() for dataset in datasets]}."
            )

    def train_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self.concatenated_datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self.concatenated_datasets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self.concatenated_datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers
        )
