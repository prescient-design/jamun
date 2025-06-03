from typing import Callable, Optional, Sequence, Tuple
import functools
import os

import mdtraj as md
import numpy as np
import torch
import torch.utils.data
import torch_geometric

from jamun import utils


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


@utils.singleton
class MDtrajIterableDataset(torch.utils.data.IterableDataset):
    """PyTorch iterable dataset for MDtraj trajectories."""

    def __init__(
        self,
        root: str,
        traj_files: Sequence[str],
        pdb_file: str,
        label: str,
        transform: Optional[Callable] = None,
        subsample: Optional[int] = None,
        loss_weight: float = 1.0,
        chunk_size: int = 100,
        start_at_random_frame: bool = False,
        verbose: bool = False,
    ):
        self.root = root
        self._label = label
        self.transform = transform
        self.loss_weight = loss_weight
        self.chunk_size = chunk_size
        self.start_at_random_frame = start_at_random_frame

        self.traj_files = [os.path.join(self.root, filename) for filename in traj_files]

        if subsample is None or subsample == 0:
            subsample = 1
        self.subsample = subsample

        pdb_file = os.path.join(self.root, pdb_file)
        topology = md.load_topology(pdb_file)

        self.graph, self.top, self.top_withH = preprocess_topology(topology)
        self.graph.dataset_label = self.label()
        self.graph.loss_weight = torch.tensor([loss_weight], dtype=torch.float32)

        # self.save_topology_pdb()

        if verbose:
            utils.dist_log(
                f"Dataset {self.label()}: Iteratively loading trajectory files {traj_files} and PDB file {pdb_file}."
            )

    def label(self):
        return self._label

    def save_topology_pdb(self):
        os.makedirs("dataset_pdbs", exist_ok=True)
        filename = f"dataset_pdbs/{self.label()}.pdb"
        traj = next(md.iterload(self.traj_files[0], top=self.top, chunk=self.chunk_size))
        utils.save_pdb(traj[0], filename)

    def __iter__(self):
        traj_files = self.traj_files
        if self.start_at_random_frame:
            traj_files = np.random.permutation(traj_files)

        for traj_file in traj_files:
            for traj in md.iterload(traj_file, top=self.top, chunk=self.chunk_size, stride=self.subsample):
                for frame in traj:
                    graph = self.graph.clone()
                    graph.pos = torch.tensor(frame.xyz[0])
                    if self.transform:
                        graph = self.transform(graph)
                    yield graph

    @functools.cached_property
    def topology(self) -> md.Topology:
        return self.top

    @functools.cached_property
    def trajectory(self) -> md.Trajectory:
        return md.load(self.traj_files, top=self.top)


@utils.singleton
class MDtrajDataset(torch.utils.data.Dataset):
    """PyTorch dataset for MDtraj trajectories."""

    def __init__(
        self,
        root: str,
        traj_files: Sequence[str],
        pdb_file: str,
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

        pdb_file = os.path.join(self.root, pdb_file)
        traj_files = [os.path.join(self.root, filename) for filename in traj_files]
       
        self.traj_files = traj_files
        self.pdb_file = pdb_file

        if traj_files[0].endswith(".npz") or traj_files[0].endswith(".npy"):
            self.traj = md.load(pdb_file)
            self.traj.xyz = np.vstack(
                [np.load(os.path.join(self.root, filename))["positions"] for filename in traj_files]
            )

            assert self.traj.xyz.shape == (self.traj.n_frames, self.traj.n_atoms, 3)

            self.traj.time = np.arange(self.traj.n_frames)
        else:
            self.traj = md.load(traj_files, top=pdb_file)

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
        self.traj = self.traj.atom_slice(topology.select("protein and not type H"))

        self.graph.pos = torch.tensor(self.traj.xyz[0], dtype=torch.float32)
        self.graph.loss_weight = torch.tensor([loss_weight], dtype=torch.float32)
        self.graph.dataset_label = self.label()

        # self.save_topology_pdb()

        if verbose:
            utils.dist_log(f"Dataset {self.label()}: Loading trajectory files {traj_files} and PDB file {pdb_file}.")
            utils.dist_log(
                f"Dataset {self.label()}: Loaded {self.traj.n_frames} frames starting from index {start_frame} with subsample {subsample}."
            )

    def label(self):
        return self._label

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

    @property
    def trajectory_files(self) -> Sequence[str]:
        return self.traj_files

    @property
    def topology_file(self) -> str:
        return self.pdb_file
