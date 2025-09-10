import functools
import logging
import os
from collections.abc import Callable, Sequence

import mdtraj as md
import numpy as np
import torch
import torch.utils.data
import torch_geometric
from mdtraj.formats.pdb.pdbstructure import PdbStructure

from jamun import utils


def make_graph_from_topology(
    topology: md.Topology,
) -> torch_geometric.data.Data:
    """Create a PyTorch Geometric graph from an MDTraj topology."""
    # Encode the atom types, residue codes, and residue sequence indices.
    atom_type_index = torch.tensor(
        [utils.encode_atom_type(x.element.symbol) for x in topology.atoms], dtype=torch.int32
    )
    residue_code_index = torch.tensor([utils.encode_residue(x.residue.name) for x in topology.atoms], dtype=torch.int32)
    residue_sequence_index = torch.tensor([x.residue.index for x in topology.atoms], dtype=torch.int32)
    atom_code_index = torch.tensor([utils.encode_atom_code(x.name) for x in topology.atoms], dtype=torch.int32)

    # Get the bonded edges from the topology.
    bonded_edge_index = torch.tensor([[bond[0].index, bond[1].index] for bond in topology.bonds], dtype=torch.long).T

    # Create the graph.
    # Positions will be updated later.
    graph = utils.DataWithResidueInformation(
        atom_type_index=atom_type_index,
        residue_code_index=residue_code_index,
        residue_sequence_index=residue_sequence_index,
        atom_code_index=atom_code_index,
        residue_index=residue_sequence_index,
        num_residues=residue_sequence_index.max().item() + 1,
        bonded_edge_index=bonded_edge_index,
        pos=None,
    )
    graph.residues = [x.residue.name for x in topology.atoms]
    graph.atom_names = [x.name for x in topology.atoms]
    graph.num_nodes = topology.n_atoms
    return graph


def preprocess_topology(
    topology: md.Topology,
    keep_hydrogens: bool,
) -> tuple[md.Topology, np.ndarray]:
    """Preprocess the MDtraj topology, returning a PyTorch Geometric graph, the topology with protein only, and the topology with hydrogenated protein."""
    if keep_hydrogens:
        # Select all atoms in the protein.
        select_with_H = topology.select("protein")
        return topology.subset(select_with_H), select_with_H

    # Select all heavy atoms in the protein.
    # This also removes all waters.
    select = topology.select("protein and not type H")
    return topology.subset(select), select


@utils.singleton
class MDtrajIterableDataset(torch.utils.data.IterableDataset):
    """PyTorch iterable dataset for MDtraj trajectories."""

    def __init__(
        self,
        root: str,
        traj_files: Sequence[str],
        pdb_file: str,
        label: str,
        transform: Callable | None = None,
        subsample: int | None = None,
        loss_weight: float = 1.0,
        chunk_size: int = 100,
        start_at_random_frame: bool = False,
        verbose: bool = False,
        keep_hydrogens: bool = False,
        coarse_grained: bool = False,
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

        self.pdb_file = os.path.join(self.root, pdb_file)
        self.original_topology = md.load_topology(self.pdb_file)

        if coarse_grained:
            # In the coarse-grained case, we create bonds between consecutive residues.
            atom_indices = [atom.index for atom in self.original_topology.atoms]
            for i in range(len(atom_indices) - 1):
                self.original_topology.add_bond(
                    self.original_topology.atom(atom_indices[i]), self.original_topology.atom(atom_indices[i + 1])
                )

            py_logger = logging.getLogger(__name__)
            py_logger.warning(
                f"Dataset {self.label()}: No bonds found in topology. Assuming a coarse-grained model and creating bonds between consecutive residues."
            )

        self.top_with_H, self.topology_slice_with_H = preprocess_topology(self.original_topology, keep_hydrogens=True)
        self.top_without_H, self.topology_slice_without_H = preprocess_topology(
            self.original_topology, keep_hydrogens=False
        )
        self.graph_with_H = make_graph_from_topology(self.top_with_H)
        self.graph_without_H = make_graph_from_topology(self.top_without_H)

        if keep_hydrogens:
            self.graph = self.graph_with_H
            self.top = self.top_with_H
            self.topology_slice = self.topology_slice_with_H
        else:
            self.graph = self.graph_without_H
            self.top = self.top_without_H
            self.topology_slice = self.topology_slice_without_H

        self.graph.dataset_label = self.label()
        self.graph.loss_weight = torch.tensor([loss_weight], dtype=torch.float32)

        # self.save_topology_pdb()

        if verbose:
            utils.dist_log(
                f"Dataset {self.label()}: Iteratively loading trajectory files {traj_files} and PDB file {pdb_file}."
            )

    def label(self):
        return self._label

    def save_topology_pdb(self, filename: str | None = None):
        """Save the final topology as a PDB file."""
        if filename is None:
            os.makedirs("dataset_pdbs", exist_ok=True)
            filename = f"dataset_pdbs/{self.label()}.pdb"
        traj = next(md.iterload(self.traj_files[0], top=self.top, chunk=self.chunk_size))
        utils.save_pdb(traj[0], filename)

    def __iter__(self):
        traj_files = self.traj_files
        if self.start_at_random_frame:
            traj_files = np.random.permutation(traj_files)

        for traj_file in traj_files:
            for traj in md.iterload(
                traj_file,
                top=self.original_topology,
                chunk=self.chunk_size,
                stride=self.subsample,
                atom_indices=self.topology_slice,
            ):
                for frame in traj:
                    graph = self.graph.clone("pos")
                    graph.pos = torch.tensor(frame.xyz[0])
                    if self.transform:
                        graph = self.transform(graph)
                    yield graph

    @functools.cached_property
    def topology(self) -> md.Topology:
        return self.top

    @functools.cached_property
    def trajectory(self) -> md.Trajectory:
        return md.load(self.traj_files, top=self.original_topology, atom_indices=self.topology_slice)

    @property
    def trajectory_files(self) -> Sequence[str]:
        return self.traj_files

    @property
    def topology_file(self) -> str:
        return self.pdb_file


@utils.singleton
class MDtrajDataset(torch.utils.data.Dataset):
    """PyTorch dataset for MDtraj trajectories."""

    def __init__(
        self,
        root: str,
        traj_files: Sequence[str],
        pdb_file: str,
        label: str,
        num_frames: int | None = None,
        start_frame: int | None = None,
        transform: Callable | None = None,
        subsample: int | None = None,
        loss_weight: float = 1.0,
        verbose: bool = False,
        keep_hydrogens: bool = False,
        coarse_grained: bool = False,
        atom_indices: Sequence[int] | None = None,
        infer_atom_indices_from_temperature_factors: bool = False,
        lambda_for_atom_indices: int | None = None,
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

        # Infer atom indices from temperature factors, used in FEP datasets.
        if infer_atom_indices_from_temperature_factors:
            with open(pdb_file) as f:
                struct = PdbStructure(f, load_all_models=True)
            temperature_factors = np.asarray([atom.temperature_factor for atom in struct.iter_atoms()])
            lambda_0_indices = np.where(temperature_factors <= 0.5)[0]
            lambda_1_indices = np.where(temperature_factors >= 0.5)[0]

            if lambda_for_atom_indices == 0:
                atom_indices = lambda_0_indices
            elif lambda_for_atom_indices == 1:
                atom_indices = lambda_1_indices
            else:
                raise ValueError(
                    f"Invalid value for lambda_for_atom_indices: {lambda_for_atom_indices}. "
                    "Expected 0 or 1 to select atoms based on temperature factors."
                )

            py_logger = logging.getLogger("jamun")
            py_logger.info(
                f"Dataset {self.label()}: Using atom indices based on temperature factors: {len(atom_indices)} atoms with indices {atom_indices}."
            )

        # If atom indices are provided, slice the trajectory to only include those atoms.
        if atom_indices is None:
            self.atom_indices = None
        else:
            self.atom_indices = np.array(atom_indices, dtype=int)
            self.traj = self.traj.atom_slice(self.atom_indices)

        # Subsample the trajectory.
        self.traj = self.traj[start_frame : start_frame + num_frames : subsample]

        if coarse_grained:
            # In the coarse-grained case, we create bonds between consecutive residues.
            atom_indices = [atom.index for atom in self.traj.topology.atoms]
            for i in range(len(atom_indices) - 1):
                self.traj.topology.add_bond(
                    self.traj.topology.atom(atom_indices[i]), self.traj.topology.atom(atom_indices[i + 1])
                )

            py_logger = logging.getLogger(__name__)
            py_logger.warning(
                f"Dataset {self.label()}: No bonds found in topology. Assuming a coarse-grained model and creating bonds between consecutive residues."
            )

        self.top_with_H, self.topology_slice_with_H = preprocess_topology(self.traj.topology, keep_hydrogens=True)
        self.top_without_H, self.topology_slice_without_H = preprocess_topology(
            self.traj.topology, keep_hydrogens=False
        )
        self.graph_with_H = make_graph_from_topology(self.top_with_H)
        self.graph_without_H = make_graph_from_topology(self.top_without_H)

        if keep_hydrogens:
            self.graph = self.graph_with_H
            self.top = self.top_with_H
            self.topology_slice = self.topology_slice_with_H
        else:
            self.graph = self.graph_without_H
            self.top = self.top_without_H
            self.topology_slice = self.topology_slice_without_H

        self.traj = self.traj.atom_slice(self.topology_slice)

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

    def save_topology_pdb(self, filename: str | None = None):
        """Save the final topology as a PDB file."""
        if filename is None:
            os.makedirs("dataset_pdbs", exist_ok=True)
            filename = f"dataset_pdbs/{self.label()}.pdb"
        utils.save_pdb(self.traj[0], filename)

    def __getitem__(self, idx):
        if idx >= self.traj.n_frames:
            idx = self.traj.n_frames - 1
        graph = self.graph.clone("pos")
        graph.pos = torch.tensor(self.traj.xyz[idx])
        if self.transform:
            graph = self.transform(graph)
        return graph

    def __len__(self):
        return max(32, self.traj.n_frames)

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
