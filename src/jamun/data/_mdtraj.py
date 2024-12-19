from typing import Callable, Dict, Optional, Sequence
import functools
import logging
import os

import lightning.pytorch as pl
import mdtraj as md
import torch
import torch.utils.data
import torch_geometric
import numpy as np

from jamun import utils


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

        if start_frame is None:
            start_frame = 0

        py_logger = logging.getLogger("jamun")
        pdbfile = os.path.join(self.root, pdbfile)
        if trajfiles[0].endswith(".npz") or trajfiles[0].endswith(".npy"):
            self.traj = md.load(pdbfile)
            self.traj.xyz = np.vstack([np.load(os.path.join(self.root, filename))["positions"] for filename in trajfiles])

            assert self.traj.xyz.shape[1] == self.traj.n_atoms
            assert self.traj.xyz.shape[2] == 3

            self.traj.time = np.arange(self.traj.n_frames)
        else:
            self.traj = md.load([os.path.join(self.root, filename) for filename in trajfiles], top=pdbfile)

        if num_frames == -1 or num_frames is None:
            num_frames = self.traj.n_frames - start_frame

        if subsample == None or subsample == 0:
            subsample = 1

        # Subsample the trajectory.
        self.traj = self.traj[start_frame : start_frame + num_frames : subsample]

        self.top = self.traj.topology.subset(self.traj.topology.select("protein and not type H"))
        self.top_withH = self.traj.topology.subset(self.traj.topology.select("protein"))
        self.traj = self.traj.atom_slice(self.traj.topology.select("protein and not type H"))

        # Encode the atom types, residue codes, and residue sequence indices.
        atom_type_index = torch.tensor([utils.encode_atom_type(x.element.symbol) for x in self.top.atoms], dtype=torch.int32)
        residue_code_index = torch.tensor([utils.encode_residue(x.residue.name) for x in self.top.atoms], dtype=torch.int32)
        residue_sequence_index = torch.tensor([x.residue.index for x in self.top.atoms], dtype=torch.int32)
        atom_code_index = torch.tensor([utils.encode_atom_code(x.name) for x in self.top.atoms], dtype=torch.int32)

        bonds = torch.tensor([[bond[0].index, bond[1].index] for bond in self.top.bonds], dtype=torch.long).T
        positions = torch.tensor(self.traj.xyz[0][self.traj.top.select("protein and not type H")], dtype=torch.float)
        loss_weight = torch.tensor([self.loss_weight], dtype=torch.float)

        # Create the graph.
        # Positions will be updated in __getitem__.
        self.graph = utils.DataWithResidueInformation(
            atom_type_index=atom_type_index,
            residue_code_index=residue_code_index,
            residue_sequence_index=residue_sequence_index,
            atom_code_index=atom_code_index,
            residue_index=residue_sequence_index,
            num_residues=residue_sequence_index.max().item() + 1,
            edge_index=bonds,
            pos=positions,
        )
        self.graph.residues = [x.residue.name for x in self.top.atoms]
        self.graph.atom_names = [x.name for x in self.top.atoms]
        self.graph.dataset_label = self.label()
        self.graph.loss_weight = loss_weight

        py_logger.info(
            f"Dataset {self.label()}: Loaded {self.traj.n_frames} frames starting from index {start_frame} with subsample {subsample}."
        )
        self.save_modified_pdb()

    def save_modified_pdb(self):
        os.makedirs("pdbs", exist_ok=True)
        filename = f"pdbs/{self.label()}-modified.pdb"
        
        with open(filename, "w") as f:
            f.write("MODEL        0\n")

            for j, positions in enumerate(self.graph.pos):
                f.write(
                    "ATOM  %5d %-4s %3s %1s%4d    %s%s%s  1.00 %5s      %-4s%2s  \n"
                    % (
                        j + 1,
                        self.top.atom(j).name,
                        self.top.atom(j).residue.name,
                        self.top.atom(j).residue.chain.index + 1,
                        self.top.atom(j).residue.index + 1,
                        "%8.3f" % (positions[0] * 10),
                        "%8.3f" % (positions[1] * 10),
                        "%8.3f" % (positions[2] * 10),
                        1,
                        0,
                        utils.ResidueMetadata.ATOM_TYPES[self.graph.atom_type_index[j].item()],
                    )
                )

            f.write(
                "TER   %5d      %3s %s%4d\n"
                % (
                    j + 2,
                    self.top.atom(j).residue.name,
                    self.top.atom(j).residue.chain.index,
                    self.top.atom(j).residue.index + 1,
                )
            )

            bonds = [[i + 1] for i in range(self.top.n_atoms)]
            for bond in self.graph.edge_index.T:
                bonds[bond[0]].append(bond[1] + 1)
                bonds[bond[1]].append(bond[0] + 1)
            for bond in bonds:
                s = "".join(["%5d" % atom for atom in bond])
                f.write("CONECT%s\n" % s)

            f.write("ENDMDL\n")
            f.write("END\n")

    def __getitem__(self, idx):
        graph = self.graph.clone()
        graph.pos = torch.tensor(self.traj.xyz[idx])
        if self.transform:
            graph = self.transform(graph)
        return graph

    def __len__(self):
        return self.traj.n_frames

    @functools.cached_property
    def structure(self) -> md.Trajectory:
        return self.traj

    @functools.cached_property
    def trajectory(self) -> md.Trajectory:
        return self.traj


class MDtrajDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for MDtraj datasets."""

    def __init__(
        self,
        datasets: Dict[str, Sequence[MDtrajDataset]],
        seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 2,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.datasets = datasets
        self.concatenated_datasets = {}

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        py_logger = logging.getLogger("jamun")
        for split, datasets in self.datasets.items():
            if datasets is None:
                continue

            self.concatenated_datasets[split] = torch.utils.data.ConcatDataset(datasets)
            py_logger.info(
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
