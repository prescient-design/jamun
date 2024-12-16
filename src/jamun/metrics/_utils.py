from typing import List
import logging
import tempfile
import os
from pathlib import Path

import einops
import wandb
import mdtraj as md
import torch
import torch_geometric
import torchmetrics

from jamun import utils_md
from jamun.data import MDtrajDataset
from jamun.utils_residue import ResidueMetadata


def validate_sample(sample: torch_geometric.data.Batch, dataset: MDtrajDataset) -> None:
    """Validate that the sample is compatible with the dataset."""
    if sample.dataset_label != dataset.label():
        raise ValueError(
            f"Sample dataset label {sample.dataset_label} does not match expected label {dataset.label()}."
        )

    expected_atom_types = [atom.element.symbol for atom in dataset.structure.topology.atoms]
    actual_atom_types = [ResidueMetadata.ATOM_TYPES[idx] for idx in sample.atom_type_index]
    if expected_atom_types != actual_atom_types:
        raise ValueError(
            f"Atom types in init_graph ({actual_atom_types}) do not match "
            f"expected atom types in structure ({expected_atom_types})."
        )


class TrajectoryMetric(torchmetrics.Metric):
    """A metric assigned per-dataset that validates and accumulates trajectory samples for this dataset."""

    def __init__(self, dataset: MDtrajDataset, sample_key: str = "xhat_traj"):
        super().__init__()

        self.dataset = dataset
        self.sample_key = sample_key

        self.add_state("samples", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("num_chains_seen", default=torch.tensor(0), dist_reduce_fx="sum")

    def on_sample_start(self):
        """Called at the start of sampling."""
        pass

    def on_after_sample_batch(self) -> None:
        """Called after a batch of samples has been processed."""
        self.num_chains_seen = len(self.samples)

    def on_sample_end(self):
        """Called at the end of sampling."""
        pass

    def update(self, sample: torch_geometric.data.Batch) -> None:
        """Update the metric with a new sample."""
        # py_logger = logging.getLogger("jamun")
        # py_logger.info(f"Dataset {self.dataset.label()}: Obtained samples of shape {sample[self.sample_key].shape}.")
        validate_sample(sample, self.dataset)
        samples = sample[self.sample_key]

        # Reshape samples to be of shape (1, num_atoms, num_frames, 3).
        if samples.ndim != 3:
            raise ValueError(f"Invalid sample shape: {samples.shape}, expected (num_atoms, num_frames, 3).")

        samples = samples[None, ...]
        if len(self.samples) == 0:
            self.samples = samples.clone()
        else:
            self.samples = torch.cat([self.samples, samples])

        # self.samples has shape (batch_size, num_atoms, num_frames, 3).
        assert self.samples.ndim == 4
        # py_logger.info(f"Dataset {self.dataset.label()}: Current samples shape: {self.samples.shape}.")

    def sample_tensors(self, *, new: bool) -> torch.Tensor:
        """Return the samples as a torch.Tensor."""
        if new:
            return self.samples[self.num_chains_seen :]
        return self.samples

    def joined_sample_tensor(self) -> torch.Tensor:
        """Return the samples as a torch.Tensor, concatenated across all batches."""
        return einops.rearrange(
            self.samples, "batch_size num_atoms num_frames coords -> num_atoms (batch_size num_frames) coords"
        )

    def sample_trajectories(self, *, new: bool) -> List[md.Trajectory]:
        """Convert the samples to MD trajectories."""
        if new:
            samples = self.samples[self.num_chains_seen :]
        else:
            samples = self.samples

        trajectories = utils_md.coordinates_to_trajectories(samples, self.dataset.structure)
        return trajectories

    def joined_sample_trajectory(self) -> md.Trajectory:
        """Convert the samples to a single MD trajectory."""
        py_logger = logging.getLogger("jamun")

        trajectories = utils_md.coordinates_to_trajectories(self.samples, self.dataset.structure)
        py_logger.info(f"{self.dataset.label()}: Joining {len(trajectories)} trajectories into 1.")

        joined_trajectory = md.join(trajectories, check_topology=True)
        return joined_trajectory
