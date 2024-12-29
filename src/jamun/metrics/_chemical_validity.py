from typing import Dict, List, Sequence, Tuple

import mdtraj as md
import numpy as np
import wandb
from lightning.pytorch.utilities import rank_zero_only

from jamun import utils
from jamun.metrics import TrajectoryMetric

# Van der Waals radii in nm.
VDW_RADII = {
    "C": 0.170,
    "O": 0.152,
    "N": 0.155,
    "H": 0.120,
    "F": 0.147,
    "S": 0.180,
    "other": 0.150,
}

# Approximate covalent radii in nm.
COVALENT_RADII = {
    "C": 0.076,
    "O": 0.066,
    "N": 0.071,
    "H": 0.031,
    "F": 0.057,
    "S": 1.005,
    "other": 0.070,
}


def _validate_positions_and_atom_types(positions: np.ndarray, atom_types: np.ndarray) -> None:
    num_frames, num_atoms, dim = positions.shape
    if dim != 3:
        raise ValueError("Only 3D positions are supported")
    if len(atom_types) != num_atoms:
        raise ValueError(f"Expected {num_atoms} atom types, got {len(atom_types)}.")
    if not isinstance(atom_types[0], str):
        raise ValueError("Atom types should be single characters.")


def check_volume_exclusion(traj: md.Trajectory, tolerance: float) -> List[List[Tuple[int, int]]]:
    """Check for volume exclusion between atoms in a trajectory."""
    positions = traj.xyz
    atoms, _ = traj.topology.to_dataframe()
    atom_types = atoms["element"].values
    adjacency_pairs = [(bond.atom1.index, bond.atom2.index) for bond in traj.topology.bonds]
    num_nonbonded_atom_pairs = (len(atom_types) * (len(atom_types) - 1)) // 2 - len(adjacency_pairs)
    issues = _check_volume_exclusion(positions, atom_types, adjacency_pairs, tolerance)
    return [len(frame_issues) / num_nonbonded_atom_pairs for frame_issues in issues]


def _check_volume_exclusion(
    positions: np.ndarray, atom_types: np.ndarray, adjacency_pairs: Sequence[Tuple[int, int]], tolerance: float
) -> List[List[Tuple[int, int]]]:
    """Check for volume exclusion between atoms in a trajectory."""
    positions = np.asarray(positions)
    atom_types = np.asarray(atom_types)
    adjacency_pairs = set(adjacency_pairs)

    _validate_positions_and_atom_types(positions, atom_types)
    num_frames, num_atoms, _ = positions.shape

    issues = [[] for _ in range(num_frames)]
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            # Only check non-bonded atoms.
            if (i, j) in adjacency_pairs:
                continue

            dists = np.linalg.norm(positions[:, i] - positions[:, j], axis=1)

            vdW_dist = VDW_RADII[atom_types[i]] + VDW_RADII[atom_types[j]]

            for frame_num, dist in enumerate(dists):
                if dist < (1 - tolerance) * vdW_dist:
                    issues[frame_num].append((i, j))
                    # print(f"Frame {frame_num}: Atoms {i} {atom_types[i]}  and {j} {atom_types[j]} are too close: {dist} < {vdW_dist}.")

    return issues


def check_bond_lengths(traj: md.Trajectory, tolerance: float) -> List[List[Tuple[int, int]]]:
    """Check for bond lengths in a trajectory."""
    positions = traj.xyz
    atoms, _ = traj.topology.to_dataframe()
    atom_types = atoms["element"].values
    adjacency_pairs = [(bond.atom1.index, bond.atom2.index) for bond in traj.topology.bonds]
    num_bonds = len(adjacency_pairs)
    issues = _check_bond_lengths(positions, atom_types, adjacency_pairs, tolerance)
    return [len(issues) / num_bonds for issues in issues]


def _check_bond_lengths(
    positions: np.ndarray, atom_types: np.ndarray, adjacency_pairs: Sequence[Tuple[int, int]], tolerance: float
) -> List[List[Tuple[int, int]]]:
    """Check for bond lengths in a trajectory."""
    positions = np.asarray(positions)
    atom_types = np.asarray(atom_types)
    adjacency_pairs = set(adjacency_pairs)

    _validate_positions_and_atom_types(positions, atom_types)
    num_frames, num_atoms, _ = positions.shape

    issues = [[] for _ in range(num_frames)]
    for i, j in adjacency_pairs:
        dists = np.linalg.norm(np.array(positions[:, i]) - np.array(positions[:, j]), axis=1)

        covalent_dist = COVALENT_RADII[atom_types[i]] + COVALENT_RADII[atom_types[j]]

        for frame_num, dist in enumerate(dists):
            if dist > (1 + tolerance) * covalent_dist or dist < (1 - tolerance) * covalent_dist:
                issues[frame_num].append((i, j))
                # print(f"Frame {frame_num}: Atoms {i} {atom_types[i]}  and {j} {atom_types[j]} have bond length {dist} instead of {covalent_dist}.")

    return issues


class ChemicalValidityMetrics(TrajectoryMetric):
    """Computes chemical validity metrics using bond lengths and volume exclusion tests."""

    def __init__(
        self,
        volume_exclusion_tolerance: float,
        bond_length_tolerance: float,
        num_molecules_per_trajectory: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.bond_length_tolerance = bond_length_tolerance
        self.volume_exclusion_tolerance = volume_exclusion_tolerance
        self.num_molecules_per_trajectory = num_molecules_per_trajectory

    def on_sample_start(self):
        true_trajectory = self.dataset.trajectory
        subsampling_factor = max(len(true_trajectory) // self.num_molecules_per_trajectory, 1)
        true_trajectory_subset = true_trajectory[::subsampling_factor]

        # Check for overlapping atoms in the true trajectory.
        metrics = {}
        avg_volume_exclusion_issues = check_volume_exclusion(
            true_trajectory_subset,
            self.volume_exclusion_tolerance,
        )
        avg_volume_exclusion_issues_table = wandb.Table(
            data=[[val] for val in avg_volume_exclusion_issues], columns=["avg_volume_exclusion_issues"]
        )
        metrics[f"{self.dataset.label()}/mean_volume_exclusion_issues/true_traj"] = np.mean(avg_volume_exclusion_issues)
        if rank_zero_only.rank == 0:
            utils.wandb_dist_log(
                {
                    f"{self.dataset.label()}/volume_exclusion_issues/true_traj": wandb.plot.histogram(
                        avg_volume_exclusion_issues_table,
                        "avg_volume_exclusion_issues",
                        title="Average number of volume exclusion issues (true trajectory)",
                    )
                }
            )

        # Check for invalid bonds in the true trajectory.
        avg_bond_length_issues = check_bond_lengths(
            true_trajectory_subset,
            self.bond_length_tolerance,
        )
        avg_bond_length_issues_table = wandb.Table(
            data=[[val] for val in avg_bond_length_issues], columns=["avg_bond_length_issues"]
        )
        metrics[f"{self.dataset.label()}/mean_bond_length_issues/true_traj"] = np.mean(avg_bond_length_issues)
        
        if rank_zero_only.rank == 0:
            utils.wandb_dist_log(
                {
                    f"{self.dataset.label()}/bond_length_issues/true_traj": wandb.plot.histogram(
                        avg_bond_length_issues_table,
                        "avg_bond_length_issues",
                        title="Average number of bond length issues (true trajectory)",
                    )
                }
            )
        return metrics

    def compute(self) -> Dict[str, float]:
        metrics = {}
        pred_trajectories = self.sample_trajectories(new=True)
        for trajectory_index, pred_trajectory in enumerate(pred_trajectories, start=self.num_chains_seen):
            # Subsample the trajectory.
            subsampling_factor = max(len(pred_trajectory) // self.num_molecules_per_trajectory, 1)
            pred_trajectory_subset = pred_trajectory[::subsampling_factor]

            # Check for overlapping atoms.
            avg_volume_exclusion_issues = check_volume_exclusion(
                pred_trajectory_subset,
                self.volume_exclusion_tolerance,
            )
            avg_volume_exclusion_issues_table = wandb.Table(
                data=[[val] for val in avg_volume_exclusion_issues], columns=["avg_volume_exclusion_issues"]
            )
            metrics[f"{self.dataset.label()}/mean_volume_exclusion_issues/pred_traj_{trajectory_index}"] = np.mean(
                avg_volume_exclusion_issues
            )
            if rank_zero_only.rank == 0:
                utils.wandb_dist_log(
                    {
                        f"{self.dataset.label()}/volume_exclusion_issues/pred_traj_{trajectory_index}": wandb.plot.histogram(
                            avg_volume_exclusion_issues_table,
                            "avg_volume_exclusion_issues",
                            title=f"Average number of volume exclusion issues (predicted trajectory {trajectory_index})",
                        )
                    }
                )

            # Check for invalid bonds.
            avg_bond_length_issues = check_bond_lengths(
                pred_trajectory_subset,
                self.bond_length_tolerance,
            )
            avg_bond_length_issues_table = wandb.Table(
                data=[[val] for val in avg_bond_length_issues], columns=["avg_bond_length_issues"]
            )
            metrics[f"{self.dataset.label()}/mean_bond_length_issues/pred_traj_{trajectory_index}"] = np.mean(
                avg_bond_length_issues
            )
            if rank_zero_only.rank == 0:
                utils.wandb_dist_log(
                    {
                        f"{self.dataset.label()}/bond_length_issues/pred_traj_{trajectory_index}": wandb.plot.histogram(
                            avg_bond_length_issues_table,
                            "avg_bond_length_issues",
                            title=f"Average number of bond length issues (predicted trajectory {trajectory_index})",
                        )
                    }
                )
        return metrics
