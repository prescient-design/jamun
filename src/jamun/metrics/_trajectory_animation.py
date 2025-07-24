import os
import tempfile

import mdtraj as md
import wandb
from lightning.pytorch.utilities import rank_zero_only

from jamun import utils
from jamun.metrics._utils import TrajectoryMetric


def _save_pdb_to_wandb(trajectory: md.Trajectory, label: str):
    """Save a PDB of the trajectory as a wandb artifact."""
    if rank_zero_only.rank != 0:
        return

    temp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb").name
    utils.save_pdb(trajectory, temp_pdb)

    # Replace slashes in the label to avoid issues with wandb.
    label = label.replace("/", "_").replace("=", "-")
    artifact = wandb.Artifact(label, type="animated_trajectory_pdb")
    artifact.add_file(temp_pdb, "animated_trajectory.pdb")
    wandb.log_artifact(artifact)
    os.remove(temp_pdb)


def _log_trajectory_animation_to_wandb(trajectory: md.Trajectory, label: str):
    """Save an animation of the trajectory as a wandb artifact."""
    if rank_zero_only.rank != 0:
        return

    view = utils.animate_trajectory_with_py3Dmol(trajectory)
    temp_html = tempfile.NamedTemporaryFile(suffix=".temp_html").name
    view.write_html(temp_html)
    with open(temp_html) as f:
        wandb.log({label: wandb.Html(f)})
    os.remove(temp_html)


class TrajectoryVisualizer(TrajectoryMetric):
    """A metric to animate MD trajectories."""

    def __init__(self, num_frames_to_animate: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_frames_to_animate = num_frames_to_animate

    def on_sample_start(self):
        true_trajectory = self.dataset.trajectory
        true_trajectory_subset = true_trajectory[: self.num_frames_to_animate]

        # _save_pdb_to_wandb(true_trajectory_subset, label=f"{self.dataset.label()}/animated_trajectory_pdb/true_traj")
        _log_trajectory_animation_to_wandb(
            true_trajectory_subset, label=f"{self.dataset.label()}/trajectory_animation/true_traj"
        )

    def compute(self) -> dict[str, float]:
        true_trajectory = self.dataset.trajectory
        pred_trajectories = self.sample_trajectories(new=True)
        pred_trajectory_joined = self.joined_sample_trajectory()

        for trajectory_index, pred_trajectory in enumerate(
            pred_trajectories + [pred_trajectory_joined], start=self.num_chains_seen
        ):
            if trajectory_index == len(pred_trajectories) + self.num_chains_seen:
                trajectory_index = "joined"

            pred_trajectory_subset = pred_trajectory[: self.num_frames_to_animate]
            pred_trajectory_subset = pred_trajectory_subset.superpose(true_trajectory[0])
            _log_trajectory_animation_to_wandb(
                pred_trajectory_subset,
                label=f"{self.dataset.label()}/trajectory_animation/pred_traj_{trajectory_index}",
            )
        return {}
