import os
from typing import Dict, Union

import numpy as np
import wandb
from lightning.pytorch.utilities import rank_zero_only

from jamun import utils
from jamun.metrics._utils import TrajectoryMetric


class SaveTrajectory(TrajectoryMetric):
    """A metric that saves the predicted and true samples."""

    def __init__(self, save_true_trajectory: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = os.path.join("sampler", self.dataset.label())
        self.pred_samples_dir = os.path.join(self.output_dir, "predicted_samples")
        self.true_samples_dir = os.path.join(self.output_dir, "true_samples")

        # Create the output directories.
        self.save_true_trajectory = save_true_trajectory
        if self.save_true_trajectory:
            self.true_samples_extensions = ["pdb", "dcd"]
            for ext in self.true_samples_extensions:
                os.makedirs(os.path.join(self.true_samples_dir, ext), exist_ok=True)

        self.pred_samples_extensions = ["npy", "pdb", "dcd"]
        for ext in self.pred_samples_extensions:
            os.makedirs(os.path.join(self.pred_samples_dir, ext), exist_ok=True)


    def filename_pred(self, trajectory_index: Union[int, str], extension: str) -> str:
        """Returns the filename for the predicted samples."""
        if extension not in self.pred_samples_extensions:
            raise ValueError(f"Invalid extension: {extension}")
        filenames = {
            "npy": os.path.join(self.pred_samples_dir, "npy", f"{trajectory_index}.npy"),
            "pdb": os.path.join(self.pred_samples_dir, "pdb", f"{trajectory_index}.pdb"),
            "dcd": os.path.join(self.pred_samples_dir, "dcd", f"{trajectory_index}.dcd"),
        }
        return filenames[extension]

    def filename_true(self, trajectory_index: Union[int, str], extension: str) -> str:
        """Returns the filename for the true samples."""
        if extension not in self.true_samples_extensions:
            raise ValueError(f"Invalid extension: {extension}")
        filenames = {
            "pdb": os.path.join(self.true_samples_dir, "pdb", f"{trajectory_index}.pdb"),
            "dcd": os.path.join(self.true_samples_dir, "dcd", f"{trajectory_index}.dcd"),
        }
        return filenames[extension]

    def on_sample_start(self):
        # Save topology from the true trajectory.
        true_trajectory = self.dataset.trajectory
        utils.save_pdb(true_trajectory[0], os.path.join(self.output_dir, "topology.pdb"))

        if not self.save_true_trajectory:
            return

        utils.save_pdb(true_trajectory, self.filename_true(0, "pdb"))
        true_trajectory.save_dcd(self.filename_true(0, "dcd"))

    def on_sample_end(self):
        if rank_zero_only.rank != 0:
            return

        # Save the joined samples at the very end of sampling to wandb.
        label = self.dataset.label()
        label = label.replace("/", "_").replace("=", "-")

        for ext in ["npy", "pdb", "dcd"]:
            filename = self.filename_pred("joined", ext)
            artifact = wandb.Artifact(f"{label}_pred_samples_joined", type="pred_samples_joined")
            artifact.add_file(filename, f"pred_samples_joined.{ext}")
            wandb.log_artifact(artifact)

    def compute(self) -> Dict[str, float]:
        # Save the predicted samples as numpy files.
        samples_np = self.sample_tensors(new=True).cpu().detach().numpy()
        for trajectory_index, sample in enumerate(samples_np):
            np.save(self.filename_pred(trajectory_index, "npy"), sample)

        samples_joined_np = self.joined_sample_tensor().cpu().detach().numpy()
        np.save(self.filename_pred("joined", "npy"), samples_joined_np)

        # Save the predict sample trajectory as a PDB and DCD file.
        pred_trajectories = self.sample_trajectories(new=True)
        for trajectory_index, pred_trajectory in enumerate(pred_trajectories, start=self.num_chains_seen):
            utils.save_pdb(pred_trajectory, self.filename_pred(trajectory_index, "pdb"))
            pred_trajectory.save_dcd(self.filename_pred(trajectory_index, "dcd"))

        pred_trajectory_joined = self.joined_sample_trajectory()
        utils.save_pdb(pred_trajectory_joined, self.filename_pred("joined", "pdb"))
        pred_trajectory_joined.save_dcd(self.filename_pred("joined", "dcd"))

        return {}
