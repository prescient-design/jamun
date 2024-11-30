from typing import Dict, Union

import logging
import tempfile
import os

import numpy as np
import mdtraj as md
import wandb
import matplotlib.pyplot as plt

from jamun import utils_md
from jamun.data import MDtrajDataset
from jamun.metrics import TrajectoryMetric


class SaveTrajectory(TrajectoryMetric):
    """A metric that saves the predicted and true samples."""

    def __init__(self, output_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_samples_dir = os.path.join(output_dir, self.dataset.label(), "predicted_samples") 
        self.true_samples_dir = os.path.join(output_dir, self.dataset.label(), "true_samples")
    
        # Create the output directories.
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
        # Save the true trajectory as a PDB and DCD file.
        true_trajectory = self.dataset.trajectory
        utils_md.save_pdb(true_trajectory, self.filename_true(0, "pdb"))
        true_trajectory.save_dcd(self.filename_true(0, "dcd"))
        
    def on_sample_end(self):
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
            utils_md.save_pdb(pred_trajectory, self.filename_pred(trajectory_index, "pdb"))
            pred_trajectory.save_dcd(self.filename_pred(trajectory_index, "dcd"))
        
        pred_trajectory_joined = self.joined_sample_trajectory()
        utils_md.save_pdb(pred_trajectory_joined, self.filename_pred("joined", "pdb"))
        pred_trajectory_joined.save_dcd(self.filename_pred("joined", "dcd"))
        
        return {}