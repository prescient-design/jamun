from typing import Dict, Optional
import os
import tempfile
import math
import logging

import mdtraj as md
import wandb

from jamun import utils
from jamun.metrics import TrajectoryMetric


class MDSampleVisualizer(TrajectoryMetric):
    """A metric to visualize static MD samples."""

    def __init__(self, num_samples_to_plot: int, subsample: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Round up to the nearest perfect square.
        num_rows = int(math.ceil(math.sqrt(num_samples_to_plot)))
        self.num_samples_to_plot = num_rows**2
        self.num_rows = num_rows
        if subsample is None:
            subsample = 1
        self.subsample = subsample

    def align_and_subsample(self, traj: md.Trajectory) -> md.Trajectory:
        traj_subset = traj[: self.num_samples_to_plot * self.subsample : self.subsample]
        traj_subset = traj_subset.superpose(self.dataset.trajectory[0])
        return traj_subset

    def plot(self, traj: md.Trajectory, label: str):
        # Convert the trajectories to RDKit mols.
        all_mols = utils.to_rdkit_mols(traj)

        # Warn if there are not enough samples to plot.
        if len(all_mols) < self.num_samples_to_plot:
            logger = logging.getLogger("jamun")
            logger.warning(f"Only {len(all_mols)} samples available for visualization.")

        # Create a dictionary of the RDKit mols, indexed by row.
        mols = {}
        for row in range(self.num_rows):
            mols[row] = all_mols[row * self.num_rows : (row + 1) * self.num_rows]

        # Plot with py3Dmol.
        view = utils.plot_molecules_with_py3Dmol(mols, show_keys=False)

        # Log the HTML file to Weights & Biases.
        temp_html = tempfile.NamedTemporaryFile(suffix=".html").name
        view.write_html(temp_html)
        with open(temp_html) as f:
            wandb.run.log({f"{self.dataset.label()}/visualize_samples/3D_view/{label}": wandb.Html(f)})
        os.remove(temp_html)

    def compute(self) -> Dict[str, float]:
        pred_trajectories = self.sample_trajectories(new=True)
        for trajectory_index, pred_trajectory in enumerate(pred_trajectories, start=self.num_chains_seen):
            logger = logging.getLogger("jamun")
            logger.info(
                f"Visualizing trajectory {trajectory_index} ({pred_trajectory}) for dataset {self.dataset.label()}."
            )
            pred_trajectory_subset = self.align_and_subsample(pred_trajectory)
            self.plot(pred_trajectory_subset, f"pred_traj_{trajectory_index}")
        return {}

    def on_sample_start(self) -> None:
        true_traj_subset = self.align_and_subsample(self.dataset.trajectory)
        self.plot(true_traj_subset, "true_traj")
