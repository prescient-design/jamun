from typing import Dict

import numpy as np
import mdtraj as md
import wandb
import matplotlib.pyplot as plt

from jamun import utils_md
from jamun.data import MDtrajDataset
from jamun.metrics import TrajectoryMetric


class ScoreDistributionMetrics(TrajectoryMetric):
    """A metric to plot the distribution of the score."""

    def __init__(self, dataset: MDtrajDataset):
        super().__init__(dataset=dataset, sample_key="score_traj")

    def compute(self) -> Dict[str, float]:
        pred_scores = self.sample_tensors(new=True)
        pred_scores = pred_scores.cpu().numpy()

        for trajectory_index, pred_score_trajectory in enumerate(pred_scores, start=self.num_chains_seen):
            # pred_score_trajectory has shape (num_atoms, num_frames, 3).
            # pred_scores_norm has shape (num_atoms, num_frames).
            pred_scores_norm = np.linalg.norm(pred_score_trajectory, axis=-1)

            # Mean, min, max, and IQR of the scores over the atoms.
            mean_pred_scores_norm = np.mean(pred_scores_norm, axis=0)
            min_pred_scores_norm = np.min(pred_scores_norm, axis=0)
            max_pred_scores_norm = np.max(pred_scores_norm, axis=0)
            first_quartile_pred_scores_norm = np.percentile(pred_scores_norm, 25, axis=0)
            third_quartile_pred_scores_norm = np.percentile(pred_scores_norm, 75, axis=0)

            # Plot the score distributions.
            plt.plot(max_pred_scores_norm, label="Max")
            plt.plot(mean_pred_scores_norm, label="Mean")
            plt.plot(min_pred_scores_norm, label="Min")
            plt.fill_between(
                range(len(mean_pred_scores_norm)),
                first_quartile_pred_scores_norm,
                third_quartile_pred_scores_norm,
                alpha=0.3,
                label="IQR",
                color="gray",
            )
            plt.xlabel("Frame")
            plt.ylabel("Score Norms")
            plt.title(f"Score norms across frames for trajectory {trajectory_index}")
            plt.legend()
            wandb.log(
                {
                    f"{self.dataset.label()}/score_distribution/pred_traj_{trajectory_index}": wandb.Image(plt),
                }
            )
            plt.clf()

        return {}
