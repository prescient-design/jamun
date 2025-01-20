import logging
from typing import Dict, Optional

import mdtraj as md
import pandas as pd
import posebusters
import wandb

from jamun import utils
from jamun.metrics._utils import TrajectoryMetric


def run_posebusters_on_trajectory(trajectory: md.Trajectory) -> Optional[pd.DataFrame]:
    """Run PoseBusters on each frame of a trajectory."""
    mols = utils.to_rdkit_mols(trajectory)
    if len(mols) == 0:
        return None

    buster = posebusters.PoseBusters(config="mol")
    results = buster.bust(mols, None, None, full_report=False)

    return results


class PoseBustersMetrics(TrajectoryMetric):
    """Computes chemical validity metrics using PoseBusters."""

    def __init__(
        self,
        num_molecules_per_trajectory: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_molecules_per_trajectory = num_molecules_per_trajectory

    def on_sample_start(self):
        true_trajectory = self.dataset.trajectory
        subsampling_factor = max(len(true_trajectory) // self.num_molecules_per_trajectory, 1)

        df = run_posebusters_on_trajectory(true_trajectory[::subsampling_factor])

        metrics = {}
        if df is None:
            py_logger = logging.getLogger("posebusters")
            py_logger.info(f"{self.dataset.label()}/PoseBusters found no molecules in the trajectory.")
            return metrics

        mean_fail_rates = 1 - df.mean()
        utils.wandb_dist_log(
            {
                f"{self.dataset.label()}/posebusters/mean_fail_rates/true_traj": wandb.Table(
                    data=[mean_fail_rates.values], columns=list(mean_fail_rates.index)
                )
            }
        )
        for key, value in mean_fail_rates.items():
            metrics[f"{self.dataset.label()}/posebusters/{key}/true_traj"] = value

        return metrics

    def compute(self) -> Dict[str, float]:
        metrics = {}
        pred_trajectories = self.sample_trajectories(new=True)
        for trajectory_index, pred_trajectory in enumerate(pred_trajectories, start=self.num_chains_seen):
            # Run PoseBusters on a subset of the trajectory.
            subsampling_factor = max(len(pred_trajectory) // self.num_molecules_per_trajectory, 1)
            df = run_posebusters_on_trajectory(pred_trajectory[::subsampling_factor])
            if df is None:
                py_logger = logging.getLogger("posebusters")
                py_logger.info("PoseBusters found no molecules in the trajectory.")
            else:
                mean_fail_rates = 1 - df.mean()
                utils.wandb_dist_log(
                    {
                        f"{self.dataset.label()}/posebusters/mean_fail_rates/pred_traj_{trajectory_index}": wandb.Table(
                            data=[mean_fail_rates.values], columns=list(mean_fail_rates.index)
                        )
                    }
                )
                for key, value in mean_fail_rates.items():
                    metrics[f"{self.dataset.label()}/posebusters/{key}/pred_traj_{trajectory_index}"] = value

        return metrics
