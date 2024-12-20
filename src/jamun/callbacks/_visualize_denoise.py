from typing import List, Sequence

import lightning.pytorch as pl
import torch_geometric.data
from lightning.pytorch.utilities import rank_zero_only

from jamun.data import MDtrajDataset
from jamun.metrics import MDVisualizeDenoiseMetrics


class MDVisualizeDenoise(pl.Callback):
    """Callback to denoise and visualize MDTraj datasets during training."""

    def __init__(
        self,
        datasets: Sequence[MDtrajDataset],
        sigma_list: List[float],
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.sigma_list = sigma_list
        self.visualizers = {
            dataset.label(): MDVisualizeDenoiseMetrics(
                dataset=dataset,
                sigma_list=sigma_list,
            )
            for dataset in sorted(datasets, key=lambda dataset: dataset.label())
        }
        self.every_n_epochs = every_n_epochs

    def setup(self, trainer, pl_module, stage):
        self.to(pl_module.device)

    def to(self, device):
        for visualizer in self.visualizers.values():
            visualizer.to(device)

    def reset(self):
        for visualizer in self.visualizers.values():
            visualizer.reset()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if (pl_module.current_epoch % self.every_n_epochs) != 0:
            return

        x = batch
        for sigma in self.sigma_list:
            xhat, y = pl_module.noise_and_denoise(
                x, sigma, align_noisy_input=pl_module.align_noisy_input_during_evaluation
            )
            for xhat_graph, y_graph, x_graph in zip(
                torch_geometric.data.Batch.to_data_list(xhat),
                torch_geometric.data.Batch.to_data_list(y),
                torch_geometric.data.Batch.to_data_list(x),
                strict=True,
            ):
                assert x_graph.dataset_label == y_graph.dataset_label == xhat_graph.dataset_label
                visualizer = self.visualizers[x_graph.dataset_label]
                visualizer.update(
                    xhat=xhat_graph,
                    y=y_graph,
                    x=x_graph,
                    sigma=sigma,
                )

    def on_validation_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch % self.every_n_epochs) != 0:
            return

        for visualizer in self.visualizers.values():
            if visualizer.has_samples:
                trajectories, scaled_rmsd_per_sigma = visualizer.compute()

                if rank_zero_only.rank == 0 and trajectories is not None:
                    visualizer.log(trajectories, scaled_rmsd_per_sigma)

            visualizer.reset()
