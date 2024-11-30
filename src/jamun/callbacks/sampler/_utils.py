from typing import Sequence, Callable

import torch_geometric

from jamun.data import MDtrajDataset
from jamun.metrics import TrajectoryMetric


def get_unique_datasets(datasets: Sequence[MDtrajDataset]) -> Sequence[MDtrajDataset]:
    """Get unique datasets by label."""
    labels = set()
    unique_datasets = []
    for dataset in list(datasets):
        label = dataset.label()
        if label not in labels:
            labels.add(label)
            unique_datasets.append(dataset)
    return unique_datasets


class TrajectoryMetricCallback:
    """A wrapper to compute metrics over trajectories."""

    def __init__(
        self,
        datasets: Sequence[MDtrajDataset],
        metric_fn: Callable[[], TrajectoryMetric],
    ):
        datasets = sorted(get_unique_datasets(datasets), key=lambda dataset: dataset.label())
        self.meters = {
            dataset.label(): metric_fn(
                dataset=dataset,
            )
            for dataset in datasets
        }

    def on_sample_start(self, sampler):
        for meter in self.meters.values():
            meter.to(sampler.fabric.device)
            meter.on_sample_start()

    def on_after_sample_batch(
        self,
        sample: Sequence[torch_geometric.data.Batch],
        sampler,
    ):
        for sample_graph in sample:
            self.meters[sample_graph.dataset_label].update(sample_graph)
        
        for meter in self.meters.values():
            sampler.fabric.log_dict(meter.compute())
            meter.on_after_sample_batch()

    def on_sample_end(self, sampler):
        for meter in self.meters.values():
            meter.on_sample_end()