import logging
from typing import Sequence

from jamun.callbacks.sampler import TrajectoryMetricCallback
from jamun.data import MDtrajDataset
from jamun.metrics import MDSampleVisualizer


class SampleVisualizerCallback(TrajectoryMetricCallback):
    """A callback to visualize static samples from MD trajectories."""

    def __init__(
        self,
        datasets: Sequence[MDtrajDataset],
        *args,
        **kwargs,
    ):
        super().__init__(
            datasets=datasets,
            metric_fn=lambda dataset: MDSampleVisualizer(*args, dataset=dataset, **kwargs),
        )
        py_logger = logging.getLogger("jamun")
        py_logger.info(f"Initialized MDSampleVisualizerCallback with datasets of labels: {list(self.meters.keys())}.")
