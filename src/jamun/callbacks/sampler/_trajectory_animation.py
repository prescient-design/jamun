from typing import Sequence
import logging


from jamun.metrics import TrajectoryVisualizer
from jamun.data import MDtrajDataset
from jamun.callbacks.sampler import TrajectoryMetricCallback


class TrajectoryVisualizerCallback(TrajectoryMetricCallback):
    """A callback to animate MD trajectories."""

    def __init__(
        self,
        datasets: Sequence[MDtrajDataset],
        *args,
        **kwargs,
    ):
        super().__init__(
            datasets=datasets,
            metric_fn=lambda dataset: TrajectoryVisualizer(*args, dataset=dataset, **kwargs),
        )
        py_logger = logging.getLogger("jamun")
        py_logger.info(f"Initialized TrajectoryVisualizerCallback with datasets of labels: {list(self.meters.keys())}.")
