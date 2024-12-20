import logging
from typing import Sequence

from jamun.callbacks.sampler import TrajectoryMetricCallback
from jamun.data import MDtrajDataset
from jamun.metrics import SaveTrajectory


class SaveTrajectoryCallback(TrajectoryMetricCallback):
    """A wrapper to save sampled trajectories to disk."""

    def __init__(
        self,
        datasets: Sequence[MDtrajDataset],
        *args,
        **kwargs,
    ):
        super().__init__(
            datasets=datasets,
            metric_fn=lambda dataset: SaveTrajectory(*args, dataset=dataset, **kwargs),
        )
        py_logger = logging.getLogger("jamun")
        py_logger.info(f"Initialized SaveTrajectoryCallback with datasets of labels: {list(self.meters.keys())}.")
        py_logger.info(f"Saving true and predicted samples to {kwargs['output_dir']}.")
