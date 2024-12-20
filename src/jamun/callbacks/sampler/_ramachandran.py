import logging
from typing import Sequence

from jamun.callbacks.sampler import TrajectoryMetricCallback
from jamun.data import MDtrajDataset
from jamun.metrics import RamachandranPlotMetrics


class RamachandranPlotMetricsCallback(TrajectoryMetricCallback):
    """A wrapper to compute Ramachandran plot metrics."""

    def __init__(
        self,
        datasets: Sequence[MDtrajDataset],
    ):
        super().__init__(
            datasets=datasets,
            metric_fn=lambda dataset: RamachandranPlotMetrics(dataset=dataset),
        )
        py_logger = logging.getLogger("jamun")
        py_logger.info(
            f"Initialized RamachandranPlotMetricsCallback with datasets of labels: {list(self.meters.keys())}."
        )
