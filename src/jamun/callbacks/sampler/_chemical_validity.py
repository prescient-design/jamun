import logging
from typing import Sequence

from jamun.callbacks.sampler._utils import TrajectoryMetricCallback
from jamun.data import MDtrajDataset
from jamun.metrics import ChemicalValidityMetrics


class ChemicalValidityMetricsCallback(TrajectoryMetricCallback):
    """A wrapper to compute chemical validity metrics."""

    def __init__(
        self,
        datasets: Sequence[MDtrajDataset],
        *args,
        **kwargs,
    ):
        super().__init__(
            datasets=datasets,
            metric_fn=lambda dataset: ChemicalValidityMetrics(*args, dataset=dataset, **kwargs),
        )
        py_logger = logging.getLogger("jamun")
        py_logger.info(
            f"Initialized ChemicalValidityMetricsCallback with datasets of labels: {list(self.meters.keys())}."
        )
