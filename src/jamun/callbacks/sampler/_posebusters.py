import logging
from collections.abc import Sequence

from jamun.callbacks.sampler._utils import TrajectoryMetricCallback
from jamun.data import MDtrajDataset
from jamun.metrics import PoseBustersMetrics


class PoseBustersCallback(TrajectoryMetricCallback):
    """A wrapper to compute PoseBusters metrics."""

    def __init__(
        self,
        datasets: Sequence[MDtrajDataset],
        *args,
        **kwargs,
    ):
        super().__init__(
            datasets=datasets,
            metric_fn=lambda dataset: PoseBustersMetrics(*args, dataset=dataset, **kwargs),
        )
        py_logger = logging.getLogger("jamun")
        py_logger.info(f"Initialized PoseBustersCallback with datasets of labels: {list(self.meters.keys())}.")
