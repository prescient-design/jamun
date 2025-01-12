import dataclasses
import math
from typing import Optional

from lightning.pytorch.utilities import rank_zero_only

from jamun.utils import dist_log


class MeasurementDependentParametersCallback:
    def __init__(self, parameters_by_measurement: Optional[dict] = None, verbose: bool = False):
        self.parameters_by_measurement = parameters_by_measurement if parameters_by_measurement else {}
        self.verbose = verbose and (rank_zero_only.rank == 0)
        self.previous_params = None

    def on_before_sample(self, mcmc, t: int):
        params = self.parameters_by_measurement.get(t, None)
        if params:
            self.previous_params = dataclasses.asdict(mcmc)
            mcmc = mcmc.replace(**(self.previous_params | params))
            if self.verbose:
                dist_log(f"{t=}, {mcmc=}")

        return mcmc

    def on_after_sample(self, mcmc, t: int):
        if self.previous_params is not None:
            mcmc = mcmc.replace(**self.previous_params)
            self.previous_params = None

        return mcmc


class DeltaSqrtDecayCallback:
    """Decay delta by 1/sqrt(t) at each step."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.delta_orig = None

    def on_before_sample(self, mcmc, t: int):
        self.delta_orig = mcmc.delta
        mcmc = mcmc.replace(delta=self.delta_orig / math.sqrt(t))
        if self.verbose and rank_zero_only.rank == 0:
            dist_log(f"{t=}, {mcmc=}")
        return mcmc

    def on_after_sample(self, mcmc, t: int):
        mcmc = mcmc.replace(delta=self.delta_orig)
        return mcmc


class InterpolateParametersCallback:
    """Interpolate sampling parameters based on the iteration number."""

    def __init__(self, params: dict[str, tuple[float, float]], verbose: bool = False):
        self.params = params
        self.verbose = verbose

    def on_before_sample(self, mcmc, t: int):
        f = 1 - math.sqrt(1.0 / t)
        params_t = {k: type(v[0])((1 - f) * v[0] + f * v[1]) for k, v in self.params.items()}
        mcmc = mcmc.replace(**params_t)
        if self.verbose and rank_zero_only.rank == 0:
            dist_log(f"{t=}, {mcmc=}")

        return mcmc

    def on_after_sample(self, mcmc, t: int):
        return mcmc
