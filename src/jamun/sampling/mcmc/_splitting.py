import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from jamun.sampling.mcmc.functional import aboba, baoab, aboba_memory, baoab_memory


@dataclass
class ABOBA:
    delta: float = 1.0
    friction: float = 1.0
    M: float = 1.0
    steps: int = 128
    save_trajectory: bool = False
    save_every_n_steps: int = 1
    burn_in_steps: int = 0
    verbose: bool = False
    cpu_offload: bool = False
    v_init: str | Tensor = "zero"
    inverse_temperature: float = 1.0
    score_fn_clip: float | None = None

    def __post_init__(self):
        if isinstance(self.v_init, str):
            if self.v_init not in {"gaussian", "zero"}:
                raise RuntimeError(f"{self.v_init} not in (gaussian, zero)")

    def __call__(self, y: torch.Tensor, score_fn: Callable, **kwargs):
        kwargs = dataclasses.asdict(self) | kwargs
        return aboba(y, score_fn, **kwargs)


@dataclass
class BAOAB:
    delta: float = 1.0
    friction: float = 1.0
    M: float = 1.0
    steps: int = 128
    save_trajectory: bool = False
    save_every_n_steps: int = 1
    burn_in_steps: int = 0
    verbose: bool = False
    cpu_offload: bool = False
    v_init: str | Tensor = "zero"
    inverse_temperature: float = 1.0
    score_fn_clip: float | None = None

    def __post_init__(self):
        if isinstance(self.v_init, str):
            if self.v_init not in {"gaussian", "zero"}:
                raise RuntimeError(f"{self.v_init} not in (gaussian, zero)")

    def __call__(self, y: torch.Tensor, score_fn: Callable, **kwargs):
        kwargs = dataclasses.asdict(self) | kwargs
        return baoab(y, score_fn, **kwargs)


@dataclass
class ABOBA_memory(ABOBA):
    def __call__(self, y: torch.Tensor, y_hist: list, score_fn: Callable, **kwargs):
        kwargs = dataclasses.asdict(self) | kwargs
        return aboba_memory(y=y, y_hist=y_hist, score_fn=score_fn, **kwargs)


@dataclass
class BAOAB_memory(BAOAB):
    def __call__(self, y: torch.Tensor, y_hist: list, score_fn: Callable, **kwargs):
        kwargs = dataclasses.asdict(self) | kwargs
        return baoab_memory(y=y, y_hist=y_hist, score_fn=score_fn, **kwargs)
