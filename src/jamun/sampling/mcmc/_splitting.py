import dataclasses
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import Tensor

from jamun.sampling.mcmc.functional import aboba, baoab


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
    v_init: Union[str, Tensor] = "zero"
    inverse_temperature: float = 1.0
    score_fn_clip: Optional[float] = None

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
    v_init: Union[str, Tensor] = "zero"
    inverse_temperature: float = 1.0
    score_fn_clip: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.v_init, str):
            if self.v_init not in {"gaussian", "zero"}:
                raise RuntimeError(f"{self.v_init} not in (gaussian, zero)")

    def __call__(self, y: torch.Tensor, score_fn: Callable, **kwargs):
        kwargs = dataclasses.asdict(self) | kwargs
        return baoab(y, score_fn, **kwargs)
