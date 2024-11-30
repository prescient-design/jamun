from typing import Callable, Union, Optional, Tuple
import logging
import math

import torch
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def initialize_velocity(v_init: Union[str, torch.Tensor], y: torch.Tensor, u: float) -> torch.Tensor:
    """Initialize velocity according to the given method."""
    if isinstance(v_init, str):
        if v_init == "gaussian":
            return math.sqrt(u) * torch.randn_like(y)
        if v_init == "zero":
            return torch.zeros_like(y)
        raise RuntimeError(f"{v_init} not in (gaussian, zero)")

    if isinstance(v_init, torch.Tensor):
        return v_init

    raise RuntimeError(f"{type(v_init)=} must be either `str` or `Tensor`.")


def create_score_fn(score_fn: Callable, inverse_temperature: float, score_fn_clip: Optional[float]) -> Callable:
    """Create a score function that is clipped and scaled by the inverse temperature."""
    
    def score_fn_processed(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score function clipped and scaled by the inverse temperature."""
        orig_score = score_fn(y).to(dtype=y.dtype)
        # Clip the score by norm.
        score = orig_score
        if score_fn_clip is not None:
            norm = torch.linalg.vector_norm(score, dim=-1, keepdim=True)
            clip = torch.min(norm, torch.ones_like(norm) * score_fn_clip)
            score = (score / norm) * clip
        score = score * inverse_temperature
        return score, orig_score
    return score_fn_processed


def aboba(
    y: torch.Tensor,
    score_fn: Callable,
    steps: int,
    v_init: Union[str, torch.Tensor] = "zero",
    save_trajectory=False,
    save_every_n_steps=1,
    burn_in_steps=0,
    verbose=False,
    cpu_offload=False,
    delta: float = 1.0,
    friction: float = 1.0,
    M: float = 1.0,
    inverse_temperature: float = 1.0,
    score_fn_clip: Optional[float] = None,
    **_,
):
    """
    Note
    ----
    ABOBA splitting scheme.

    See section 7.3 of "Molecular Dynamics: With Deterministic and Stochastic Numerical Methods" (page 271)
    """
    i = 0
    y_traj = [] if save_trajectory else None
    if y_traj is not None and i >= burn_in_steps:
        y_traj.append(y.detach().cpu() if cpu_offload else y.detach())

    u = pow(M, -1)  # inverse mass
    zeta2 = math.sqrt(1 - math.exp(-2 * friction))

    v_init = initialize_velocity(v_init=v_init, y=y, u=u)
    v = v_init

    # step zero is initialization
    steps_iter = range(1, steps)
    if verbose:
        steps_iter = tqdm(steps_iter, leave=False, desc="ABOBA")

    # Wrap score function with clipping and scaling by inverse temperature.
    score_fn_processed = create_score_fn(score_fn, inverse_temperature, score_fn_clip)
    score_traj = []
    for i in steps_iter:
        y = y + (delta / 2) * v  # y_{t+1/2}

        psi, orig_score = score_fn_processed(y)
        v = v + u * (delta / 2) * psi  # v_{t+1/2}

        R = torch.randn_like(y)
        vhat = math.exp(-friction) * v + zeta2 * math.sqrt(u) * R

        v = vhat + (delta / 2) * psi  # v_{t+1}
        y = y + (delta / 2) * v  # y_{t+1}

        if y_traj is not None and ((i % save_every_n_steps) == 0) and (i >= burn_in_steps):
            y_traj.append(y.detach().cpu() if cpu_offload else y.detach())
            score_traj.append(orig_score.detach().cpu() if cpu_offload else orig_score.detach())
    
    if y_traj is not None:
        y_traj = torch.stack(y_traj)

    if score_traj is not None:
        score_traj = torch.stack(score_traj)

    return y, v, y_traj, score_traj


def baoab(
    y: torch.Tensor,
    score_fn: Callable,
    steps: int,
    v_init: Union[str, torch.Tensor] = "zero",
    save_trajectory=False,
    save_every_n_steps=1,
    burn_in_steps=0,
    verbose=False,
    cpu_offload=False,
    delta: float = 1.0,
    friction: float = 1.0,
    M: float = 1.0,
    inverse_temperature: float = 1.0,
    score_fn_clip: Optional[float] = None,
    **_,
):
    """
    Note
    ----
    BAOAB splitting scheme.

    See section 7.3 of "Molecular Dynamics: With Deterministic and Stochastic Numerical Methods" (page 271)
    """
    i = 0
    y_traj = [] if save_trajectory else None
    if y_traj is not None and i >= burn_in_steps:
        y_traj.append(y.detach().cpu() if cpu_offload else y.detach())

    u = pow(M, -1)  # inverse mass
    zeta2 = math.sqrt(1 - math.exp(-2 * friction))

    v_init = initialize_velocity(v_init=v_init, y=y, u=u)
    v = v_init

    # step zero is initialization
    steps_iter = range(1, steps)
    if verbose:
        steps_iter = tqdm(steps_iter, leave=False, desc="BAOAB")


    # Wrap score function with clipping and scaling by inverse temperature.
    score_fn_processed = create_score_fn(score_fn, inverse_temperature, score_fn_clip)
    psi, orig_score = score_fn_processed(y)
    score_traj = [orig_score.detach().cpu() if cpu_offload else orig_score.detach()]

    for i in steps_iter:
        v = v + u * (delta / 2) * psi  # v_{t+1/2}
        y = y + (delta / 2) * v  # y_{t+1/2}

        R = torch.randn_like(y)
        vhat = math.exp(-friction) * v + zeta2 * math.sqrt(u) * R
        y = y + (delta / 2) * vhat  # y_{t+1}

        psi, orig_score = score_fn_processed(y)
        v = vhat + (delta / 2) * psi  # v_{t+1}

        if y_traj is not None and ((i % save_every_n_steps) == 0) and (i >= burn_in_steps):
            y_traj.append(y.detach().cpu() if cpu_offload else y.detach())
            score_traj.append(orig_score.detach().cpu() if cpu_offload else orig_score.detach())
    
    if y_traj is not None:
        y_traj = torch.stack(y_traj)

    if score_traj is not None:
        score_traj = torch.stack(score_traj)

    return y, v, y_traj, score_traj
