from typing import Any, Dict
import logging

import wandb
from lightning.pytorch.utilities import rank_zero_only
import torch


def wandb_dist_log(data: Dict[str, Any]) -> None:
    """Log data to wandb only on rank 0."""
    if rank_zero_only.rank == 0:
        wandb.log(data)
    

def dist_log(msg: str, logger: logging.Logger = None) -> None:
    """Helper for distributed logging."""

    if logger is None:
        logger = logging.getLogger("jamun")

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        for r in range(world_size):
            if r == rank:
                logger.info(f"[rank {rank}/{world_size}]: {msg}")
            torch.distributed.barrier()
    else:
        logger.info(f"{msg}")
