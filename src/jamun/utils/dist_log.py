import logging

import torch


def dist_log(msg: str, logger=None) -> None:
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
