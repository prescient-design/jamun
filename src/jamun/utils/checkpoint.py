import logging
import os
from typing import Any, Dict

import wandb


def get_wandb_run_config(wandb_run_path: str) -> Dict[str, Any]:
    """Get the wandb run config."""
    run = wandb.Api().run(wandb_run_path)
    py_logger = logging.getLogger("jamun")
    py_logger.info(f"Loading checkpoint corresponding to wandb run {run.name} at {run.url}")
    return run.config["cfg"]


def find_checkpoint_directory(wandb_train_run_path: str) -> str:
    """Find the checkpoint directory based on the wandb run path."""
    config = get_wandb_run_config(wandb_train_run_path)
    checkpoint_dir = config["callbacks"]["model_checkpoint"]["dirpath"]
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist.")
    return checkpoint_dir


def find_checkpoint_in_directory(checkpoint_dir: str, checkpoint_type: str) -> str:
    """Find the checkpoint based on the checkpoint type, with a lot of assumptions on checkpoint naming."""
    if checkpoint_type.endswith(".ckpt"):
        return os.path.join(checkpoint_dir, checkpoint_type)

    if checkpoint_type == "last":
        return os.path.join(checkpoint_dir, "last.ckpt")

    if checkpoint_type == "best_so_far":
        # Assumes that the checkpoints are saved when the validation loss is lower.
        best_epoch = 0
        checkpoint_path = None
        for file in sorted(os.listdir(checkpoint_dir)):
            if not file.endswith(".ckpt") or not file.startswith("epoch="):
                continue

            epoch = int(file.split("epoch=")[1].split("-")[0])
            if epoch >= best_epoch:
                best_epoch = epoch
                checkpoint_path = os.path.join(checkpoint_dir, file)

        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found in the directory {checkpoint_dir}")
        return checkpoint_path

    raise ValueError(f"Invalid checkpoint type: {checkpoint_type}. Must be one of ['last', 'best', '*.ckpt'].")


def find_checkpoint(wandb_train_run_path: str = None, checkpoint_dir: str = None, checkpoint_type: str = None) -> str:
    """Find the checkpoint based on the wandb run path or the checkpoint directory."""
    py_logger = logging.getLogger("jamun")
    if (wandb_train_run_path and checkpoint_dir) or (not wandb_train_run_path and not checkpoint_dir):
        raise ValueError(
            "Exactly one of wandb_train_run_path or checkpoint_dir must be provided."
            f"Obtained: wandb_train_run_path={wandb_train_run_path}, checkpoint_dir={checkpoint_dir}"
        )

    if wandb_train_run_path:
        checkpoint_dir = find_checkpoint_directory(wandb_train_run_path)
        py_logger.info(f"Checkpoint directory found: {checkpoint_dir}")

    checkpoint_path = find_checkpoint_in_directory(checkpoint_dir, checkpoint_type)
    py_logger.info(f"Loading checkpoint_path: {checkpoint_path}")
    return checkpoint_path
