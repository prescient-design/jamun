from typing import Sequence, Dict, Any
import logging
import os
import sys
import traceback

import dotenv
dotenv.load_dotenv(".env", verbose=True)

import hydra
import lightning.pytorch as pl
import torch
import torch_geometric
import wandb
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

import jamun
from jamun.data import MDtrajDataset
from jamun.hydra import instantiate_dict_cfg
from jamun.hydra.utils import format_resolver

OmegaConf.register_new_resolver("format", format_resolver)


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
    return checkpoint_dir


def load_checkpoint(checkpoint_dir: str, checkpoint_type: str) -> str:
    """Load the checkpoint based on the checkpoint type, with a lot of assumptions on checkpoint naming."""
    if checkpoint_type.endswith(".ckpt"):
        return os.path.join(checkpoint_dir, checkpoint_type)

    if checkpoint_type == "last":
        return os.path.join(checkpoint_dir, "last.ckpt")

    if checkpoint_type == "best_so_far":
        best_epoch = 0
        checkpoint_path = None
        for file in os.listdir(checkpoint_dir):
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


def get_initial_graphs(
    datasets: Sequence[MDtrajDataset], num_init_samples_per_dataset: int, repeat: int = 1
) -> torch_geometric.data.Batch:
    """Get initial graphs for sampling."""
    init_graphs = []
    for dataset in datasets:
        random_indices = torch.randperm(len(dataset))[:num_init_samples_per_dataset]
        for index in random_indices:
            init_graph = dataset[index]
            for _ in range(repeat):
                init_graphs.append(init_graph)
    return torch_geometric.data.Batch.from_data_list(init_graphs)


def run(cfg):
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    py_logger = logging.getLogger("jamun")

    if rank_zero_only.rank == 0:
        py_logger.info(f"{OmegaConf.to_yaml(log_cfg)}")
        py_logger.info(f"{os.getcwd()=}")
        py_logger.info(f"{torch.__config__.parallel_info()}")
        py_logger.info(f"{os.sched_getaffinity(0)=}")

    if matmul_prec := cfg.get("float32_matmul_precision"):
        torch.set_float32_matmul_precision(matmul_prec)
        if rank_zero_only.rank == 0:
            py_logger.info(f"setting float_32_matmul_precision to {matmul_prec}")

    loggers = instantiate_dict_cfg(cfg.get("logger"), verbose=(rank_zero_only.rank == 0))
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, pl.loggers.WandbLogger):
            wandb_logger = logger

    if wandb_logger and rank_zero_only.rank == 0:
        py_logger.info(f"{wandb_logger.experiment.name=}")

    if rank_zero_only.rank == 0 and wandb_logger:
        wandb_logger.experiment.config.update({"cfg": log_cfg, "version": jamun.__version__, "cwd": os.getcwd()})

    callbacks = instantiate_dict_cfg(cfg.get("callbacks"), verbose=(rank_zero_only.rank == 0))
    sampler = hydra.utils.instantiate(cfg.sampler, callbacks=callbacks, loggers=loggers)

    if seed := cfg.get("seed"):
        # During sampling, we want ranks to generate different chains.
        pl.seed_everything(seed + sampler.fabric.global_rank)

    # Load the checkpoint either given the wandb run path or the checkpoint path.
    wandb_train_run_path = cfg.get("wandb_train_run_path")
    checkpoint_dir = cfg.get("checkpoint_dir")
    if (wandb_train_run_path and checkpoint_dir) or (not wandb_train_run_path and not checkpoint_dir):
        raise ValueError(
            "Exactly one of wandb_train_run_path or checkpoint_dir must be provided."
            f"Obtained: wandb_train_run_path={wandb_train_run_path}, checkpoint_dir={checkpoint_dir}"
        )

    if wandb_train_run_path:
        checkpoint_dir = find_checkpoint_directory(wandb_train_run_path)
        py_logger.info(f"Checkpoint directory found: {checkpoint_dir}")

    # Overwrite the checkpoint path in the config.
    checkpoint_path = load_checkpoint(checkpoint_dir, cfg.checkpoint_type)
    py_logger.info(f"Loading checkpoint_path: {checkpoint_path}")
    cfg.model.checkpoint_path = checkpoint_path

    model = hydra.utils.instantiate(cfg.model)
    batch_sampler = hydra.utils.instantiate(cfg.batch_sampler)
    init_datasets = hydra.utils.instantiate(cfg.init_datasets)

    init_graphs = get_initial_graphs(
        init_datasets,
        num_init_samples_per_dataset=cfg.num_init_samples_per_dataset,
        repeat=cfg.repeat_init_samples,
    )

    sampler.sample(
        model=model,
        batch_sampler=batch_sampler,
        init_graphs=init_graphs,
        num_batches=cfg.num_batches,
        continue_chain=cfg.continue_chain,
    )

    if wandb_logger:
        wandb_logger.finalize(status="finished")


# see https://github.com/facebookresearch/hydra/issues/2664
@hydra.main(version_base=None, config_path="../hydra_config", config_name="sample_MD")
def main(cfg):
    try:
        run(cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
