import logging
import os
import pathlib
import sys
import traceback

import dotenv
import hydra
import lightning
import torch
import wandb
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

import jamun
from jamun.hydra import instantiate_dict_cfg
from jamun.hydra.utils import format_resolver
from jamun.utils import compute_average_squared_distance_from_data, dist_log

dotenv.load_dotenv(".env", verbose=True)
OmegaConf.register_new_resolver("format", format_resolver)


def compute_average_squared_distance_from_config(cfg: OmegaConf) -> float:
    """Computes the average squared distance for normalization from the data."""
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("compute_normalization")
    train_dataloader = datamodule.train_dataloader()
    cutoff = cfg.model.max_radius
    average_squared_distance = compute_average_squared_distance_from_data(train_dataloader, cutoff)
    average_squared_distance = float(average_squared_distance)
    return average_squared_distance


def run(cfg):
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    py_logger = logging.getLogger("jamun")

    if rank_zero_only.rank == 0:
        py_logger.info(f"{OmegaConf.to_yaml(log_cfg)}")
        py_logger.info(f"{os.getcwd()=}")
        py_logger.info(f"{torch.__config__.parallel_info()}")

    dist_log(f"{os.sched_getaffinity(0)=}")

    # Compute data normalization.
    if cfg.get("compute_average_squared_distance_from_data"):
        average_squared_distance = compute_average_squared_distance_from_config(cfg)
        py_logger.info(
            f"Overwriting average_squared_distance in config from {cfg.model.average_squared_distance} to {average_squared_distance}."
        )
        cfg.model.average_squared_distance = average_squared_distance

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    if matmul_prec := cfg.get("float32_matmul_precision"):
        py_logger.info(f"setting float_32_matmul_precision to {matmul_prec}")
        torch.set_float32_matmul_precision(matmul_prec)

    loggers = instantiate_dict_cfg(cfg.get("logger"), verbose=(rank_zero_only.rank == 0))
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, lightning.pytorch.loggers.WandbLogger):
            wandb_logger = logger

    if wandb_logger:
        py_logger.info(f"{wandb_logger.experiment.name=}")

    callbacks = instantiate_dict_cfg(cfg.get("callbacks"), verbose=(rank_zero_only.rank == 0))

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    # TODO support wandb notes/description
    if rank_zero_only.rank == 0 and wandb_logger:
        wandb_logger.experiment.config.update({"cfg": log_cfg, "version": jamun.__version__, "cwd": os.getcwd()})

    trainer.fit(model, datamodule=datamodule)

    if wandb_logger and isinstance(trainer.profiler, lightning.pytorch.profilers.PyTorchProfiler):
        profile_art = wandb.Artifact("trace", type="profile")
        for trace in pathlib.Path(trainer.profiler.dirpath).glob("*.pt.trace.json"):
            profile_art.add_file(trace)
        profile_art.save()

    if rank_zero_only.rank == 0:
        py_logger.info(f"{torch.cuda.max_memory_allocated()=:0.2e}")

    if wandb_logger:
        wandb.finish()


# Needed for submitit error output.
# See https://github.com/facebookresearch/hydra/issues/2664
@hydra.main(version_base=None, config_path="../hydra_config", config_name="train")
def main(cfg):
    try:
        run(cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise
