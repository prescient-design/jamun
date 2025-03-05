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
from jamun.utils import compute_average_squared_distance_from_data, dist_log, find_checkpoint

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
    dist_log(f"{OmegaConf.to_yaml(log_cfg)}")
    dist_log(f"{os.getcwd()=}")
    dist_log(f"{torch.__config__.parallel_info()}")
    dist_log(f"{os.sched_getaffinity(0)=}")

    # Compute data normalization.
    if cfg.get("compute_average_squared_distance_from_data"):
        average_squared_distance = compute_average_squared_distance_from_config(cfg)
        dist_log(
            f"Overwriting average_squared_distance in config from {cfg.model.average_squared_distance} to {average_squared_distance}."
        )
        cfg.model.average_squared_distance = average_squared_distance

    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    if matmul_prec := cfg.get("float32_matmul_precision"):
        dist_log(f"setting float_32_matmul_precision to {matmul_prec}")
        torch.set_float32_matmul_precision(matmul_prec)

    loggers = instantiate_dict_cfg(cfg.get("logger"), verbose=(rank_zero_only.rank == 0))
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, lightning.pytorch.loggers.WandbLogger):
            wandb_logger = logger

    # Train the model
    trainer = lightning.Trainer(logger=loggers, **cfg.trainer)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

# Needed for submitit error output.
# See https://github.com/facebookresearch/hydra/issues/2664
@hydra.main(version_base=None, config_path="../hydra_config", config_name="train")
def main(cfg):
    try:
        run(cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


# if __name__ == "__main__":
#     cfg = OmegaConf.load("/homefs/home/davidsd5/jamun/jamun/configs/experiment/train_macrocycle.yaml")
#     run(cfg)