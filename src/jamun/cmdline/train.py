import os
import pathlib
import sys
import traceback

import dotenv
import e3nn
import hydra
import lightning
import torch
import wandb
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

e3nn.set_optimization_defaults(jit_script_fx=False)

import jamun  # noqa: E402
from jamun.hydra import instantiate_dict_cfg  # noqa: E402
from jamun.hydra.utils import format_resolver  # noqa: E402
from jamun.utils import compute_average_squared_distance_from_datasets, dist_log, find_checkpoint  # noqa: E402
from jamun.utils._normalizations import normalization_factors  # noqa: E402

dotenv.load_dotenv(".env", verbose=True)
OmegaConf.register_new_resolver("format", format_resolver)


def compute_average_squared_distance_from_config(cfg: OmegaConf) -> float:
    """Computes the average squared distance for normalization from the data."""
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("compute_normalization")
    train_datasets = datamodule.datasets["train"]
    cutoff = cfg.model.max_radius
    average_squared_distance = compute_average_squared_distance_from_datasets(train_datasets, cutoff)
    return average_squared_distance


def run(cfg):
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    dist_log(f"{OmegaConf.to_yaml(log_cfg)}")
    dist_log(f"{os.getcwd()=}")
    dist_log(f"{torch.__config__.parallel_info()}")
    dist_log(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    dist_log(f"{os.sched_getaffinity(0)=}")
    
    # Set the start method to spawn to avoid issues with the default fork method.
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Set random seed for reproducible training
    if seed := cfg.get("seed"):
        lightning.seed_everything(seed)
        dist_log(f"Set random seed to {seed} for reproducible training")

    # Compute data normalization.
    if cfg.get("compute_average_squared_distance_from_data"):
        average_squared_distance = compute_average_squared_distance_from_config(cfg)
        dist_log(
            f"Overwriting average_squared_distance in config from {cfg.model.average_squared_distance} to {average_squared_distance}."
        )
        cfg.model.average_squared_distance = average_squared_distance

    # Compute normalization factors for conditioner c_in parameter
    if cfg.model.get("conditioner") and cfg.model.conditioner.get("_target_") == "jamun.model.conditioners.DenoisedConditioner":
        if hasattr(cfg.model.sigma_distribution, "sigma"):
            sigma = cfg.model.sigma_distribution.sigma
            average_squared_distance = cfg.model.average_squared_distance
            c_in, c_skip, c_out, c_noise = normalization_factors(sigma, average_squared_distance)
            c_in_float = float(c_in)
            
            dist_log(f"Computing normalization factors for DenoisedConditioner with sigma={sigma}")
            dist_log(f"  average_squared_distance: {average_squared_distance}")
            dist_log(f"  c_in: {c_in_float}")
            dist_log(f"  c_skip: {c_skip}")
            dist_log(f"  c_out: {c_out}")
            dist_log(f"  c_noise: {c_noise}")
            
            cfg.model.conditioner.c_in = c_in_float
            dist_log(f"Set cfg.model.conditioner.c_in to {c_in_float}")

    # # do this for the sweep
    # if cfg.model.N_measurements_hidden is not None:
    #     dist_log(f"Number of hidden measurements: {cfg.model.N_measurements_hidden}")
    #     dist_log(f"Overwriting N_measurements...")
    #     cfg.model.N_measurements = 100 // cfg.model.N_measurements_hidden
    #     dist_log(f"New num of measurements: {cfg.model.N_measurements=}")
    # breakpoint()
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    if matmul_prec := cfg.get("float32_matmul_precision"):
        dist_log(f"Setting float_32_matmul_precision to {matmul_prec}")
        torch.set_float32_matmul_precision(matmul_prec)

    # breakpoint()
    # # If running under Slurm, ensure the number of devices matches the allocation.
    # if "SLURM_GPUS_PER_TASK" in os.environ and torch.cuda.is_available():
    #     dist_log(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    #     try:
    #         num_gpus = int(os.environ["SLURM_GPUS_PER_TASK"])
    #         dist_log(f"Slurm-allocated GPUs per task: {num_gpus}")
    #         # Explicitly create a list of device IDs [0, 1, ..., n-1] for Lightning.
    #         device_ids = list(range(num_gpus))
    #         # This will override any value from the config file, ensuring it matches the Slurm allocation.
    #         cfg.trainer.devices = device_ids
    #         dist_log(f"Explicitly set cfg.trainer.devices to {cfg.trainer.devices}")
    #     except (ValueError, KeyError):
    #         dist_log("Could not parse or find SLURM_GPUS_PER_TASK.")

    loggers = instantiate_dict_cfg(cfg.get("logger"), verbose=(rank_zero_only.rank == 0))
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, lightning.pytorch.loggers.WandbLogger):
            wandb_logger = logger

    if wandb_logger:
        dist_log(f"{wandb_logger.experiment.name=}")

    callbacks = instantiate_dict_cfg(cfg.get("callbacks"), verbose=(rank_zero_only.rank == 0))

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)
    # breakpoint()
    # TODO support wandb notes/description
    if rank_zero_only.rank == 0 and wandb_logger:
        wandb_logger.experiment.config.update({"cfg": log_cfg, "version": jamun.__version__, "cwd": os.getcwd()})

    # Load training checkpoint, if provided.
    if resume_checkpoint_cfg := cfg.get("resume_from_checkpoint"):
        # Load the checkpoint either given the wandb run path or the checkpoint path.
        checkpoint_path = find_checkpoint(
            wandb_train_run_path=resume_checkpoint_cfg.get("wandb_train_run_path"),
            checkpoint_dir=resume_checkpoint_cfg.get("checkpoint_dir"),
            checkpoint_type=resume_checkpoint_cfg["checkpoint_type"],
        )
    else:
        checkpoint_path = None
    print(f'Saving checkpoints @ {checkpoint_path}')

    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    # breakpoint()
    if wandb_logger and isinstance(trainer.profiler, lightning.pytorch.profilers.PyTorchProfiler):
        profile_art = wandb.Artifact("trace", type="profile")
        for trace in pathlib.Path(trainer.profiler.dirpath).glob("*.pt.trace.json"):
            profile_art.add_file(trace)
        profile_art.save()

    dist_log(f"{torch.cuda.max_memory_allocated()=:0.2e}")

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
