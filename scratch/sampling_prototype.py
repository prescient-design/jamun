
# %% Imports and Basic Setup
import functools
import logging
import os
import sys
from typing import Union, Sequence

import dotenv
import tqdm
import torch
import e3nn
import e3tools.nn
import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf
import lightning.pytorch as pl
import torch_geometric.data

import jamun
import jamun.data
import jamun.distributions
import jamun.model
import jamun.model.arch
import jamun.sampling
from jamun.utils import compute_average_squared_distance_from_datasets, find_checkpoint
from jamun.hydra import instantiate_dict_cfg
from jamun.data import MDtrajDataModule, MDtrajDataset

# --- Basic Setup ---
logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("jamun_sampling_script")

torch.cuda.is_available()
torch.set_float32_matmul_precision("high")
e3nn.set_optimization_defaults(jit_script_fx=False)

# %% Environment and Paths
dotenv.load_dotenv("../.env", verbose=True) # Adjust path if script is not in scratch/
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")
JAMUN_ROOT_PATH = os.getenv("JAMUN_ROOT_PATH")

project_root = "/homefs/home/sules/jamun" # Adjust if necessary
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    py_logger.info(f"Added '{project_root}' to sys.path for module discovery.")
else:
    py_logger.info(f"'{project_root}' is already in sys.path.")

# %% Load Configuration from Specific File
py_logger.info("Loading configuration from specific training run...")
config_file_path = "../outputs/train/dev/runs/2025-06-11_20-16-04/final_resolved_script_config.yaml"

# Load the configuration directly from the YAML file
cfg = OmegaConf.load(config_file_path)
py_logger.info("Loaded configuration from training run:")
py_logger.info(f"Config file: {config_file_path}")

# Modify config for sampling purposes
# Override model target to use our local architectures
cfg.model._target_ = "scratch.denoiser_test.Denoiser"
cfg.model.arch._target_ = "scratch.e3conv_test.E3Conv"

# Add sampling-specific configuration
if not hasattr(cfg, 'sampling'):
    cfg.sampling = OmegaConf.create({})

# Set sampling parameters (adapt from sample.yaml structure)
cfg.sampling.repeat_init_samples = 1
cfg.sampling.num_batches = 10
cfg.sampling.continue_chain = True
cfg.sampling.num_init_samples_per_dataset = 5
cfg.sampling.seed = 42

# Add sampler configuration
cfg.sampling.sampler = OmegaConf.create({
    "_target_": "jamun.sampling.Sampler",
    "precision": "32-true"
})

# Add batch sampler configuration  
cfg.sampling.batch_sampler = OmegaConf.create({
    "_target_": "jamun.sampling.SingleMeasurementSampler",
    "num_steps": 100,
    "sigma_min": 0.001,
    "sigma_max": 0.1
})

py_logger.info("Modified configuration for sampling:")
py_logger.info(OmegaConf.to_yaml(cfg))

# %% Helper Functions
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

# %% Load Model 
py_logger.info("Loading model from checkpoint...")

cfg.model._target_ = 'scratch.denoiser_test.Denoiser'
model = hydra.utils.instantiate(cfg.model)

# Device Setup
if torch.cuda.is_available():
    device = torch.device("cuda")
    py_logger.info("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    py_logger.info("CUDA not available. Using CPU.")

model = model.to(device)
py_logger.info(f"Model moved to device: {device}")

# %% Load from ckpoint
sys.path.append("../")
checkpoint_dir = "../outputs/train/dev/runs/2025-06-11_20-16-04/wandb/latest-run/checkpoints"
try:
    checkpoint_path = find_checkpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_type="last"  # or "best"
    )
    py_logger.info(f"Found checkpoint: {checkpoint_path}")
except Exception as e:
    py_logger.error(f"Could not find checkpoint in {checkpoint_dir}: {e}")
    # Try to find checkpoint in the checkpoints subdirectory
    checkpoint_subdir = os.path.join(checkpoint_dir, "run-*/checkpoints")
    import glob
    checkpoint_files = glob.glob(os.path.join(checkpoint_subdir, "*.ckpt"))
    if checkpoint_files:
        checkpoint_path = checkpoint_files[-1]  # Use the last one
        py_logger.info(f"Using checkpoint: {checkpoint_path}")
    else:
        py_logger.error("No checkpoint files found!")
        # sys.exit(1)

# %% Load from checkpoint
if checkpoint_path:
    
    checkpoint = torch.load(checkpoint_path, map_location=model.device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    py_logger.info("Successfully loaded model from checkpoint.")
    py_logger.info(f"Model architecture type: {type(model.g)}")
else:
    py_logger.error("No checkpoint files found!")
    # sys.exit(1)

# %% checkpoint path update
cfg.model.checkpoint_path = checkpoint_path
py_logger.info(f"Updated checkpoint path: {cfg.model.checkpoint_path}")

# %% Setup Initial Datasets for Sampling
py_logger.info("Setting up initial datasets for sampling...")

# Use the same data configuration as the training
init_datasets = jamun.data.parse_datasets_from_directory(
    root=f"{JAMUN_DATA_PATH}/timewarp/2AA-1-large/train/",
    traj_pattern="^(.*)-traj-arrays.npz",
    pdb_file="AA-traj-state0.pdb",
    filter_codes=['AA'],
    as_iterable=False,
    subsample=20,  # Smaller subset for sampling
    max_datasets=1,
)

py_logger.info(f"Loaded {len(init_datasets)} samples for initial configurations")

# %% Generate Initial Graphs
py_logger.info("Generating initial graphs for sampling...")
init_graphs = get_initial_graphs(
    init_datasets,
    num_init_samples_per_dataset=cfg.sampling.num_init_samples_per_dataset,
    repeat=cfg.sampling.repeat_init_samples,
)
py_logger.info(f"Generated {len(init_graphs)} initial graphs")

# %% Setup Sampling Components
py_logger.info("Setting up sampling components...")

# Set random seed
if cfg.sampling.seed:
    pl.seed_everything(cfg.sampling.seed)
    py_logger.info(f"Set random seed to {cfg.sampling.seed}")

# Setup loggers for sampling
loggers_list = []
if cfg.get("logger"):
    # Create a sampling-specific logger config
    sampling_logger_cfg = OmegaConf.create({
        "wandb": {
            "_target_": "lightning.pytorch.loggers.WandbLogger",
            "project": "jamun-sampling",
            "entity": None,
            "offline": False,
            "group": "sampling_test",
            "notes": "Sampling from trained model",
            "save_dir": "./outputs/sample/"
        }
    })
    loggers_list = instantiate_dict_cfg(sampling_logger_cfg)

# Instantiate sampler
try:
    sampler = hydra.utils.instantiate(cfg.sampling.sampler, callbacks=[], loggers=loggers_list)
    py_logger.info("Successfully instantiated sampler")
except Exception as e:
    py_logger.error(f"Error instantiating sampler: {e}")
    # Fallback to direct instantiation
    sampler = jamun.sampling.Sampler(precision="32-true", callbacks=[], loggers=loggers_list)
    py_logger.info("Using fallback sampler instantiation")

# Instantiate batch sampler
try:
    batch_sampler = hydra.utils.instantiate(cfg.sampling.batch_sampler)
    py_logger.info("Successfully instantiated batch sampler")
except Exception as e:
    py_logger.error(f"Error instantiating batch sampler: {e}")
    # Fallback to direct instantiation
    batch_sampler = jamun.sampling.Sampler(
        num_steps=100,
        sigma_min=0.001, 
        sigma_max=0.1
    )
    py_logger.info("Using fallback batch sampler instantiation")

# %% Run Sampling
py_logger.info("Starting sampling...")
try:
    sampler.sample(
        model=model,
        batch_sampler=batch_sampler,
        init_graphs=init_graphs,
        num_batches=cfg.sampling.num_batches,
        continue_chain=cfg.sampling.continue_chain,
    )
    py_logger.info("Sampling completed successfully!")

except Exception as e:
    py_logger.error(f"Sampling FAILED: {e}")
    import traceback
    traceback.print_exc()

# %% Cleanup and Finish
py_logger.info("Sampling script finished.")

# Finalize wandb if used
if loggers_list:
    for logger in loggers_list:
        if isinstance(logger, pl.loggers.WandbLogger):
            logger.finalize(status="finished")
            py_logger.info("Finalized WandB logger")

py_logger.info("All done!") 