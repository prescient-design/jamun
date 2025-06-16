# %% Imports and Basic Setup
import functools
import logging
import os
import sys
from typing import Union

import dotenv
import tqdm # Often used in notebooks, can be optional in scripts
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
# Assuming these are in jamun.utils and jamun.hydra respectively
from jamun.utils import compute_average_squared_distance_from_datasets, find_checkpoint
from jamun.hydra import instantiate_dict_cfg

# --- Basic Setup ---
logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("jamun_script")

torch.cuda.is_available() # Good to check, but PL trainer will also handle device
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

# %% Load Hydra Configuration
py_logger.info("Loading Hydra configuration...")
# Adjust config_path relative to the script's location if it's not in scratch/
# If jamun_training_script.py is in scratch/, and configs are in jamun/configs/
# then config_path should be "../configs"
with initialize(config_path=".", job_name="conditioning_initial_run"): # Corrected job_name from previous context
    cfg = compose(
        config_name="config", # Main config file
        overrides=[
            "model.arch._target_=scratch.e3conv_test.E3Conv", # Relative to project_root
            "model._target_=scratch.denoiser_test.Denoiser", # Relative to project_root
            "+model.arch.N_structures=2",
            "trainer.max_epochs=100", # Example: train for 10 epochs
            # Add other overrides, e.g. "logger=null" if you don't want default loggers for a quick test
        ]
    )
py_logger.info("Loaded configuration:")
py_logger.info(OmegaConf.to_yaml(cfg))


# %% Initial Dataset Setup (for model properties like average_squared_distance)
py_logger.info("Setting up initial dataset for model properties...")
initial_datasets_for_props = {
    "props_dataset": jamun.data.parse_datasets_from_directory( # Renamed key for clarity
        root=f"{JAMUN_DATA_PATH}/timewarp/2AA-1-large/train/",
        traj_pattern="^(.*)-traj-arrays.npz",
        pdb_file="AA-traj-state0.pdb",
        filter_codes=['AA'],
        as_iterable=False,
        subsample=100, # Keep this small for this purpose
        max_datasets=1,
    )
}

# %% Model Instantiation
py_logger.info("Instantiating model...")
try:
    if not hasattr(cfg.model, "average_squared_distance") or cfg.model.average_squared_distance is None:
        py_logger.info("Computing average_squared_distance for the model...")
        average_squared_distance = compute_average_squared_distance_from_datasets(
            initial_datasets_for_props['props_dataset'], # Use the small dataset for this
            cfg.model.max_radius
        )
        cfg.model.average_squared_distance = average_squared_distance
        py_logger.info(f"Set cfg.model.average_squared_distance to {cfg.model.average_squared_distance}")
    
    # Provide conditioner if needed
    if not hasattr(cfg.model, "conditioner"):
        OmegaConf.set_struct(cfg.model, False)  # Allow modification
        cfg.model.conditioner = OmegaConf.create({})
        cfg.model.conditioner._target_ = 'scratch.conditioners.SelfConditioner'  # Use the SelfConditioner from scratch.conditioners
        OmegaConf.set_struct(cfg.model, True)  # Lock structure again
        py_logger.info(f"Set cfg.model.conditioner to instantiate {cfg.model.conditioner._target_}.")

    model = hydra.utils.instantiate(cfg.model)
    
    py_logger.info("Successfully instantiated model.")
    py_logger.info(f"Instantiated model architecture type: {type(model.g)}")
except Exception as e:
    py_logger.error(f"Error during model instantiation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# %% Device Setup for Model
# Determine the target device, preferring GPU if available.
if torch.cuda.is_available():
    device = torch.device("cuda")
    py_logger.info("CUDA is available. Attempting to use GPU.")
else:
    device = torch.device("cpu")
    py_logger.info("CUDA not available. Using CPU.")

# Move the model to the determined device.
model = model.to(device)

# Verify and log the model's actual device.
# After .to(device), the model's internal device attribute should update.
# We can also check a parameter's device as a fallback verification.
final_model_device = None
if hasattr(model, 'device') and model.device is not None:
    final_model_device = model.device
elif next(model.parameters(), None) is not None:
    final_model_device = next(model.parameters()).device

py_logger.info(f"Model '{type(model).__name__}' is now on device: {final_model_device}")


# %% Setup for Actual Training
py_logger.info("Setting up for actual training...")

# 1. Prepare datasets for training, validation, and testing
# Using the same dataset source for all splits as a placeholder.
# In a real scenario, these would be different datasets or splits.
# You might want to parse a larger dataset here for actual training.
py_logger.info("Parsing dataset for training...")
training_dataset_source = jamun.data.parse_datasets_from_directory(
    root=f"{JAMUN_DATA_PATH}/timewarp/2AA-1-large/train/", # Consider using full dataset
    traj_pattern="^(.*)-traj-arrays.npz",
    pdb_file="AA-traj-state0.pdb",
    filter_codes=['AA'],
    as_iterable=False, # Set to True for very large datasets if memory is an issue
    subsample=cfg.data.datamodule.datasets.train[0].subsample, # Use subsample from config or None for full
    max_datasets=1, # Use from config or None for all
)

datasets_for_training = {
    "train": training_dataset_source,
    "val": training_dataset_source,    # Replace with actual validation set
    "test": training_dataset_source,   # Replace with actual test set
}
py_logger.info(f"Prepared datasets for training: { {k: type(v).__name__ for k, v in datasets_for_training.items()} }")
if isinstance(training_dataset_source, torch_geometric.data.Dataset):
    py_logger.info(f"Training dataset size: {len(training_dataset_source)}")


# 2. Initialize DataModule for training
datamodule_for_training = jamun.data.MDtrajDataModule(
    datasets=datasets_for_training,
    batch_size=cfg.data.datamodule.batch_size,
    num_workers=cfg.data.datamodule.num_workers,
)

# 3. Model is already instantiated and on device
py_logger.info(f"Model '{type(model).__name__}' is ready for training.")

# %% Loggers and Callbacks Setup
py_logger.info("Setting up loggers and callbacks...")

if cfg.get("logger") and cfg.logger.get("wandb"):
    try:
        # This requires JAMUN_ROOT_PATH, task_name, run_group, and run_key to be correctly
        # defined and resolvable in your configuration.
        wandb_save_dir = str(cfg.paths.run_path) # Resolve the path from OmegaConf

        # Update the logger config before instantiation
        OmegaConf.update(cfg, "logger.wandb.save_dir", wandb_save_dir, merge=False)
        py_logger.info(f"Explicitly setting WandbLogger save_dir to: {wandb_save_dir}")
        
        # Ensure the target directory for wandb files exists.
        # WandbLogger will create its 'wandb/' subdirectory and run-specific folders inside this save_dir.
        os.makedirs(wandb_save_dir, exist_ok=True)
        py_logger.info(f"Ensured WandbLogger save_dir exists: {wandb_save_dir}")

    except Exception as e:
        py_logger.error(f"Could not resolve or set wandb save_dir from cfg.paths.run_path: {e}")
        py_logger.warning(f"Wandb will use default save directory (likely ./wandb in CWD: {os.getcwd()}).")

# 1. Instantiate Loggers and Callbacks from Hydra config
loggers_list = []
if cfg.get("logger"):
    if hasattr(cfg.logger, '_target_'): 
         loggers_list.append(hydra.utils.instantiate(cfg.logger))
    else: 
        # Assuming instantiate_dict_cfg iterates and calls hydra.utils.instantiate for each logger config
        loggers_list = instantiate_dict_cfg(cfg.logger) 
py_logger.info(f"Instantiated loggers: {[type(l).__name__ for l in loggers_list]}")

# %% 2. Determine and set the ModelCheckpoint directory path
final_checkpoint_dir = None
# Check if the first logger is a WandbLogger and provides a directory
if (loggers_list and 
    isinstance(loggers_list[0], pl.loggers.WandbLogger) and 
    hasattr(loggers_list[0], 'experiment') and loggers_list[0].experiment and 
    hasattr(loggers_list[0].experiment, 'dir') and loggers_list[0].experiment.dir):
    
    wandb_run_root_dir = loggers_list[0].experiment.dir
    # Adjust if wandb_run_root_dir points to a 'files' subdirectory
    if os.path.basename(wandb_run_root_dir) == "files":
        wandb_run_root_dir = os.path.dirname(wandb_run_root_dir)
    final_checkpoint_dir = os.path.join(wandb_run_root_dir, "checkpoints")
    py_logger.info(f"Using WandB logger's experiment directory for checkpoints: {final_checkpoint_dir}")
else:
    # Default path if no suitable WandB logger is found or no loggers are configured
    final_checkpoint_dir = os.path.join(os.getcwd(), "outputs", "checkpoints")
    if not loggers_list:
        py_logger.info(f"No loggers configured. Defaulting checkpoint directory: {final_checkpoint_dir}")
    else:
        py_logger.info(f"First logger is not a suitable WandB logger. Defaulting checkpoint directory: {final_checkpoint_dir}")

# Update the config if model_checkpoint callback is defined
if cfg.get("callbacks") and cfg.callbacks.get("model_checkpoint"):
    # Use OmegaConf.update to safely set the possibly nested key,
    # this will create it if it doesn't exist or overwrite if it does.
    OmegaConf.update(cfg, "callbacks.model_checkpoint.dirpath", final_checkpoint_dir, merge=False)
    py_logger.info(f"Set cfg.callbacks.model_checkpoint.dirpath to: {final_checkpoint_dir}")
    # Ensure the directory exists
    os.makedirs(final_checkpoint_dir, exist_ok=True)
else:
    py_logger.info("ModelCheckpoint callback not configured in cfg.callbacks, dirpath not set.")


# %% Instantiate Callbacks
callbacks_list = []
if cfg.get("callbacks"):
    if hasattr(cfg.callbacks, '_target_'): # Single callback config
        callbacks_list.append(hydra.utils.instantiate(cfg.callbacks))
    else: # Dictionary of callback configs
        callbacks_list = instantiate_dict_cfg(cfg.callbacks) # This will now use the modified dirpath
py_logger.info(f"Instantiated callbacks: {[type(c).__name__ for c in callbacks_list]}")

# 2. Instantiate PyTorch Lightning Trainer
trainer_config = cfg.trainer
if not hasattr(trainer_config, "_target_") and isinstance(trainer_config, dict):
    trainer_config = OmegaConf.merge(trainer_config, {"_target_": "lightning.pytorch.Trainer"})

trainer: pl.Trainer = hydra.utils.instantiate(
    trainer_config,
    logger=loggers_list if loggers_list else True,
    callbacks=callbacks_list,
)
py_logger.info(f"Instantiated Trainer: {type(trainer)}")
py_logger.info(f"Trainer will run for {trainer.max_epochs} epochs.")

# 3. Handle checkpoint resumption (optional)
checkpoint_path = None
if resume_checkpoint_cfg := cfg.get("resume_from_checkpoint"):
    if resume_checkpoint_cfg.get("enabled", False):
        py_logger.info(f"Attempting to resume from checkpoint with config: {resume_checkpoint_cfg}")
        try:
            checkpoint_path = find_checkpoint(
                wandb_train_run_path=resume_checkpoint_cfg.get("wandb_train_run_path"),
                checkpoint_dir=resume_checkpoint_cfg.get("checkpoint_dir"),
                checkpoint_type=resume_checkpoint_cfg.get("checkpoint_type", "last"),
            )
            if checkpoint_path:
                py_logger.info(f"Found checkpoint to resume from: {checkpoint_path}")
            else:
                py_logger.warning("No checkpoint found for resumption based on config.")
        except Exception as e:
            py_logger.error(f"Error finding checkpoint: {e}. Starting training from scratch.")
            checkpoint_path = None
    else:
        py_logger.info("Checkpoint resumption is configured but not enabled.")

# %% Start Training
py_logger.info("Starting training...")
try:
    trainer.fit(
        model=model, # Use the main model instance
        datamodule=datamodule_for_training,
        ckpt_path=checkpoint_path if checkpoint_path else None
    )
    py_logger.info("Training finished.")

    if cfg.get("run_test_after_train", False):
        py_logger.info("Running test phase...")
        trainer.test(model=model, datamodule=datamodule_for_training)
        py_logger.info("Test phase finished.")

except Exception as e:
    py_logger.error(f"Training FAILED: {e}")
    traceback.print_exc()

py_logger.info("Script finished.")

# %% Log the final configuration and save it locally
wandb_logger_instance = None
# loggers_list should still be in scope from when it was passed to the Trainer
for logger_from_list in loggers_list: # Use a different variable name to avoid conflict if logger is defined elsewhere
    if isinstance(logger_from_list, pl.loggers.WandbLogger):
        wandb_logger_instance = logger_from_list
        break

if wandb_logger_instance and hasattr(wandb_logger_instance, 'experiment') and wandb_logger_instance.experiment:
    py_logger.info(f"WandbLogger experiment active (run_id: {wandb_logger_instance.experiment.id}). Logging final script config to wandb.")
    # Convert the current OmegaConf object 'cfg' to a plain dictionary
    # This 'cfg' includes all modifications made throughout the script
    final_script_cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    try:
        wandb_logger_instance.experiment.config.update(
            {"cfg": final_script_cfg_dict, "jamun_version_at_end": jamun.__version__, "script_cwd_at_end": os.getcwd()}
        )
        py_logger.info("Updated wandb.config with final_script_cfg.")
    except Exception as e:
        py_logger.error(f"Failed to update wandb.config with final script config: {e}")
else:
    if cfg.get("logger") and cfg.logger.get("wandb"):
        py_logger.warning("WandbLogger was configured but not found or experiment not active at script end. Final script config not logged to wandb.config.")

# 2. Explicitly save the final state of the OmegaConf object 'cfg' to a local file

final_config_output_dir = None
if cfg.get("logger") and cfg.logger.get("wandb") and cfg.logger.wandb.get("save_dir"):
    final_config_output_dir = cfg.logger.wandb.save_dir
elif 'wandb_save_dir' in locals() and wandb_save_dir: # If it was set in a previous cell
     final_config_output_dir = wandb_save_dir
else:
    # Fallback if a specific run directory isn't easily available
    # This might not be ideal as it won't be co-located with W&B run files if save_dir wasn't set
    final_config_output_dir = os.path.join(os.getcwd(), "outputs", cfg.get("task_name", "unknown_task"), cfg.get("run_key", "unknown_run"))
    os.makedirs(final_config_output_dir, exist_ok=True)


if final_config_output_dir:
    final_config_path = os.path.join(final_config_output_dir, "final_resolved_script_config.yaml")
    try:
        with open(final_config_path, 'w') as f:
            OmegaConf.save(config=cfg, f=f)
        py_logger.info(f"Final script configuration saved locally to: {final_config_path}")
    except Exception as e:
        py_logger.error(f"Failed to save final script configuration locally: {e}")
else:
    py_logger.warning("Could not determine a definitive output directory for final_resolved_script_config.yaml. Not saving locally.")

loggers_list[0].experiment.finish() if loggers_list and isinstance(loggers_list[0], pl.loggers.WandbLogger) else None
py_logger.info("Script finished.")
# %%
