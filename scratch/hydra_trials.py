# %% imports 
import hydra
from omegaconf import OmegaConf
import os
import sys
import dotenv
import logging 

# --- Basic Setup ---
logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("jamun_sampling_script")

# Add project root to path for custom modules
project_root = "/homefs/home/sules/jamun"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    py_logger.info(f"Added '{project_root}' to sys.path for module discovery.")
else:
    py_logger.info(f"'{project_root}' is already in sys.path.")

dotenv.load_dotenv("../.env", verbose=True)
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")
JAMUN_ROOT_PATH = os.getenv("JAMUN_ROOT_PATH")

def print_config_sections(cfg):
    """Print different sections of the configuration."""
    print("\nFull Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    if hasattr(cfg, 'model'):
        print("\nModel Configuration:")
        print(OmegaConf.to_yaml(cfg.model))
    
    if hasattr(cfg, 'init_datasets'):
        print("\nDataset Configuration:")
        print(OmegaConf.to_yaml(cfg.init_datasets))
    
    if hasattr(cfg, 'sampler'):
        print("\nSampler Configuration:")
        print(OmegaConf.to_yaml(cfg.sampler))

def run(cfg):
    """Main function to run the config loading and printing."""
    # Print the loaded configuration
    print_config_sections(cfg)
    
    # Print specific config values
    if hasattr(cfg, 'model'):
        print("\nModel target:", cfg.model._target_)
    if hasattr(cfg, 'sampler'):
        print("Sampler target:", cfg.sampler._target_)

def main():
    # Initialize Hydra
    with hydra.initialize(config_path="../src/jamun/hydra_config"):
        # Compose the base configuration
        base_cfg = hydra.compose(config_name="sample")
    
    # Compose the experiment configuration
    experiment_cfg = hydra.compose(config_name="sample", overrides=["experiment=sample_uncapped_single_shape_conditioning"])
    
    # Merge configurations (experiment overrides base)
    cfg = OmegaConf.merge(base_cfg, experiment_cfg)
    
    # Run with the merged configuration
    run(cfg)

if __name__ == "__main__":
    main()
