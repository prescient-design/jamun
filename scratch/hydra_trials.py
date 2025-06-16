# %% imports 
import hydra
from omegaconf import OmegaConf
import os
import sys
import dotenv
import logging 
import traceback

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
    try:
        # Print the loaded configuration
        print_config_sections(cfg)
        
        # Print specific config values
        if hasattr(cfg, 'model'):
            print("\nModel target:", cfg.model._target_)
        if hasattr(cfg, 'sampler'):
            print("Sampler target:", cfg.sampler._target_)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

@hydra.main(version_base=None, config_path="../src/jamun/hydra_config", config_name="sample")
def main(cfg):
    try:
        run(cfg)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
