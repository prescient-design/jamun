import sys
import os
import torch
import logging
import dotenv
import hydra
from omegaconf import OmegaConf
from denoiser_test import Denoiser
from jamun.utils.checkpoint import find_checkpoint

# Setup logging
logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
logger = logging.getLogger("load_wandb_checkpoint")

dotenv.load_dotenv("../.env", verbose=True) # Adjust path if script is not in scratch/
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")
JAMUN_ROOT_PATH = os.getenv("JAMUN_ROOT_PATH")

project_root = "/homefs/home/sules/jamun" # Adjust if necessary
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added '{project_root}' to sys.path for module discovery.")
else:
    logger.info(f"'{project_root}' is already in sys.path.")

@hydra.main(version_base=None, config_path="../src/jamun/hydra_config", config_name="sample")
def load_model_from_wandb(cfg, wandb_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load a model checkpoint from wandb.
    
    Args:
        cfg: Hydra configuration
        wandb_path (str): Path to the wandb checkpoint (e.g., "entity/project/run_id")
        device (str): Device to load the model on
    
    Returns:
        Denoiser: The loaded model
    """
    # Find the checkpoint path using the utility function
    checkpoint_path = find_checkpoint(
        wandb_train_run_path=wandb_path,
        checkpoint_type="last"  # or "best_so_far" if you want the best checkpoint
    )
    logger.info(f"Found checkpoint at: {checkpoint_path}")
    
    # Update the config with the checkpoint path
    cfg.model.checkpoint_path = checkpoint_path
    
    # Load the model using Hydra
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model

if __name__ == "__main__":
    # Example usage
    wandb_path = "sule-shashank/jamun/y4rm5488"  # Replace with actual wandb path
    model = load_model_from_wandb(wandb_path) 