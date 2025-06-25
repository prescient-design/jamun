import e3nn
e3nn.set_optimization_defaults(jit_script_fx=False)
import dotenv
import sys
import os
import hydra
from omegaconf import OmegaConf
import torch
import torch_geometric
from jamun.hydra import instantiate_dict_cfg
import pdb
import jamun
from jamun.utils import compute_average_squared_distance_from_datasets

dotenv.load_dotenv("../.env", verbose=True) # Adjust path if script is not in scratch/
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")
JAMUN_ROOT_PATH = os.getenv("JAMUN_ROOT_PATH")

project_root = "/homefs/home/sules/jamun" # Adjust if necessary
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added '{project_root}' to sys.path for module discovery.")
else:
    print(f"'{project_root}' is already in sys.path.")

def compute_average_squared_distance_from_config(cfg: OmegaConf) -> float:
    """Computes the average squared distance for normalization from the data."""
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("compute_normalization")
    train_datasets = datamodule.datasets["train"]
    cutoff = cfg.model.max_radius
    average_squared_distance = compute_average_squared_distance_from_datasets(train_datasets, cutoff)
    return average_squared_distance


@hydra.main(version_base=None, config_path="../src/jamun/hydra_config", config_name="train")
def main(cfg):
    # Load the test config
    average_squared_distance = compute_average_squared_distance_from_config(cfg)
    cfg.model.average_squared_distance = average_squared_distance
    # breakpoint()
    
    # # First merge test config into base config, then override with test config
    # cfg = OmegaConf.merge(cfg, test_cfg)
    # cfg = OmegaConf.merge(cfg, test_cfg, override=True)
    # breakpoint()
    
    print("Loading datamodule...")
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup('test')
    # breakpoint()
    
    print("Loading model...")
    model = hydra.utils.instantiate(cfg.model)
    breakpoint()
    
    # Get a single batch
    print("Getting a batch of data...")
    train_loader = datamodule.train_dataloader()
    _, batch = next(enumerate(train_loader))
    # breakpoint()
    
    # # Move to CPU
    # batch = batch.to("cpu")
    # model = model.to("cpu")
    
    print(f"Batch shape: {batch.pos.shape}")
    print(f"Hidden state shape: {[h.shape for h in batch.hidden_state]}")
    breakpoint()
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        sigma = model.sigma_distribution.sample()
        x_target, xhat, y = model.noise_and_denoise(batch, sigma, align_noisy_input=True)
        
    print(f"Input shape: {batch.pos.shape}")
    print(f"Noisy shape: {y.pos.shape}")
    print(f"Output shape: {xhat.pos.shape}")
    
    # Test loss computation
    print("Testing loss computation...")
    loss, aux = model.compute_loss(x_target, xhat, sigma)
    print(f"Loss: {loss.mean().item():.4f}")
    print(f"Metrics: {aux}")

if __name__ == "__main__":
    main() 