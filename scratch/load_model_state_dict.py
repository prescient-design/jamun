import hydra
import torch
from omegaconf import OmegaConf

# Load the config
config_path = "/data2/sules/jamun-conditional-runs//outputs/train/dev/runs/2025-08-05_04-24-31/wandb/run-20250805_042516-yqn9mm7x/files/config.yaml"
cfg = OmegaConf.load(config_path)

# Find the checkpoint file
checkpoint_path = (
    "/data2/sules/jamun-conditional-runs//outputs/train/dev/runs/2025-08-05_04-24-31/checkpoints/last.ckpt"
)
# checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
# checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])  # or choose specific one
breakpoint()
# Instantiate the model using the config
model = hydra.utils.instantiate(cfg.model)
breakpoint()
# Load the state dict from checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])

print(f"Loaded model: {type(model).__name__}")
print(f"From checkpoint: {checkpoint_path}")
