import sys
sys.path.insert(0, "src")

import torch
from jamun.model.denoiser_conditional import Denoiser
print(f"Loading model from checkpoint...")
# Load model
model = Denoiser.load_from_checkpoint("/data2/sules/jamun-conditional-runs/outputs/train/dev/runs/2025-07-08_22-31-01/checkpoints/last.ckpt")

# Print key info
print(f"Model: {type(model).__name__}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Conditioner: {model.conditioner}")
print(f"Architecture: {model.g}")
print(f"Hyperparams: {dict(model.hparams)}") 