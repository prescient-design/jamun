#!/usr/bin/env python3
"""
Simple test script for the debugged denoiser_conditional using default hydra config.
Tests with sigma = 0.0 and sigma = 0.1.
"""

import e3nn

e3nn.set_optimization_defaults(jit_script_fx=False)
import os
import sys

import dotenv
import hydra
import torch
from omegaconf import OmegaConf

from jamun.utils import compute_average_squared_distance_from_datasets

breakpoint()  # Start debugging

dotenv.load_dotenv("../.env", verbose=True)
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")
JAMUN_ROOT_PATH = os.getenv("JAMUN_ROOT_PATH")

project_root = "/homefs/home/sules/jamun"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added '{project_root}' to sys.path for module discovery.")

breakpoint()  # After setup


def compute_average_squared_distance_from_config(cfg: OmegaConf) -> float:
    """Computes the average squared distance for normalization from the data."""
    breakpoint()  # Start of function
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("compute_normalization")
    train_datasets = datamodule.datasets["train"]
    cutoff = cfg.model.max_radius
    average_squared_distance = compute_average_squared_distance_from_datasets(train_datasets, cutoff)
    breakpoint()  # After computation
    return average_squared_distance


@hydra.main(version_base=None, config_path="../src/jamun/hydra_config", config_name="train")
def main(cfg):
    breakpoint()  # Start of main
    print("=" * 60)
    print("Testing debugged denoiser_conditional")
    print("=" * 60)

    # Compute average squared distance
    print("Computing average squared distance...")
    breakpoint()  # Before distance computation
    average_squared_distance = compute_average_squared_distance_from_config(cfg)
    cfg.model.average_squared_distance = average_squared_distance
    print(f"Average squared distance: {average_squared_distance:.6f}")

    # Load datamodule
    print("Loading datamodule...")
    breakpoint()  # Before datamodule
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("test")

    # Load model
    print("Loading model...")
    breakpoint()  # Before model loading
    model = hydra.utils.instantiate(cfg.model)

    # Get a single batch
    print("Getting a batch of data...")
    breakpoint()  # Before getting batch
    train_loader = datamodule.train_dataloader()
    _, batch = next(enumerate(train_loader))

    print("Batch info:")
    print(f"  Position shape: {batch.pos.shape}")
    print(f"  Number of atoms: {batch.pos.shape[0]}")
    print(f"  Hidden state shapes: {[h.shape for h in batch.hidden_state]}")
    print(f"  Number of hidden states: {len(batch.hidden_state)}")

    breakpoint()  # After batch info

    # Test with sigma = 0.0
    print("\n" + "=" * 40)
    print("Testing with sigma = 0.0 (no noise)")
    print("=" * 40)

    breakpoint()  # Before sigma=0.0 test

    with torch.no_grad():
        breakpoint()  # Before noise_and_denoise
        sigma = torch.tensor(0.0)
        x_target, xhat, y = model.noise_and_denoise(batch, sigma, align_noisy_input=True)

        print(f"Input shape: {batch.pos.shape}")
        print(f"Noisy shape: {y.pos.shape}")
        print(f"Output shape: {xhat.pos.shape}")

        breakpoint()  # After noise_and_denoise

        # Compute loss
        loss, aux = model.compute_loss(x_target, xhat, sigma)
        print(f"Loss: {loss.mean().item():.6f}")
        print(f"Metrics: {aux}")

        # Check if positions are preserved (should be identical with sigma=0)
        pos_diff = torch.abs(batch.pos - y.pos).max()
        print(f"Max position difference (sigma=0): {pos_diff.item():.8f}")

    breakpoint()  # After sigma=0.0 test

    # Test with sigma = 0.1
    print("\n" + "=" * 40)
    print("Testing with sigma = 0.1 (with noise)")
    print("=" * 40)

    breakpoint()  # Before sigma=0.1 test

    with torch.no_grad():
        sigma = torch.tensor(0.1)
        breakpoint()  # Before noise_and_denoise with sigma=0.1
        x_target, xhat, y = model.noise_and_denoise(batch, sigma, align_noisy_input=True)

        print(f"Input shape: {batch.pos.shape}")
        print(f"Noisy shape: {y.pos.shape}")
        print(f"Output shape: {xhat.pos.shape}")

        # Compute loss
        loss, aux = model.compute_loss(x_target, xhat, sigma)
        print(f"Loss: {loss.mean().item():.6f}")
        print(f"Metrics: {aux}")

        # Check noise level
        pos_diff = torch.abs(batch.pos - y.pos).max()
        print(f"Max position difference (sigma=0.1): {pos_diff.item():.6f}")

        # Check denoising quality
        denoise_diff = torch.abs(batch.pos - xhat.pos).max()
        print(f"Max denoising difference: {denoise_diff.item():.6f}")

    breakpoint()  # After sigma=0.1 test

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)

    breakpoint()  # End of main


if __name__ == "__main__":
    main()
