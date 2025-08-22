#!/usr/bin/env python3
"""
Script to analyze validation trajectories using a trained model.
Generates Ramachandran plots for clean, noisy, and denoised samples.
"""

import os
import glob
import tempfile
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
import torch
import torch_geometric
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import mdtraj as md
import wandb
import pdb
pdb.set_trace()
from jamun.data import parse_datasets_from_directory, MDtrajDataset
from jamun.metrics._visualize_denoise import VisualizeDenoiseMetrics, plot_ramachandran_grid
from jamun import utils
from jamun.utils.checkpoint import find_checkpoint
from jamun.model.denoiser_conditional import Denoiser
# from jamun.model.denoiser import Denoiser as Denoiser_unconditional


def load_model_from_wandb(wandb_run_path: str, checkpoint_type: str = "last", checkpoint_path: str = None):
    """Load model from wandb run."""
    print(f"Loading model from {wandb_run_path}...")
    
    # Use jamun utilities to find the checkpoint
    checkpoint_path_wandb = find_checkpoint(
        wandb_train_run_path=wandb_run_path,
        checkpoint_type=checkpoint_type
    )

    if checkpoint_path is None:
        checkpoint_path = checkpoint_path_wandb
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load the model
    model = Denoiser.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model


def create_dataset_from_trajectory(traj_file: str, pdb_file: str, total_lag_time: int = 2):
    """Create a dataset from a single trajectory file."""
    # Create temporary directory structure expected by parse_datasets_from_directory
    temp_dir = tempfile.mkdtemp()
    
    # Copy trajectory file to temp directory
    traj_name = Path(traj_file).stem
    temp_traj_path = os.path.join(temp_dir, f"{traj_name}.xtc")
    temp_pdb_path = os.path.join(temp_dir, f"{traj_name}.pdb")
    
    # Create symlinks
    os.symlink(traj_file, temp_traj_path)
    os.symlink(pdb_file, temp_pdb_path)
    
    # Parse dataset
    datasets = parse_datasets_from_directory(
        root=temp_dir,
        traj_pattern="^(.*).xtc",
        pdb_pattern="^(.*).pdb",
        as_iterable=False,
        subsample=1,
        total_lag_time=total_lag_time,
        lag_subsample_rate=1,
        max_datasets=1,
        label_override=traj_name
    )
    
    return datasets[0] if datasets else None


def process_trajectory(model, dataset: MDtrajDataset, sigma: float = 0.04):
    """Process a single trajectory through the model."""
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=torch_geometric.data.Batch.from_data_list
    )
    
    # Store all samples
    all_clean = []
    all_noisy = []
    all_denoised = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(model.device)
            
            # # Ensure all batch attributes are the correct dtype
            # if hasattr(batch, 'pos'):
            #     batch.pos = batch.pos.float()
            # if hasattr(batch, 'batch'):
            #     batch.batch = batch.batch.long()
            # if hasattr(batch, 'edge_index'):
            #     batch.edge_index = batch.edge_index.long()  # Keep as long for indexing!
            # if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            #     batch.edge_attr = batch.edge_attr.float()
                
            # Convert sigma to tensor with correct dtype and device
            sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device=model.device)
            
            # Run noise and denoise
            _, xhat, y = model.noise_and_denoise(
                batch, sigma_tensor, align_noisy_input=model.align_noisy_input_during_evaluation
            )
            
            # Convert to data lists
            clean_samples = torch_geometric.data.Batch.to_data_list(batch)
            noisy_samples = torch_geometric.data.Batch.to_data_list(y)
            denoised_samples = torch_geometric.data.Batch.to_data_list(xhat)
            
            all_clean.extend(clean_samples)
            all_noisy.extend(noisy_samples)
            all_denoised.extend(denoised_samples)
    
    return all_clean, all_noisy, all_denoised


def samples_to_trajectory(samples: List, dataset: MDtrajDataset):
    """Convert list of samples to MDTraj trajectory."""
    coordinates = []
    for sample in samples:
        coords = sample.pos.cpu().numpy()
        coordinates.append(coords)
    
    coords_array = np.array(coordinates)  # Shape: (n_frames, n_atoms, 3)
    
    # Create trajectory
    traj = md.Trajectory(coords_array, dataset.topology)
    return traj


def create_ramachandran_plot(clean_traj, noisy_traj, denoised_traj, title: str, save_path: str):
    """Create and save Ramachandran plot for three trajectories."""
    trajs = {
        "x": clean_traj,
        "y": noisy_traj, 
        "xhat": denoised_traj
    }
    
    try:
        fig, axes = plot_ramachandran_grid(trajs, title)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved Ramachandran plot: {save_path}")
    except Exception as e:
        print(f"✗ Error creating Ramachandran plot for {title}: {e}")


def main():
    # Configuration
    wandb_run_path = "sule-shashank/jamun/4p0ejn0z"
    val_dir = "/data2/sules/ALA_ALA_enhanced_full_grid/val"
    pdb_file = "/data/bucket/kleinhej/capped_diamines/timewarp_splits/train/ALA_ALA.pdb"
    checkpoint_type = 'epoch=49-step=52900-v1.ckpt'
    # checkpoint_path = "/data2/sules/jamun-conditional-runs/outputs/train/dev/runs/2025-07-31_00-43-14/checkpoints/epoch=9-step=10051.ckpt"
    total_lag_time = 5
    sigma = 0.04
    output_dir = "val_ramachandrans"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    checkpoint_path = find_checkpoint(wandb_run_path, checkpoint_type=checkpoint_type)
    model = Denoiser.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to('cuda:0')
    print(f"Model loaded and moved to cuda:0")
    # config_path = "/data2/sules/jamun-conditional-runs//outputs/train/dev/runs/2025-08-05_04-24-31/wandb/run-20250805_042516-yqn9mm7x/files/config.yaml"    
    # cfg = OmegaConf.load(config_path)
    # checkpoint_path = "/data2/sules/jamun-conditional-runs//outputs/train/dev/runs/2025-08-05_04-24-31/checkpoints/last.ckpt"
    # model = hydra.utils.instantiate(cfg.cfg.value.model)
    # checkpoint = torch.load(checkpoint_path, weights_only=False)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()
    # model.to('cuda:0')
    # print(f"Model loaded and moved to cuda:0")
    print(f"Model device: {next(model.parameters()).device}")
    # Get all trajectory files
    traj_files = glob.glob(os.path.join(val_dir, "*.xtc"))
    traj_files.sort()
    
    print(f"Found {len(traj_files)} trajectory files")
    # do one trial run 
    dataset = create_dataset_from_trajectory(traj_files[0], pdb_file, total_lag_time)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=torch_geometric.data.Batch.from_data_list
    )
    _, batch = next(enumerate(dataloader))

    batch = batch.to(model.device)
    sigma_tensor = torch.tensor(sigma, dtype=torch.float32, device=model.device)
    _, xhat, y = model.noise_and_denoise(
        batch, sigma_tensor, align_noisy_input=model.align_noisy_input_during_evaluation
    )
    # Store all samples for concatenated analysis
    all_clean_samples = []
    all_noisy_samples = []
    all_denoised_samples = []
    breakpoint()
    # Process each trajectory with progress bar
    for traj_file in tqdm(traj_files, desc="Processing trajectories"):
        traj_name = Path(traj_file).stem
        
        try:
            # Create dataset
            dataset = create_dataset_from_trajectory(traj_file, pdb_file, total_lag_time)
            if dataset is None:
                tqdm.write(f"Failed to create dataset for {traj_name}")
                continue
            
            # Process trajectory
            # breakpoint()
            clean_samples, noisy_samples, denoised_samples = process_trajectory(model, dataset, sigma)
            
            # Store samples for concatenated analysis
            all_clean_samples.extend(clean_samples)
            all_noisy_samples.extend(noisy_samples)
            all_denoised_samples.extend(denoised_samples)
            
        except Exception as e:
            tqdm.write(f"Error processing {traj_name}: {e}")
            continue
    
    # Create concatenated analysis
    if all_clean_samples:
        print("\nCreating concatenated Ramachandran plot...")
        
        # Use the last dataset for topology (they should all be the same)
        concat_clean_traj = samples_to_trajectory(all_clean_samples, dataset)
        concat_noisy_traj = samples_to_trajectory(all_noisy_samples, dataset)
        concat_denoised_traj = samples_to_trajectory(all_denoised_samples, dataset)
        
        # Create concatenated plot
        concat_plot_path = os.path.join(output_dir, "concatenated_ramachandran.png")
        create_ramachandran_plot(
            concat_clean_traj, 
            concat_noisy_traj, 
            concat_denoised_traj, 
            "Concatenated Trajectories", 
            concat_plot_path
        )
        
        print(f"\nAnalysis complete! Concatenated Ramachandran plot saved in {output_dir}/")
        print(f"Processed {len(all_clean_samples)} total samples from {len(traj_files)} trajectories")
    else:
        print("No samples were processed successfully.")


if __name__ == "__main__":
    main() 