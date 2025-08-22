#!/usr/bin/env python3
"""
Script to analyze validation errors for runs from the graph_type_comparison_experiment_enhanced_sampling_data_onlyfan_aug17 group.

This script:
1. Scrapes all runs from the specified WandB group
2. For each run, extracts lag_subsample_rate and total_lag_time
3. Loads validation data with the same parameters and max_datasets=1
4. Computes validation errors for each model
5. Creates a 3D histogram plot with lag_subsample_rate (x), total_lag_time (y), and validation error (height)
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import wandb
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm

# Add jamun to path
sys.path.insert(0, '/homefs/home/sules/jamun/src')

import jamun
from jamun.model.denoiser_conditional import Denoiser
from jamun.utils.checkpoint import find_checkpoint, get_wandb_run_config
from jamun.data import parse_datasets_from_directory, parse_repeated_position_datasets_from_directory
from jamun.data._dloader import MDtrajDataModule
import torch_geometric

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_data_path():
    """Get the data path from environment or common locations."""
    data_path = os.getenv("JAMUN_DATA_PATH")
    if data_path is None:
        # Try common locations
        possible_paths = [
            "/data/bucket/kleinhej/",
            "/data2/sules/",
            "/homefs/home/sules/data/"
        ]
        for path in possible_paths:
            if Path(path).exists():
                data_path = path
                break
    
    if data_path is None:
        raise ValueError("JAMUN_DATA_PATH not set and cannot find data. Please set JAMUN_DATA_PATH environment variable.")
    
    logger.info(f"Using data path: {data_path}")
    return data_path

def scrape_wandb_runs(group_name: str, project: str = "sule-shashank/jamun") -> List[wandb.Api.run]:
    """Scrape all runs from the specified WandB group."""
    logger.info(f"Scraping runs from group: {group_name}")
    
    api = wandb.Api()
    runs = api.runs(project, filters={'group': group_name})
    runs_list = list(runs)
    
    logger.info(f"Found {len(runs_list)} runs in group '{group_name}'")
    
    return runs_list

def extract_run_parameters(runs: List[wandb.Api.run]) -> List[Dict[str, Any]]:
    """Extract lag_subsample_rate and total_lag_time from each run."""
    logger.info("Extracting lag_subsample_rate and total_lag_time from runs...")
    
    run_params = []
    
    for run in tqdm(runs, desc="Processing runs", unit="run"):
        try:
            config = run.config
            if 'cfg' not in config:
                logger.warning(f"Run {run.name} missing 'cfg' in config")
                continue
            
            cfg = config['cfg']
            
            # Extract lag_subsample_rate
            lag_subsample_rate = None
            try:
                lag_subsample_rate = cfg['data']['datamodule']['datasets']['train']['lag_subsample_rate']
            except (KeyError, TypeError):
                try:
                    # Try alternative path
                    lag_subsample_rate = cfg['data']['datamodule']['datasets']['lag_subsample_rate']
                except (KeyError, TypeError):
                    logger.warning(f"Could not extract lag_subsample_rate for run {run.name}")
                    continue
            
            # Extract total_lag_time
            total_lag_time = None
            try:
                total_lag_time = cfg['data']['datamodule']['datasets']['train']['total_lag_time']
            except (KeyError, TypeError):
                try:
                    # Try alternative path
                    total_lag_time = cfg['data']['datamodule']['datasets']['total_lag_time']
                except (KeyError, TypeError):
                    logger.warning(f"Could not extract total_lag_time for run {run.name}")
                    continue
            
            # Extract model target for filtering
            model_target = cfg.get('model', {}).get('_target_')
            
            # Extract data target for filtering
            data_target = None
            try:
                data_target = cfg['data']['datamodule']['datasets']['train']['_target_']
            except (KeyError, TypeError):
                try:
                    data_target = cfg['data']['datamodule']['datasets']['_target_']
                except (KeyError, TypeError):
                    logger.warning(f"Could not extract data target for run {run.name}")
            
            run_info = {
                'run': run,
                'run_path': '/'.join(run.path),
                'run_name': run.name,
                'lag_subsample_rate': lag_subsample_rate,
                'total_lag_time': total_lag_time,
                'model_target': model_target,
                'data_target': data_target,
                'cfg': cfg
            }
            
            run_params.append(run_info)
            logger.info(f"Added run: {run.name} (lag_subsample_rate={lag_subsample_rate}, total_lag_time={total_lag_time})")
            
        except Exception as e:
            logger.warning(f"Error processing run {run.name}: {e}")
            continue
    
    logger.info(f"Successfully extracted parameters from {len(run_params)} runs")
    return run_params

def load_validation_data(total_lag_time: int, lag_subsample_rate: int, val_root: str = "/data2/sules/ALA_ALA_enhanced_full_grid/val/") -> torch_geometric.loader.DataLoader:
    """Load validation data for error computation with specific lag parameters."""
    logger.info(f"Loading validation data with total_lag_time={total_lag_time}, lag_subsample_rate={lag_subsample_rate}")
    
    if not os.path.exists(val_root):
        raise ValueError(f"Validation directory not found: {val_root}")
    
    try:
        datasets = parse_repeated_position_datasets_from_directory(
            root=val_root,
            traj_pattern="^(.*).xtc",
            pdb_pattern="^(.*).pdb",
            filter_codes=None,
            as_iterable=False,
            subsample=1,
            total_lag_time=total_lag_time,
            lag_subsample_rate=lag_subsample_rate,
            max_datasets=10  # Use 10 datasets for validation
        )
        
        if not datasets:
            raise ValueError("No validation datasets found")
        
        # Create data module
        data_module = MDtrajDataModule(
            datasets={'val': datasets},
            batch_size=32,
            num_workers=0,
            persistent_workers=False
        )
        data_module.setup('val')
        
        val_loader = data_module.val_dataloader()
        logger.info(f"Loaded validation data with {len(datasets)} datasets")
        
        return val_loader
        
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        raise

def load_model_and_compute_scaled_rmse(run_info: Dict[str, Any], val_loader: torch_geometric.loader.DataLoader) -> float:
    """Load a model from checkpoint and compute validation scaled RMSE."""
    run_path = run_info['run_path']
    run_name = run_info['run_name']
    
    logger.info(f"Loading model for run: {run_name}")
    
    try:
        # Find checkpoint
        checkpoint_path = find_checkpoint(
            wandb_train_run_path=run_path,
            checkpoint_type="last"
        )
        
        # Load model
        model = Denoiser.load_from_checkpoint(checkpoint_path, strict=False)
        model.eval()
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info(f"Model loaded successfully for run: {run_name}")
        
        # Compute validation scaled RMSE
        total_scaled_rmse = 0.0
        total_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Computing validation", leave=False, unit="batch"):
                batch = batch.to(device)
                
                # Use the model's validation logic
                sigma = model.sigma_distribution.sample().to(device)
                loss, aux = model.noise_and_compute_loss(
                    batch, 
                    sigma,
                    align_noisy_input=model.align_noisy_input_during_evaluation
                )
                
                # Extract scaled RMSE from aux dictionary
                if 'scaled_rmsd' in aux:
                    scaled_rmse_value = aux['scaled_rmsd'].mean().item()
                    total_scaled_rmse += scaled_rmse_value
                elif 'rmsd' in aux and 'mse' in aux:
                    # Compute scaled RMSE manually if not available
                    rmsd_value = aux['rmsd'].mean().item()
                    # scaled_rmsd = rmsd / (sigma * sqrt(D)) - but we'll use rmsd as fallback
                    total_scaled_rmse += rmsd_value
                    logger.warning(f"Using RMSD instead of scaled RMSD for {run_name}")
                else:
                    logger.warning(f"Scaled RMSD not found in aux dictionary for {run_name}. Available keys: {list(aux.keys())}")
                    # Fallback to loss if scaled RMSD not available
                    total_scaled_rmse += loss.mean().item()
                
                total_batches += 1
        
        avg_scaled_rmse = total_scaled_rmse / total_batches if total_batches > 0 else float('inf')
        logger.info(f"Validation scaled RMSE for {run_name}: {avg_scaled_rmse:.6f}")
        
        return avg_scaled_rmse
        
    except Exception as e:
        logger.error(f"Error computing validation error for run {run_name}: {e}")
        return float('inf')

def create_3d_histogram(results: List[Dict[str, Any]], output_path: str = "graph_type_comparison_3d_histogram.png"):
    """Create a 3D histogram plot of validation scaled RMSE errors."""
    logger.info("Creating 3D histogram plot...")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Debug: Print available columns and data ranges
    logger.info(f"Available DataFrame columns: {list(df.columns)}")
    logger.info(f"Lag subsample rate range: {df['lag_subsample_rate'].min()} - {df['lag_subsample_rate'].max()}")
    logger.info(f"Total lag time range: {df['total_lag_time'].min()} - {df['total_lag_time'].max()}")
    logger.info(f"Validation scaled RMSE range: {df['validation_scaled_rmse'].min()} - {df['validation_scaled_rmse'].max()}")
    
    # Get unique values for x and y axes
    unique_lag_subsample_rates = sorted(df['lag_subsample_rate'].unique())
    unique_total_lag_times = sorted(df['total_lag_time'].unique(), reverse=True)  # Reverse order: 8 to 2
    
    logger.info(f"Unique lag subsample rates: {unique_lag_subsample_rates}")
    logger.info(f"Unique total lag times: {unique_total_lag_times}")
    
    # Create meshgrid for positioning bars
    x_positions = np.arange(len(unique_lag_subsample_rates))
    y_positions = np.arange(len(unique_total_lag_times))
    X, Y = np.meshgrid(x_positions, y_positions, indexing='ij')
    
    # Initialize heights array
    heights = np.zeros((len(unique_lag_subsample_rates), len(unique_total_lag_times)))
    
    # Fill heights array with validation scaled RMSE values
    for _, row in df.iterrows():
        x_idx = unique_lag_subsample_rates.index(row['lag_subsample_rate'])
        y_idx = unique_total_lag_times.index(row['total_lag_time'])
        heights[x_idx, y_idx] = row['validation_scaled_rmse']
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Flatten arrays for bar3d
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_bottom = np.zeros_like(x_flat)
    heights_flat = heights.flatten()
    
    # Create bars with proper color normalization
    dx = dy = 0.8  # Width of bars
    
    # Create proper normalization for colors
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=heights_flat.min(), vmax=heights_flat.max())
    colors = plt.cm.viridis(norm(heights_flat))
    
    ax.bar3d(x_flat, y_flat, z_bottom, dx, dy, heights_flat, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Set labels and title with font size 16 for axis labels, 14 for tick labels
    ax.set_xlabel('Lag Subsample Rate', fontsize=16)
    ax.set_ylabel('Total Lag Time', fontsize=16)
    ax.set_zlabel('Test Scaled RMSE', fontsize=16)
    ax.set_title('3D Histogram: Test Scaled RMSE vs Lag Parameters\nGroup: graph_type_comparison_experiment_enhanced_sampling_data_onlyfan_aug17', fontsize=14)
    
    # Set custom tick labels with font size 14
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(rate) for rate in reversed(unique_lag_subsample_rates)], fontsize=14)
    ax.set_yticks(y_positions)
    # Reverse the tick labels while keeping positions from 2 to 8
    ax.set_yticklabels([str(time) for time in reversed(unique_total_lag_times)], fontsize=14)
    ax.tick_params(axis='z', labelsize=14)
    
    # Add colorbar with the same normalization
    mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    mappable.set_array(heights_flat)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Test Scaled RMSE', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    # Adjust view angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"3D histogram saved to: {output_path}")
    
    # Also save data to CSV
    csv_path = output_path.replace('.png', '.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Data saved to: {csv_path}")
    
    # Save as structured numpy array
    npy_path = "graph_type_comparison_validation_errors.npy"
    data = np.array(list(zip(df['lag_subsample_rate'].values, 
                            df['total_lag_time'].values, 
                            df['validation_scaled_rmse'].values)),
                   dtype=[('lag_subsample_rate', 'i4'), 
                          ('total_lag_time', 'i4'), 
                          ('validation_scaled_rmse', 'f8')])
    np.save(npy_path, data)
    logger.info(f"Structured array saved to: {npy_path}")

def main():
    """Main function to execute the graph type comparison analysis."""
    logger.info("Starting graph type comparison validation analysis...")
    
    # Configuration
    group_name = "graph_type_comparison_experiment_enhanced_sampling_data_onlyfan_aug17"
    project = "sule-shashank/jamun"
    
    try:
        # Step 1: Scrape WandB runs
        runs = scrape_wandb_runs(group_name, project)
        
        # Step 2: Extract parameters from runs
        run_params = extract_run_parameters(runs)
        
        if not run_params:
            logger.error("No runs with valid parameters found in the specified group")
            return
        
        # Step 3: Compute validation errors for each run
        results = []
        
        # Create a cache for validation loaders to avoid reloading same data
        validation_cache = {}
        
        # Create progress bar with more detailed description
        total_runs = len(run_params)
        pbar = tqdm(total=total_runs, desc="Processing runs", unit="run")
        
        for i, run_info in enumerate(run_params):
            try:
                lag_subsample_rate = run_info['lag_subsample_rate']
                total_lag_time = run_info['total_lag_time']
                
                # Update progress bar with current parameters
                pbar.set_description(f"Processing lag_sub={lag_subsample_rate}, lag_time={total_lag_time}")
                
                # Create cache key
                cache_key = (total_lag_time, lag_subsample_rate)
                
                # Load validation data (use cache if available)
                if cache_key not in validation_cache:
                    logger.info(f"Loading validation data for lag_time={total_lag_time}, lag_subsample={lag_subsample_rate}")
                    val_loader = load_validation_data(total_lag_time, lag_subsample_rate)
                    validation_cache[cache_key] = val_loader
                else:
                    val_loader = validation_cache[cache_key]
                    logger.info(f"Using cached validation data for lag_time={total_lag_time}, lag_subsample={lag_subsample_rate}")
                
                # Compute validation scaled RMSE
                validation_scaled_rmse = load_model_and_compute_scaled_rmse(run_info, val_loader)
                
                result = {
                    'run_name': run_info['run_name'],
                    'run_path': run_info['run_path'],
                    'lag_subsample_rate': lag_subsample_rate,
                    'total_lag_time': total_lag_time,
                    'validation_scaled_rmse': validation_scaled_rmse,
                    'model_target': run_info['model_target'],
                    'data_target': run_info['data_target']
                }
                results.append(result)
                
                logger.info(f"Processed {run_info['run_name']}: Scaled RMSE={validation_scaled_rmse:.6f}")
                
            except Exception as e:
                logger.error(f"Error processing run {run_info['run_name']}: {e}")
                continue
            finally:
                # Update progress bar
                pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
        if not results:
            logger.error("No successful results obtained")
            return
        
        # Step 4: Create 3D histogram
        create_3d_histogram(results)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("SUMMARY")
        logger.info("="*50)
        logger.info(f"Total runs processed: {len(results)}")
        
        # Group by lag parameters and show statistics
        df = pd.DataFrame(results)
        param_groups = df.groupby(['lag_subsample_rate', 'total_lag_time'])['validation_scaled_rmse']
        
        logger.info("\nValidation Scaled RMSE by lag parameters:")
        for (lag_subsample, lag_time), group in param_groups:
            mean_scaled_rmse = group.mean()
            std_scaled_rmse = group.std() if len(group) > 1 else 0
            logger.info(f"lag_subsample={lag_subsample}, lag_time={lag_time}: Scaled RMSE={mean_scaled_rmse:.6f} ± {std_scaled_rmse:.6f} (n={len(group)})")
        
        # Find best performing combination
        best_result = min(results, key=lambda x: x['validation_scaled_rmse'])
        logger.info(f"\nBest performing combination:")
        logger.info(f"Run: {best_result['run_name']}")
        logger.info(f"Lag subsample rate: {best_result['lag_subsample_rate']}")
        logger.info(f"Total lag time: {best_result['total_lag_time']}")
        logger.info(f"Validation Scaled RMSE: {best_result['validation_scaled_rmse']:.6f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
