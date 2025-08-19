#!/usr/bin/env python3
"""
Script to analyze validation errors for Denoiser models from WandB runs.

This script:
1. Scrapes all runs from the WandB group "noise_check_experiment_multimeasurement_vs_correlation"
2. Filters runs by model target "jamun.model.Denoiser"
3. Loads the models and computes validation errors
4. Plots validation errors from 2 to 10 (assuming this refers to some parameter range)
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from pathlib import Path
from typing import List, Dict, Any, Optional
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

def filter_denoiser_runs(runs: List[wandb.Api.run]) -> List[Dict[str, Any]]:
    """Filter runs by specific criteria for spatiotemporal multimeasurement analysis."""
    logger.info("Filtering runs by specific criteria:")
    logger.info("- cfg.model._target_ = 'jamun.model.denoiser_conditional.Denoiser'")
    logger.info("- cfg.data.datamodule.datasets.train.subsample = 1")
    logger.info("- cfg.data.datamodule.datasets.train._target_ = 'jamun.data.parse_repeated_position_datasets_from_directory'")
    
    denoiser_runs = []
    
    for run in tqdm(runs, desc="Filtering runs", unit="run"):
        try:
            config = run.config
            if 'cfg' not in config:
                logger.warning(f"Run {run.name} missing 'cfg' in config")
                continue
            
            cfg = config['cfg']
            
            # Check model target
            model_target = cfg.get('model', {}).get('_target_')
            if model_target != 'jamun.model.denoiser_conditional.Denoiser':
                continue
            
            # Check subsample = 1
            try:
                subsample = cfg['data']['datamodule']['datasets']['train']['subsample']
                if subsample != 1:
                    continue
            except (KeyError, TypeError):
                logger.warning(f"Could not extract subsample for run {run.name}")
                continue
            
            # Check data target
            try:
                data_target = cfg['data']['datamodule']['datasets']['train']['_target_']
                if data_target != 'jamun.data.parse_repeated_position_datasets_from_directory':
                    continue
            except (KeyError, TypeError):
                logger.warning(f"Could not extract data target for run {run.name}")
                continue
            
            # If we get here, all criteria are met
            # Extract additional parameters that might be useful for plotting
            run_info = {
                'run': run,
                'run_path': '/'.join(run.path),
                'run_name': run.name,
                'model_target': model_target,
                'data_target': data_target,
                'subsample': subsample,
                'cfg': cfg
            }
            
            # Try to extract parameters that might be varied (for plotting from 2 to 10)
            sigma = cfg.get('model', {}).get('sigma_distribution', {}).get('sigma')
            if sigma is not None:
                run_info['sigma'] = sigma
            
            # Extract total_lag_time specifically for data loading
            total_lag_time = None
            try:
                # Try the specific path you mentioned
                total_lag_time = cfg['data']['datamodule']['datasets']['train']['total_lag_time']
                run_info['total_lag_time'] = total_lag_time
            except (KeyError, TypeError):
                try:
                    # Fallback to the general datasets path
                    total_lag_time = cfg['data']['datamodule']['datasets']['total_lag_time']
                    run_info['total_lag_time'] = total_lag_time
                except (KeyError, TypeError):
                    logger.warning(f"Could not extract total_lag_time for run {run.name}")
                    logger.warning(f"Available config keys: {list(cfg.get('data', {}).get('datamodule', {}).get('datasets', {}).keys())}")
            
            # Look for other potential varying parameters
            for param_path in [
                ['model', 'arch', 'num_layers'],
                ['model', 'arch', 'hidden_dim'],
                ['data', 'datamodule', 'batch_size']
            ]:
                value = cfg
                for key in param_path:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = None
                        break
                if value is not None:
                    param_name = '_'.join(param_path)
                    run_info[param_name] = value
            
            denoiser_runs.append(run_info)
            logger.info(f"Added run: {run.name} (sigma={sigma}, total_lag_time={total_lag_time})")
            
        except Exception as e:
            logger.warning(f"Error processing run {run.name}: {e}")
            continue
    
    logger.info(f"Found {len(denoiser_runs)} Denoiser runs")
    return denoiser_runs

def load_validation_data(total_lag_time: int, val_root: str = "/data2/sules/ALA_ALA_enhanced_full_grid/val/", num_frames: int = 100) -> torch_geometric.loader.DataLoader:
    """Load validation data for error computation."""
    logger.info(f"Loading validation data from: {val_root} with total_lag_time={total_lag_time}")
    
    if not os.path.exists(val_root):
        raise ValueError(f"Validation directory not found: {val_root}")
    
    try:
        datasets = parse_repeated_position_datasets_from_directory(
            root=val_root,
            traj_pattern="^(.*).xtc",
            pdb_pattern="^(.*).pdb",
            filter_codes=None,  # Don't filter by codes, use all available data
            as_iterable=False,
            subsample=1,
            total_lag_time=total_lag_time,
            lag_subsample_rate=1,
            max_datasets=5  # Use a few datasets for validation
        )
        
        if not datasets:
            raise ValueError("No validation datasets found")
        
        # Create data module
        data_module = MDtrajDataModule(
            datasets={'val': datasets},
            batch_size=32,
            num_workers=0,  # Use 0 for debugging
            persistent_workers=False
        )
        data_module.setup('val')
        
        val_loader = data_module.val_dataloader()
        logger.info(f"Loaded validation data with {len(datasets)} datasets")
        
        return val_loader
        
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        raise

def load_model_and_compute_rmsd(run_info: Dict[str, Any], val_loader: torch_geometric.loader.DataLoader) -> float:
    """Load a model from checkpoint and compute validation RMSD²."""
    run_path = run_info['run_path']
    run_name = run_info['run_name']
    
    logger.info(f"Loading model for run: {run_name}")
    
    try:
        # Find checkpoint
        checkpoint_path = find_checkpoint(
            wandb_train_run_path=run_path,
            checkpoint_type="best_so_far"
        )
        
        # Load model
        model = Denoiser.load_from_checkpoint(checkpoint_path, strict=False)
        model.eval()
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info(f"Model loaded successfully for run: {run_name}")
        
        # Compute validation squared RMSD
        total_rmsd_squared = 0.0
        total_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Computing validation", leave=False, unit="batch"):
                batch = batch.to(device)
                
                # Use the model's validation logic
                sigma = model.sigma_distribution.sample().to(device)
                loss, aux = model.noise_and_compute_loss(
                    batch.pos, 
                    batch, 
                    batch.batch, 
                    batch.num_graphs,
                    sigma,
                    align_noisy_input=model.align_noisy_input_during_evaluation
                )
                
                # Extract RMSD from aux dictionary and square it
                if 'rmsd' in aux:
                    rmsd_value = aux['rmsd'].mean().item()
                    total_rmsd_squared += rmsd_value ** 2  # Square the RMSD
                else:
                    logger.warning(f"RMSD not found in aux dictionary for {run_name}. Available keys: {list(aux.keys())}")
                    # Fallback to loss if RMSD not available
                    total_rmsd_squared += loss.mean().item()
                
                total_batches += 1
        print(f"total_rmsd_squared: {total_rmsd_squared}")
        print(f"total_batches: {total_batches}")
        breakpoint()
        avg_rmsd_squared = total_rmsd_squared / total_batches if total_batches > 0 else float('inf')
        logger.info(f"Validation RMSD² for {run_name}: {avg_rmsd_squared:.6f}")
        
        return avg_rmsd_squared
        
    except Exception as e:
        logger.error(f"Error computing validation error for run {run_name}: {e}")
        return float('inf')

def plot_validation_errors(results: List[Dict[str, Any]], output_path: str = "validation_errors_plot.png"):
    """Plot validation RMSD² values."""
    logger.info("Creating validation RMSD² plot...")
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Debug: Print available columns
    logger.info(f"Available DataFrame columns: {list(df.columns)}")
    
    # Determine the parameter to plot on x-axis (2 to 10 range)
    x_param = None
    for param in ['sigma', 'model_arch_num_layers', 'model_arch_hidden_dim']:
        if param in df.columns:
            values = df[param].dropna()
            if len(values) > 1 and values.min() >= 2 and values.max() <= 10:
                x_param = param
                break
    
    if x_param is None:
        # If no parameter in 2-10 range, just use indices
        logger.warning("No parameter found in range 2-10, using run indices")
        df['index'] = range(len(df))
        x_param = 'index'
    
    # Sort by the x parameter
    df = df.sort_values(x_param)
    
    # Find the correct validation column name for plotting
    validation_col = None
    for col in ['validation_rmsd_squared', 'validation_rmsd', 'validation_error']:
        if col in df.columns:
            validation_col = col
            break
    
    if validation_col is None:
        logger.error(f"No validation column found for plotting. Available columns: {list(df.columns)}")
        return
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.scatter(df[x_param], df[validation_col], alpha=0.7, s=60)
    
    # Add labels for each point
    for i, row in df.iterrows():
        plt.annotate(row['run_name'][:8], 
                    (row[x_param], row[validation_col]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    plt.xlabel(x_param.replace('_', ' ').title())
    plt.ylabel('Validation RMSD²')
    plt.title('Validation RMSD² for Spatiotemporal Multimeasurement Models\nGroup: noise_check_experiment_multimeasurement_vs_correlation')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis limits to 2-10 if appropriate
    if x_param != 'index' and df[x_param].min() >= 2 and df[x_param].max() <= 10:
        plt.xlim(2, 10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to: {output_path}")
    
    # Also save data to CSV
    csv_path = output_path.replace('.png', '.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Data saved to: {csv_path}")
    
    # Save validation RMSD² paired with total_lag_time as npy file
    # Find the correct validation column name
    validation_col = None
    for col in ['validation_rmsd_squared', 'validation_rmsd', 'validation_error']:
        if col in df.columns:
            validation_col = col
            break
    
    if validation_col is None:
        logger.error(f"No validation column found in DataFrame. Available columns: {list(df.columns)}")
        return
    
    if 'total_lag_time' in df.columns:
        # Create structured array with total_lag_time and validation values
        data = np.array(list(zip(df['total_lag_time'].values, df[validation_col].values)),
                       dtype=[('total_lag_time', 'i4'), ('validation_rmsd_squared', 'f8')])
        npy_path = "validation_errors_spatiotemporal_multimeasurement.npy"
        np.save(npy_path, data)
        logger.info(f"Validation data with total_lag_time saved to: {npy_path}")
        logger.info(f"Data format: structured array with fields 'total_lag_time' and 'validation_rmsd_squared'")
        logger.info(f"Used validation column: {validation_col}")
    else:
        # Fallback to just validation values if total_lag_time not available
        validation_values = df[validation_col].values
        npy_path = "validation_errors_spatiotemporal_multimeasurement.npy"
        np.save(npy_path, validation_values)
        logger.info(f"Validation data saved to: {npy_path}")
        logger.info(f"Used validation column: {validation_col}")
        logger.warning("total_lag_time not available, saved only validation values")

def main():
    """Main function to execute the spatiotemporal multimeasurement analysis."""
    logger.info("Starting spatiotemporal multimeasurement validation analysis...")
    logger.info("Filtering criteria:")
    logger.info("- model target = jamun.model.denoiser_conditional.Denoiser")
    logger.info("- subsample = 1")
    logger.info("- data target = jamun.data.parse_repeated_position_datasets_from_directory")
    
    # Configuration
    group_name = "noise_check_experiment_multimeasurement_vs_correlation"
    project = "sule-shashank/jamun"
    
    try:
        # Step 1: Scrape WandB runs
        runs = scrape_wandb_runs(group_name, project)
        
        # Step 2: Filter by model target
        denoiser_runs = filter_denoiser_runs(runs)
        
        if not denoiser_runs:
            logger.error("No Denoiser runs found in the specified group")
            return
        
        # Step 3: Group runs by total_lag_time and load validation data
        runs_by_lag_time = {}
        for run_info in denoiser_runs:
            lag_time = run_info.get('total_lag_time')
            if lag_time is None:
                logger.warning(f"Skipping run {run_info['run_name']} - no total_lag_time found")
                continue
            if lag_time not in runs_by_lag_time:
                runs_by_lag_time[lag_time] = []
            runs_by_lag_time[lag_time].append(run_info)
        
        logger.info(f"Found runs with {len(runs_by_lag_time)} different total_lag_time values: {list(runs_by_lag_time.keys())}")
        
        # Step 4: Compute validation errors for each model
        results = []
        total_runs = sum(len(runs_for_lag_time) for runs_for_lag_time in runs_by_lag_time.values())
        
        with tqdm(total=total_runs, desc="Processing models", unit="model") as pbar:
            for lag_time, runs_for_lag_time in runs_by_lag_time.items():
                logger.info(f"Loading validation data for total_lag_time={lag_time}")
                val_loader = load_validation_data(total_lag_time=lag_time)
                
                for run_info in runs_for_lag_time:
                    pbar.set_description(f"Processing {run_info['run_name'][:20]}...")
                    
                    validation_rmsd_squared = load_model_and_compute_rmsd(run_info, val_loader)
                    
                    result = {
                        'run_name': run_info['run_name'],
                        'run_path': run_info['run_path'],
                        'validation_rmsd_squared': validation_rmsd_squared,
                        **{k: v for k, v in run_info.items() if k not in ['run', 'cfg']}
                    }
                    results.append(result)
                    pbar.update(1)
        
        # Step 5: Plot results
        plot_validation_errors(results)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("SUMMARY")
        logger.info("="*50)
        logger.info(f"Total runs processed: {len(results)}")
        logger.info("Top 5 best performing models (lowest RMSD²):")
        sorted_results = sorted(results, key=lambda x: x['validation_rmsd_squared'])
        for i, result in enumerate(sorted_results[:5]):
            logger.info(f"{i+1}. {result['run_name']}: {result['validation_rmsd_squared']:.6f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
