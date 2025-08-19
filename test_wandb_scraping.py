#!/usr/bin/env python3
"""
Test script to check WandB run scraping functionality.
"""

import wandb
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test WandB scraping."""
    group_name = "noise_check_experiment_multimeasurement_vs_correlation"
    project = "sule-shashank/jamun"
    
    logger.info(f"Testing WandB scraping for group: {group_name}")
    
    try:
        api = wandb.Api()
        runs = api.runs(project, filters={'group': group_name})
        runs_list = list(runs)
        
        logger.info(f"Found {len(runs_list)} total runs in group")
        
        denoiser_count = 0
        for i, run in enumerate(runs_list):
            try:
                config = run.config
                if 'cfg' in config:
                    cfg = config['cfg']
                    model_target = cfg.get('model', {}).get('_target_')
                    
                    logger.info(f"Run {i+1}: {run.name}")
                    logger.info(f"  Model target: {model_target}")
                    logger.info(f"  State: {run.state}")
                    
                    if model_target == 'jamun.model.Denoiser':
                        denoiser_count += 1
                        logger.info(f"  ✓ This is a Denoiser run!")
                        
                        # Extract sigma value
                        sigma = cfg.get('model', {}).get('sigma_distribution', {}).get('sigma')
                        if sigma is not None:
                            logger.info(f"  Sigma: {sigma}")
                    
                    logger.info("")
                    
                    if i >= 10:  # Limit to first 10 runs for testing
                        logger.info("... (limiting to first 10 runs for testing)")
                        break
                        
            except Exception as e:
                logger.warning(f"Error processing run {run.name}: {e}")
                continue
        
        logger.info(f"Found {denoiser_count} Denoiser runs out of {min(len(runs_list), 10)} examined")
        
    except Exception as e:
        logger.error(f"Error in WandB scraping: {e}")
        raise

if __name__ == "__main__":
    main()
