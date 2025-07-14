#!/usr/bin/env python3
"""
Script to run the three ALA_ALA denoiser experiments.

Experiment Setup:
=================

1. Model 1: Denoiser with SelfConditioner, 2 structures, noise level sigma=0.04
   - Uses real lagged frames from trajectory as hidden states
   - SelfConditioner just repeats the current position
   
2. Model 2: Denoiser with SelfConditioner, 2 structures, noise level sigma/sqrt(2)≈0.0283
   - Same as Model 1 but with reduced noise level
   - Uses real lagged frames from trajectory as hidden states
   
3. Model 3: Denoiser with PositionConditioner, 2 structures, noise level sigma=0.04
   - Hidden states are repeated copies of current position (not real trajectory frames)
   - PositionConditioner aligns these copies to current position
   - Noise is added by the denoiser during training

Usage:
======
python run_denoiser_experiments.py [model_number]

Where model_number is 1, 2, or 3. If no number is provided, all models will be run.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_experiment(model_num: int, root_path: str = "/data2/sules/jamun-denoiser-experiments"):
    """Run a specific experiment model."""
    config_name = f"ala_ala_denoiser_experiment_model{model_num}"
    
    # Map model numbers to descriptions
    descriptions = {
        1: "Model 1: SelfConditioner, sigma=0.04",
        2: "Model 2: SelfConditioner, sigma/sqrt(2)≈0.0283", 
        3: "Model 3: PositionConditioner with repeated position copies, sigma=0.04"
    }
    
    print(f"\n{'='*60}")
    print(f"Starting {descriptions[model_num]}")
    print(f"Config: {config_name}")
    print(f"Output path: {root_path}/model{model_num}")
    print(f"{'='*60}\n")
    
    cmd = [
        "python", "jamun_train.py",
        "--config-dir=configs",
        f"experiment={config_name}",
        f"++paths.root_path={root_path}/model{model_num}",
        "++trainer.max_epochs=500",
        "++trainer.log_every_n_steps=10"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print(f"\n✅ Model {model_num} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Model {model_num} failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Model {model_num} interrupted by user")
        return False

def main():
    """Main function to run experiments."""
    print(__doc__)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        try:
            model_num = int(sys.argv[1])
            if model_num not in [1, 2, 3]:
                raise ValueError()
            models_to_run = [model_num]
        except ValueError:
            print("Error: Please provide a valid model number (1, 2, or 3)")
            sys.exit(1)
    else:
        models_to_run = [1, 2, 3]
        print("No model specified. Running all three models...")
    
    # Check if we're in the right directory
    if not Path("jamun_train.py").exists():
        print("Error: jamun_train.py not found. Please run this script from the jamun root directory.")
        sys.exit(1)
    
    # Run experiments
    start_time = time.time()
    results = {}
    
    for model_num in models_to_run:
        print(f"\n\nStarting Model {model_num}...")
        results[model_num] = run_experiment(model_num)
        
        if len(models_to_run) > 1 and model_num != models_to_run[-1]:
            print(f"\nWaiting 10 seconds before starting next model...")
            time.sleep(10)
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\n\n{'='*60}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {elapsed/3600:.2f} hours")
    print()
    
    for model_num in models_to_run:
        status = "✅ SUCCESS" if results[model_num] else "❌ FAILED"
        print(f"Model {model_num}: {status}")
    
    print(f"\nResults saved to: /data2/sules/jamun-denoiser-experiments/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 