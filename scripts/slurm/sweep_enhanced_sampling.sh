#!/bin/bash
#SBATCH --job-name=sweep_enhanced_sampling
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-23  # Adjust this range based on number of runs in your sweep
#SBATCH --output=logs/%A_%a_sweep_enhanced_sampling.log
#SBATCH --error=logs/%A_%a_sweep_enhanced_sampling.err

# Set up environment
set -e
export JAMUN_ROOT_PATH=/homefs/home/sules/jamun
cd $JAMUN_ROOT_PATH

# Create logs directory if it doesn't exist
mkdir -p logs

# Initialize conda 
echo "Initializing conda..."
source ~/.bashrc
eval "$(conda shell.bash hook)"

# Activate conda environment
echo "Activating jamun environment..."
conda activate jamun

# Configuration
WANDB_GROUP="fake_enhanced_data_jul_11_sweep"
ENTITY="sule-shashank"
PROJECT="jamun"
RUN_INDEX=$SLURM_ARRAY_TASK_ID

echo "========================================="
echo "Enhanced Sampling Sweep - Production Run"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Starting time: $(date)"
echo "Processing run index: $RUN_INDEX from group: $WANDB_GROUP"
echo "========================================="

# Python script to fetch wandb runs and execute jamun_sample
python -c "
import wandb
import subprocess
import sys
import os

# Configuration
entity = '$ENTITY'
project = '$PROJECT'
group = '$WANDB_GROUP'
run_index = $RUN_INDEX

try:
    # Initialize wandb API
    api = wandb.Api()
    
    # Get all runs from the specified group
    print(f'Fetching runs from {entity}/{project} with group \"{group}\"...')
    runs = api.runs(f'{entity}/{project}', filters={'group': group})
    runs_list = list(runs)
    
    print(f'Found {len(runs_list)} runs in group \"{group}\"')
    
    # Check if run_index is valid
    if run_index >= len(runs_list) or run_index < 0:
        print(f'Error: Run index {run_index} is out of bounds. The group has {len(runs_list)} runs (indices 0 to {len(runs_list) - 1}).', file=sys.stderr)
        sys.exit(1)
    
    # Get the specific run
    run = runs_list[run_index]
    run_path = '/'.join(run.path)
    
    print(f'\\nProcessing run: {run.name} ({run_path})')
    print(f'Run URL: {run.url}')
    print(f'Run state: {run.state}')
    
    # Extract parameters from the run config
    config = run.config
    cfg_key = 'cfg'  # This is the key used in jamun configs
    
    if cfg_key not in config:
        print(f'Error: Config key \"{cfg_key}\" not found in run config. Available keys: {list(config.keys())}', file=sys.stderr)
        sys.exit(1)
    
    cfg = config[cfg_key]
    
    # Extract required parameters
    try:
        conditioner = cfg['model']['conditioner']['_target_']
        sigma = cfg['model']['sigma_distribution']['sigma']
        total_lag_time = cfg['data']['datamodule']['datasets']['train']['total_lag_time']
        
        print(f'\\nExtracted parameters:')
        print(f'  Conditioner: {conditioner}')
        print(f'  Sigma: {sigma}')
        print(f'  Total Lag Time: {total_lag_time}')
        
    except KeyError as e:
        print(f'Error: Could not extract required parameter: {e}', file=sys.stderr)
        print(f'Available config structure: {cfg.keys()}', file=sys.stderr)
        sys.exit(1)
    
    # Ensure all required parameters are present
    if not all(v is not None for v in [run_path, conditioner, total_lag_time, sigma]):
        print(f'Error: Could not extract all required parameters for run at index {run_index}.', file=sys.stderr)
        sys.exit(1)
    
    # Create a meaningful run group name for sampling
    conditioner_name = conditioner.split('.')[-1]  # Get class name without module path
    sampling_group = 'sample_enhanced_sampling_from_jul_11'
    
    # Create tags for better organization
    tags = [
        f'sweep_run_{run_index}',
        f'conditioner_{conditioner_name}',
        f'sigma_{sigma}',
        f'lag_time_{total_lag_time}',
        'enhanced_sampling',
        'sample_from_sweep'
    ]
    tags_string = '[' + ', '.join(f'\"{tag}\"' for tag in tags) + ']'
    
    print('\\n========================================')
    print(f'Starting sampling for run at index {run_index}: {run_path}')
    print(f'  Conditioner: {conditioner}')
    print(f'  Sigma: {sigma}')
    print(f'  Total Lag Time: {total_lag_time}')
    print(f'  Sample Group: {sampling_group}')
    print('========================================\\n')
    
    # Build the jamun_sample command
    # Note: We don't override model.conditioner._target_ because the checkpoint 
    # already contains the correct conditioner configuration with all parameters
    cmd = [
        'jamun_sample',
        '--config-dir=configs',
        'experiment=sample_enhanced_sampling_single_shape.yaml',
        f'++wandb_train_run_path={run_path}',
        f'++init_datasets.total_lag_time={total_lag_time}',
        f'++sigma={sigma}',
        f'++delta={sigma}',
        f'++logger.wandb.group={sampling_group}',
        f'++logger.wandb.tags={tags_string}',
        f'++logger.wandb.notes=\"Sampling from enhanced sampling sweep run {run_index} - {conditioner_name} sigma={sigma} lag={total_lag_time}\"',
        f'++run_key=sweep_sample_{run_index}_{conditioner_name}_sigma_{sigma}_lag_{total_lag_time}'
    ]
    
    print('Executing command:')
    print(' '.join(cmd))
    print('\\n' + '='*50 + '\\n')
    
    # Execute the command
    result = subprocess.run(cmd, check=True, env=os.environ.copy())
    
    print('\\n' + '='*50)
    print(f'Successfully completed sampling for run index {run_index}')
    print(f'End time: {os.popen(\"date\").read().strip()}')
    
except subprocess.CalledProcessError as e:
    print(f'Error running jamun_sample: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo "========================================="
echo "Finished processing run index: $RUN_INDEX"
echo "End time: $(date)"
echo "=========================================" 