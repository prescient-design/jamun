#!/bin/bash

# Debug version of the enhanced sampling sweep script
# Usage: ./debug_sweep_enhanced_sampling.sh <RUN_INDEX>

set -e

# Check if run index is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <RUN_INDEX>"
    echo "Please provide the 0-based index of the run to process."
    exit 1
fi

RUN_INDEX=$1

# Set up environment
export JAMUN_ROOT_PATH=/homefs/home/sules/jamun
cd $JAMUN_ROOT_PATH

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

echo "========================================="
echo "DEBUG MODE - Enhanced Sampling Sweep"
echo "Working directory: $(pwd)"
echo "Processing run index: $RUN_INDEX from group: $WANDB_GROUP"
echo "========================================="

# Python script to fetch wandb runs and build jamun_sample command
python -c "
import wandb
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
    print(f'\\nAvailable config keys: {list(config.keys())}')
    
    cfg_key = 'cfg'  # This is the key used in jamun configs
    
    if cfg_key not in config:
        print(f'Error: Config key \"{cfg_key}\" not found in run config.', file=sys.stderr)
        sys.exit(1)
    
    cfg = config[cfg_key]
    print(f'Config structure keys: {list(cfg.keys())}')
    
    # Extract required parameters with detailed debugging
    try:
        print(f'\\nExtracting parameters...')
        
        # Extract conditioner
        conditioner = cfg['model']['conditioner']['_target_']
        print(f'  ✓ Conditioner: {conditioner}')
        
        # Extract sigma
        sigma = cfg['model']['sigma_distribution']['sigma']
        print(f'  ✓ Sigma: {sigma}')
        
        # Extract total_lag_time
        total_lag_time = cfg['data']['datamodule']['datasets']['train']['total_lag_time']
        print(f'  ✓ Total Lag Time: {total_lag_time}')
        
        # Optional: Extract other useful parameters
        model_arch_N_structures = cfg.get('model', {}).get('arch', {}).get('N_structures', total_lag_time)
        print(f'  ✓ Model N_structures: {model_arch_N_structures}')
        
    except KeyError as e:
        print(f'Error: Could not extract required parameter: {e}', file=sys.stderr)
        print(f'Available model keys: {cfg.get(\"model\", {}).keys()}', file=sys.stderr)
        if 'model' in cfg:
            print(f'Available model.conditioner keys: {cfg[\"model\"].get(\"conditioner\", {}).keys()}', file=sys.stderr)
            print(f'Available model.sigma_distribution keys: {cfg[\"model\"].get(\"sigma_distribution\", {}).keys()}', file=sys.stderr)
        if 'data' in cfg:
            print(f'Available data keys: {cfg[\"data\"].keys()}', file=sys.stderr)
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
    print(f'Parameters extracted successfully!')
    print(f'  Run Path: {run_path}')
    print(f'  Conditioner: {conditioner}')
    print(f'  Conditioner Name: {conditioner_name}')
    print(f'  Sigma: {sigma}')
    print(f'  Total Lag Time: {total_lag_time}')
    print(f'  Sample Group: {sampling_group}')
    print('========================================\\n')
    
    # Build the jamun_sample command
    # Note: We don't override model.conditioner._target_ because the checkpoint 
    # already contains the correct conditioner configuration with all parameters
    cmd_parts = [
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
    
    print('DEBUG: Generated jamun_sample command:')
    print('=' * 80)
    cmd_string = ' \\\\\\n    '.join(cmd_parts)
    print(cmd_string)
    print('=' * 80)
    
    print('\\nDEBUG: Single line command:')
    print('=' * 80)
    print(' '.join(cmd_parts))
    print('=' * 80)
    
    print(f'\\nDEBUG: Successfully processed run index {run_index}')
    
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo "========================================="
echo "DEBUG: Finished processing run index: $RUN_INDEX"
echo "=========================================" 