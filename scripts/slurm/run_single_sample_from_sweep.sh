#!/bin/bash

# This script runs sampling for a specific run from a wandb sweep, selected by an index.

set -e

# --- Configuration ---
SWEEP_ID="sule-shashank/jamun/evgtrff4"

# --- Argument Parsing ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <RUN_INDEX>"
    echo "Please provide the 0-based index of the run to process."
    exit 1
fi
RUN_INDEX=$1

# --- Main Logic ---
echo "Fetching run at index $RUN_INDEX from sweep $SWEEP_ID..."

python -c "
import wandb
import subprocess
import sys

# The sweep ID and run index are passed as command-line arguments
if len(sys.argv) < 3:
    print('Usage: python_script.py <SWEEP_ID> <RUN_INDEX>', file=sys.stderr)
    sys.exit(1)
sweep_id = sys.argv[1]
run_index = int(sys.argv[2])

try:
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    # wandb runs are often ordered from newest to oldest; reverse to make index stable
    runs_list = list(reversed(list(sweep.runs)))
    
    if run_index >= len(runs_list) or run_index < 0:
        print(f'Error: Run index {run_index} is out of bounds. The sweep has {len(runs_list)} runs (indices 0 to {len(runs_list) - 1}).', file=sys.stderr)
        sys.exit(1)
    
    run = runs_list[run_index]
    run_path = '/'.join(run.path)
    conditioner = run.config.get('cfg', {}).get('model', {}).get('conditioner', {}).get('_target_')
    total_lag_time = run.config.get('cfg', {}).get('data', {}).get('datamodule', {}).get('datasets', {}).get('train', {}).get('total_lag_time')
    sigma = run.config.get('cfg', {}).get('model', {}).get('sigma_distribution', {}).get('sigma')
    
    # Ensure all required parameters are present
    if not all(v is not None for v in [run_path, conditioner, total_lag_time, sigma]):
        print(f'Error: Could not extract all required parameters for run at index {run_index}.', file=sys.stderr)
        sys.exit(1)
    
    print('========================================')
    print(f'Starting sampling for run at index {run_index}: {run_path}')
    print(f'  Conditioner: {conditioner}')
    print(f'  Total Lag Time: {total_lag_time}')
    print(f'  Sigma: {sigma}')
    print('========================================')
    
    # Execute jamun_sample with the extracted parameters
    tags_string = '[' + f'\"{str(conditioner)}\", \"{str(total_lag_time)}\", \"{str(sigma)}\"' + ']'
    cmd = [
        'jamun_sample',
        '--config-dir=configs',
        'experiment=sample_capped_single_shape_conditioning.yaml',
        f'wandb_train_run_path={run_path}',
        f'++init_datasets.total_lag_time={total_lag_time}',
        f'++sigma={sigma}',
        f'++delta={sigma}',
        f'++logger.wandb.group=sampling_from_sweep_{run_index}',
        f'++logger.wandb.tags={tags_string}'
    ]
    
    result = subprocess.run(cmd, check=True)
    
    print('----------------------------------------')
    print(f'Finished sampling for run index {run_index}.')
    
except subprocess.CalledProcessError as e:
    print(f'Error running jamun_sample: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'Error fetching data from wandb: {e}', file=sys.stderr)
    sys.exit(1)
" "$SWEEP_ID" "$RUN_INDEX" 