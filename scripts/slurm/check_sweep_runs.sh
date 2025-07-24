#!/bin/bash

# Helper script to check how many runs are in the enhanced sampling sweep
# and preview the array range for the SLURM script

set -e

# Configuration
WANDB_GROUP="fake_enhanced_data_jul_11_sweep"
ENTITY="sule-shashank"
PROJECT="jamun"

echo "========================================="
echo "Enhanced Sampling Sweep - Run Check"
echo "Group: $WANDB_GROUP"
echo "Entity/Project: $ENTITY/$PROJECT"
echo "========================================="

# Initialize conda 
echo "Initializing conda..."
source ~/.bashrc
eval "$(conda shell.bash hook)"

# Activate conda environment
echo "Activating jamun environment..."
conda activate jamun

# Python script to fetch wandb runs and show summary
python -c "
import wandb
import sys
from collections import defaultdict

# Configuration
entity = '$ENTITY'
project = '$PROJECT'
group = '$WANDB_GROUP'

try:
    # Initialize wandb API
    api = wandb.Api()
    
    # Get all runs from the specified group
    print(f'Fetching runs from {entity}/{project} with group \"{group}\"...')
    runs = api.runs(f'{entity}/{project}', filters={'group': group})
    runs_list = list(runs)
    
    print(f'\\nFound {len(runs_list)} runs in group \"{group}\"')
    
    if len(runs_list) == 0:
        print('No runs found in this group!')
        sys.exit(1)
    
    # Collect parameter combinations
    param_combinations = []
    conditioner_counts = defaultdict(int)
    sigma_counts = defaultdict(int)
    lag_time_counts = defaultdict(int)
    
    for i, run in enumerate(runs_list):
        try:
            config = run.config
            cfg = config.get('cfg', {})
            
            conditioner = cfg.get('model', {}).get('conditioner', {}).get('_target_', 'Unknown')
            sigma = cfg.get('model', {}).get('sigma_distribution', {}).get('sigma', 'Unknown')
            total_lag_time = cfg.get('data', {}).get('datamodule', {}).get('datasets', {}).get('train', {}).get('total_lag_time', 'Unknown')
            
            conditioner_name = conditioner.split('.')[-1] if conditioner != 'Unknown' else 'Unknown'
            
            param_combinations.append({
                'index': i,
                'name': run.name,
                'run_path': '/'.join(run.path),
                'conditioner': conditioner_name,
                'sigma': sigma,
                'lag_time': total_lag_time,
                'state': run.state
            })
            
            conditioner_counts[conditioner_name] += 1
            sigma_counts[sigma] += 1
            lag_time_counts[total_lag_time] += 1
            
        except Exception as e:
            print(f'Warning: Could not extract parameters for run {i}: {e}')
            param_combinations.append({
                'index': i,
                'name': run.name,
                'run_path': '/'.join(run.path),
                'conditioner': 'Error',
                'sigma': 'Error',
                'lag_time': 'Error',
                'state': run.state
            })
    
    # Print summary
    print('\\n========================================')
    print('PARAMETER DISTRIBUTION SUMMARY:')
    print('========================================')
    
    print('\\nConditioner types:')
    for conditioner, count in sorted(conditioner_counts.items()):
        print(f'  {conditioner}: {count} runs')
    
    print('\\nSigma values:')
    for sigma, count in sorted(sigma_counts.items()):
        print(f'  {sigma}: {count} runs')
    
    print('\\nLag time values:')
    for lag_time, count in sorted(lag_time_counts.items()):
        print(f'  {lag_time}: {count} runs')
    
    # Print first 5 runs as examples
    print('\\n========================================')
    print('FIRST 5 RUNS (EXAMPLES):')
    print('========================================')
    print(f'{'Index':<6} {'Name':<25} {'Conditioner':<18} {'Sigma':<8} {'Lag':<5} {'State':<10}')
    print('-' * 75)
    
    for combo in param_combinations[:5]:
        print(f'{combo[\"index\"]:<6} {combo[\"name\"]:<25} {combo[\"conditioner\"]:<18} {combo[\"sigma\"]:<8} {combo[\"lag_time\"]:<5} {combo[\"state\"]:<10}')
    
    if len(param_combinations) > 5:
        print(f'... and {len(param_combinations) - 5} more runs')
    
    print('\\n========================================')
    print('SLURM ARRAY CONFIGURATION:')
    print('========================================')
    print(f'Total runs: {len(runs_list)}')
    print(f'Array range: 0-{len(runs_list) - 1}')
    print(f'\\nUpdate your SLURM script with:')
    print(f'#SBATCH --array=0-{len(runs_list) - 1}')
    print('\\nTo submit the job:')
    print('sbatch scripts/slurm/sweep_enhanced_sampling.sh')
    print('\\nTo submit a subset (e.g., first 5 runs):')
    print('sbatch --array=0-4 scripts/slurm/sweep_enhanced_sampling.sh')
    
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo "========================================="
echo "Run check completed!"
echo "=========================================" 