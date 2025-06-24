#!/usr/bin/env bash

#SBATCH --partition gpu3
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=4 # Number of agents to run in parallel on this node
#SBATCH --gpus-per-task=1   # Assign one GPU to each agent
#SBATCH --cpus-per-task=8
#SBATCH --time 3-0
#SBATCH --mem-per-cpu=32G

# Check if a Sweep ID is provided as an argument
if [ -z "$1" ]; then
    echo "Error: Please provide the W&B Sweep ID as the first argument."
    echo "Usage: sbatch scripts/slurm/sweep.sh <SWEEP_ID>"
    exit 1
fi

SWEEP_ID=$1

# Set up the environment
eval "$(conda shell.bash hook)"
conda activate jamun

set -eux

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "Running on hostname: $(hostname)"
echo "Starting ${SLURM_NTASKS} agents for sweep: ${SWEEP_ID}"

# Launch multiple wandb agents in parallel using srun.
# Each agent will poll the sweep server, get a configuration, and run one training job.
# PyTorch Lightning will automatically use the single GPU assigned by Slurm to each task.
srun wandb agent "${SWEEP_ID}" 