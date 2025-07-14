#!/bin/bash
#SBATCH --job-name=sweep_sampling
#SBATCH --partition gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1 # Number of agents to run in parallel on this node
#SBATCH --gpus-per-node=1   # Assign one GPU to each agent
#SBATCH --cpus-per-task=12
#SBATCH --time 1-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array 2-31

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $HOSTNAME"
echo "Starting time: $(date)"

# Activate conda environment (adjust the environment name as needed)
source ~/.bashrc
conda activate jamun

# Change to the working directory
cd /homefs/home/sules/jamun

# Run the sampling script with the array task ID as the run index
echo "Running sampling for sweep run index: $SLURM_ARRAY_TASK_ID"
bash run_single_sample_from_sweep.sh $SLURM_ARRAY_TASK_ID

echo "Finished sampling for run index: $SLURM_ARRAY_TASK_ID"
echo "End time: $(date)" 