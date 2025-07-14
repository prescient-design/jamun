#!/bin/bash

#SBATCH --job-name=noise_check
#SBATCH --partition gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1 # Number of agents to run in parallel on this node
#SBATCH --gpus-per-node=1   # Assign one GPU to each agent
#SBATCH --cpus-per-task=12
#SBATCH --time 1-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array 3-4

# Print job information
echo "Starting job $SLURM_JOB_ID, array task $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Job started at: $(date)"

# Set up environment
source ~/.bashrc
conda activate jamun

# Change to project directory
cd /homefs/home/sules/jamun

# Run training with the corresponding model config
echo "Training with experiment: ala_ala_denoiser_experiment_model${SLURM_ARRAY_TASK_ID}"
jamun_train --config-dir=configs experiment=ala_ala_denoiser_experiment_model${SLURM_ARRAY_TASK_ID}

echo "Job completed at: $(date)" 