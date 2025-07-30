#!/bin/bash
#SBATCH --job-name=model3_beta_comparison
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # Number of agents to run in parallel on this node
#SBATCH --gpus-per-node=1   # Assign one GPU to each agent
#SBATCH --cpus-per-task=12
#SBATCH --time=1-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array=1-2

# Create logs directory if it doesn't exist
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate jamun

set -eux

echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "hostname = $(hostname)"

export HYDRA_FULL_ERROR=1

# Generate unique run key for this experiment
RUN_KEY=$(openssl rand -hex 12)
echo "RUN_KEY = ${RUN_KEY}"

nvidia-smi

# Navigate to project directory
cd /homefs/home/sules/jamun

# Determine which beta configuration to run based on array task ID
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    # First run: Adam betas (0.9, 0.9)
    echo "Running model3 experiment with Adam betas (0.9, 0.9)"
    jamun_train --config-dir=configs \
        experiment=ala_ala_denoiser_experiment_model3 \
        ++run_key=$RUN_KEY \
        '++model.optim.betas=[0.9,0.9]' \
        ++logger.wandb.name="model3_beta09_09"
else
    # Second run: Adam betas (0.9, 0.999) - PyTorch default
    echo "Running model3 experiment with Adam betas (0.9, 0.999)"
    jamun_train --config-dir=configs \
        experiment=ala_ala_denoiser_experiment_model3 \
        ++run_key=$RUN_KEY \
        '++model.optim.betas=[0.9,0.999]' \
        ++logger.wandb.name="model3_beta09_0999"
fi

echo "Model3 beta comparison experiment $SLURM_ARRAY_TASK_ID completed" 