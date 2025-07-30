#!/usr/bin/env bash

#SBATCH --job-name=model3_beta_simple
#SBATCH --array=1-2
#SBATCH --output=logs/model3_simple_%A_%a.out
#SBATCH --error=logs/model3_simple_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

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
    jamun_train --config-dir=src/jamun/hydra_config \
        experiment=ala_ala_denoiser_experiment_model3 \
        ++run_key=$RUN_KEY \
        'model.optim.betas=[0.9,0.9]' \
        ++logger.wandb.name="model3_beta09_09"
else
    # Second run: Adam betas (0.9, 0.999) - PyTorch default
    echo "Running model3 experiment with Adam betas (0.9, 0.999)"
    jamun_train --config-dir=src/jamun/hydra_config \
        experiment=ala_ala_denoiser_experiment_model3 \
        ++run_key=$RUN_KEY \
        'model.optim.betas=[0.9,0.999]' \
        ++logger.wandb.name="model3_beta09_0999"
fi

echo "Model3 beta comparison experiment $SLURM_ARRAY_TASK_ID completed" 