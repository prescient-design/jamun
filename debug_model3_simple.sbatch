#!/usr/bin/env bash

#SBATCH --job-name=debug_model3
#SBATCH --array=1-2
#SBATCH --output=logs/debug_model3_%A_%a.out
#SBATCH --error=logs/debug_model3_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate jamun

set -eux

echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "hostname = $(hostname)"

export HYDRA_FULL_ERROR=1

cd /homefs/home/sules/jamun

# Test basic command first
echo "Testing jamun_train command..."
jamun_train --help

echo "Testing config validation..."
jamun_train --config-dir=src/jamun/hydra_config \
    experiment=ala_ala_denoiser_experiment_model3 \
    --cfg job

# Now test with minimal overrides
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    echo "Testing with betas=[0.9,0.9]"
    jamun_train --config-dir=src/jamun/hydra_config \
        experiment=ala_ala_denoiser_experiment_model3 \
        trainer.max_epochs=1 \
        data.datamodule.datasets.train.num_frames=10 \
        'model.optim.betas=[0.9,0.9]' \
        --dry-run
else
    echo "Testing with betas=[0.9,0.999]"
    jamun_train --config-dir=src/jamun/hydra_config \
        experiment=ala_ala_denoiser_experiment_model3 \
        trainer.max_epochs=1 \
        data.datamodule.datasets.train.num_frames=10 \
        'model.optim.betas=[0.9,0.999]' \
        --dry-run
fi

echo "Debug test completed for task $SLURM_ARRAY_TASK_ID" 