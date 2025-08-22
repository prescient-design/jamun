#!/usr/bin/env bash

#SBATCH --partition=b200
#SBATCH --job-name=enhanced_long_comparison
 #SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-5

# Initialize conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate jamun

# Verify conda activation worked
which python
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
nvidia-smi

echo "Running array job ${SLURM_ARRAY_TASK_ID}"

# NOTE: We generate this in submit script instead of using time-based default to ensure consistency across ranks.
RUN_KEY=$(openssl rand -hex 12)
echo "RUN_KEY = ${RUN_KEY}"

# Define configurations for each job
case ${SLURM_ARRAY_TASK_ID} in
    0)
        echo "Job 0: Standard JAMUN on enhanced_long, noise 0.04"
        CONFIG="train_enhanced_standard_jamun"
        DATA_PATH="/data2/sules/ALA_ALA_enhanced_long"
        WANDB_GROUP="model_comparison_enhanced_long_take2"
        NOISE_LEVEL="0.04"
        RUN_NAME="enhanced_long_standard_noise0.04"
        ;;
    1)
        echo "Job 1: Spatiotemporal JAMUN on enhanced_long, noise 0.04"
        CONFIG="train_enhanced_spatiotemporal_conditioner"
        DATA_PATH="/data2/sules/ALA_ALA_enhanced_long"
        WANDB_GROUP="model_comparison_enhanced_long_take2"
        NOISE_LEVEL="0.04"
        RUN_NAME="enhanced_long_spatiotemporal_noise0.04"
        ;;
    2)
        echo "Job 2: Standard JAMUN on enhanced_long, noise 0.06"
        CONFIG="train_enhanced_standard_jamun"
        DATA_PATH="/data2/sules/ALA_ALA_enhanced_long"
        WANDB_GROUP="model_comparison_enhanced_long_take2"
        NOISE_LEVEL="0.06"
        RUN_NAME="enhanced_long_standard_noise0.06"
        ;;
    3)
        echo "Job 3: Spatiotemporal JAMUN on enhanced_long, noise 0.06"
        CONFIG="train_enhanced_spatiotemporal_conditioner"
        DATA_PATH="/data2/sules/ALA_ALA_enhanced_long"
        WANDB_GROUP="model_comparison_enhanced_long_take2"
        NOISE_LEVEL="0.06"
        RUN_NAME="enhanced_long_spatiotemporal_noise0.06"
        ;;
    4)
        echo "Job 4: Standard JAMUN on enhanced_long_state_split, noise 0.04"
        CONFIG="train_enhanced_standard_jamun"
        DATA_PATH="/data2/sules/ALA_ALA_enhanced_long_state_split"
        WANDB_GROUP="withheld_state_take2"
        NOISE_LEVEL="0.04"
        RUN_NAME="enhanced_long_state_split_standard_noise0.04"
        ;;
    5)
        echo "Job 5: Spatiotemporal JAMUN on enhanced_long_state_split, noise 0.04"
        CONFIG="train_enhanced_spatiotemporal_conditioner"
        DATA_PATH="/data2/sules/ALA_ALA_enhanced_long_state_split"
        WANDB_GROUP="withheld_state_take2"
        NOISE_LEVEL="0.04"
        RUN_NAME="enhanced_long_state_split_spatiotemporal_noise0.04"
        ;;
    *)
        echo "Unknown job ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

# Build the command with base config
CMD="jamun_train --config-dir=configs experiment=${CONFIG}.yaml"

# Add common training parameters
CMD="$CMD ++data.datamodule.datasets.train.root=${DATA_PATH}/train"
CMD="$CMD ++data.datamodule.datasets.val.root=${DATA_PATH}/val"
CMD="$CMD ++data.datamodule.datasets.train.num_frames=10000"
CMD="$CMD ++data.datamodule.datasets.val.num_frames=10000"
CMD="$CMD ++data.datamodule.datasets.train.lag_subsample_rate=1"
CMD="$CMD ++data.datamodule.datasets.val.lag_subsample_rate=1"
CMD="$CMD ++data.datamodule.datasets.train.subsample=10"
CMD="$CMD ++data.datamodule.datasets.val.subsample=10"
CMD="$CMD ++data.datamodule.datasets.train.total_lag_time=5"
CMD="$CMD ++data.datamodule.datasets.val.total_lag_time=5"

# Add test run parameters (change for full training)
CMD="$CMD ++data.datamodule.datasets.train.max_datasets=200"
CMD="$CMD ++data.datamodule.datasets.val.max_datasets=50"
CMD="$CMD ++trainer.max_epochs=100"
CMD="$CMD ++trainer.val_check_interval=0.2"

# Add noise level
CMD="$CMD ++model.sigma_distribution.sigma=${NOISE_LEVEL}"

# Add wandb configuration
CMD="$CMD ++logger.wandb.group=${WANDB_GROUP}"
CMD="$CMD ++logger.wandb.tags=[${RUN_NAME},enhanced_long_comparison,job_${SLURM_ARRAY_TASK_ID}]"
CMD="$CMD ++run_key=${RUN_KEY}"

# # Add experiment name
CMD="$CMD ++logger.wandb.name=${RUN_NAME}"

echo "Running command: $CMD"
exec $CMD
