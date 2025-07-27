#!/bin/bash

#SBATCH --job-name=conditioner_lag_sweep
#SBATCH --array=0-5
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=1-0

# Experiment: Sweep over SelfConditioner, PositionConditioner, and DenoisedConditioner
# with different total lag times (2, 5, 8) on enhanced sampling data with 2 layers
# Testing mode: 1 epoch, max_datasets=1

# Array of config names to run (base configs)
CONFIG_NAMES=(
    "train_enhanced_self_conditioner"
    "train_enhanced_position_conditioner" 
    "train_enhanced_denoised_conditioner"
)

# Array of conditioner names for logging
CONDITIONER_NAMES=(
    "SelfConditioner"
    "PositionConditioner"
    "DenoisedConditioner"
)

# Array of lag times to test
LAG_TIMES=(2 5)

# Calculate which conditioner and lag time based on array index
# Array index 0-5 maps to:
# 0-1: SelfConditioner with lag_time 2,5
# 2-3: PositionConditioner with lag_time 2,5  
# 4-5: DenoisedConditioner with lag_time 2,5
CONDITIONER_IDX=$((SLURM_ARRAY_TASK_ID / 2))
LAG_TIME_IDX=$((SLURM_ARRAY_TASK_ID % 2))

CONFIG_NAME=${CONFIG_NAMES[$CONDITIONER_IDX]}
CONDITIONER_NAME=${CONDITIONER_NAMES[$CONDITIONER_IDX]}
LAG_TIME=${LAG_TIMES[$LAG_TIME_IDX]}

echo "=== SLURM Array Job ${SLURM_ARRAY_TASK_ID}: Training ${CONDITIONER_NAME} with lag_time=${LAG_TIME} ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Conditioner Index: ${CONDITIONER_IDX}"
echo "Lag Time Index: ${LAG_TIME_IDX}"
echo "Config: ${CONFIG_NAME}"
echo "Lag Time: ${LAG_TIME}"
echo "Starting at $(date)"
echo ""

# Set environment variables
export JAMUN_DATA_PATH=/data/bucket/kleinhej/
export WANDB_PROJECT=jamun

# Activate conda environment
source ~/.bashrc
conda activate jamun

# Create logs directory if it doesn't exist
mkdir -p logs

# Build command with testing overrides and lag time sweep
CMD="jamun_train --config-dir=configs experiment=${CONFIG_NAME}"
CMD="${CMD} ++data.datamodule.datasets.train.total_lag_time=${LAG_TIME}"
CMD="${CMD} ++data.datamodule.datasets.val.total_lag_time=${LAG_TIME}"
CMD="${CMD} ++trainer.max_epochs=100"

echo "Running command:"
echo "${CMD}"
echo ""

eval ${CMD}

echo ""
echo "=== ${CONDITIONER_NAME} (lag_time=${LAG_TIME}) Training Complete ==="
echo "Finished at $(date)"
echo "Check Weights & Biases group 'conditioner_lag_sweep_test' for results" 