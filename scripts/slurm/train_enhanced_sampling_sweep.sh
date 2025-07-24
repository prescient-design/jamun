#!/usr/bin/env bash

#SBATCH --partition gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --time 3-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-2

# Initialize conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate jamun

# Verify conda activation worked
which python
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

set -eux

echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "hostname = $(hostname)"

export HYDRA_FULL_ERROR=1

# NOTE: We generate this in submit script instead of using time-based default to ensure consistency across ranks.
RUN_KEY=$(openssl rand -hex 12)
echo "RUN_KEY = ${RUN_KEY}"

# Define parameter arrays
CONDITIONERS=("jamun.model.conditioners.PositionConditioner")
CONDITIONER_NAMES=("PositionConditioner")
LAG_SUBSAMPLE_RATES=(1 5 10)

# Fixed parameters
SIGMA=0.04
TOTAL_LAG_TIME=5
SUBSAMPLE=1

# Calculate parameter indices from SLURM_ARRAY_TASK_ID
# Total combinations: 1 conditioner * 3 lag_subsample_rates = 3
LAG_SUBSAMPLE_IDX=${SLURM_ARRAY_TASK_ID}

# Get parameter values
CONDITIONER=${CONDITIONERS[0]}
CONDITIONER_NAME=${CONDITIONER_NAMES[0]}
LAG_SUBSAMPLE_RATE=${LAG_SUBSAMPLE_RATES[$LAG_SUBSAMPLE_IDX]}

echo "Parameter combination ${SLURM_ARRAY_TASK_ID}:"
echo "  Conditioner: ${CONDITIONER_NAME}"
echo "  Sigma: ${SIGMA}"
echo "  Total lag time: ${TOTAL_LAG_TIME}"
echo "  Lag subsample rate: ${LAG_SUBSAMPLE_RATE}"
echo "  Subsample: ${SUBSAMPLE}"

nvidia-smi

# Run training with parameter overrides
jamun_train --config-dir=configs \
  experiment=train_test_single_shape_enhanced_sampling.yaml \
  ++trainer.max_epochs=100 \
  ++data.datamodule.datasets.train.subsample=${SUBSAMPLE} \
  ++data.datamodule.datasets.val.subsample=${SUBSAMPLE} \
  ++data.datamodule.datasets.test.subsample=${SUBSAMPLE} \
  ++model.conditioner._target_=${CONDITIONER} \
  ++model.sigma_distribution.sigma=${SIGMA} \
  ++data.datamodule.datasets.train.total_lag_time=${TOTAL_LAG_TIME} \
  ++data.datamodule.datasets.train.lag_subsample_rate=${LAG_SUBSAMPLE_RATE} \
  ++model.arch.N_structures=${TOTAL_LAG_TIME} \
  ++logger.wandb.group="fake_enhanced_data_jul17_sweep_lag_times" \
  ++logger.wandb.tags=["'${SLURM_JOB_ID}'","'${RUN_KEY}'","train","enhanced_sampling","${CONDITIONER_NAME}","sigma_${SIGMA}","lag_${TOTAL_LAG_TIME}","lag_subsample_${LAG_SUBSAMPLE_RATE}"] \
  ++run_key=$RUN_KEY 