#!/usr/bin/env bash

#SBATCH --partition gpu2
#SBATCH --qos=preempt
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --time 1-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0

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
CONDITIONERS=("jamun.model.conditioners.PositionConditioner" "jamun.model.conditioners.SelfConditioner")
CONDITIONER_NAMES=("PositionConditioner" "SelfConditioner")
SIGMAS=(0.01 0.04 0.08 0.1)
LAG_TIMES=(2 5 8)

# Calculate parameter indices from SLURM_ARRAY_TASK_ID
# Total combinations: 2 conditioners * 4 sigmas * 3 lag_times = 24
COND_IDX=$((SLURM_ARRAY_TASK_ID / 12))
SIGMA_IDX=$(((SLURM_ARRAY_TASK_ID % 12) / 3))
LAG_IDX=$((SLURM_ARRAY_TASK_ID % 3))

# Get parameter values
CONDITIONER=${CONDITIONERS[$COND_IDX]}
CONDITIONER_NAME=${CONDITIONER_NAMES[$COND_IDX]}
SIGMA=${SIGMAS[$SIGMA_IDX]}
LAG_TIME=${LAG_TIMES[$LAG_IDX]}

echo "TRIAL RUN - Parameter combination ${SLURM_ARRAY_TASK_ID}:"
echo "  Conditioner: ${CONDITIONER_NAME}"
echo "  Sigma: ${SIGMA}"
echo "  Total lag time: ${LAG_TIME}"

nvidia-smi

# Run training with parameter overrides (reduced epochs for trial)
jamun_train --config-dir=configs \
  experiment=train_test_single_shape_enhanced_sampling.yaml \
  ++trainer.max_epochs=2 \
  ++data.datamodule.datasets.train.subsample=10 \
  ++data.datamodule.datasets.val.subsample=10 \
  ++data.datamodule.datasets.test.subsample=10 \
  ++data.datamodule.datasets.train.max_datasets=5 \
  ++data.datamodule.datasets.val.max_datasets=5 \
  ++data.datamodule.datasets.test.max_datasets=5 \
  ++model.conditioner._target_=${CONDITIONER} \
  ++model.sigma_distribution.sigma=${SIGMA} \
  ++data.datamodule.datasets.train.total_lag_time=${LAG_TIME} \
  ++model.arch.N_structures=${LAG_TIME} \
  ++logger.wandb.group="fake_enhanced_data_trial" \
  ++logger.wandb.tags=["'${SLURM_JOB_ID}'","'${RUN_KEY}'","trial","enhanced_sampling","${CONDITIONER_NAME}","sigma_${SIGMA}","lag_${LAG_TIME}"] \
  ++run_key=$RUN_KEY 