#!/usr/bin/env bash

#SBATCH --partition g6e
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 4
#SBATCH --time 1-0
#SBATCH --array 0-2
#SBATCH --mem=60G

eval "$(conda shell.bash hook)"
conda activate jamun

set -eux

echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "hostname = $(hostname)"

export HYDRA_FULL_ERROR=1
# export TORCH_COMPILE_DEBUG=1
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

# NOTE: We generate this in submit script instead of using time-based default to ensure consistency across ranks.
RUN_KEY=$(openssl rand -hex 12)
echo "RUN_KEY = ${RUN_KEY}"

nvidia-smi

srun --cpus-per-task 4 --cpu-bind=cores,verbose \
  jamun_train --config-dir=/homefs/home/daigavaa/jamun/configs \
    experiment=train_uncapped_4AA_alignment.yaml \
    ++trainer.devices=$SLURM_GPUS_PER_NODE \
    ++trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    ++model.use_alignment_estimators=true \
    ++model.alignment_correction_order=$SLURM_ARRAY_TASK_ID \
    ++logger.wandb.tags=["'${SLURM_JOB_ID}'","'${RUN_KEY}'","train","align-4AA","'${SLURM_ARRAY_TASK_ID}'"] \
    ++run_key=$RUN_KEY
