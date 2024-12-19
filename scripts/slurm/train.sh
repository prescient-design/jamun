#!/usr/bin/env bash

#SBATCH --partition gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --gpus-per-node 2
#SBATCH --cpus-per-task 8
#SBATCH --time 7-0

# if using mamba or conda, comment the following:
source .venv/bin/activate
# if using mamba or conda, uncomment the following:
# eval "$(mamba shell.bash hook)"
# mamba activate jamun-env


set -eux

echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
hostname

export HYDRA_FULL_ERROR=1
export TORCH_COMPILE_DEBUG=1
export TORCH_LOGS="+dynamo"
export TORCHDYNAMO_VERBOSE=1

# NOTE we generate this in submit script instead of using time based default to ensure consistency across ranks
RUN_KEY=$(openssl rand -hex 12)

nvidia-smi

srun --cpus-per-task 8 --cpu-bind=cores,verbose \
  jamun_train --config-dir=/homefs/home/daigavaa/jamun/configs \
    experiment=train_uncapped_4AA.yaml \
    ++trainer.devices=$SLURM_GPUS_PER_NODE \
    ++trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    ++trainer.limit_train_batches=1.0 \
    ++logger.wandb.tags=["'${SLURM_JOB_ID}'","train"] \
    ++run_key=$RUN_KEY
