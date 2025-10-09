#!/usr/bin/env bash

#SBATCH --partition g6e
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --time 3-0
#SBATCH --mem=32G

eval "$(conda shell.bash hook)"
conda activate jamun

set -eux

echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "hostname = $(hostname)"

export HYDRA_FULL_ERROR=1
export CUDA_HOME=/usr/local/cuda-12.8/

# export TORCH_COMPILE_DEBUG=1
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

# NOTE: We generate this in submit script instead of using time-based default to ensure consistency across ranks.
RUN_KEY=$(openssl rand -hex 12)
echo "RUN_KEY = ${RUN_KEY}"

nvidia-smi

srun --cpus-per-task 8 --cpu-bind=cores,verbose \
  jamun_train --config-dir=/homefs/home/daigavaa/jamun/configs \
    experiment=train_atlas.yaml \
    ++trainer.devices=$SLURM_GPUS_PER_NODE \
    ++trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    ++logger.wandb.tags=["'${SLURM_JOB_ID}'","'${RUN_KEY}'","train","atlas"] \
    ++run_key=$RUN_KEY
