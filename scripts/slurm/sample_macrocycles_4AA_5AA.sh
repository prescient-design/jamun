#!/usr/bin/env bash

#SBATCH --partition gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --time 72:00:00
#SBATCH --array 0-4
#SBATCH --mem-per-cpu=32G

#eval "$(conda shell.bash hook)"
source /homefs/home/davidsd5/miniforge3/bin/activate jamun
eval "$(conda shell.bash hook)"
#conda activate jamun

set -eux

echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "hostname = $(hostname)"

max_datasets=20
max_datasets_offset=$((SLURM_ARRAY_TASK_ID * 20))

export HYDRA_FULL_ERROR=1
# export TORCH_COMPILE_DEBUG=1
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

# NOTE: We generate this in submit script instead of using time-based default to ensure consistency across ranks.
RUN_KEY=$(openssl rand -hex 12)
echo "RUN_KEY = ${RUN_KEY}"

nvidia-smi

srun --cpus-per-task 8 --cpu-bind=cores,verbose \
    /homefs/home/davidsd5/miniforge3/envs/jamun/bin/jamun_sample --config-dir=/homefs/home/davidsd5/jamun/jamun/configs \
        experiment=sample_macrocycle_4AA_5AA.yaml \
        ++init_datasets.max_datasets=${max_datasets} \
        ++init_datasets.max_datasets_offset=${max_datasets_offset} \
        ++sampler.devices=$SLURM_GPUS_PER_NODE \
        ++sampler.num_nodes=$SLURM_JOB_NUM_NODES \
        ++logger.wandb.tags=["'${SLURM_JOB_ID}'","'${RUN_KEY}'","sample","macrocycle"] \
        ++run_key=$RUN_KEY
