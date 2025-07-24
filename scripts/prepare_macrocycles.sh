#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=prepare_macrocycles
#SBATCH --output=logs/%j_prepare_macrocycles.log
#SBATCH --error=logs/%j_prepare_macrocycles.err
#SBATCH --array=0-1000

eval "$(conda shell.bash hook)"
conda activate jamun

index=$((SLURM_ARRAY_TASK_ID))

python prepare_macrocycles.py --index ${index} "$@"  # Passes through any command line arguments
