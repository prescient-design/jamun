#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=analysis_sweep
#SBATCH --output=logs/%j_analysis_sweep.log
#SBATCH --error=logs/%j_analysis_sweep.err
#SBATCH --array=0-199

eval "$(conda shell.bash hook)"
conda activate jamun

row_index=$((SLURM_ARRAY_TASK_ID))

python analysis_sweep.py --row-index ${row_index} "$@"  # Passes through any command line arguments