#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=tbg_analysis_sweep
#SBATCH --output=logs/%j_tbg_analysis_sweep.log
#SBATCH --error=logs/%j_tbg_analysis_sweep.err
#SBATCH --array=0-100
#SBATCH --time 1:00:00

eval "$(conda shell.bash hook)"
conda activate jamun

row_index=$((SLURM_ARRAY_TASK_ID))

python tbg_analysis_sweep.py --row-index ${row_index} "$@"  # Passes through any command line arguments