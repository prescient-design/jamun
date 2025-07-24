#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=analysis_sweep
#SBATCH --output=logs/%j_analysis_sweep.log
#SBATCH --error=logs/%j_analysis_sweep.err
#SBATCH --array=0-199
#SBATCH --time 1:00:00

eval "$(conda shell.bash hook)"
conda activate jamun

row_index=$((SLURM_ARRAY_TASK_ID))

# Use the new combined script with the row_index from SLURM_ARRAY_TASK_ID
# The first argument should be the analysis type (boltz, jamun, or mdgen)
# The remaining arguments should include --output-dir, --experiment, etc.
analysis_type=$1
shift 1  # Remove the first argument from $@

echo "Analysis type: ${analysis_type}"
echo "Running analysis_sweep with row index: ${row_index}"

python analysis_sweep.py ${analysis_type} --row-index ${row_index} "$@"
