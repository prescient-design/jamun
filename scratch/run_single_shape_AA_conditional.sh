#!/usr/bin/env bash

#SBATCH --partition gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1 # Adjusted to 1 as typically one training script runs per job
#SBATCH --gpus-per-node 1   # Assuming your script uses 1 GPU. Adjust if it uses more.
#SBATCH --cpus-per-task 8   # Number of CPUs for your task
#SBATCH --time 08:00:00   # 7 days runtime limit
#SBATCH --mem-per-cpu=32G   # Memory per CPU
#SBATCH --job-name=train_prototype # Descriptive job name
#SBATCH --output=slurm_logs/train_prototype_%A_%a.out # Standard output file
#SBATCH --error=slurm_logs/train_prototype_%A_%a.err  # Standard error file

# --- Environment Setup ---
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "hostname = $(hostname)"
echo "Running on partition: ${SLURM_JOB_PARTITION}"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-"Not set"}" # SLURM usually sets CUDA_VISIBLE_DEVICES

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate jamun
echo "Conda environment 'jamun' activated."
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# --- Create Log Directory ---
# Create a directory for SLURM logs if it doesn't exist
# This should be relative to where you submit the job from, or an absolute path.
# Assuming you submit from /homefs/home/sules/jamun/
LOG_DIR_BASE="/homefs/home/sules/jamun/slurm_logs" 
mkdir -p "${LOG_DIR_BASE}"
# The %A_%a in sbatch output/error directives will be replaced by JobID and TaskID

# --- Application Execution ---
# Navigate to the directory containing your script, if necessary
# Assuming training_prototype.py is in /homefs/home/sules/jamun/scratch/
SCRIPT_DIR="/homefs/home/sules/jamun/scratch"
PYTHON_SCRIPT="training_prototype.py"

echo "Changing directory to ${SCRIPT_DIR}"
cd "${SCRIPT_DIR}" || { echo "Failed to cd to ${SCRIPT_DIR}"; exit 1; }

echo "Starting Python script: ${PYTHON_SCRIPT}"
# Run the Python script
# Add any necessary command-line arguments for your script here
python "${PYTHON_SCRIPT}"

echo "Python script finished."
echo "Job finished at: $(date)"
