#!/bin/bash

#SBATCH --partition gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --time 24:00:00


# Get peptide name from command line argument
PEPTIDE_NAME=$1

# Print job information
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Peptide name: ${PEPTIDE_NAME}"

# Load conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate bioemu

set -eux


# Define paths
PDB_PATH=~/bioemu-samples/${PEPTIDE_NAME}/topology.pdb
XTC_PATH=~/bioemu-samples/${PEPTIDE_NAME}/samples_10k.xtc

# Check if files exist
if [ ! -f "${PDB_PATH}" ]; then
    echo "Error: PDB file not found at ${PDB_PATH}"
    exit 1
fi

if [ ! -f "${XTC_PATH}" ]; then
    echo "Error: XTC file not found at ${XTC_PATH}"
    exit 1
fi

# Run the bioemu.sidechain_relax module
python -m bioemu.sidechain_relax --pdb-path ${PDB_PATH} --xtc-path ${XTC_PATH} --no-md-equil --prefix ${PEPTIDE_NAME}

# Check the exit status
if [ $? -eq 0 ]; then
    echo "Sidechain relaxation completed successfully"
else
    echo "Error: Sidechain relaxation failed"
    exit 1
fi

echo "Job finished at $(date)"
