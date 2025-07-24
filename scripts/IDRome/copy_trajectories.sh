#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=1G
#SBATCH --job-name=copy_trajectories
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Assign arguments to variables
pdb_path="$1"
pdb_path_renamed="$2"
xtc_path="$3"
xtc_path_renamed="$4"

# Copy PDB file
echo "Copying ${pdb_path} to ${pdb_path_renamed}"
cp "${pdb_path}" "${pdb_path_renamed}"
if [ $? -eq 0 ]; then
    echo "PDB file copy successful"
else
    echo "PDB file copy failed with exit code $?"
    exit 1
fi

# Copy XTC file
echo "Copying ${xtc_path} to ${xtc_path_renamed}"
cp "${xtc_path}" "${xtc_path_renamed}"
if [ $? -eq 0 ]; then
    echo "XTC file copy successful"
else
    echo "XTC file copy failed with exit code $?"
    exit 1
fi

exit 0
