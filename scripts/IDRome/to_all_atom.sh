#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=8G
#SBATCH --job-name=to_all_atom
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Assign arguments to variables
temp_path="$1"
pulchra_temp_path="$2"
final_path="$3"

eval "$(conda shell.bash hook)"
conda activate jamun

# Run PULCHRA
pulchra "$temp_path"

# Move the rebuilt PDB to final location
mv "$pulchra_temp_path" "$final_path"

# Clean up input file
rm "$temp_path"