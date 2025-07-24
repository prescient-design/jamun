#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --mem=4G
#SBATCH --job-name=relax_structures
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Assign arguments to variables
pdb_path="$1"
output_dir="$2"

eval "$(conda shell.bash hook)"
conda activate jamun

# Run the simulation
python scripts/generate_data/run_simulation.py "$pdb_path" \
    --output-dir "$output_dir" \
    --energy-minimization-only \
    --energy-minimization-steps=5000

# Exit with the simulation's exit code
exit $?
