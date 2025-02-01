#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

python analysis/run_analysis.py "$@"  # Passes through any command line arguments
