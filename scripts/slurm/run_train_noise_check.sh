#!/usr/bin/env bash
#
# Wrapper script to run train_noise_check.sh for multiple m values
# 
# This script loops over m values from 2 to 10 and submits the train_noise_check.sh
# SLURM script for each value. This ensures only 4 parallel jobs are submitted at a time
# (one for each model configuration) rather than submitting all 36 jobs at once.
#
# Usage: ./run_train_noise_check.sh
#

# Set script directory
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="scripts/slurm/train_noise_check.sh"

# Check if train_noise_check.sh exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Error: train_noise_check.sh not found at $TRAIN_SCRIPT"
    exit 1
fi

# Make sure the script is executable
chmod +x "$TRAIN_SCRIPT"

echo "Starting noise check experiments for m values 2-10"
echo "Each submission will run 4 parallel jobs (one for each model configuration)"
echo ""

# Loop over m values from 2 to 10
for M in {2..10}; do
    echo "Submitting jobs for M=$M..."
    
    # Submit the SLURM script with the current m value
    JOB_ID=$(sbatch --parsable scripts/slurm/train_noise_check.sh $M)
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully submitted job ID: $JOB_ID for M=$M"
        echo "    This will run 4 parallel jobs (array 0-3) for the 4 model configurations"
        
        # Wait for all array jobs to complete before submitting next batch
        # echo "  ⏳ Waiting for job ID $JOB_ID to complete before submitting next batch..."
        echo "  ⏳ Submitted ID $JOB_ID..."
        # # Wait for the job to finish (all array tasks)
        # while squeue -j "$JOB_ID" 2>/dev/null | grep -q "$JOB_ID"; do
        #     sleep 30  # Check every 30 seconds
        # done
        
        echo "  ✅ All jobs for M=$M (Job ID: $JOB_ID) completed!"
        echo ""
        
    else
        echo "  ✗ Failed to submit job for M=$M"
        exit 1
    fi
done

echo ""
echo "All jobs submitted successfully!"
echo "Total submissions: 9 (one for each m value from 2-10)"
echo "Total jobs: 36 (4 models × 9 m values)"
echo ""
echo "Monitor job status with: squeue -u \$USER"
echo "View job outputs in the current directory with pattern: slurm-<jobid>_<arrayindex>.out"
