#!/bin/bash
#SBATCH --job-name=sweep_delta_friction
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-24

# Load environment
source ~/.bashrc
conda activate jamun
cd /homefs/home/sules/jamun
mkdir -p logs

# Precomputed values for sigma=0.04
# Delta: 5 values from sigma/sqrt(5) to sqrt(5)*sigma
DELTAS=(0.017889 0.026833 0.040000 0.059665 0.089443)

# Friction: -log of 5 values from 0.01 to 0.99  
FRICTIONS=(2.52572864 1.2552661 0.71334989 0.36384343 0.10536052)

# Get parameter values based on array index
DELTA_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
FRICTION_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
DELTA=${DELTAS[$DELTA_INDEX]}
FRICTION=${FRICTIONS[$FRICTION_INDEX]}

echo "Running: delta=$DELTA, friction=$FRICTION (job $SLURM_ARRAY_TASK_ID)"

# Run experiment
jamun_sample \
    --config-dir=configs \
    experiment=sample_enhanced_conditioning_sweep \
    ++delta=$DELTA \
    ++friction=$FRICTION \
    ++logger.wandb.name="sweep_d${DELTA}_f${FRICTION}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"