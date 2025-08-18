#!/usr/bin/env bash

#SBATCH --partition=gpu2
#SBATCH --job-name=capped_2AA_comparison
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-4

# Initialize conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate jamun

# Verify conda activation worked
which python
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
nvidia-smi

echo "Running array job ${SLURM_ARRAY_TASK_ID}"

# NOTE: We generate this in submit script instead of using time-based default to ensure consistency across ranks.
RUN_KEY=$(openssl rand -hex 12)
echo "RUN_KEY = ${RUN_KEY}"

# Define configurations for each job
case ${SLURM_ARRAY_TASK_ID} in
    0)
        echo "Job 0: Standard JAMUN on 2AA capped diamines"
        CONFIG="train_capped_2AA"
        OVERRIDES=""
        ;;
    1)
        echo "Job 1: Position conditioner on 2AA capped diamines"
        CONFIG="train_capped_2AA_position_conditioner"
        OVERRIDES=""
        ;;
    2)
        echo "Job 2: Self conditioner on 2AA capped diamines"
        CONFIG="train_capped_2AA_self_conditioner"
        OVERRIDES=""
        ;;
    3)
        echo "Job 3: Spatiotemporal conditioner with temporal embedding and mean pooling on 2AA capped diamines"
        CONFIG="train_capped_2AA_spatiotemporal_conditioner"
        OVERRIDES="++model.conditioner.spatiotemporal_model.temporal_to_spatial_pooler._target_=jamun.model.pooling.TemporalToSpatialNodeAttrMean"
        ;;
    4)
        echo "Job 4: Spatiotemporal conditioner with ones temporal encoding and mean pooling on 2AA capped diamines"
        CONFIG="train_capped_2AA_spatiotemporal_conditioner"
        OVERRIDES="++model.conditioner.spatiotemporal_model.temporal_to_spatial_pooler._target_=jamun.model.pooling.TemporalToSpatialNodeAttrMean ++model.conditioner.spatiotemporal_model.temporal_module.node_attr_temporal_encoding_function=ones ++model.conditioner.spatiotemporal_model.temporal_module.edge_attr_temporal_encoding_function=ones"
        ;;
    *)
        echo "Unknown job ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

# Build the command with base config
CMD="jamun_train --config-dir=configs experiment=${CONFIG}.yaml"

# Add overrides if any
if [ -n "$OVERRIDES" ]; then
    CMD="$CMD $OVERRIDES"
fi

# Add common training overrides
CMD="$CMD ++trainer.max_epochs=100"
CMD="$CMD ++logger.wandb.group=capped_2AA_model_comparison"
CMD="$CMD ++run_key=${RUN_KEY}"

# Add dataset overrides for debugging (quick completion)
# CMD="$CMD ++data.datamodule.datasets.train.max_datasets=1"
# CMD="$CMD ++data.datamodule.datasets.val.max_datasets=1"

# Add job-specific wandb tags
WANDB_TAG="job_${SLURM_ARRAY_TASK_ID}"
CMD="$CMD ++logger.wandb.tags=[${WANDB_TAG},capped_2AA_comparison,generalization_test]"

echo "Running command: $CMD"
exec $CMD
