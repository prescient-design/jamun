#!/usr/bin/env bash

#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-5

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

# Define configurations for each job
case ${SLURM_ARRAY_TASK_ID} in
    0)
        echo "Job 0: Initialized spatial module + TemporalToSpatialNodeAttrMean"
        CONFIG="train_enhanced_spatiotemporal_conditioner"
        POOLER="jamun.model.pooling.TemporalToSpatialNodeAttrMean"
        TRAINABLE_OVERRIDE=""
        ;;
    1)
        echo "Job 1: Initialized spatial module + TemporalToSpatialNodeAttr"
        CONFIG="train_enhanced_spatiotemporal_conditioner"
        POOLER="jamun.model.pooling.TemporalToSpatialNodeAttr"
        TRAINABLE_OVERRIDE=""
        ;;
    2)
        echo "Job 2: Pretrained trainable spatial module + TemporalToSpatialNodeAttrMean"
        CONFIG="train_enhanced_pretrained_spatiotemporal_conditioner"
        POOLER="jamun.model.pooling.TemporalToSpatialNodeAttrMean"
        TRAINABLE_OVERRIDE="model.conditioner.spatiotemporal_model.spatial_module.trainable=true"
        ;;
    3)
        echo "Job 3: Pretrained trainable spatial module + TemporalToSpatialNodeAttr"
        CONFIG="train_enhanced_pretrained_spatiotemporal_conditioner"
        POOLER="jamun.model.pooling.TemporalToSpatialNodeAttr"
        TRAINABLE_OVERRIDE="model.conditioner.spatiotemporal_model.spatial_module.trainable=true"
        ;;
    4)
        echo "Job 4: Pretrained non-trainable spatial module + TemporalToSpatialNodeAttrMean"
        CONFIG="train_enhanced_pretrained_spatiotemporal_conditioner"
        POOLER="jamun.model.pooling.TemporalToSpatialNodeAttrMean"
        TRAINABLE_OVERRIDE="model.conditioner.spatiotemporal_model.spatial_module.trainable=false"
        ;;
    5)
        echo "Job 5: Pretrained non-trainable spatial module + TemporalToSpatialNodeAttr"
        CONFIG="train_enhanced_pretrained_spatiotemporal_conditioner"
        POOLER="jamun.model.pooling.TemporalToSpatialNodeAttr"
        TRAINABLE_OVERRIDE="model.conditioner.spatiotemporal_model.spatial_module.trainable=false"
        ;;
    *)
        echo "Unknown job ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

# Build the command with overrides
CMD="jamun_train --config-dir=configs experiment=${CONFIG}.yaml"

# Add pooler override (keeping the irreps_out parameter from base config)
CMD="$CMD ++model.conditioner.spatiotemporal_model.temporal_to_spatial_pooler._target_=${POOLER}"

# Add trainable override if needed
if [ -n "$TRAINABLE_OVERRIDE" ]; then
    CMD="$CMD $TRAINABLE_OVERRIDE"
fi

# Add dataset and training overrides
# CMD="$CMD data.datamodule.datasets.train.max_datasets=1"
# CMD="$CMD data.datamodule.datasets.val.max_datasets=1"
CMD="$CMD ++trainer.max_epochs=100"
CMD="$CMD ++wandb.logger.group=spatiotemporal_comparison"

# Add job-specific wandb tags
WANDB_TAG="job_${SLURM_ARRAY_TASK_ID}"
# CMD="$CMD ++wandb.logger.tags=[${WANDB_TAG},spatiotemporal_comparison]"

echo "Running command: $CMD"
exec $CMD