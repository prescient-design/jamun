#!/usr/bin/env bash

#SBATCH --partition=gpu3
#SBATCH --job-name=model_comparison
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array=2-4

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
        echo "Job 0: Standard JAMUN"
        CONFIG="train_enhanced_standard_jamun"
        OVERRIDES=""
        ;;
    1)
        echo "Job 1: Position conditioner"
        CONFIG="train_enhanced_position_conditioner"
        OVERRIDES=""
        ;;
    2)
        echo "Job 2: Self conditioner"
        CONFIG="train_enhanced_self_conditioner"
        OVERRIDES=""
        ;;
    3)
        echo "Job 3: Spatiotemporal conditioner with mean pooling and trainable pretrained denoiser"
        CONFIG="train_enhanced_pretrained_spatiotemporal_conditioner"
        OVERRIDES="++model.conditioner.spatiotemporal_model.temporal_to_spatial_pooler._target_=jamun.model.pooling.TemporalToSpatialNodeAttrMean ++model.conditioner.spatiotemporal_model.spatial_module.trainable=true"
        ;;
    4)
        echo "Job 4: Spatiotemporal conditioner with mean pooling, trainable pretrained denoiser, and ones temporal encoding"
        CONFIG="train_enhanced_pretrained_spatiotemporal_conditioner"
        OVERRIDES="++model.conditioner.spatiotemporal_model.temporal_to_spatial_pooler._target_=jamun.model.pooling.TemporalToSpatialNodeAttrMean ++model.conditioner.spatiotemporal_model.spatial_module.trainable=true ++model.conditioner.spatiotemporal_model.temporal_module.node_attr_temporal_encoding_function=ones ++model.conditioner.spatiotemporal_model.temporal_module.edge_attr_temporal_encoding_function=ones"
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
CMD="$CMD ++logger.wandb.group=model_comparison_full_swarm"
CMD="$CMD ++data.datamodule.datasets.train.root=/data2/sules/ALA_ALA_enhanced_full_swarm/train"
CMD="$CMD ++data.datamodule.datasets.val.root=/data2/sules/ALA_ALA_enhanced_full_swarm/val"

# Add dataset overrides for debugging (quick completion)
# CMD="$CMD ++data.datamodule.datasets.train.max_datasets=1"
# CMD="$CMD ++data.datamodule.datasets.val.max_datasets=1"

# Add job-specific wandb tags
WANDB_TAG="job_${SLURM_ARRAY_TASK_ID}"
CMD="$CMD ++logger.wandb.tags=[${WANDB_TAG},model_comparison]"

echo "Running command: $CMD"
exec $CMD
