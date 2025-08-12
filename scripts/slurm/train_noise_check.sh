#!/usr/bin/env bash
#
# Noise check experiment script
# 
# Implements 5 out of 7 requested model configurations for m=2,3,4,5,6,7,8,9,10:
# 1. Standard JAMUN with noise level sigma/sqrt(m)
# 2. Spatiotemporal with temporal embedding (pretrained denoiser, trainable=true)
# 3. Spatiotemporal with ones embedding (pretrained denoiser, trainable=true)
# 4. Spatiotemporal with temporal embedding, repeated position dataset
# 5. Spatiotemporal with ones embedding, repeated position dataset
#
# NOT IMPLEMENTED (requires custom dataset class):
# 6. Spatiotemporal with temporal embedding, random lag times
# 7. Spatiotemporal with ones embedding, random lag times
#
# Total jobs: 9 m-values × 5 models = 45 jobs (array 0-44)

#SBATCH --partition=gpu3
#SBATCH --job-name=noise_check
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-44

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

# Define experiment parameters
# m values: 2, 3, 4, 5, 6, 7, 8, 9, 10 (9 values)
# Model types: 5 types (random lag times require custom implementation)
# Total combinations: 9 * 5 = 45 jobs (0-44)

M_VALUES=(2 3 4 5 6 7 8 9 10)
MODEL_TYPES=(
    "standard_jamun"
    "spatiotemporal_temporal_embedding"
    "spatiotemporal_ones_embedding"
    "spatiotemporal_temporal_embedding_repeated_pos"
    "spatiotemporal_ones_embedding_repeated_pos"
)

# Calculate m and model indices
NUM_MODELS=5
M_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_MODELS))
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_MODELS))

M=${M_VALUES[$M_INDEX]}
MODEL_TYPE=${MODEL_TYPES[$MODEL_INDEX]}

echo "Job ${SLURM_ARRAY_TASK_ID}: M=${M}, Model=${MODEL_TYPE}"

# Calculate noise level: sigma / sqrt(m) where base sigma = 0.04
BASE_SIGMA=0.04
NOISE_LEVEL=$(python3 -c "import math; print(${BASE_SIGMA} / math.sqrt(${M}))")

echo "Noise level: ${NOISE_LEVEL}"

# Configure base parameters based on model type
case ${MODEL_TYPE} in
    "standard_jamun")
        echo "Model 1: Standard JAMUN with noise level sigma/sqrt(m)"
        CONFIG="train_enhanced_standard_jamun"
        OVERRIDES="++model.sigma_distribution.sigma=${NOISE_LEVEL}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.max_datasets=1"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.max_datasets=1"
        WANDB_TAG="standard_jamun_m${M}"
        ;;
    "spatiotemporal_temporal_embedding")
        echo "Model 2: Spatiotemporal with temporal embedding (pretrained denoiser, trainable=true)"
        CONFIG="train_enhanced_pretrained_spatiotemporal_conditioner"
        OVERRIDES="++model.conditioner.spatiotemporal_model.spatial_module.trainable=true"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.max_datasets=1"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.max_datasets=1"

        WANDB_TAG="spatiotemporal_temporal_m${M}"
        ;;
    "spatiotemporal_ones_embedding")
        echo "Model 3: Spatiotemporal with ones embedding (pretrained denoiser, trainable=true)"
        CONFIG="train_enhanced_pretrained_spatiotemporal_conditioner"
        OVERRIDES="++model.conditioner.spatiotemporal_model.spatial_module.trainable=true"
        OVERRIDES="${OVERRIDES} ++model.conditioner.spatiotemporal_model.temporal_module.node_attr_temporal_encoding_function=ones"
        OVERRIDES="${OVERRIDES} ++model.conditioner.spatiotemporal_model.temporal_module.edge_attr_temporal_encoding_function=ones"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.max_datasets=1"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.max_datasets=1"

        WANDB_TAG="spatiotemporal_ones_m${M}"
        ;;
    "spatiotemporal_temporal_embedding_repeated_pos")
        echo "Model 4: Spatiotemporal with temporal embedding, repeated position dataset"
        CONFIG="train_enhanced_pretrained_spatiotemporal_conditioner"
        OVERRIDES="++model.conditioner.spatiotemporal_model.spatial_module.trainable=true"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train._target_=jamun.data.parse_repeated_position_datasets_from_directory"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val._target_=jamun.data.parse_repeated_position_datasets_from_directory"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.max_datasets=1"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.max_datasets=1"

        WANDB_TAG="spatiotemporal_temporal_repeated_m${M}"
        ;;
    "spatiotemporal_ones_embedding_repeated_pos")
        echo "Model 5: Spatiotemporal with ones embedding, repeated position dataset"
        CONFIG="train_enhanced_pretrained_spatiotemporal_conditioner"
        OVERRIDES="++model.conditioner.spatiotemporal_model.spatial_module.trainable=true"
        OVERRIDES="${OVERRIDES} ++model.conditioner.spatiotemporal_model.temporal_module.node_attr_temporal_encoding_function=ones"
        OVERRIDES="${OVERRIDES} ++model.conditioner.spatiotemporal_model.temporal_module.edge_attr_temporal_encoding_function=ones"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train._target_=jamun.data.parse_repeated_position_datasets_from_directory"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val._target_=jamun.data.parse_repeated_position_datasets_from_directory"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.max_datasets=1"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.max_datasets=1"

        WANDB_TAG="spatiotemporal_ones_repeated_m${M}"
        ;;
    *)
        echo "Unknown model type: ${MODEL_TYPE}"
        exit 1
        ;;
esac

# Build the command with base config
CMD="jamun_train --config-dir=configs experiment=${CONFIG}.yaml"

# Add overrides
if [ -n "$OVERRIDES" ]; then
    CMD="$CMD $OVERRIDES"
fi

# Add common training overrides
CMD="$CMD ++trainer.max_epochs=1"
CMD="$CMD ++logger.wandb.group=noise_check_experiment"

# Add job-specific wandb tags and run name
WANDB_RUN_NAME="noise_check_${WANDB_TAG}"
CMD="$CMD ++logger.wandb.tags=[${WANDB_TAG},noise_check,m_${M}]"
CMD="$CMD ++logger.wandb.name=${WANDB_RUN_NAME}"

# Add notes about the experiment
WANDB_NOTES="Noise check experiment: ${MODEL_TYPE} with m=${M}"
if [[ ${MODEL_TYPE} == "standard_jamun" ]]; then
    WANDB_NOTES="${WANDB_NOTES}, noise_level=${NOISE_LEVEL}"
fi
CMD="$CMD ++logger.wandb.notes=\"${WANDB_NOTES}\""

echo "Running command: $CMD"
exec $CMD
