#!/usr/bin/env bash
#
# Noise check experiment script
# 
# Implements 4 model configurations for a given m value (passed as command line argument):
# 1. Standard JAMUN with repeated position dataset and noise level sigma/sqrt(m)
# 2. Spatiotemporal JAMUN with repeated position dataset and total lag time m
# 3. Spatiotemporal JAMUN with total lag time m
# 4. Spatiotemporal JAMUN with total lag time m, hub_n_spoke graph type, ones encoding
#
# Usage: sbatch train_noise_check.sh <m_value>
# Total jobs: 4 models (array 0-3)

#SBATCH --partition=gpu2
#SBATCH --job-name=noise_check
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array=1-3

# Initialize conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate jamun

# Verify conda activation worked
which python
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
nvidia-smi

# Get m value from command line argument
if [ $# -eq 0 ]; then
    echo "Error: Please provide m value as command line argument"
    echo "Usage: sbatch train_noise_check.sh <m_value>"
    exit 1
fi

M=$1
echo "Running array job ${SLURM_ARRAY_TASK_ID} with M=${M}"

# Define experiment parameters
# Model types: 4 types
# Total jobs: 4 models (array 0-3)

MODEL_TYPES=(
    "standard_jamun_repeated_pos"
    "spatiotemporal_repeated_pos"
    "spatiotemporal_default"
    "spatiotemporal_hub_spoke_ones"
)

# Calculate model index directly from array task ID
MODEL_INDEX=${SLURM_ARRAY_TASK_ID}
MODEL_TYPE=${MODEL_TYPES[$MODEL_INDEX]}

echo "Job ${SLURM_ARRAY_TASK_ID}: M=${M}, Model=${MODEL_TYPE}"

# Generate unique run key to prevent checkpoint overwrites
RUN_KEY=$(openssl rand -hex 12)
echo "RUN_KEY = ${RUN_KEY}"

# Calculate noise level: sigma / sqrt(m) where base sigma = 0.04
BASE_SIGMA=0.04
NOISE_LEVEL=$(python3 -c "import math; print(${BASE_SIGMA} / math.sqrt(${M}))")

echo "Noise level: ${NOISE_LEVEL}"

# Configure base parameters based on model type
case ${MODEL_TYPE} in
    # "standard_jamun_repeated_pos")
    #     echo "Model 1: Standard JAMUN with repeated position dataset and noise level sigma/sqrt(m)"
    #     CONFIG="train_enhanced_standard_jamun"
    #     OVERRIDES="++model.sigma_distribution.sigma=${NOISE_LEVEL}"
    #     OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train._target_=jamun.data.parse_repeated_position_datasets_from_directory"
    #     OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val._target_=jamun.data.parse_repeated_position_datasets_from_directory"
    #     OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.total_lag_time=${M}"
    #     OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.total_lag_time=${M}"
    #     OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.max_datasets=500"
    #     OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.max_datasets=100"
    #     WANDB_TAG="standard_jamun_repeated_pos_m${M}"
    #     ;;
    "spatiotemporal_repeated_pos")
        echo "Model 2: Spatiotemporal JAMUN with repeated position dataset and total lag time m"
        CONFIG="train_enhanced_spatiotemporal_conditioner"
        OVERRIDES="++data.datamodule.datasets.train._target_=jamun.data.parse_repeated_position_datasets_from_directory"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val._target_=jamun.data.parse_repeated_position_datasets_from_directory"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.max_datasets=500"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.max_datasets=100"
        WANDB_TAG="spatiotemporal_repeated_pos_m${M}"
        ;;
    "spatiotemporal_default")
        echo "Model 3: Spatiotemporal JAMUN with total lag time m"
        CONFIG="train_enhanced_spatiotemporal_conditioner"
        OVERRIDES="++data.datamodule.datasets.train.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.max_datasets=500"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.max_datasets=100"
        WANDB_TAG="spatiotemporal_default_m${M}"
        ;;
    "spatiotemporal_hub_spoke_ones")
        echo "Model 4: Spatiotemporal JAMUN with total lag time m, hub_n_spoke graph type, ones encoding"
        CONFIG="train_enhanced_spatiotemporal_conditioner"
        OVERRIDES="++data.datamodule.datasets.train.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.total_lag_time=${M}"
        OVERRIDES="${OVERRIDES} ++model.conditioner.spatiotemporal_model.graph_type=hub_n_spoke"
        OVERRIDES="${OVERRIDES} ++model.conditioner.spatiotemporal_model.temporal_module.node_attr_temporal_encoding_function=ones"
        OVERRIDES="${OVERRIDES} ++model.conditioner.spatiotemporal_model.temporal_module.edge_attr_temporal_encoding_function=ones"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.train.max_datasets=500"
        OVERRIDES="${OVERRIDES} ++data.datamodule.datasets.val.max_datasets=100"
        WANDB_TAG="spatiotemporal_hub_spoke_ones_m${M}"
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
CMD="$CMD ++trainer.max_epochs=50"
CMD="$CMD ++run_key=${RUN_KEY}"
CMD="$CMD ++logger.wandb.group=noise_check_experiment_multimeasurement_vs_correlation"

# Add job-specific wandb tags and run name
WANDB_RUN_NAME="noise_check_${WANDB_TAG}"
CMD="$CMD ++logger.wandb.tags=[${WANDB_TAG},noise_check,m_${M}]"
CMD="$CMD ++logger.wandb.name=${WANDB_RUN_NAME}"

# Add notes about the experiment
WANDB_NOTES="Noise_check_experiment:_${MODEL_TYPE}_with_m=${M}"
if [[ ${MODEL_TYPE} == "standard_jamun" ]]; then
    WANDB_NOTES="${WANDB_NOTES}, noise_level=${NOISE_LEVEL}"
fi
CMD="$CMD ++logger.wandb.notes=\"${WANDB_NOTES}\""

echo "Running command: $CMD"
exec $CMD
