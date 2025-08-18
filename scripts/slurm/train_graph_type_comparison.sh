#!/usr/bin/env bash
#
# Graph type comparison experiment script
# 
# Iterates over:
# - Lag subsample rates: 1, 2, 3, 4
# - Total lag times: 2, 4, 6, 8
# - Configs: train_test_single_shape.yaml, train_test_single_shape_conditional.yaml, train_test_single_shape_spatiotemporal_conditioner.yaml
# - For spatiotemporal: hub_n_spoke vs complete graph types (both with ones encoding)
#
# Total jobs: 4 lag_subsample_rates × 4 total_lag_times × (1 conditional + 2 spatiotemporal variants) = 48 jobs
#
#SBATCH --partition=gpu3
#SBATCH --job-name=graph_type_comparison
#SBATCH --qos=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0
#SBATCH --mem-per-cpu=32G
#SBATCH --array=0-15

# Set up the environment
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate jamun

# Define experiment parameters
LAG_SUBSAMPLE_RATES=(1 2 3 4)
TOTAL_LAG_TIMES=(2 4 6 8)
CONFIGS=(
    # "train_test_single_shape"
    "train_test_single_shape_conditional"
    "train_test_single_shape_spatiotemporal_conditioner_default"
    "train_test_single_shape_spatiotemporal_conditioner_hub_spoke"
    # "train_test_single_shape_spatiotemporal_conditioner_complete"
)

NUM_CONFIGS=1
NUM_LAG_TIMES=4
NUM_LAG_SUBSAMPLE_RATES=4

# Calculate indices
LAG_SUBSAMPLE_INDEX=$((SLURM_ARRAY_TASK_ID / (NUM_LAG_TIMES * NUM_CONFIGS)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (NUM_LAG_TIMES * NUM_CONFIGS)))
LAG_TIME_INDEX=$((REMAINDER / NUM_CONFIGS))
CONFIG_INDEX=$((REMAINDER % NUM_CONFIGS))

LAG_SUBSAMPLE_RATE=${LAG_SUBSAMPLE_RATES[$LAG_SUBSAMPLE_INDEX]}
TOTAL_LAG_TIME=${TOTAL_LAG_TIMES[$LAG_TIME_INDEX]}
CONFIG_TYPE=${CONFIGS[$CONFIG_INDEX]}

echo "Job ${SLURM_ARRAY_TASK_ID}: lag_subsample_rate=${LAG_SUBSAMPLE_RATE}, total_lag_time=${TOTAL_LAG_TIME}, config=${CONFIG_TYPE}"

# Generate unique run key
RUN_KEY=$(openssl rand -hex 12)
echo "RUN_KEY = ${RUN_KEY}"

# Calculate max_datasets using the formula: floor(250 * (49 / (lag_subsample_rate * (total_lag_time - 1) - 1)))
DENOMINATOR=$((LAG_SUBSAMPLE_RATE * (TOTAL_LAG_TIME - 1) - 1))
if [ $DENOMINATOR -le 0 ]; then
    echo "Warning: Invalid denominator ($DENOMINATOR) for max_datasets calculation. Setting to 250."
    MAX_DATASETS=250
else
    # Using bc for floating point calculation and floor function
    MAX_DATASETS=$(echo "scale=0; 250 * 49 / $DENOMINATOR" | bc)
fi
echo "Calculated max_datasets = $MAX_DATASETS (lag_subsample_rate=$LAG_SUBSAMPLE_RATE, total_lag_time=$TOTAL_LAG_TIME)"

# Set base configuration and overrides based on config type
case $CONFIG_INDEX in
    # 0)  # Standard JAMUN
    #     CONFIG="train_test_single_shape"
    #     OVERRIDES=""
    #     WANDB_TAG="standard_jamun_lag_${LAG_SUBSAMPLE_RATE}_time_${TOTAL_LAG_TIME}"
    #     ;;
    # 0)  # Position Conditioner
    #     CONFIG="train_enhanced_position_conditioner"
    #     OVERRIDES="++data.datamodule.datasets.train.total_lag_time=${TOTAL_LAG_TIME}"
    #     OVERRIDES="$OVERRIDES ++data.datamodule.datasets.train.lag_subsample_rate=${LAG_SUBSAMPLE_RATE}"
    #     OVERRIDES="$OVERRIDES ++model.arch.N_structures=${TOTAL_LAG_TIME}"
    #     OVERRIDES="$OVERRIDES ++model.conditioner.N_structures=${TOTAL_LAG_TIME}"
    #     WANDB_TAG="position_conditioner_lag_${LAG_SUBSAMPLE_RATE}_time_${TOTAL_LAG_TIME}"
    #     ;;
    0)  # SpatioTemporal Conditioner - Default (fan graph, temporal encoding)
        CONFIG="train_enhanced_spatiotemporal_conditioner"
        OVERRIDES="++data.datamodule.datasets.train.total_lag_time=${TOTAL_LAG_TIME}"
        OVERRIDES="$OVERRIDES ++data.datamodule.datasets.train.lag_subsample_rate=${LAG_SUBSAMPLE_RATE}"
        WANDB_TAG="spatiotemporal_default_fan_temporal_lag_${LAG_SUBSAMPLE_RATE}_time_${TOTAL_LAG_TIME}"
        ;;
    # 2)  # SpatioTemporal Conditioner - Hub & Spoke
    #     CONFIG="train_enhanced_spatiotemporal_conditioner"
    #     OVERRIDES="++data.datamodule.datasets.train.total_lag_time=${TOTAL_LAG_TIME}"
    #     OVERRIDES="$OVERRIDES ++data.datamodule.datasets.train.lag_subsample_rate=${LAG_SUBSAMPLE_RATE}"
    #     OVERRIDES="$OVERRIDES ++model.conditioner.spatiotemporal_model.graph_type=hub_n_spoke"
    #     OVERRIDES="$OVERRIDES ++model.conditioner.spatiotemporal_model.temporal_module.node_attr_temporal_encoding_function=ones"
    #     OVERRIDES="$OVERRIDES ++model.conditioner.spatiotemporal_model.temporal_module.edge_attr_temporal_encoding_function=ones"
    #     WANDB_TAG="spatiotemporal_hub_spoke_ones_lag_${LAG_SUBSAMPLE_RATE}_time_${TOTAL_LAG_TIME}"
    #     ;;
    # 4)  # SpatioTemporal Conditioner - Complete
    #     CONFIG="train_test_single_shape_spatiotemporal_conditioner"
    #     OVERRIDES="++data.datamodule.datasets.train.total_lag_time=${TOTAL_LAG_TIME}"
    #     OVERRIDES="$OVERRIDES ++data.datamodule.datasets.train.lag_subsample_rate=${LAG_SUBSAMPLE_RATE}"
    #     OVERRIDES="$OVERRIDES ++model.conditioner.spatiotemporal_model.graph_type=complete"
    #     OVERRIDES="$OVERRIDES ++model.conditioner.spatiotemporal_model.temporal_module.node_attr_temporal_encoding_function=ones"
    #     OVERRIDES="$OVERRIDES ++model.conditioner.spatiotemporal_model.temporal_module.edge_attr_temporal_encoding_function=ones"
    #     WANDB_TAG="spatiotemporal_complete_ones_lag_${LAG_SUBSAMPLE_RATE}_time_${TOTAL_LAG_TIME}"
    #     ;;
esac

# Build command
CMD="jamun_train --config-dir=configs experiment=${CONFIG}.yaml"

# Add overrides if any
if [ -n "$OVERRIDES" ]; then
    CMD="$CMD $OVERRIDES"
fi

# Calculate validation max_datasets using multiplier of 50
VAL_MAX_DATASETS=$(echo "scale=0; 50 * 49 / $DENOMINATOR" | bc)
if [ $DENOMINATOR -le 0 ]; then
    VAL_MAX_DATASETS=50
fi

# Add common overrides
CMD="$CMD ++run_key=${RUN_KEY}"
CMD="$CMD ++data.datamodule.datasets.train.max_datasets=${MAX_DATASETS}"
CMD="$CMD ++data.datamodule.datasets.val.max_datasets=${VAL_MAX_DATASETS}"
CMD="$CMD ++logger.wandb.group=graph_type_comparison_experiment_enhanced_sampling_data_onlyfan_aug17"
CMD="$CMD ++logger.wandb.tags=[${WANDB_TAG},graph_comparison,lag_subsample_${LAG_SUBSAMPLE_RATE},total_lag_${TOTAL_LAG_TIME}]"

# Set wandb run name
WANDB_RUN_NAME="graph_comparison_${WANDB_TAG}_enhanced_sampling_data"
CMD="$CMD ++logger.wandb.name=${WANDB_RUN_NAME}"

echo "Running command: $CMD"
exec $CMD
