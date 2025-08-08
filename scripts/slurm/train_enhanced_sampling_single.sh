#!/usr/bin/env bash

#SBATCH --partition gpu2
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --time 12:00:00
#SBATCH --mem-per-cpu=32G

# Initialize conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate jamun

# Verify conda activation worked
which python
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
nvidia-smi

# Run training with parameter overrides
jamun_train --config-dir=configs experiment=train_enhanced_pretrained_spatiotemporal_conditioner.yaml model.conditioner.spatiotemporal_model.temporal_to_spatial_pooler._target_=jamun.model.pooling.TemporalToSpatialNodeAttrMean model.conditioner.spatiotemporal_model.spatial_module.trainable=false trainer.max_epochs=10 logger.wandb.tags=[job_4,spatiotemporal_comparison]