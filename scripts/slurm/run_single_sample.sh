#!/bin/bash

# This script runs sampling for a specific run from a wandb sweep, selected by an index.

jamun_sample --config-dir=configs experiment=sample_capped_single_shape_conditioning ++wandb_train_run_path=sule-shashank/jamun/zchesftt ++logger.wandb.notes=jumping-sweep-29
jamun_sample --config-dir=configs experiment=sample_capped_single_shape_conditioning ++wandb_train_run_path=sule-shashank/jamun/jqp09yv1 ++logger.wandb.notes=stellar-sweep-25