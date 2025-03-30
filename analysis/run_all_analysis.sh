#!/bin/bash

source ../.env

# for experiment in Our_2AA Timewarp_4AA MDGen_4AA Our_5AA; do
#     sbatch analysis_sweep.sh --experiment ${experiment} --output-dir ${JAMUN_ANALYSIS_PATH}  --csv sample_runs.csv
#     sleep 1 
# done

for experiment in Timewarp_4AA_0.2A Timewarp_4AA_0.4A Timewarp_4AA_0.8A; do
    sbatch analysis_sweep.sh --experiment ${experiment} --output-dir ${JAMUN_ANALYSIS_PATH}  --csv sample_runs.csv
    sleep 5
done

# for experiment in Timewarp_2AA; do
#     sbatch tbg_analysis_sweep.sh --experiment ${experiment} --output-dir ${JAMUN_ANALYSIS_PATH}
# done