#!/bin/bash

output_dir=/data/bucket/kleinhej/jamun-analysis/

for experiment in Our_2AA Timewarp_2AA Timewarp_4AA MDGen_4AA Our_5AA; do
    sbatch analysis_sweep.sh --experiment ${experiment} --output-dir ${output_dir}  --csv sample_runs.csv 
done

for experiment in Timewarp_2AA; do
    sbatch tbg_analysis_sweep.sh --experiment ${experiment} --output-dir ${output_dir}
done