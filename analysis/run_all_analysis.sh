#!/bin/bash

for experiment in Our_2AA Timewarp_2AA Timewarp_4AA MDGen_4AA Our_5AA; do
    sbatch analysis_sweep.sh --experiment ${experiment} --csv sample_runs.csv --output-dir /data/bucket/kleinhej/jamun-analysis/
done