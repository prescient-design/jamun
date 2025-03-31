#!/bin/bash

source ../.env

for experiment in Our_2AA Timewarp_4AA MDGen_4AA Our_5AA; do
    sbatch analysis_sweep.sh --experiment ${experiment} --output-dir ${JAMUN_ANALYSIS_PATH}  --csv sample_runs.csv
    sleep 5
done

for experiment in Timewarp_4AA_0.2A Timewarp_4AA_0.4A Timewarp_4AA_0.8A; do
    sbatch analysis_sweep.sh --experiment ${experiment} --output-dir ${JAMUN_ANALYSIS_PATH}  --csv sample_runs.csv
    sleep 5
done

for experiment in Chignolin; do
    sbatch -a 0-1 analysis_sweep.sh --experiment ${experiment} --output-dir ${JAMUN_ANALYSIS_PATH}  --csv sample_runs.csv
    sleep 5
done

for experiment in Timewarp_2AA; do
    sbatch analysis_sweep.sh --experiment ${experiment} --output-dir ${JAMUN_ANALYSIS_PATH}  --csv sample_runs.csv
    sleep 5
    sbatch tbg_analysis_sweep.sh --experiment ${experiment} --output-dir ${JAMUN_ANALYSIS_PATH}
    sleep 5
    sbatch tbg_analysis_sweep.sh --shorten-trajectory-factor 20 --experiment ${experiment} --output-dir ${JAMUN_ANALYSIS_PATH}
    sleep 5
    sbatch tbg_analysis_sweep.sh --shorten-trajectory-factor 200 --experiment ${experiment} --output-dir ${JAMUN_ANALYSIS_PATH}
    sleep 5
done

sbatch mdgen_analysis_sweep.sh --experiment MDGen_4AA --output-dir ${JAMUN_ANALYSIS_PATH} --peptide-type 4AA
sleep 5
