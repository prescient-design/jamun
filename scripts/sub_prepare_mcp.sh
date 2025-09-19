N=$(find /data/davidsd5/both_cremps -maxdepth 1 -type f -name '*.sdf' | wc -l)
MAX=$(( (N + 49)/50 - 1 ))
mkdir -p logs
sbatch --array=0-$MAX /homefs/home/davidsd5/jamun/jamun/scripts/prepare_macrocycles.sh --input-dir /data/davidsd5/both_cremps --output-dir /data/davidsd5/both_cremps/prepro
