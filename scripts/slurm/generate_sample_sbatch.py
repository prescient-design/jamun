#!/usr/bin/env python3
"""
Script to generate an SBATCH script for jamun_sample based on a config file.

Usage:
    ./generate_sample_sbatch.py CONFIG_FILENAME [OPTIONS]

Arguments:
    CONFIG_FILENAME - Name of the sampling config file (e.g., sample_capped_2AA.yaml)

Options:
    --partition PARTITION   - SLURM partition (default: gpu2)
    --nodes N               - Number of nodes (default: 1)
    --gpus N                - GPUs per node (default: 1)
    --cpus N                - CPUs per task (default: 8)
    --time TIME             - Time limit (default: 72:00:00)
    --array-range RANGE     - Job array range (default: 0-5)
    --env-name NAME         - Conda environment name (default: jamun)
    --max-datasets N        - Max datasets per task (default: 50)
    --output OUTPUT         - Output file (default: CONFIG_NAME.sbatch)
"""

import os
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SBATCH script for jamun_sample")
    parser.add_argument("config_file", help="Config file name (e.g., sample_capped_2AA.yaml)")
    parser.add_argument("--partition", default="gpu2", help="SLURM partition")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per node")
    parser.add_argument("--cpus", type=int, default=8, help="CPUs per task")
    parser.add_argument("--time", default="72:00:00", help="Time limit")
    parser.add_argument("--array-range", default="0-5", help="Job array range")
    parser.add_argument("--env-name", default="jamun", help="Conda environment name")
    parser.add_argument("--max-datasets", type=int, default=50, help="Max datasets per task")
    parser.add_argument("--output", help="Output script file")

    return parser.parse_args()


def generate_sample_sbatch_script(args):
    experiment_dir = os.path.dirname(args.config_file)
    if not experiment_dir.endswith("experiment"):
        raise ValueError(
            f"Config file '{args.config_file}' must be in the 'experiment' directory."
        )
    config_dir = os.path.abspath(os.path.dirname(experiment_dir))
    config_file = os.path.basename(args.config_file)

    # Extract config name without extension
    config_name = os.path.splitext(args.config_file)[0]

    # Default output filename if not specified
    if args.output is None:
        args.output = f"{config_name}.sbatch"

    # Create the SBATCH script
    script = f'''#!/usr/bin/env bash

#SBATCH --partition {args.partition}
#SBATCH --nodes {args.nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node {args.gpus}
#SBATCH --cpus-per-task {args.cpus}
#SBATCH --time {args.time}
#SBATCH --array {args.array_range}

eval "$(conda shell.bash hook)"
conda activate {args.env_name}

set -eux

echo "SLURM_JOB_ID = ${{SLURM_JOB_ID}}"
echo "hostname = $(hostname)"

max_datasets={args.max_datasets}
max_datasets_offset=$((SLURM_ARRAY_TASK_ID * {args.max_datasets}))

export HYDRA_FULL_ERROR=1
# export TORCH_COMPILE_DEBUG=1
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

# NOTE: We generate this in submit script instead of using time-based default to ensure consistency across ranks.
RUN_KEY=$(openssl rand -hex 12)
echo "RUN_KEY = ${{RUN_KEY}}"

nvidia-smi

srun --cpus-per-task {args.cpus} --cpu-bind=cores,verbose \\
    jamun_sample --config-dir={config_dir} \\
        experiment={config_file} \\
        ++init_datasets.max_datasets=${{max_datasets}} \\
        ++init_datasets.max_datasets_offset=${{max_datasets_offset}} \\
        ++sampler.devices=$SLURM_GPUS_PER_NODE \\
        ++sampler.num_nodes=$SLURM_JOB_NUM_NODES \\
        ++logger.wandb.tags=["'${{SLURM_JOB_ID}}'","'${{RUN_KEY}}'","sample","{config_file}"] \\
        ++run_key=$RUN_KEY
'''

    return script


def main():
    args = parse_args()

    # Validate config file format
    if not args.config_file.endswith(".yaml"):
        print(f"Warning: Config file '{args.config_file}' doesn't have a .yaml extension. Adding it.")
        args.config_file = args.config_file + ".yaml"

    # Generate the script
    sbatch_script = generate_sample_sbatch_script(args)

    # Write to output file
    with open(args.output, "w") as f:
        f.write(sbatch_script)

    os.chmod(args.output, 0o755)  # Make executable

    print(f"Sampling SBATCH script created: {args.output}")
    print(f"You can submit your job with: sbatch {args.output}")


if __name__ == "__main__":
    main()
