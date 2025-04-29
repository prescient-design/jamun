#!/usr/bin/env python3
"""
Script to generate an SBATCH script for jamun_train based on a config file.

Usage:
    ./generate_train_sbatch.py CONFIG_FILENAME [OPTIONS]

Arguments:
    CONFIG_FILENAME - Name of the training config file (e.g., train_uncapped_2AA.yaml)

Options:
    --partition PARTITION   - SLURM partition (default: gpu2)
    --nodes N               - Number of nodes (default: 1)
    --tasks N               - Tasks per node (default: 2)
    --gpus N                - GPUs per node (default: 2)
    --cpus N                - CPUs per task (default: 8)
    --time TIME             - Time limit (default: 7-0)
    --mem MEM               - Memory per CPU (default: 32G)
    --env-name NAME         - Conda environment name (default: jamun)
    --output OUTPUT         - Output file (default: CONFIG_NAME.sbatch)
"""

import os
import sys
import argparse
import subprocess


def find_project_root():
    """Returns the root directory of the project."""
    return subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SBATCH script for jamun_train")
    parser.add_argument("config_file", help="Config file name (e.g., train_uncapped_2AA.yaml)")
    parser.add_argument("--config-dir", default=os.path.join(find_project_root(), "configs"), help="Config directory")
    parser.add_argument("--partition", default="gpu2", help="SLURM partition")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--tasks", type=int, default=2, help="Tasks per node")
    parser.add_argument("--gpus", type=int, default=2, help="GPUs per node")
    parser.add_argument("--cpus", type=int, default=8, help="CPUs per task")
    parser.add_argument("--time", default="7-0", help="Time limit (7-0 means 7 days)")
    parser.add_argument("--mem", default="32G", help="Memory per CPU")
    parser.add_argument("--env-name", default="jamun", help="Conda environment name")
    parser.add_argument("--output", help="Output script file")

    return parser.parse_args()


def generate_train_sbatch_script(args):
    # Extract config name without extension
    config_name = os.path.splitext(args.config_file)[0]

    # Default output filename if not specified
    if args.output is None:
        args.output = f"{config_name}.sbatch"

    # Extract config tag (remove 'train_' prefix if present)
    config_tag = config_name
    if config_name.startswith("train_"):
        config_tag = config_name[6:]

    # Create the SBATCH script
    script = f'''#!/usr/bin/env bash

#SBATCH --partition {args.partition}
#SBATCH --nodes {args.nodes}
#SBATCH --ntasks-per-node {args.tasks}
#SBATCH --gpus-per-node {args.gpus}
#SBATCH --cpus-per-task {args.cpus}
#SBATCH --time {args.time}
#SBATCH --mem-per-cpu={args.mem}

eval "$(conda shell.bash hook)"
conda activate {args.env_name}

set -eux

echo "SLURM_JOB_ID = ${{SLURM_JOB_ID}}"
echo "hostname = $(hostname)"

export HYDRA_FULL_ERROR=1
# export TORCH_COMPILE_DEBUG=1
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1

# NOTE: We generate this in submit script instead of using time-based default to ensure consistency across ranks.
RUN_KEY=$(openssl rand -hex 12)
echo "RUN_KEY = ${{RUN_KEY}}"

nvidia-smi

srun --cpus-per-task {args.cpus} --cpu-bind=cores,verbose \\
    jamun_train --config-dir={args.config_dir} \\
        experiment={args.config_file} \\
        ++trainer.devices=$SLURM_GPUS_PER_NODE \\
        ++trainer.num_nodes=$SLURM_JOB_NUM_NODES \\
        ++logger.wandb.tags=["'${{SLURM_JOB_ID}}'","'${{RUN_KEY}}'","train","{config_tag}"] \\
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
    sbatch_script = generate_train_sbatch_script(args)

    # Write to output file
    with open(args.output, "w") as f:
        f.write(sbatch_script)

    os.chmod(args.output, 0o755)  # Make executable

    print(f"Training SBATCH script created: {args.output}")
    print(f"You can submit your job with: sbatch {args.output}")


if __name__ == "__main__":
    main()
