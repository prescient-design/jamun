"""Script to copy over IDRome v4 data."""

from typing import Tuple, Optional, List
import argparse
import logging
import os
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("process_IDRome")


def find_trajectory_directories(root_dir: str) -> List[str]:
    """Find all directories containing .pdb files."""
    traj_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(filename == "top.pdb" for filename in filenames) and any(
            filename == "traj.xtc" for filename in filenames
        ):
            traj_dirs.append(dirpath)
    traj_dirs = list(sorted(traj_dirs))
    return traj_dirs


def copy_trajectory_files(args, use_sbatch: bool = True) -> None:
    """Copy trajectory files to a new directory."""
    traj_dir, input_dir, output_dir = args

    traj_dir_short = traj_dir.replace(input_dir + "/", "")
    name = traj_dir_short.replace("/", "_")

    os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    pdb_path = os.path.join(traj_dir, "top.pdb")
    pdb_path_renamed = os.path.join(output_dir, name, "top.pdb")
    xtc_path = os.path.join(traj_dir, "traj.xtc")
    xtc_path_renamed = os.path.join(output_dir, name, "traj.xtc")

    if os.path.exists(pdb_path_renamed) and os.path.exists(xtc_path_renamed):
        py_logger.info(f"Skipping {pdb_path_renamed} and {xtc_path_renamed} as they already exist.")
        return

    # Path to the sbatch script (assuming it's in the same directory as this Python file).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_script = os.path.join(script_dir, "copy_trajectories.sh")

    # Submit the job.
    cmd = [sbatch_script, pdb_path, pdb_path_renamed, xtc_path, xtc_path_renamed]
    if use_sbatch:
        cmd = ["sbatch"] + cmd

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if use_sbatch:
            job_id = result.stdout.strip().split()[-1]
            py_logger.info(f"Successfully submitted job {job_id}")
        else:
            py_logger.info(
                f"Successfully copied {pdb_path} and {xtc_path} to {pdb_path_renamed} and {xtc_path_renamed}"
            )
    except subprocess.CalledProcessError as e:
        py_logger.error(f"Failed to submit sbatch job: {e}")
        py_logger.error(f"stdout: {e.stdout}")
        py_logger.error(f"stderr: {e.stderr}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy over IDRome v4 data.")
    parser.add_argument(
        "--input-dir", help="Directory where original trajectories were downloaded.", type=str, required=True
    )
    parser.add_argument("--output-dir", "-o", help="Output directory to save trajectories.", type=str, required=True)
    parser.add_argument(
        "--num-workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers"
    )
    args = parser.parse_args()

    # Run in parallel.
    traj_dirs = find_trajectory_directories(args.input_dir)
    preprocess_args = list(
        zip(
            traj_dirs,
            [args.input_dir] * len(traj_dirs),
            [args.output_dir] * len(traj_dirs),
        )
    )
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(executor.map(copy_trajectory_files, preprocess_args))
