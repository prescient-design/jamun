"""Script to convert IDRome v4 coarse-grained data to all-atom."""

import argparse
import logging
import os
import multiprocessing
import subprocess
import time
import mdtraj as md
from concurrent.futures import ProcessPoolExecutor

from jamun.utils.slurm import wait_for_jobs

# Set up logging
logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("process_IDRome")


def extract_frames(name: str, input_dir: str, output_dir: str) -> None:
    """Convert IDRome v4 data to all-atom with PULCHRA."""
    pdb_path = os.path.join(input_dir, name, "top.pdb")
    xtc_path = os.path.join(input_dir, name, "traj.xtc")
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    top = md.load_topology(pdb_path)
    traj = md.load_xtc(xtc_path, top=top)

    for frame in range(traj.n_frames):
        output_path = os.path.join(output_dir, name, f"{frame}.pdb")
        if not os.path.exists(output_path):
            traj[frame].save_pdb(output_path)

    py_logger.info(f"Successfully processed {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from IDRome v4 trajectories.")
    parser.add_argument("--name", help="Name of the trajectory.", type=str, required=True)
    parser.add_argument(
        "--input-dir", help="Directory of trajectories (stored in each folder).", type=str, required=True
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory to save all-atom trajectories (stored in each folder).",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    extract_frames(args.name, args.input_dir, args.output_dir)
