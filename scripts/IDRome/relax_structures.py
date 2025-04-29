"""Script to relax IDRome v4 all-atom data."""

import argparse
import logging
import os
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("process_IDRome")


def relax_structure(args, use_sbatch: bool = True) -> None:
    """Convert IDRome v4 data to all-atom with PULCHRA."""
    name, pdb_path, output_dir = args

    os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_script = os.path.join(script_dir, "relax_structures.sh")

    cmd = [
        sbatch_script,
        pdb_path,
        output_dir,
    ]
    if use_sbatch:
        cmd = ["sbatch"] + cmd

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Extract job ID from output (typical output: "Submitted batch job 12345").
        if use_sbatch:
            job_id = result.stdout.strip().split()[-1]
            py_logger.info(f"Successfully submitted job {job_id}")
        else:
            py_logger.info(f"Successfully processed {name}")
    except subprocess.CalledProcessError as e:
        py_logger.error(f"Failed to submit sbatch job: {e}")
        py_logger.error(f"stdout: {e.stdout}")
        py_logger.error(f"stderr: {e.stderr}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IDRome v4 data to all-atom.")
    parser.add_argument(
        "--input-dir", help="Directory of all-atom trajectories (stored in each folder).", type=str, required=True
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory to save relaxed all-atom trajectories (stored in each folder).",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num-workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers"
    )
    args = parser.parse_args()

    # Run in parallel.
    names = list(sorted(os.listdir(args.input_dir)))
    pdb_paths = [list(sorted(os.listdir(os.path.join(args.input_dir, name)))) for name in names]
    names = [name for name in names for _ in range(len(pdb_paths))]
    preprocess_args = list(
        zip(
            names,
            pdb_paths,
            [args.output_dir] * len(names),
        )
    )
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(executor.map(relax_structure, preprocess_args))
