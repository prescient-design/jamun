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


def to_all_atom(args, use_sbatch: bool = True) -> None:
    """Convert IDRome v4 data to all-atom with PULCHRA."""
    name, input_dir, output_dir = args
    pdb_path = os.path.join(input_dir, name, 'top.pdb')
    xtc_path = os.path.join(input_dir, name, 'traj.xtc')
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    top = md.load_topology(pdb_path)
    traj = md.load_xtc(xtc_path, top=top)

    for frame in range(traj.n_frames):
        output_path = os.path.join(output_dir, name, f"{frame}.pdb")
        if not os.path.exists(output_path):
            traj[frame].save_pdb(output_path)

    py_logger.info(f"Successfully processed {name}")
    return None
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # sbatch_script = os.path.join(script_dir, "to_all_atom_batched.sh")

    # if not use_sbatch:
    #     raise NotImplementedError("This script is not yet implemented without sbatch.")

    # # Submit the job array
    # cmd = [
    #     'sbatch',
    #     f'--array=0-{traj.n_frames-1}%{1}',
    #     sbatch_script,
    #     temp_dir,
    #     name,
    #     output_dir,
    #     str(traj.n_frames)
    # ]

    # try:
    #     result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    #     # Sleep for a short time to ensure the job is registered.
    #     time.sleep(1)

    #     # Extract job ID from output (typical output: "Submitted batch job 12345").
    #     if use_sbatch:
    #         job_id = result.stdout.strip().split()[-1]
    #         py_logger.info(f"Successfully submitted job {job_id} for {name}")
    #         return job_id
    #     else:
    #         py_logger.info(f"Successfully processed {name}")

    # except subprocess.CalledProcessError as e:
    #     py_logger.error(f"Failed to submit sbatch job: {e}")
    #     py_logger.error(f"stdout: {e.stdout}")
    #     py_logger.error(f"stderr: {e.stderr}")
    #     raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert IDRome v4 data to all-atom.')
    parser.add_argument('--input-dir', help='Directory of trajectories (stored in each folder).', type=str, required=True)
    parser.add_argument('--output-dir', '-o', help='Output directory to save all-atom trajectories (stored in each folder).', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=multiprocessing.cpu_count() - 1,
                        help='Number of parallel workers')
    args = parser.parse_args()

    # Run in parallel.
    names = list(sorted(os.listdir(args.input_dir)))
    py_logger.info(f"Processing {len(names)} trajectories.")

    preprocess_args = list(
        zip(
            names,
            [args.input_dir] * len(names),
            [args.output_dir] * len(names),
        )
    )

    # Submit jobs.
    # job_ids = []
    # for i in range(len(preprocess_args)):
    #     job_ids.append(to_all_atom(preprocess_args[i]))
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        job_ids = list(executor.map(to_all_atom, preprocess_args))

    # Wait for all jobs to finish.
    if job_ids:
        wait_for_jobs(job_ids)
    else:
        py_logger.info("No jobs were submitted.")

