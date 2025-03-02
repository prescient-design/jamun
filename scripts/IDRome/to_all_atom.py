"""Script to convert IDRome v4 coarse-grained data to all-atom."""

import argparse
import logging
import os
import multiprocessing
import subprocess
import tempfile
import mdtraj as md
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("process_IDRome")


def to_all_atom(args, use_sbatch: bool = False) -> None:
    """Convert IDRome v4 data to all-atom with PULCHRA."""
    name, input_dir, output_dir = args
    pdb_path = os.path.join(input_dir, name, 'top.pdb')
    xtc_path = os.path.join(input_dir, name, 'traj.xtc')
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sbatch_script = os.path.join(script_dir, "to_all_atom.sh")

    top = md.load_topology(pdb_path)
    traj = md.load_xtc(xtc_path, top=top)
    temp_dir = tempfile.mkdtemp()
    for frame in range(traj.n_frames):
        temp_path = os.path.join(temp_dir, f'{name}_{frame}.pdb')
        pulchra_temp_path = os.path.join(temp_dir, f'{name}_{frame}.rebuilt.pdb')
        final_path = os.path.join(output_dir, name, f"{frame}.pdb")

        if os.path.exists(final_path):
            py_logger.info(f"Skipping {final_path} as it already exists.")
            continue

        traj[frame].save_pdb(temp_path)
        cmd = [sbatch_script, temp_path, pulchra_temp_path, final_path]
        if use_sbatch:
            cmd = ['sbatch'] + cmd

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            # Extract job ID from output (typical output: "Submitted batch job 12345").
            if use_sbatch:
                job_id = result.stdout.strip().split()[-1]
                py_logger.info(f"Successfully submitted job {job_id}")
            else:
                py_logger.info(f"Successfully processed frame {frame} for {name}")

        except subprocess.CalledProcessError as e:
            py_logger.error(f"Failed to submit sbatch job: {e}")
            py_logger.error(f"stdout: {e.stdout}")
            py_logger.error(f"stderr: {e.stderr}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert IDRome v4 data to all-atom.')
    parser.add_argument('--input-dir', help='Directory of trajectories (stored in each folder).', type=str, required=True)
    parser.add_argument('--output-dir', '-o', help='Output directory to save all-atom trajectories (stored in each folder).', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=multiprocessing.cpu_count(),
                      help='Number of parallel workers')
    args = parser.parse_args()

    # Run in parallel.
    names = list(sorted(os.listdir(args.input_dir)))[::-1]
    preprocess_args = list(
        zip(
            names,
            [args.input_dir] * len(names),
            [args.output_dir] * len(names),
        )
    )
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(executor.map(to_all_atom, preprocess_args))


