from typing import Tuple, Optional, List, Dict
import collections
import time
import subprocess
import argparse
import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

sys.path.append("./")

from jamun.utils.slurm import wait_for_jobs
import load_trajectory


def run_analysis(args) -> Tuple[str, Optional[str], Optional[str]]:
    """Run analysis for a single peptide."""
    peptide, trajectory, reference, run_path, experiment, output_dir, use_sbatch = args

    cmd = []
    if use_sbatch:
        cmd += ["sbatch", "--partition=cpu", "--mem=64G"]        
    cmd += [
        "analysis/run_analysis.sh",
        f"--peptide={peptide}",
        f"--trajectory={trajectory}",
        f"--run-path={run_path}",
        f"--reference={reference}",
        f"--experiment={experiment}",
        f"--output-dir={output_dir}",
    ]
    print(f"Running command: {' '.join(cmd)}")
    try:
        launched = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return (peptide, None, e.stderr)

    if use_sbatch:
        job_id = launched.stdout.strip().split()[-1]
        return (peptide, job_id, None)
    else:
        return (peptide, None, None)


def main():
    parser = argparse.ArgumentParser(description="Run analysis for multiple peptides")
    parser.add_argument("--csv", type=str, required=True, help="CSV file containing wandb runs")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment type")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--num-workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers to launch",
    )
    parser.add_argument(
        "--no-use-sbatch", action="store_true", help="Do not use sbatch to launch analysis",
    )

    args = parser.parse_args()

    # Make output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Read wandb run paths from CSV.
    df = pd.read_csv(args.csv)

    # Choose type of trajectory to analyze.
    df = df[df["experiment"] == args.experiment]

    # Get run paths.
    df["run_path"] = df["wandb_sample_run_path"].map(load_trajectory.get_run_path_for_wandb_run)
    df["peptide"] = df["run_path"].map(load_trajectory.get_peptides_in_JAMUN_run)

    # Create one row for each peptide.
    df = df.explode("peptide")

    # Prepare arguments for parallel processing.
    analysis_args = list(
        zip(
            df["peptide"],
            df["trajectory"],
            df["reference"],
            df["run_path"],
            [args.experiment] * len(df),
            [args.output_dir] * len(df),
            [not args.no_use_sbatch] * len(df),
        )
    )

    # Launch analyses in parallel.
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(executor.map(run_analysis, analysis_args))

    # Process results.
    if not args.no_use_sbatch:

        job_ids = []
        for peptide, job_id, _ in results:
            print(f"Job {job_id} launched for peptide {peptide}.")
            job_ids.append(job_id)

        # Check job status.
        wait_for_jobs(job_ids)
        
    else:
        for peptide, _, error in results:
            if error:
                print(f"Error processing peptide {peptide}: {error}")
            else:
                print(f"Successfully processed peptide {peptide}")


if __name__ == "__main__":
    main()
