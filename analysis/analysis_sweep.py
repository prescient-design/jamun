from typing import Tuple, Optional
import subprocess
import argparse
import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

sys.path.append("./")

import load_trajectory


def run_analysis(args, use_srun: bool = True) -> Tuple[str, Optional[str]]:
    """Run analysis for a single peptide."""
    peptide, trajectory, reference, run_path, experiment, output_dir, same_sampling_time = args

    cmd = []
    if use_srun:
        cmd += ["srun", "--partition=cpu", "--mem=64G"]
    cmd += [
        "python",
        "analysis/run_analysis.py",
        f"--peptide={peptide}",
        f"--trajectory={trajectory}",
        f"--run-path={run_path}",
        f"--reference={reference}",
        f"--experiment={experiment}",
        f"--output-dir={output_dir}",
    ]
    if same_sampling_time:
        cmd += ["--same-sampling-time"]
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return (peptide, None)
    except subprocess.CalledProcessError as e:
        return (peptide, str(e))


def main():
    parser = argparse.ArgumentParser(description="Run analysis for multiple peptides")
    parser.add_argument("--csv", type=str, required=True, help="CSV file containing wandb runs")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment type")
    parser.add_argument(
        "--same-sampling-time", action="store_true", help="If set, will subset reference trajectory to match the length of the trajectory in actual sampling time.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--num-workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers"
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
            [args.same_sampling_time] * len(df),
        )
    )

    # Run analyses in parallel.
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(executor.map(run_analysis, analysis_args))

    # Process results.
    for peptide, error in results:
        if error:
            print(f"Error processing peptide {peptide}: {error}")
        else:
            print(f"Successfully processed peptide {peptide}")


if __name__ == "__main__":
    main()
