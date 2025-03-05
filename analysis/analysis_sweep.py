import subprocess
import argparse
import os
import sys
import pandas as pd

sys.path.append("./")

import load_trajectory


def run_analysis(peptide: str, trajectory: str, reference: str, run_path: str, experiment: str, output_dir: str) -> None:
    """Run analysis for a single peptide."""
    cmd = [
        "python",
        "run_analysis.py",
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
        raise RuntimeError(f"Error running command: {' '.join(cmd)}: {e.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Run analysis for multiple peptides")
    parser.add_argument("--csv", type=str, required=True, help="CSV file containing wandb runs")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment type")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--row-index", type=int, help="Row index to analyze",
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

    # Choose row to analyze.
    df = df.iloc[[args.row_index]]
    
    run_analysis(
        peptide=df["peptide"].iloc[0],
        trajectory=df["trajectory"].iloc[0],
        reference=df["reference"].iloc[0],
        run_path=df["run_path"].iloc[0],
        experiment=args.experiment,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
