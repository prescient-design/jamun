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
        "analysis/run_analysis.py",
        f"--peptide={peptide}",
        f"--trajectory={trajectory}",
        f"--run-path={run_path}",
        f"--reference={reference}",
        f"--experiment={experiment}",
        f"--output-dir={output_dir}", 
    ]
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}: {e.stderr}")
        print("Skipping to the next command.")

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
    
    print(f"wandb_sample_run_path: {df['wandb_sample_run_path'].unique()}")
    # Get run paths.
    #df["run_path"] = df["wandb_sample_run_path"].map(load_trajectory.get_run_path_for_wandb_run)
    df["run_path"] = "/data/davidsd5/jamun_run/outputs/sample/dev/runs/29bf247f7d538e55ba2f2f7f"
    df["peptide"] = df["run_path"].map(load_trajectory.get_peptides_in_JAMUN_run)
    # Create one row for each peptide.
    df = df.explode("peptide")

    # Choose row to analyze.
    print(df)
    # Iterate over all rows in the DataFrame and run analysis for each.
    for _, row in df.iterrows():
        run_analysis(
            peptide=row["peptide"],
            trajectory=row["trajectory"],
            reference=row["reference"],
            run_path=row["run_path"],
            experiment=args.experiment,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
