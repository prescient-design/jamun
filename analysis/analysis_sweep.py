import argparse
import os
import sys
import subprocess
from typing import Optional, List

import pandas as pd

sys.path.append("./")

import load_trajectory
import jamun.utils

def run_analysis(
    peptide: str,
    trajectory: str,
    reference: str,
    run_path: Optional[str],
    experiment: str,
    output_dir: str,
    shorten_trajectory_factor: Optional[int] = None,
) -> None:
    """Run analysis for a single peptide."""
    cmd = [
        "python",
        "run_analysis.py",
        f"--peptide={peptide}",
        f"--trajectory={trajectory}",
        f"--reference={reference}",
        f"--experiment={experiment}",
        f"--output-dir={output_dir}",
    ]

    if run_path is not None:
        cmd.append(f"--run-path={run_path}")

    if shorten_trajectory_factor is not None:
        cmd.append(f"--shorten-trajectory-factor={shorten_trajectory_factor}")

    print(f"Running command: {' '.join(cmd)}")
    try:
        launched = subprocess.run(cmd, check=True, stdout=None, stderr=None)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running command: {' '.join(cmd)}: {e.stderr}")


def get_dataframe_of_runs(csv: str, experiment: Optional[str] = None) -> pd.DataFrame:
    """Read the CSV file and filter for the specified experiment."""

    # Read wandb run paths from CSV.
    df = pd.read_csv(csv)

    # Choose type of trajectory to analyze.
    if experiment is not None:
        df = df[df["experiment"] == experiment]

    # Get run paths.
    df["run_path"] = df["wandb_sample_run_path"].map(jamun.utils.get_run_path_for_wandb_run)
    df["peptide"] = df["run_path"].map(load_trajectory.get_peptides_in_JAMUN_run)

    # Create one row for each peptide.
    df = df.explode("peptide")

    return df


def analyze_Boltz1_trajectories(args):
    """Analyze Boltz-1 samples based on the provided arguments."""
    # Make output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Read all peptides.
    data_path = load_trajectory.get_data_path(args.data_path)
    samples_path = os.path.join(data_path, "boltz-preprocessed")
    peptides = [
        os.path.splitext(filename)[0] for filename in os.listdir(samples_path) if filename.endswith(".pdb")
    ]
    peptides = list(sorted(peptides))
    print(f"Peptides: {peptides}")

    # Choose row to analyze.
    peptide = peptides[args.row_index]

    run_analysis(
        peptide=peptide,
        trajectory="BoltzSamples",
        reference="JAMUNReference_5AA",
        run_path=None,
        experiment=args.experiment,
        output_dir=args.output_dir,
        shorten_trajectory_factor=args.shorten_trajectory_factor,
    )


def analyze_JAMUN_trajectories(args):
    """Analyze JAMUN trajectories based on the provided arguments."""
    # Make output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Read the CSV file for experiments.
    df = get_dataframe_of_runs(args.csv, experiment=args.experiment)

    # Choose row to analyze.
    df = df.iloc[[args.row_index]]

    run_analysis(
        peptide=df["peptide"].iloc[0],
        trajectory=df["trajectory"].iloc[0],
        reference=df["reference"].iloc[0],
        run_path=df["run_path"].iloc[0],
        experiment=args.experiment,
        output_dir=args.output_dir,
        shorten_trajectory_factor=args.shorten_trajectory_factor,
    )


def analyze_MDGen_trajectories(args):
    """Analyze MDGen trajectories based on the provided arguments."""
    # Make output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Read all peptides.
    data_path = load_trajectory.get_data_path(args.data_path)

    if args.peptide_type == "4AA":
        samples_path = os.path.join(data_path, "mdgen-samples", "4AA_test")
    elif args.peptide_type == "5AA":
        samples_path = os.path.join(data_path, "mdgen-samples", "5AA_test")
    else:
        raise ValueError(f"Invalid peptide type: {args.peptide_type}")

    # List all peptides.
    peptides = [
        os.path.basename(filename).split("_")[0] for filename in os.listdir(samples_path) if filename.endswith(".pdb")
    ]
    peptides = list(sorted(peptides))
    print(f"Peptides: {peptides}")

    # Choose row to analyze.
    peptide = peptides[args.row_index]

    if args.peptide_type == "4AA":
        trajectory = "MDGenSamples_4AA"
    if args.peptide_type == "5AA":
        trajectory = "MDGenSamples_5AA"

    run_analysis(
        peptide=peptide,
        trajectory=trajectory,
        reference="MDGenReference",
        run_path=None,
        experiment=args.experiment,
        output_dir=args.output_dir,
        shorten_trajectory_factor=args.shorten_trajectory_factor,
    )


def analyze_BioEmu_trajectories(args):
    """Analyze BioEmu samples based on the provided arguments."""
    # Make output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Read all peptides.
    peptides = sorted(load_trajectory.load_all_trajectories_with_info(
        trajectory_name="BioEmuSamples", data_path=args.data_path
    ).keys())
    print(f"Peptides: {peptides}")

    # Choose row to analyze.
    peptide = peptides[args.row_index]

    run_analysis(
        peptide=peptide,
        trajectory="BioEmuSamples",
        reference="JAMUNReference_5AA",
        run_path=None,
        experiment=args.experiment,
        output_dir=args.output_dir,
        shorten_trajectory_factor=args.shorten_trajectory_factor,
    )


def analyze_TBG_trajectories(args):
    """Analyze TBG trajectories based on the provided arguments."""

    # Make output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Read all test peptides.
    data_path = load_trajectory.get_data_path(args.data_path)
    datasets = load_trajectory.get_TimewarpReference_datasets(data_path, peptide_type="2AA", split="test")
    peptides = list(sorted(datasets.keys()))

    # Choose row to analyze.
    peptide = peptides[args.row_index]

    run_analysis(
        peptide=peptide,
        trajectory="TBG",
        reference="TimewarpReference",
        run_path=None,
        experiment=args.experiment,
        output_dir=args.output_dir,
        shorten_trajectory_factor=args.shorten_trajectory_factor,
    )


def main():
    parser = argparse.ArgumentParser(description="Run analysis of trajectories for multiple peptides")

    # Create subparsers for different analysis types
    subparsers = parser.add_subparsers(dest="analysis_type", help="Type of analysis to run", required=True)

    # Boltz trajectory analysis
    boltz_parser = subparsers.add_parser("boltz", help="Analyze Boltz-1 trajectories")
    boltz_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    boltz_parser.add_argument("--row-index", type=int, required=True, help="Row index to analyze")
    boltz_parser.add_argument("--experiment", type=str, required=True, help="Experiment type")
    boltz_parser.add_argument(
        "--shorten-trajectory-factor", type=int, default=None, help="Factor to shorten trajectory by. Defaults to None."
    )
    boltz_parser.add_argument(
        "--data-path", type=str, help="Path to JAMUN data directory. Defaults to JAMUN_DATA_PATH environment variable."
    )

    # JAMUN trajectory analysis
    jamun_parser = subparsers.add_parser("jamun", help="Analyze JAMUN trajectories")
    jamun_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    jamun_parser.add_argument("--row-index", type=int, required=True, help="Row index to analyze")
    jamun_parser.add_argument("--experiment", type=str, required=True, help="Experiment type")
    jamun_parser.add_argument(
        "--shorten-trajectory-factor", type=int, default=None, help="Factor to shorten trajectory by. Defaults to None."
    )
    jamun_parser.add_argument(
        "--csv", type=str, required=True, help="CSV file containing information about wandb sampling runs"
    )

    # MDGen trajectory analysis
    mdgen_parser = subparsers.add_parser("mdgen", help="Analyze MDGen trajectories")
    mdgen_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    mdgen_parser.add_argument("--row-index", type=int, required=True, help="Row index to analyze")
    mdgen_parser.add_argument("--experiment", type=str, required=True, help="Experiment type")
    mdgen_parser.add_argument(
        "--shorten-trajectory-factor", type=int, default=None, help="Factor to shorten trajectory by. Defaults to None."
    )
    mdgen_parser.add_argument(
        "--data-path", type=str, help="Path to JAMUN data directory. Defaults to JAMUN_DATA_PATH environment variable."
    )
    mdgen_parser.add_argument(
        "--peptide-type", type=str, required=True, help="Peptide type", choices=["4AA", "5AA"]
    )

    # BioEmu sample analysis
    bioemu_parser = subparsers.add_parser("bioemu", help="Analyze BioEmu trajectories")
    bioemu_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    bioemu_parser.add_argument("--row-index", type=int, required=True, help="Row index to analyze")
    bioemu_parser.add_argument("--experiment", type=str, required=True, help="Experiment type")
    bioemu_parser.add_argument(
        "--shorten-trajectory-factor", type=int, default=None, help="Factor to shorten trajectory by. Defaults to None."
    )
    bioemu_parser.add_argument(
        "--data-path", type=str, help="Path to JAMUN data directory. Defaults to JAMUN_DATA_PATH environment variable."
    )

    tbg_parser = subparsers.add_parser("tbg", help="Analyze TBG trajectories")
    tbg_parser = argparse.ArgumentParser(description="Run analysis of TBG trajectories for multiple peptides")
    tbg_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    tbg_parser.add_argument(
        "--row-index",
        type=int,
        help="Row index to analyze",
    )
    tbg_parser.add_argument(
        "--data-path", type=str, help="Path to JAMUN data directory. Defaults to JAMUN_DATA_PATH environment variable."
    )
    tbg_parser.add_argument(
        "--shorten-trajectory-factor", type=int, default=None, help="Factor to shorten trajectory by. Defaults to None."
    )
    tbg_parser.add_argument("--experiment", type=str, required=True, help="Experiment type")

    args = parser.parse_args()


    if args.analysis_type == "boltz":
        analyze_Boltz1_trajectories(args)
    elif args.analysis_type == "jamun":
        analyze_JAMUN_trajectories(args)
    elif args.analysis_type == "mdgen":
        analyze_MDGen_trajectories(args)
    elif args.analysis_type == "bioemu":
        analyze_BioEmu_trajectories(args)
    elif args.analysis_type == "tbg":
        analyze_TBG_trajectories(args)
    else:
        raise ValueError(f"Invalid analysis type: {args.analysis_type}")


if __name__ == "__main__":
    main()
