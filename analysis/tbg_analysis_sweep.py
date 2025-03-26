import subprocess
import argparse
import os
import sys
import pandas as pd

sys.path.append("./")

import load_trajectory
import analysis_sweep

def main():
    parser = argparse.ArgumentParser(description="Run analysis of TBG trajectories for multiple peptides")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--row-index", type=int, help="Row index to analyze",
    )
    parser.add_argument(
        "--data-path", type=str, help="Path to JAMUN data directory. Defaults to JAMUN_DATA_PATH environment variable."
    )
    parser.add_argument("--experiment", type=str, required=True, help="Experiment type")

    args = parser.parse_args()

    # Make output directory if it doesn't exist.
    os.makedirs(args.output_dir, exist_ok=True)

    # Read all peptides.
    data_path = load_trajectory.get_data_path(args.data_path)
    datasets = load_trajectory.get_TBG_datasets(data_path)
    peptides = list(sorted(datasets.keys()))

    # Choose row to analyze.
    peptide = peptides[args.row_index]

    analysis_sweep.run_analysis(
        peptide=peptide,
        trajectory="TBG",
        reference="TimewarpReference",
        run_path=None,
        experiment=args.experiment,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
