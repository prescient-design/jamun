import argparse
import os
import sys

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
    parser.add_argument(
        "--shorten-trajectory-factor", type=int, default=None, help="Factor to shorten trajectory by. Defaults to None."
    )
    parser.add_argument("--peptide-type", type=str, required=True, help="Peptide type", choices=["4AA", "5AA"])
    parser.add_argument("--experiment", type=str, required=True, help="Experiment type")

    args = parser.parse_args()

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
    peptides = [os.path.basename(filename).split("_")[0] for filename in os.listdir(samples_path) if filename.endswith(".pdb")]
    peptides = list(sorted(peptides))
    print(f"Peptides: {peptides}")

    # Choose row to analyze.
    peptide = peptides[args.row_index]

    if args.peptide_type == "4AA":
        trajectory = "MDGenSamples_4AA"
    if args.peptide_type == "5AA":
        trajectory = "MDGenSamples_5AA"

    analysis_sweep.run_analysis(
        peptide=peptide,
        trajectory=trajectory,
        reference="MDGenReference",
        run_path=None,
        experiment=args.experiment,
        output_dir=args.output_dir,
        shorten_trajectory_factor=args.shorten_trajectory_factor
    )


if __name__ == "__main__":
    main()
