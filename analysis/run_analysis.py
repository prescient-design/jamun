from typing import Dict, Any, Tuple
import os
import dotenv
import logging
import argparse
import sys
import pickle

import pandas as pd
import mdtraj as md

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("analysis")


sys.path.append("./")
import utils as analysis_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze molecular dynamics trajectories for peptide sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--peptide", type=str, required=True, help="Peptide sequence to analyze (e.g., FAFG)")

    parser.add_argument(
        "--trajectory",
        type=str,
        choices=["JAMUN", "2AA_JAMUNReference", "5AA_JAMUNReference", "MDGenReference", "TimewarpReference"],
        help="Type of trajectory to analyze",
    )

    parser.add_argument(
        "--reference",
        type=str,
        choices=["2AA_JAMUNReference", "5AA_JAMUNReference", "MDGenReference", "TimewarpReference"],
        help="Type of reference trajectory to compare against",
    )

    parser.add_argument(
        "--wandb-runs",
        type=str,
        nargs="+",
        default=[
            "prescient-design/jamun/w5qiuq63",
            "prescient-design/jamun/flby06tj",
        ],
        # default=[
        #     "prescient-design/jamun/xv2dsan8",
        #     "prescient-design/jamun/a8fukafx",
        #     "prescient-design/jamun/odb1bs62",
        #     "prescient-design/jamun/5dklwb4r"
        # ],
        help="Weights & Biases run paths for JAMUN sampling runs",
    )

    parser.add_argument(
        "--data-path", type=str, help="Path to JAMUN data directory. Defaults to JAMUN_DATA_PATH environment variable."
    )

    parser.add_argument("--output-dir", type=str, default="analysis_results", help="Directory to save analysis results")

    return parser.parse_args()


def load_trajectories_by_name(
    name: str,
    args: argparse.Namespace,
):
    # Set up data path
    if args.data_path:
        JAMUN_DATA_PATH = args.data_path
    else:
        JAMUN_DATA_PATH = os.environ.get("JAMUN_DATA_PATH", dotenv.get_key(".env", "JAMUN_DATA_PATH"))
        if not JAMUN_DATA_PATH:
            raise ValueError("JAMUN_DATA_PATH must be provided either via --data-path or environment variable")
    py_logger.info(f"Using JAMUN_DATA_PATH: {JAMUN_DATA_PATH}")

    filter_codes = [args.peptide]
    if name == "JAMUN":
        run_paths = [analysis_utils.get_run_path_for_wandb_run(path) for path in args.wandb_runs]
        return analysis_utils.get_JAMUN_trajectories(run_paths, filter_codes=filter_codes)
    elif name == "MDGenReference":
        return analysis_utils.get_MDGenReference_trajectories(
            JAMUN_DATA_PATH,
            filter_codes=filter_codes,
        )
    elif name == "TimewarpReference":
        return analysis_utils.get_TimewarpReference_trajectories(
            JAMUN_DATA_PATH,
            filter_codes=filter_codes,
        )
    elif name == "2AA_JAMUNReference":
        return analysis_utils.get_2AA_JAMUNReference_trajectories(
            JAMUN_DATA_PATH,
            filter_codes=filter_codes,
        )
    elif name == "5AA_JAMUNReference":
        return analysis_utils.get_5AA_JAMUNReference_trajectories(
            JAMUN_DATA_PATH,
            filter_codes=filter_codes,
        )
    else:
        raise ValueError(f"Trajectory type {args.trajectory} not supported")


def load_trajectories(args) -> Tuple[md.Trajectory, md.Trajectory]:
    """Load trajectories based on command line arguments."""

    trajs_md = load_trajectories_by_name(args.trajectory, args)
    if not trajs_md:
        raise ValueError(f"No {args.trajectory} trajectories found for peptide {args.peptide}")

    ref_trajs_md = load_trajectories_by_name(args.reference, args)
    if not ref_trajs_md:
        raise ValueError(f"No {args.reference} trajectories found for peptide {args.peptide}")

    return trajs_md[args.peptide], ref_trajs_md[args.peptide]


def analyze_trajectories(traj_md: md.Trajectory, ref_traj_md: md.Trajectory) -> Dict[str, Any]:
    """Run analysis on the trajectories and return results dictionary."""
    results = {}

    # Featurization, with and without cossin
    ref_traj_feat_cossin, ref_traj_featurized_cossin = analysis_utils.featurize_trajectory(ref_traj_md, cossin=True)
    ref_traj_feat, ref_traj_featurized = analysis_utils.featurize_trajectory(ref_traj_md, cossin=False)

    traj_feat_cossin, traj_featurized_cossin = analysis_utils.featurize_trajectory(traj_md, cossin=True)
    traj_feat, traj_featurized = analysis_utils.featurize_trajectory(traj_md, cossin=False)
    results["featurization"] = {
        "ref_traj_feat_cossin": ref_traj_feat_cossin,
        "ref_traj_featurized_cossin": ref_traj_featurized_cossin,
        "ref_traj_feat": ref_traj_feat,
        "ref_traj_featurized": ref_traj_featurized,
        "traj_feat_cossin": traj_feat_cossin,
        "traj_featurized_cossin": traj_featurized_cossin,
        "traj_feat": traj_feat,
        "traj_featurized": traj_featurized,
    }
    py_logger.info(f"Featurization complete.")

    # TICA analysis
    traj_tica, ref_tica, tica = analysis_utils.compute_TICA(traj_featurized_cossin, ref_traj_featurized_cossin)
    results["TICA"] = {
        "traj_tica": traj_tica,
        "ref_tica": ref_tica,
        "tica": tica,
    }
    py_logger.info(f"TICA computed.")

    # Compute PMFs
    results["PMFs"] = analysis_utils.compute_PMFs(traj_md, ref_traj_md)
    py_logger.info(f"PMFs computed.")

    # Compute bond lengths
    results["bond_lengths"] = analysis_utils.compute_bond_lengths(traj_md, ref_traj_md)
    py_logger.info(f"Bond lengths computed.")

    # Compute JSDs
    results["JSD_stats"] = analysis_utils.compute_JSD_stats(traj_featurized, ref_traj_featurized, traj_feat)
    py_logger.info(f"JSD stats computed.")

    # Compute JSDs
    results["JSD_stats_against_time"] = analysis_utils.compute_JSDs_stats_against_time(
        traj_featurized, ref_traj_featurized, traj_feat
    )
    py_logger.info(f"JSD stats as a function of time computed.")

    # Compute TICA stats
    results["TICA_stats"] = analysis_utils.compute_TICA_stats(traj_tica, ref_tica)
    py_logger.info(f"TICA stats computed.")

    # Compute autocorrelation stats
    results["autocorrelation_stats"] = analysis_utils.compute_autocorrelation_stats(traj_tica, ref_tica)
    py_logger.info(f"Autocorrelation stats computed.")

    # Compute MSM stats
    results["MSM_stats"] = analysis_utils.compute_MSM_stats(traj_featurized_cossin, ref_traj_featurized_cossin, tica)
    py_logger.info(f"MSM stats computed.")

    return results


def save_results(results, args):
    """Save analysis results to pickle file."""
    output_dir = os.path.join(args.output_dir, args.trajectory, f"ref={args.reference}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.peptide}.pkl")

    with open(output_path, "wb") as f:
        pickle.dump({"results": results, "args": vars(args)}, f)

    py_logger.info(f"Results saved to: {os.path.abspath(output_path)}")


def main():
    args = parse_args()

    # Load trajectories
    traj, ref_traj = load_trajectories(args)
    py_logger.info(f"Successfully loaded trajectories for {args.peptide}:")
    py_logger.info(f"{args.trajectory} trajectory loaded: {traj}")
    py_logger.info(f"{args.reference} reference trajectory loaded: {ref_traj}")

    # Run analysis
    results = analyze_trajectories(traj, ref_traj)

    # Save results
    save_results(results, args)


if __name__ == "__main__":
    main()
