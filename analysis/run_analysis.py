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


# TODO: Fix imports
sys.path.append("./")
import utils as analysis_utils
import load_trajectory as load_trajectory


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze molecular dynamics trajectories for peptide sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--peptide",
        type=str,
        required=True,
        help="Peptide sequence to analyze (e.g., FAFG)"
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        choices=["JAMUN", "JAMUNReference_2AA", "JAMUNReference_5AA", "MDGenReference", "TimewarpReference"],
        help="Type of trajectory to analyze",
    )
    parser.add_argument(
        "--reference",
        type=str,
        choices=["JAMUNReference_2AA", "JAMUNReference_5AA", "MDGenReference", "TimewarpReference"],
        help="Type of reference trajectory to compare against",
    )
    parser.add_argument(
        "--run-path",
        type=str,
        help="Path to JAMUN run directory containing trajectory files",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        help="Weights & Biases run paths for JAMUN sampling run. Can be used instead of --run-path",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to JAMUN data directory. Defaults to JAMUN_DATA_PATH environment variable."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--no-delete-intermediates",
        action="store_true",
        default=False,
        help="Don't delete intermediate results to reduce memory usage",
    )
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
        if not args.run_path and not args.wandb_run:
            raise ValueError("Must provide either --run-path or --wandb-run for JAMUN trajectory")
        if args.run_path and args.wandb_run:
            raise ValueError("Must provide only one of --run-path or --wandb-run for JAMUN trajectory")

        if args.wandb_run:
            run_paths = [load_trajectory.get_run_path_for_wandb_run(args.wandb_run)]
        else:
            run_paths = [args.run_path]
        return load_trajectory.get_JAMUN_trajectories(run_paths, filter_codes=filter_codes)
    elif name == "MDGenReference":
        return load_trajectory.get_MDGenReference_trajectories(
            JAMUN_DATA_PATH,
            filter_codes=filter_codes,
        )
    elif name == "TimewarpReference":
        return load_trajectory.get_TimewarpReference_trajectories(
            JAMUN_DATA_PATH,
            filter_codes=filter_codes,
        )
    elif name == "JAMUNReference_2AA":
        return load_trajectory.get_JAMUNReference_2AA_trajectories(
            JAMUN_DATA_PATH,
            filter_codes=filter_codes,
        )
    elif name == "JAMUNReference_5AA":
        return load_trajectory.get_JAMUNReference_5AA_trajectories(
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

    # Featurize trajectories.
    results["featurization"] = analysis_utils.featurize(traj_md, ref_traj_md)
    
    traj_results = results["featurization"]["traj"]
    traj_feats = traj_results["feats"]["torsions"]
    traj_featurized_dict = traj_results["traj_featurized"]
    traj_featurized = traj_featurized_dict["torsions"]

    ref_traj_results = results["featurization"]["ref_traj"]
    ref_traj_feats = ref_traj_results["feats"]["torsions"]
    ref_traj_featurized_dict = ref_traj_results["traj_featurized"]
    ref_traj_featurized = ref_traj_featurized_dict["torsions"]
    py_logger.info(f"Featurization complete.")

    # Compute feature histograms.
    results["feature_histograms"] = analysis_utils.compute_feature_histograms(
        traj_featurized_dict,
        ref_traj_featurized_dict,
    )
    py_logger.info(f"Feature histograms computed.")

    # Compute PMFs.
    results["PMFs"] = analysis_utils.compute_PMFs(
        traj_featurized,
        ref_traj_featurized,
        traj_feats,
    )
    py_logger.info(f"PMFs computed.")

    # Compute JSDs.
    results["JSD_stats"] = analysis_utils.compute_JSD_stats(
        traj_featurized,
        ref_traj_featurized,
        traj_feats,
    )
    py_logger.info(f"JSD stats computed.")

    # Compute JSDs.
    results["JSD_stats_against_time"] = analysis_utils.compute_JSDs_stats_against_time(
        traj_featurized,
        ref_traj_featurized,
        traj_feats,
    )
    py_logger.info(f"JSD stats as a function of time computed.")

    traj_featurized_cossin = traj_featurized_dict["torsions_cossin"]
    ref_traj_featurized_cossin = ref_traj_featurized_dict["torsions_cossin"]

    # TICA analysis.
    results["TICA"] = analysis_utils.compute_TICA(
        traj_featurized_cossin,
        ref_traj_featurized_cossin,
    )
    py_logger.info(f"TICA computed.")

    traj_tica = results["TICA"]["traj_tica"]
    ref_traj_tica = results["TICA"]["ref_traj_tica"]

    # Compute MSM stats.
    results["MSM_stats"] = analysis_utils.compute_MSM_stats(
        traj_tica,
        ref_traj_tica,
    )
    py_logger.info(f"MSM stats computed.")

    # Compute TICA stats.
    results["TICA_stats"] = analysis_utils.compute_TICA_stats(
        traj_tica,
        ref_traj_tica,
    )
    py_logger.info(f"TICA stats computed.")

    # Compute autocorrelation stats.
    results["autocorrelation_stats"] = analysis_utils.compute_autocorrelation_stats(
        traj_tica,
        ref_traj_tica,
    )
    py_logger.info(f"Autocorrelation stats computed.")

    return results


def save_results(results, args):
    """Save analysis results to pickle file."""

    # Delete intermediate results, to reduce memory usage.
    if not args.no_delete_intermediates:
        del results["featurization"]["traj"]["traj_featurized"]
        del results["featurization"]["ref_traj"]["traj_featurized"]
        del results["TICA"]["traj_tica"]
        del results["TICA"]["ref_traj_tica"]

    output_dir = os.path.join(args.output_dir, args.trajectory, f"ref={args.reference}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.peptide}.pkl")

    # Print size of each object in results
    # def print_size(d, name):
    #     for k, v in d.items():
    #         if isinstance(v, dict):
    #             print_size(v, f"{name}.{k}")
    #         else:
    #             py_logger.info(f"{name}.{k}: {sys.getsizeof(v)} bytes")

    # print_size(results, "results")

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
