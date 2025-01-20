from typing import Dict, Optional
import os
import dotenv
import argparse
import sys
import pickle
from pathlib import Path

import pandas as pd
import mdtraj as md


sys.path.append("./")
import utils as analysis_utils

def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze molecular dynamics trajectories for peptide sequences.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--peptide',
        type=str,
        required=True,
        help='Peptide sequence to analyze (e.g., FAFG)'
    )
    
    parser.add_argument(
        '--trajectory',
        type=str,
        choices=['JAMUN'],
        default='JAMUN',
        help='Type of trajectory to analyze (currently only JAMUN supported)'
    )
    
    parser.add_argument(
        '--reference',
        type=str,
        choices=['MDGen', 'Timewarp'],
        default='Timewarp',
        help='Type of reference trajectory to compare against'
    )
    
    parser.add_argument(
        '--wandb-runs',
        type=str,
        nargs='+',
        default=[
            "prescient-design/jamun/xv2dsan8",
            "prescient-design/jamun/a8fukafx",
            "prescient-design/jamun/odb1bs62",
            "prescient-design/jamun/5dklwb4r"
        ],
        help='Weights & Biases run paths for JAMUN sampling runs'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to JAMUN data directory. Defaults to JAMUN_DATA_PATH environment variable.'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis_results',
        help='Directory to save analysis results'
    )
    
    parser.add_argument(
        '--t-steps',
        type=int,
        default=10,
        help='Number of time steps for JS distance analysis'
    )

    return parser.parse_args()


def load_trajectories(args):
    """Load trajectories based on command line arguments."""
    
    # Set up data path
    if args.data_path:
        JAMUN_DATA_PATH = args.data_path
    else:
        JAMUN_DATA_PATH = os.environ.get("JAMUN_DATA_PATH", dotenv.get_key("../.env", "JAMUN_DATA_PATH"))
        if not JAMUN_DATA_PATH:
            raise ValueError("JAMUN_DATA_PATH must be provided either via --data-path or environment variable")
    
    print(f"Using JAMUN_DATA_PATH: {JAMUN_DATA_PATH}")
    
    # Load trajectories
    filter_codes = [args.peptide]
    run_paths = [analysis_utils.get_run_path_for_wandb_run(path) for path in args.wandb_runs]
    
    # Load JAMUN trajectories
    if args.trajectory == 'JAMUN':
        trajs = analysis_utils.get_JAMUN_trajectories(run_paths, filter_codes=filter_codes)
    else:
        raise ValueError(f"Trajectory type {args.trajectory} not supported")

    if not trajs:
        raise ValueError(f"No {args.trajectory} trajectories found for peptide {args.peptide}")

    # Load reference trajectories
    if args.reference == 'Timewarp':
        ref_trajs = analysis_utils.get_Timewarp_trajectories(
            JAMUN_DATA_PATH, 
            peptide_type="4AA", 
            filter_codes=list(trajs.keys()), 
            split="test"
        )
    elif args.reference == 'MDGen':
        ref_trajs = analysis_utils.get_MDGen_trajectories(
            JAMUN_DATA_PATH,
            filter_codes=filter_codes,
            split="val"
        )
    else:
        raise ValueError(f"Reference type {args.reference} not supported")
    
    if not ref_trajs:
        raise ValueError(f"No {args.reference} trajectories found for peptide {args.peptide}")
        
    return trajs, ref_trajs


def analyze_trajectories(traj_md: md.Trajectory, ref_traj_md: md.Trajectory):
    """Run analysis on the trajectories and return results dictionary."""
    results = {}
        
    # Featurization, with and without cossin
    ref_feat_cossin, ref_traj_featurized_cossin = analysis_utils.featurize_trajectory(
        ref_traj_md, cossin=True
    )
    ref_feat, ref_traj_featurized = analysis_utils.featurize_trajectory(
        ref_traj_md, cossin=False
    )

    traj_feat_cossin, traj_featurized_cossin = analysis_utils.featurize_trajectory(
        traj_md, cossin=True
    )
    traj_feat, traj_featurized = analysis_utils.featurize_trajectory(
        traj_md, cossin=False
    )
    results['featurization'] = {
        'ref_feat_cossin': ref_feat_cossin,
        'ref_traj_featurized_cossin': ref_traj_featurized_cossin,
        'ref_feat': ref_feat,
        'ref_traj_featurized': ref_traj_featurized,
        'traj_feat_cossin': traj_feat_cossin,
        'traj_featurized_cossin': traj_featurized_cossin,
        'traj_feat': traj_feat,
        'traj_featurized': traj_featurized
    }

    # TICA analysis
    traj_tica, ref_tica, tica = analysis_utils.compute_TICA(traj_featurized_cossin, ref_traj_featurized_cossin)
    results['TICA'] = {
        'traj_tica': traj_tica,
        'ref_tica': ref_tica,
        'tica': tica,
    }

    # Compute JSDs
    results['JSD_stats'] = analysis_utils.compute_JSD_stats(traj_featurized, ref_traj_featurized, traj_feat)

    # Compute TICA stats
    results['TICA_stats'] = analysis_utils.compute_TICA_stats(traj_tica, ref_tica)

    # Compute autocorrelation stats
    results['autocorrelation_stats'] = analysis_utils.compute_autocorrelation_stats(traj_tica, ref_tica)

    # Compute MSM stats
    results['MSM_stats'] = analysis_utils.compute_MSM_stats(traj_featurized_cossin, ref_traj_featurized_cossin, tica)
    
    return results


def save_results(results, args):
    """Save analysis results to pickle file."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{args.peptide}_{args.reference}_{timestamp}.pkl"
    output_path = output_dir / filename
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'args': vars(args)
        }, f)
    
    print(f"\nResults saved to: {output_path}")

def main():
    args = parse_args()
    
    try:
        # Load trajectories
        trajs, ref_trajs = load_trajectories(args)
        print(f"\nSuccessfully loaded trajectories for {args.peptide}:")
        print(f"- JAMUN trajectory shape: {trajs[args.peptide].xyz.shape}")
        print(f"- {args.reference} reference shape: {ref_trajs[args.peptide].xyz.shape}")
        
        # Run analysis
        results = analyze_trajectories(trajs, ref_trajs, args.t_steps)
        
        # Save results
        save_results(results, args)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()