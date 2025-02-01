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

import load_trajectory


def run_analysis(args) -> Tuple[str, Optional[str]]:
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
        return (peptide, str(e))

    if use_sbatch:
        job_id = launched.stdout.strip().split()[-1]
        return (peptide, job_id)
    else:
        return (peptide, None)


def wait_for_jobs(job_ids: List[str], poll_interval: int = 60):
    previous_states = collections.defaultdict(str)
    completion_count = 0
    total_jobs = len(job_ids)
    
    while True:
        cmd = [
            "sacct", 
            "-j", ",".join(job_ids),
            "--format=JobID,State", 
            "--noheader",
            "--parsable2"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        current_states: Dict[str, str] = {}
        
        # Parse current states
        for line in result.stdout.strip().split('\n'):
            if not line: continue
            jobid, state = line.split('|')
            if '.' not in jobid:  # Only main jobs
                current_states[jobid] = state
                # If job just completed (wasn't completed before)
                if state == 'COMPLETED' and previous_states[jobid] != 'COMPLETED':
                    completion_count += 1
                    print(f"Job {jobid} completed successfully. Progress: {completion_count}/{total_jobs}")

        # Update states for next iteration
        previous_states.update(current_states)
        
        # Group jobs by state for summary
        states_summary = collections.defaultdict(int)
        for state in current_states.values():
            states_summary[state] += 1
            
        print(f"\nStatus summary:")
        print(f"Completed: {completion_count}/{total_jobs} ({completion_count/total_jobs*100:.1f}%)")
        print(f"Current states: {dict(states_summary)}")
        
        # Check if all jobs reached terminal state
        all_done = all(state in ['COMPLETED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'CANCELLED'] 
                      for state in current_states.values())
        
        if all_done:
            print("\nAll jobs finished!")
            failures = [jid for jid, state in current_states.items() if state != 'COMPLETED']
            if failures:
                print(f"Failed jobs: {failures}")
            break
            
        time.sleep(poll_interval)
    
    return completion_count


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
        for peptide, job_id in results:
            print(f"Job {job_id} launched for peptide {peptide}.")

        # Check job status.
        job_ids = [job_id for _, job_id in results]
        wait_for_jobs(job_ids)
        
    else:
        for peptide, error in results:
            if error:
                print(f"Error processing peptide {peptide}: {error}")
            else:
                print(f"Successfully processed peptide {peptide}")


if __name__ == "__main__":
    main()
