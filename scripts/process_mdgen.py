"""Script to process mdgen data into splits for training and testing."""

from typing import Tuple, Optional
import argparse
import logging
import os
import requests
import tqdm
import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor

import pandas as pd


def run_preprocess(args, use_srun: bool = True) -> Tuple[str, Optional[str]]:
    """Run preprocessing for a single peptide."""
    name, input_dir, output_dir = args
    
    trajectory = os.path.join(input_dir, 'data', '4AA_sims', name, f'{name}.xtc')
    topology = os.path.join(input_dir, 'data', '4AA_sims', name, f'{name}.pdb')

    cmd = []
    if use_srun:
        cmd += ['srun', '--partition=cpu', '--mem=64G']
    cmd += [
        'python',
        'scripts/chunk_trajectory.py',
        f'--trajectory={trajectory}',
        f'--topology={topology}',
        f'--output-dir={output_dir}',
    ]
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return (name, None)
    except subprocess.CalledProcessError as e:
        return (name, str(e))


def download_github_csv(url):
    """
    Download CSV file from GitHub. Works with both raw URLs and regular GitHub URLs.
    Returns a pandas DataFrame.
    """
    # Convert regular GitHub URL to raw URL if needed
    if 'github.com' in url and 'raw' not in url:
        url = url.replace('github.com', 'raw.githubusercontent.com')
        url = url.replace('/blob/', '/')
    
    try:
        # Download the CSV file
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Read CSV into pandas DataFrame
        df = pd.read_csv(url)
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create splits of .xtc files based on containing folder.')
    parser.add_argument('--input-dir', help='Directory where original trajectories were downloaded.', type=str, required=True)
    parser.add_argument('--output-dir', '-o', help='Output directory to save splits.', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=multiprocessing.cpu_count(),
                      help='Number of parallel workers')
    args = parser.parse_args()

    # Download splits.
    split_files = {
        split: rf"https://github.com/bjing2016/mdgen/blob/master/splits/4AA_{split}.csv"
    for split in ["train", "val", "test"]
    }

    for split, url in split_files.items():
        # Download the split file
        df = download_github_csv(url)
    
        split_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Run analyses in parallel.
        preprocess_args = list(
            zip(
                df['name'],
                [args.input_dir] * len(df),
                [split_dir] * len(df),
            )
        )
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(executor.map(run_preprocess, preprocess_args))

        # Process results.
        for peptide, error in results:
            if error:
                print(f"Error processing peptide {peptide}: {error}")
            else:
                print(f"Successfully processed peptide {peptide}")
