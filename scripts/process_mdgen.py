"""Script to process mdgen data into splits for training and testing."""

import argparse
import logging
import os
import requests
import shutil
import tqdm

import pandas as pd


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
    parser.add_argument('--inputdir', help='Directory where original trajectories were downloaded.', type=str, required=True)
    parser.add_argument('--outputdir', '-o', help='Output directory to save splits.', type=str, required=True)
    args = parser.parse_args()

    # Download splits.
    split_files = {
        split: rf"https://github.com/bjing2016/mdgen/blob/master/splits/4AA_{split}.csv"
    for split in ["train", "val", "test"]
    }

    for split, url in split_files.items():
        # Download the split file
        df = download_github_csv(url)
    
        split_dir = os.path.join(args.outputdir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Iterate over each row in the DataFrame, and copy the corresponding files
        for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {split} split"):
            name = row['name']
            for extension in ['xtc', 'pdb']:
                source_file = os.path.join(args.inputdir, 'data', '4AA_sims', name, f'{name}.{extension}')
                assert os.path.isfile(source_file), f"Source file {source_file} does not exist."
                destination_file = os.path.join(split_dir, f'{name}.{extension}')
                
                # Copy the file.
                shutil.copy(source_file, destination_file)
