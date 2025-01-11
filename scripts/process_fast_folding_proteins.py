import argparse
import logging
import os
import random
import shutil
import sys
from typing import List, Tuple

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("process_fast_folding_proteins")


SPLITS = {
    "train": 0.85,
    "val": 0.5,
    "test": 0.1,
}


def find_xtc_files(root_dir: str) -> List[Tuple[str, str]]:
    """
    Recursively find all .xtc files and their containing folders starting from root_dir

    Args:
        root_dir (str): The root directory to start the search from

    Returns:
        list: List of tuples containing (file_path, containing_folder)
    """
    xtc_files = []

    # Walk through directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Find all files with .xtc extension
        for filename in filenames:
            if not filename.endswith(".xtc") or filename.startswith("."):
                continue

            # Get full file path and containing folder
            file_path = os.path.join(dirpath, filename)
            containing_folder = os.path.basename(dirpath)
            xtc_files.append((file_path, containing_folder))

    return xtc_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create splits of .xtc files based on containing folder")
    parser.add_argument(
        "--inputdir", help="Directory where original trajectories were downloaded", type=str, required=True
    )
    parser.add_argument("--outputdir", "-o", help="Output directory to save splits", type=str, required=True)
    args = parser.parse_args()

    # Check if directory exists
    if not os.path.isdir(args.inputdir):
        py_logger.info(f"Error: Directory '{args.inputdir}' does not exist", file=sys.stderr)
        sys.exit(1)

    files = find_xtc_files(args.inputdir)
    if not files:
        raise ValueError("No .xtc files found in the directory.")

    # Check that the folders are unique.
    # This is because we will be using the folder names as unique identifiers for the files.
    folders = [folder for _, folder in files]
    if len(set(folders)) != len(folders):
        raise ValueError("Found duplicate folders in the directory.")

    # Randomly shuffle the files and folders.
    random.seed(42)
    random.shuffle(files)

    # Now, create the splits based on the folders.
    for split, split_ratio in SPLITS.items():
        split_dir = os.path.join(args.outputdir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Find the number of files to include in this split.
        num_files = int(len(files) * split_ratio)
        split_files = files[:num_files]

        # Save the split files as a text file.
        with open(os.path.join(split_dir, "files.txt"), "w") as f:
            for file_path, folder in split_files:
                f.write(f"{file_path}\n")

        # Copy the files to the split directory.
        for file_path, folder in split_files:
            output_file = os.path.join(split_dir, f"{folder}.xtc")
            shutil.copy(file_path, output_file)
            py_logger.info(f"Saved {file_path} as {output_file}")
