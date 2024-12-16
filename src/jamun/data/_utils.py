from typing import Optional, List, Sequence
import logging
import os
import re
import collections

import requests
import torch
from tqdm.auto import tqdm

from jamun.data._mdtraj import MDtrajDataset


def dloader_map_reduce(f, dloader, reduce_fn=torch.cat, verbose: bool = False):
    outs = []
    for batch in tqdm(dloader, disable=not verbose):
        outs.append(f(batch))
    return reduce_fn(outs)


def download_file(url: str, path: str, verbose: bool = False, block_size: Optional[int] = None):
    """Download a file from a URL to a local path."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    if block_size is None:
        block_size = 1024 * 1024

    with open(path, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, disable=not verbose) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))


def parse_datasets_from_directory(
    root: str,
    traj_pattern: str,
    pdb_pattern: str,
    max_datasets: Optional[int] = None,
    max_datasets_offset: Optional[int] = None,
    **dataset_kwargs,
) -> List[MDtrajDataset]:
    """Helper function to create MDtrajDataset objects from a directory of trajectory files."""
    py_logger = logging.getLogger("jamun")
    py_logger.info(f"Creating datasets from {root}.")

    traj_prefix, traj_pattern = os.path.split(traj_pattern)
    traj_pattern_compiled = re.compile(traj_pattern)
    if "*" in traj_prefix or "?" in traj_prefix:
        raise ValueError("traj_prefix should not contain wildcards.")

    traj_files = collections.defaultdict(list)
    pdb_files = {}
    codes = set()
    for entry in os.scandir(os.path.join(root, traj_prefix)):
        match = traj_pattern_compiled.match(entry.name)
        if not match:
            continue

        code = match.group(1)
        codes.add(code)
        traj_files[code].append(os.path.join(traj_prefix, entry.name))

    if len(codes) == 0:
        raise ValueError("No codes found in trajectory.")

    pdb_prefix, pdb_pattern = os.path.split(pdb_pattern)
    pdb_pattern_compiled = re.compile(pdb_pattern)
    if "*" in pdb_prefix or "?" in pdb_prefix:
        raise ValueError("pdb_prefix should not contain wildcards.")

    for entry in os.scandir(os.path.join(root, pdb_prefix)):
        match = pdb_pattern_compiled.match(entry.name)
        if not match:
            continue

        code = match.group(1)
        if code not in codes:
            continue
        pdb_files[code] = os.path.join(pdb_prefix, entry.name)

    # Sort the codes and offset them, if necessary.
    if max_datasets_offset is None:
        max_datasets_offset = 0
    codes = list(sorted(codes))[max_datasets_offset:]

    datasets = []
    for index, code in enumerate(codes):
        if max_datasets is not None and index >= max_datasets:
            break

        py_logger.info(f"Creating dataset for {code} with trajectories {traj_files[code]} and PDB {pdb_files[code]}.")
        dataset = MDtrajDataset(
            root,
            xtcfiles=traj_files[code],
            pdbfile=pdb_files[code],
            label=code,
            **dataset_kwargs,
        )
        datasets.append(dataset)
    return datasets


def concatenate_datasets(datasets: Sequence[Sequence[MDtrajDataset]]) -> List[MDtrajDataset]:
    """Concatenate multiple lists of datasets into one list."""
    all_datasets = []
    for datasets_list in datasets:
        all_datasets.extend(datasets_list)
    return all_datasets
