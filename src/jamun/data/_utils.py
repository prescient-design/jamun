import collections
import os
import re
from typing import List, Optional, Sequence

import hydra
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
    pdb_pattern: Optional[str] = None,
    pdb_file: Optional[Sequence[str]] = None,
    max_datasets: Optional[int] = None,
    max_datasets_offset: Optional[int] = None,
    filter_codes: Optional[Sequence[str]] = None,
    **dataset_kwargs,
) -> List[MDtrajDataset]:
    """Helper function to create MDtrajDataset objects from a directory of trajectory files."""
    if pdb_file is not None and pdb_pattern is not None:
        raise ValueError("Exactly one of pdb_file and pdb_pattern should be provided.")

    traj_prefix, traj_pattern = os.path.split(traj_pattern)
    traj_pattern_compiled = re.compile(traj_pattern)
    if "*" in traj_prefix or "?" in traj_prefix:
        raise ValueError("traj_prefix should not contain wildcards.")

    traj_files = collections.defaultdict(list)
    codes = set()
    for entry in os.scandir(os.path.join(root, traj_prefix)):
        match = traj_pattern_compiled.match(entry.name)
        if not match:
            continue

        code = match.group(1)
        codes.add(code)
        traj_files[code].append(os.path.join(traj_prefix, entry.name))

    if len(codes) == 0:
        raise ValueError("No codes found in directory.")

    pdb_files = {}
    if pdb_pattern is not None:
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
    else:
        for code in codes:
            pdb_files[code] = pdb_file

    # Filter out codes.
    if filter_codes is not None:
        codes = [code for code in codes if code in set(filter_codes)]

    # Sort the codes and offset them, if necessary.
    codes = list(sorted(codes))
    if max_datasets_offset is not None:
        codes = codes[max_datasets_offset:]
    if max_datasets is not None:
        codes = codes[:max_datasets]

    datasets = []
    for code in tqdm(codes, desc="Creating datasets"):
        dataset = MDtrajDataset(
            root,
            trajfiles=traj_files[code],
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


def create_dataset_from_pdb(pdbfile: str, label: Optional[str] = None) -> Sequence[MDtrajDataset]:
    """Create a dataset from a PDB file."""

    # Note that if pdbfile is an absolute path, the first part of the join will be ignored.
    root = os.path.join(hydra.utils.get_original_cwd(), os.path.dirname(pdbfile))
    pdbfile = os.path.basename(pdbfile)

    if label is None:
        label = pdbfile.split(".")[0]

    dataset = MDtrajDataset(
        root=root,
        trajfiles=[pdbfile],
        pdbfile=pdbfile,
        label=label,
    )
    return [dataset]
