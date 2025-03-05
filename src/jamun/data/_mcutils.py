import collections
import os
import re
import random
from typing import List, Optional, Sequence

import hydra
import requests
import torch
from tqdm.auto import tqdm

from jamun.data._cremp import MDtrajSDFDataset, MDtrajIterableSDFDataset

#from jamun.data._cremp import MDtrajPickleDataset, MDtrajSDFDataset, MDtrajIterableSDFDataset, MDtrajIterablePickleDataset

def parse_sdf_datasets_from_directory(
    root: str,
    traj_pattern: str,
    sdf_pattern: Optional[str] = None,
    sdf_file: Optional[Sequence[str]] = None,
    max_datasets: Optional[int] = None,
    max_datasets_offset: Optional[int] = None,
    filter_codes: Optional[Sequence[str]] = None,
    as_iterable: bool = False,
    **dataset_kwargs,
) -> List[MDtrajSDFDataset]:
    """Helper function to create MDtrajDataset objects from a directory of trajectory files."""
    if sdf_file is not None and sdf_pattern is not None:
        raise ValueError("Exactly one of sdf_file and sdf_pattern should be provided.")

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

    sdf_files = {}
    if sdf_pattern is not None:
        sdf_prefix, sdf_pattern = os.path.split(sdf_pattern)
        sdf_pattern_compiled = re.compile(sdf_pattern)
        if "*" in sdf_prefix or "?" in sdf_prefix:
            raise ValueError("sdf_prefix should not contain wildcards.")

        for entry in os.scandir(os.path.join(root, sdf_prefix)):
            match = sdf_pattern_compiled.match(entry.name)
            if not match:
                continue

            code = match.group(1)
            if code not in codes:
                continue
            sdf_files[code] = os.path.join(sdf_prefix, entry.name)
    else:
        for code in codes:
            sdf_files[code] = sdf_file

    # Filter out codes.
    if filter_codes is not None:
        codes = [code for code in codes if code in set(filter_codes)]

    # Sort the codes and offset them, if necessary.
    codes = list(sorted(codes))
    if max_datasets_offset is not None:
        codes = codes[max_datasets_offset:]
    if max_datasets is not None:
        codes = codes[:max_datasets]

    if as_iterable:
        dataset_class = MDtrajIterableSDFDataset
    else:
        dataset_class = MDtrajSDFDataset

    datasets = []
    for code in tqdm(codes, desc="Creating datasets"):
        dataset = dataset_class(
            sdf_path=os.path.join(root, sdf_files[code]),
            trajfiles=[os.path.join(root, trajfile) for trajfile in traj_files[code]],
            label=code,
            **dataset_kwargs,
        )
        datasets.append(dataset)
    return datasets


def concatenate_sdf_datasets(datasets: Sequence[Sequence[MDtrajSDFDataset]]) -> List[MDtrajSDFDataset]:
    """Concatenate multiple lists of SDF datasets into one list."""
    all_datasets = []
    for datasets_list in datasets:
        all_datasets.extend(datasets_list)
    return all_datasets


def create_dataset_from_sdf_jsons(sdf_files: Sequence[str], label_prefix: Optional[str] = None) -> Sequence[MDtrajSDFDataset]:
    """Create a dataset from a SDF file."""
    datasets = []
    for sdf_file in sdf_files:
        # Note that if sdf_file is an absolute path, the first part of the join will be ignored.
        root = os.path.join(hydra.utils.get_original_cwd(), os.path.dirname(sdf_file))
        sdf_file = os.path.basename(sdf_file)

        label = sdf_file.split(".")[0]
        if label_prefix is not None:
            label = f"{label_prefix}{label}"

        dataset = MDtrajSDFDataset(
            sdf_path=sdf_file,
            trajfiles=[],
            label=label,
        )
        datasets.append(dataset)

    return datasets

# def parse_pickle_datasets_from_directory(
#     root: str,
#     traj_pattern: str,
#     pickle_pattern: Optional[str] = None,
#     pickle_file: Optional[Sequence[str]] = None,
#     max_datasets: Optional[int] = None,
#     max_datasets_offset: Optional[int] = None,
#     filter_codes: Optional[Sequence[str]] = None,
#     as_iterable: bool = False,
#     **dataset_kwargs,
# ) -> List[MDtrajPickleDataset]:
#     """Helper function to create MDtrajPickleDataset objects from a directory of trajectory files."""
#     if pickle_file is not None and pickle_pattern is not None:
#         raise ValueError("Exactly one of pickle_file and pickle_pattern should be provided.")

#     traj_prefix, traj_pattern = os.path.split(traj_pattern)
#     traj_pattern_compiled = re.compile(traj_pattern)
#     if "*" in traj_prefix or "?" in traj_prefix:
#         raise ValueError("traj_prefix should not contain wildcards.")

#     traj_files = collections.defaultdict(list)
#     codes = set()
#     for entry in os.scandir(os.path.join(root, traj_prefix)):
#         match = traj_pattern_compiled.match(entry.name)
#         if not match:
#             continue

#         code = match.group(1)
#         codes.add(code)
#         traj_files[code].append(os.path.join(traj_prefix, entry.name))

#     if len(codes) == 0:
#         raise ValueError("No codes found in directory.")

#     pickle_files = {}
#     if pickle_pattern is not None:
#         pickle_prefix, pickle_pattern = os.path.split(pickle_pattern)
#         pickle_pattern_compiled = re.compile(pickle_pattern)
#         if "*" in pickle_prefix or "?" in pickle_prefix:
#             raise ValueError("pickle_prefix should not contain wildcards.")

#         for entry in os.scandir(os.path.join(root, pickle_prefix)):
#             match = pickle_pattern_compiled.match(entry.name)
#             if not match:
#                 continue

#             code = match.group(1)
#             if code not in codes:
#                 continue
#             pickle_files[code] = os.path.join(pickle_prefix, entry.name)
#     else:
#         for code in codes:
#             pickle_files[code] = pickle_file

#     # Filter out codes.
#     if filter_codes is not None:
#         codes = [code for code in codes if code in set(filter_codes)]

#     # Sort the codes and offset them, if necessary.
#     codes = list(sorted(codes))
#     if max_datasets_offset is not None:
#         codes = codes[max_datasets_offset:]
#     if max_datasets is not None:
#         codes = codes[:max_datasets]

#     if as_iterable:
#         dataset_class = MDtrajIterablePickleDataset
#     else:
#         dataset_class = MDtrajPickleDataset

#     datasets = []
#     for code in tqdm(codes, desc="Creating datasets"):
#         dataset = dataset_class(
#             pickle_path=os.path.join(root, pickle_files[code]),
#             trajfiles=[os.path.join(root, trajfile) for trajfile in traj_files[code]],
#             label=code,
#             **dataset_kwargs,
#         )
#         datasets.append(dataset)
#     return datasets


# def concatenate_pickle_datasets(datasets: Sequence[Sequence[MDtrajPickleDataset]]) -> List[MDtrajPickleDataset]:
#     """Concatenate multiple lists of pickle datasets into one list."""
#     all_datasets = []
#     for datasets_list in datasets:
#         all_datasets.extend(datasets_list)
#     return all_datasets


# def create_dataset_from_pickle_files(pickle_files: Sequence[str], label_prefix: Optional[str] = None) -> Sequence[MDtrajPickleDataset]:
#     """Create a dataset from a pickle file."""
#     datasets = []
#     for pickle_file in pickle_files:
#         # Note that if pickle_file is an absolute path, the first part of the join will be ignored.
#         root = os.path.join(hydra.utils.get_original_cwd(), os.path.dirname(pickle_file))
#         pickle_file = os.path.basename(pickle_file)

#         label = pickle_file.split(".")[0]
#         if label_prefix is not None:
#             label = f"{label_prefix}{label}"

#         dataset = MDtrajPickleDataset(
#             pickle_path=pickle_file,
#             trajfiles=[],
#             label=label,
#         )
#         datasets.append(dataset)

#     return datasets