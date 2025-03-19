from typing import Dict, List, Optional, Sequence
import os
import mdtraj as md
import tqdm
import pandas as pd
import dotenv
import logging

from jamun import data
from jamun import utils

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("analysis")


def find_project_root() -> str:
    """Returns the path to the root of the project."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(current_dir, "pyproject.toml")):
        current_dir = os.path.dirname(current_dir)
    return current_dir


def get_run_path_for_wandb_run(wandb_run_path: str) -> str:
    """Returns the path to the run directory given a wandb run path."""
    cfg = utils.get_wandb_run_config(wandb_run_path)
    run_path = os.path.join(cfg["paths"]["run_path"])
    if run_path.startswith("."):
        # Path is relative to the project root.
        run_path = os.path.join(find_project_root(), run_path)
    return os.path.abspath(run_path)


def get_peptides_in_JAMUN_run(run_path: str) -> Sequence[str]:
    """Returns the list of peptides sampled in a run and the output directory where they are stored."""

    if not os.path.exists(run_path):
        raise ValueError(f"Output directory {run_path} not found.")

    peptides_in_run = sorted(os.listdir(os.path.join(run_path, "sampler")))
    if len(peptides_in_run) == 0:
        raise ValueError(f"No sampled peptides found in {run_path}")

    return peptides_in_run


def search_for_JAMUN_files(root_path: str) -> List[str]:
    """Heuristically finds JAMUN output files in a given directory."""

    output_dir = os.path.join(root_path, "outputs")
    if not os.path.exists(output_dir):
        raise ValueError(f"No outputs directory found in {root_path}")

    # Find all folders having a "sampler" subdirectory recursively.
    run_paths = []
    for dirpath, dirnames, filenames in os.walk(output_dir):
        if "sampler" in dirnames:
            run_paths.append(dirpath)

    return run_paths


def get_sampling_rate(name: str, peptide: str, experiment: str) -> float:
    """Returns (approximate) sampling rates in seconds per sample."""

    if name == "JAMUN":
        rates_csv = os.path.join(find_project_root(), "analysis", "sampling_times", "JAMUN.csv")
        df = pd.read_csv(rates_csv)
        if experiment not in df["experiment"].values:
            return None
        ms_per_sample = df[(df["experiment"] == experiment)]["ms_per_sample"].values[0]
        return ms_per_sample / 1000

    if name == "JAMUNReference_2AA":
        rates_csv = os.path.join(find_project_root(), "analysis", "sampling_times", "JAMUNReference_2AA.csv")
        df = pd.read_csv(rates_csv)
        seconds_per_10_samples = df[(df["peptide"] == peptide)]["seconds_per_10_samples"].values[0]
        return seconds_per_10_samples / 10


def get_JAMUN_trajectory_files(run_paths: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Returns a dictionary mapping peptide names to the path of the PDB file containing the predicted structure."""

    trajectory_files = {}
    for run_path in run_paths:
        # Get the list of peptides sampled in the run and the output directory where they are stored.
        peptides_in_run = get_peptides_in_JAMUN_run(run_path)
        utils.dist_log(f"Found peptides {peptides_in_run} in run {run_path}")

        # Check that there are no duplicates amongst runs.
        for peptide in peptides_in_run:
            if peptide in trajectory_files:
                raise ValueError(
                    f"Peptide {peptide} found in multiple runs: {run_path} and {trajectory_files[peptide]['dcd']}"
                )

            # Load trajectory file as .dcd.
            trajectory_files[peptide] = {
                "dcd": f"{run_path}/sampler/{peptide}/predicted_samples/dcd/joined.dcd",
            }

            if not os.path.exists(trajectory_files[peptide]["dcd"]):
                raise ValueError(f"DCD file {trajectory_files[peptide]['dcd']} not found.")

            # Search for the corresponding PDB file.
            for pdb_file in [
                f"{run_path}/sampler/{peptide}/topology.pdb",
                f"{run_path}/sampler/{peptide}/predicted_samples/pdb/0.pdb",
                f"{run_path}/pdbs/{peptide}-modified.pdb",
                f"{run_path}/dataset_pdbs/{peptide}.pdb",
            ]:
                if os.path.exists(pdb_file):
                    trajectory_files[peptide]["pdb"] = pdb_file
                    break

            if "pdb" not in trajectory_files[peptide]:
                raise ValueError(f"No PDB file found for peptide {peptide} in run {run_path}")

    # Remove the prefix "uncapped_" and "capped_" from the peptide names.
    # for peptide in list(trajectory_files.keys()):
    #     for prefix in ["uncapped_", "capped_"]:
    #         if not peptide.startswith(prefix):
    #             continue
    #         trajectory_files[peptide[len(prefix) :]] = trajectory_files.pop(peptide)
    return trajectory_files


def get_JAMUN_trajectories(
    run_paths: Sequence[str], filter_codes: Optional[Sequence[str]] = None
) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to the sampled JAMUN trajectory."""
    trajectory_files = get_JAMUN_trajectory_files(run_paths)
    trajectories = {}
    for peptide, peptide_files in tqdm.tqdm(trajectory_files.items(), desc="Loading JAMUN trajectories"):
        if filter_codes and peptide not in filter_codes:
            continue

        trajectories[peptide] = md.load_dcd(peptide_files["dcd"], top=peptide_files["pdb"])
    return trajectories


def get_MDGenReference_trajectories(
    data_path: str, filter_codes: Optional[Sequence[str]] = None, split: str = "all"
) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to the MDGen reference trajectory."""

    def get_datasets_for_split(split: str):
        """Helper function to get datasets for a given split."""
        return data.parse_datasets_from_directory(
            root=f"{data_path}/mdgen/data/4AA_sims_partitioned_chunked/{split}/",
            traj_pattern="^(....)_.*.xtc",
            pdb_pattern="^(....).pdb",
            filter_codes=filter_codes,
        )

    if split in ["train", "val", "test"]:
        datasets = get_datasets_for_split(split)
    elif split == "all":
        datasets = get_datasets_for_split("train") + get_datasets_for_split("val") + get_datasets_for_split("test")
    else:
        raise ValueError(f"Invalid split: {split}")

    return {dataset.label(): dataset.trajectory for dataset in datasets}


def get_TimewarpReference_trajectories(
    data_path: str, filter_codes: Optional[Sequence[str]] = None, split: str = "all"
) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to the Timewarp reference trajectory."""
    # Timewarp trajectory files are in one-letter format.
    one_letter_filter_codes = ["".join([utils.convert_to_one_letter_code(aa) for aa in code]) for code in filter_codes]
    assert len(set(one_letter_filter_codes)) == len(one_letter_filter_codes), "Filter codes must be unique"

    def get_datasets_for_split(split: str):
        """Helper function to get datasets for a given split."""
        split_datasets = []
        for peptide_type_dir in ["2AA-1-large", "4AA-large"]:
            split_datasets += data.parse_datasets_from_directory(
                root=f"{data_path}/timewarp/{peptide_type_dir}/{split}/",
                traj_pattern="^(.*)-traj-arrays.npz",
                pdb_pattern="^(.*)-traj-state0.pdb",
                filter_codes=one_letter_filter_codes,
            )
        return split_datasets

    if split in ["train", "val", "test"]:
        datasets = get_datasets_for_split(split)
    elif split == "all":
        datasets = get_datasets_for_split("train") + get_datasets_for_split("val") + get_datasets_for_split("test")
    else:
        raise ValueError(f"Invalid split: {split}")

    # Remap keys.
    filter_codes_map = dict(zip(one_letter_filter_codes, filter_codes))
    return {filter_codes_map[dataset.label()]: dataset.trajectory for dataset in datasets}


def get_JAMUNReference_2AA_trajectories(
    data_path: str, filter_codes: Optional[Sequence[str]] = None, split: str = "all"
) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to our reference 2AA MDTraj trajectory."""

    def get_datasets_for_split(split: str):
        """Helper function to get datasets for a given split."""
        return data.parse_datasets_from_directory(
            root=f"{data_path}/capped_diamines/timewarp_splits/{split}",
            traj_pattern="^(.*).xtc",
            pdb_pattern="^(.*).pdb",
            filter_codes=filter_codes,
        )

    if split in ["train", "val", "test"]:
        datasets = get_datasets_for_split(split)
    elif split == "all":
        datasets = get_datasets_for_split("train") + get_datasets_for_split("val") + get_datasets_for_split("test")

    # Remap keys.
    filter_codes_map = dict(zip(filter_codes, filter_codes))
    return {filter_codes_map[dataset.label()]: dataset.trajectory for dataset in datasets}


def get_JAMUNReference_5AA_trajectories(
    data_path: str, filter_codes: Optional[Sequence[str]] = None
) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to our reference 5AA MDTraj trajectories."""
    prefix = ""
    for code in filter_codes:
        if code.startswith("uncapped_"):
            prefix = "uncapped_"
            break
        if code.startswith("uncapped_"):
            prefix = "capped_"
            break

    # Remove prefix.
    three_letter_filter_codes = [
        "_".join([utils.convert_to_three_letter_code(aa) for aa in code]) for code[len(prefix) :] in filter_codes
    ]
    assert len(set(three_letter_filter_codes)) == len(three_letter_filter_codes), "Filter codes must be unique"

    datasets = data.parse_datasets_from_directory(
        root=f"{data_path}/5AA/",
        traj_pattern="^(.*)_traj3-arrays.npz",
        pdb_pattern="^(.*)_traj3-state0.pdb",
        filter_codes=three_letter_filter_codes,
    )

    # Remap keys.
    filter_codes_map = dict(zip(three_letter_filter_codes, filter_codes))
    return {filter_codes_map[dataset.label()]: dataset.trajectory for dataset in datasets}


def get_TBG_trajectories(root: str) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to the TBG MDTraj trajectory."""
    raise NotImplementedError("TBG trajectories not implemented yet.")


def load_trajectories_by_name(
    name: str,
    peptide: str,
    data_path: str,
    run_path: Optional[str],
    wandb_run: Optional[str],
):
    """Load trajectories based on name and command line arguments."""
    if data_path:
        JAMUN_DATA_PATH = data_path
    else:
        JAMUN_DATA_PATH = os.environ.get("JAMUN_DATA_PATH")
        if JAMUN_DATA_PATH is None:
            env_file = os.path.join(find_project_root(), ".env")
            JAMUN_DATA_PATH = dotenv.get_key(env_file, "JAMUN_DATA_PATH")
        if not JAMUN_DATA_PATH:
            raise ValueError("JAMUN_DATA_PATH must be provided either via --data-path, environment variable or .env file")
    py_logger.info(f"Using JAMUN_DATA_PATH: {JAMUN_DATA_PATH}")

    filter_codes = [peptide]
    if name == "JAMUN":
        if not run_path and not wandb_run:
            raise ValueError("Must provide either --run-path or --wandb-run for JAMUN trajectory")
        if run_path and wandb_run:
            raise ValueError("Must provide only one of --run-path or --wandb-run for JAMUN trajectory")

        if wandb_run:
            run_paths = [get_run_path_for_wandb_run(wandb_run)]
        else:
            run_paths = [run_path]
        return get_JAMUN_trajectories(run_paths, filter_codes=filter_codes)
    elif name == "MDGenReference":
        return get_MDGenReference_trajectories(
            JAMUN_DATA_PATH,
            filter_codes=filter_codes,
        )
    elif name == "TimewarpReference":
        return get_TimewarpReference_trajectories(
            JAMUN_DATA_PATH,
            filter_codes=filter_codes,
        )
    elif name == "JAMUNReference_2AA":
        return get_JAMUNReference_2AA_trajectories(
            JAMUN_DATA_PATH,
            filter_codes=filter_codes,
        )
    elif name == "JAMUNReference_5AA":
        return get_JAMUNReference_5AA_trajectories(
            JAMUN_DATA_PATH,
            filter_codes=filter_codes,
        )

    raise ValueError(f"Trajectory type {name} not supported. Available options: JAMUN, MDGenReference, TimewarpReference, JAMUNReference_2AA, JAMUNReference_5AA")
