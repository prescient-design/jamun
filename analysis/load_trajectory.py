import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import subprocess

import dotenv
import mdtraj as md
import pandas as pd
import tqdm

from jamun import data, utils

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("analysis")


def find_project_root() -> str:
    """Returns the path to the root of the project."""
    return subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode("utf-8").strip()


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
    raise NotImplementedError("Sampling rate calculation is not verified yet.")

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

    return trajectory_files


def get_JAMUN_trajectories(
    run_paths: Sequence[str], filter_codes: Optional[Sequence[str]] = None
) -> Dict[str, Tuple[md.Trajectory, Dict[str, Any]]]:
    """Returns a dictionary mapping peptide names to the sampled JAMUN trajectory."""
    trajectory_files = get_JAMUN_trajectory_files(run_paths)
    trajectory_info = {}
    for peptide, peptide_files in tqdm.tqdm(trajectory_files.items(), desc="Loading JAMUN trajectories"):
        if filter_codes and peptide not in filter_codes:
            continue

        trajectory_file = peptide_files["dcd"]
        topology_file = peptide_files["pdb"]
        trajectory = md.load_dcd(trajectory_file, top=topology_file)
        info = {
            "trajectory_files": [trajectory_file],
            "topology_file": topology_file,
        }

        trajectory_info[peptide] = (trajectory, info)
    return trajectory_info


def get_MDGenReference_datasets(
    data_path: str, filter_codes: Optional[Sequence[str]] = None, split: str = "all"
) -> Dict[str, data.MDtrajDataset]:
    """Returns a dictionary mapping peptide names to the MDGen reference trajectory."""

    def get_datasets_for_split(split: str):
        """Helper function to get datasets for a given split."""
        return data.parse_datasets_from_directory(
            root=f"{data_path}/mdgen/data/4AA_sims_partitioned_chunked/{split}/",
            traj_pattern="^(....)_.*.xtc",
            pdb_pattern="^(....).pdb",
            filter_codes=filter_codes,
        )

    all_splits = ["train", "val", "test"]
    if split in all_splits:
        datasets = get_datasets_for_split(split)
    elif split == "all":
        datasets = sum([get_datasets_for_split(split) for split in all_splits], [])
    else:
        raise ValueError(f"Invalid split: {split}")

    return {dataset.label(): dataset for dataset in datasets}


def get_TimewarpReference_datasets(
    data_path: str, filter_codes: Optional[Sequence[str]] = None, split: str = "all", peptide_type: str = "all"
) -> Dict[str, data.MDtrajDataset]:
    """Returns a dictionary mapping peptide names to the Timewarp reference trajectory."""
    if peptide_type == "2AA":
        return get_TimewarpReference_2AA_datasets(data_path, filter_codes=filter_codes, split=split)
    elif peptide_type == "4AA":
        return get_TimewarpReference_4AA_datasets(data_path, filter_codes=filter_codes, split=split)
    else:
        raise ValueError(f"Invalid peptide type: {peptide_type}")


def get_TimewarpReference_2AA_datasets(
    data_path: str, filter_codes: Optional[Sequence[str]] = None, split: str = "all"
) -> Dict[str, data.MDtrajDataset]:
    """Returns a dictionary mapping peptide names to our reference 2AA MDTraj trajectory."""
    # Timewarp trajectory files are in one-letter format.
    if filter_codes is None:
        one_letter_filter_codes = None
    else:
        one_letter_filter_codes = [
            "".join([utils.convert_to_one_letter_code(aa) for aa in code]) for code in filter_codes
        ]
        assert len(set(one_letter_filter_codes)) == len(one_letter_filter_codes), "Filter codes must be unique"

    def get_datasets_for_split(split: str):
        """Helper function to get datasets for a given split."""
        split_datasets = data.parse_datasets_from_directory(
            root=f"{data_path}/timewarp/2AA-1-large/{split}/",
            traj_pattern="^(.*)-traj-arrays.npz",
            pdb_pattern="^(.*)-traj-state0.pdb",
            filter_codes=one_letter_filter_codes,
        )
        return split_datasets

    all_splits = ["train", "val", "test"]
    if split in all_splits:
        datasets = get_datasets_for_split(split)
    elif split == "all":
        datasets = sum([get_datasets_for_split(split) for split in all_splits], [])
    else:
        raise ValueError(f"Invalid split: {split}")

    if filter_codes is None:
        return {dataset.label(): dataset for dataset in datasets}

    # Remap keys.
    filter_codes_map = dict(zip(one_letter_filter_codes, filter_codes))
    return {filter_codes_map[dataset.label()]: dataset for dataset in datasets}


def get_TimewarpReference_4AA_datasets(
    data_path: str, filter_codes: Optional[Sequence[str]] = None, split: str = "all"
) -> Dict[str, data.MDtrajDataset]:
    """Returns a dictionary mapping peptide names to our reference 4AA MDTraj trajectory."""
    # Timewarp trajectory files are in one-letter format.
    if filter_codes is None:
        one_letter_filter_codes = None
    else:
        one_letter_filter_codes = [
            "".join([utils.convert_to_one_letter_code(aa) for aa in code]) for code in filter_codes
        ]
        assert len(set(one_letter_filter_codes)) == len(one_letter_filter_codes), "Filter codes must be unique"

    def get_datasets_for_split(split: str):
        """Helper function to get datasets for a given split."""
        split_datasets = data.parse_datasets_from_directory(
            root=f"{data_path}/timewarp/4AA-large/{split}/",
            traj_pattern="^(.*)-traj-arrays.npz",
            pdb_pattern="^(.*)-traj-state0.pdb",
            filter_codes=one_letter_filter_codes,
        )
        return split_datasets

    all_splits = ["train", "val", "test"]
    if split in all_splits:
        datasets = get_datasets_for_split(split)
    elif split == "all":
        datasets = sum([get_datasets_for_split(split) for split in all_splits], [])
    else:
        raise ValueError(f"Invalid split: {split}")

    if filter_codes is None:
        return {dataset.label(): dataset for dataset in datasets}

    # Remap keys.
    filter_codes_map = dict(zip(one_letter_filter_codes, filter_codes))
    return {filter_codes_map[dataset.label()]: dataset for dataset in datasets}


def get_JAMUNReference_2AA_datasets(
    data_path: str, filter_codes: Optional[Sequence[str]] = None, split: str = "all"
) -> Dict[str, data.MDtrajDataset]:
    """Returns a dictionary mapping peptide names to our reference 2AA MDTraj trajectory."""

    def get_datasets_for_split(split: str):
        """Helper function to get datasets for a given split."""
        return data.parse_datasets_from_directory(
            root=f"{data_path}/capped_diamines/timewarp_splits/{split}",
            traj_pattern="^(.*).xtc",
            pdb_pattern="^(.*).pdb",
            filter_codes=filter_codes,
            num_frames=60000,
        )

    all_splits = ["train", "val", "test"]
    if split in all_splits:
        datasets = get_datasets_for_split(split)
    elif split == "all":
        datasets = sum([get_datasets_for_split(split) for split in all_splits], [])

    return {dataset.label(): dataset for dataset in datasets}


def get_JAMUNReference_5AA_datasets(
    data_path: str, filter_codes: Optional[Sequence[str]] = None
) -> Dict[str, data.MDtrajDataset]:
    """Returns a dictionary mapping peptide names to our reference 5AA MDTraj trajectories."""
    prefix = ""

    if filter_codes is not None:
        for code in filter_codes:
            if code.startswith("uncapped_"):
                prefix = "uncapped_"
                break
            if code.startswith("uncapped_"):
                prefix = "capped_"
                break

        # Remove prefix.
        three_letter_filter_codes = [
            "_".join([utils.convert_to_three_letter_code(aa) for aa in code[len(prefix) :]]) for code in filter_codes
        ]
        assert len(set(three_letter_filter_codes)) == len(three_letter_filter_codes), "Filter codes must be unique"
    else:
        three_letter_filter_codes = None

    datasets = data.parse_datasets_from_directory(
        root=f"{data_path}/5AA/",
        traj_pattern="^(.*)_traj3-arrays.npz",
        pdb_pattern="^(.*)_traj3-state0.pdb",
        filter_codes=three_letter_filter_codes,
    )

    # Remap keys.
    if three_letter_filter_codes is not None:
        filter_codes_map = dict(zip(three_letter_filter_codes, filter_codes))
        return {filter_codes_map[dataset.label()]: dataset for dataset in datasets}

    return {dataset.label(): dataset for dataset in datasets}


def get_CrempReference_trajectories(
    data_path: str, filter_codes: Optional[Sequence[str]] = None, split: str = "all"
) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to our reference Cremp sdf trajectory."""

    def get_datasets_for_split(split: str):
        """Helper function to get datasets for a given split."""
        return data.parse_datasets_from_directory_new(
            root=f"{data_path}",
            traj_pattern= "^(.*).npz",
            topology_pattern= "^(.*).sdf",
            as_sdf=True,
            filter_codes=filter_codes,
        )

    if split in ["train", "val", "test"]:
        datasets = get_datasets_for_split(split)
    elif split == "all":
        datasets = get_datasets_for_split("train") + get_datasets_for_split("val") + get_datasets_for_split("test")
    else:
        raise ValueError(f"Invalid split: {split}")

    return {dataset.label(): dataset.trajectory for dataset in datasets}


def get_TBGSamples_datasets(
    data_path: str, filter_codes: Optional[Sequence[str]] = None
) -> Dict[str, data.MDtrajDataset]:
    """Returns a dictionary mapping peptide names to the datasets of TBG samples."""
    datasets = data.parse_datasets_from_directory(
        root=f"{data_path}/tbg-samples/",
        traj_pattern="^(.*).dcd",
        pdb_pattern="^(.*).pdb",
        filter_codes=filter_codes,
    )
    return {dataset.label(): dataset for dataset in datasets}


def get_MDGenSamples_4AA_datasets(
    data_path: str, filter_codes: Optional[Sequence[str]] = None
) -> Dict[str, data.MDtrajDataset]:
    """Returns a dictionary mapping peptide names to the datasets of MDGen samples for 4AA systems."""
    datasets = data.parse_datasets_from_directory(
        root=f"{data_path}/mdgen-samples/4AA_test",
        traj_pattern="^(.*)_i100.xtc",
        pdb_pattern="^(.*)_i100.pdb",
        filter_codes=filter_codes,
    )
    return {dataset.label(): dataset for dataset in datasets}


def get_MDGenSamples_5AA_datasets(
    data_path: str, filter_codes: Optional[Sequence[str]] = None
) -> Dict[str, data.MDtrajDataset]:
    """Returns a dictionary mapping peptide names to the datasets of MDGen samples for 5AA systems."""
    datasets = data.parse_datasets_from_directory(
        root=f"{data_path}/mdgen-samples/5AA_test",
        traj_pattern="^(.*)_i100.xtc",
        pdb_pattern="^(.*)_i100.pdb",
        filter_codes=filter_codes,
    )
    return {dataset.label(): dataset for dataset in datasets}


def get_BoltzSamples_datasets(
    data_path: str, filter_codes: Optional[Sequence[str]] = None
) -> Dict[str, data.MDtrajDataset]:
    """Returns a dictionary mapping peptide names to the datasets of Boltz-1 samples."""
    datasets = data.parse_datasets_from_directory(
        root=f"{data_path}/boltz-preprocessed/",
        traj_pattern="^(.*).xtc",
        pdb_pattern="^(.*).pdb",
        filter_codes=filter_codes,
    )
    return {dataset.label(): dataset for dataset in datasets}


def get_BioEmuSamples_datasets(
    data_path: str, filter_codes: Optional[Sequence[str]] = None
) -> Dict[str, data.MDtrajDataset]:
    """Returns a dictionary mapping peptide names to the datasets of BioEmu samples."""
    def add_suffix(label: str) -> str:
        """Adds the suffix to the label."""
        if "_sidechain_rec" not in label:
            return label + "_sidechain_rec"

    def remove_suffix(label: str) -> str:
        """Removes the suffix from the label."""
        if "_sidechain_rec" in label:
            return label.replace("_sidechain_rec", "")

    # Add suffix to filter codes.
    if filter_codes is not None:
        filter_codes = [add_suffix(code) for code in filter_codes]

    datasets = data.parse_datasets_from_directory(
        root=f"{data_path}/bioemu-samples-with-sidechains/",
        traj_pattern="^(.*).xtc",
        pdb_pattern="^(.*).pdb",
        filter_codes=filter_codes,
    )

    labels = [remove_suffix(dataset.label()) for dataset in datasets]
    return dict(zip(labels, datasets))


def get_ChignolinReference_dataset(data_path: str, split: str = "all") -> Dict[str, data.MDtrajDataset]:
    """Returns a dictionary mapping peptide names to our reference 2AA MDTraj trajectory."""

    root = Path(data_path) / "fast-folding/processed/chignolin"
    pdb_file = root / "filtered.pdb"

    if split == "all":
        traj_files = list(root.rglob("*.xtc"))
    else:
        traj_files = list((root / split).rglob("*.xtc"))

    dataset = data.MDtrajDataset(
        root=str(root), pdb_file=str(pdb_file), traj_files=list(map(str, traj_files)), label="filtered"
    )

    return {"filtered": dataset}


def get_plot_path(plot_path: Optional[str] = None):
    """Returns the default plot path if none provided."""
    if plot_path:
        return plot_path

    plot_path = os.environ.get("JAMUN_PLOT_PATH")
    if plot_path:
        return plot_path

    env_file = os.path.join(find_project_root(), ".env")
    plot_path = dotenv.get_key(env_file, "JAMUN_PLOT_PATH")
    if plot_path:
        return plot_path

    raise ValueError("plot_path must be provided as JAMUN_PLOT_PATH in environment variable or .env file")


def get_analysis_path(analysis_path: Optional[str] = None):
    """Returns the default analysis path if none provided."""
    if analysis_path:
        return analysis_path

    analysis_path = os.environ.get("JAMUN_ANALYSIS_PATH")
    if analysis_path:
        return analysis_path

    env_file = os.path.join(find_project_root(), ".env")
    analysis_path = dotenv.get_key(env_file, "JAMUN_ANALYSIS_PATH")
    if analysis_path:
        return analysis_path

    raise ValueError("analysis_path must be provided as JAMUN_ANALYSIS_PATH in environment variable or .env file")


def get_data_path(data_path: Optional[str] = None):
    """Returns the default data path if none provided."""
    if data_path:
        return data_path

    data_path = os.environ.get("JAMUN_DATA_PATH")
    if data_path:
        return data_path

    env_file = os.path.join(find_project_root(), ".env")
    data_path = dotenv.get_key(env_file, "JAMUN_DATA_PATH")
    if data_path:
        return data_path

    raise ValueError("data_path must be provided as JAMUN_DATA_PATH in environment variable or .env file")


def load_all_trajectories_with_info(
    trajectory_name: str,
    data_path: Optional[str],
    run_path: Optional[str] = None,
    wandb_run: Optional[str] = None,
    filter_codes: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[md.Trajectory, Dict[str, Any]]]:
    """Returns all trajectories, trajectory files, and topology file for this model."""
    data_path = get_data_path(data_path)
    py_logger.info(f"Using data_path: {data_path}")

    if trajectory_name == "JAMUN":
        if not run_path and not wandb_run:
            raise ValueError("Must provide either --run-path or --wandb-run for JAMUN trajectory")
        if run_path and wandb_run:
            raise ValueError("Must provide only one of --run-path or --wandb-run for JAMUN trajectory")

        if wandb_run:
            run_paths = [utils.get_run_path_for_wandb_run(wandb_run)]
        else:
            run_paths = [run_path]

        return get_JAMUN_trajectories(run_paths, filter_codes=filter_codes)

    if trajectory_name == "MDGenReference":
        datasets = get_MDGenReference_datasets(
            data_path,
            filter_codes=filter_codes,
        )
    elif trajectory_name == "TimewarpReference":
        datasets = get_TimewarpReference_datasets(
            data_path,
            filter_codes=filter_codes,
        )
    elif trajectory_name == "JAMUNReference_2AA":
        datasets = get_JAMUNReference_2AA_datasets(
            data_path,
            filter_codes=filter_codes,
        )
    elif trajectory_name == "JAMUNReference_5AA":
        datasets = get_JAMUNReference_5AA_datasets(
            data_path,
            filter_codes=filter_codes,
        )
    elif trajectory_name == "TBG":
        datasets = get_TBGSamples_datasets(
            data_path,
            filter_codes=filter_codes,
        )
    elif trajectory_name == "MDGenSamples_4AA":
        datasets = get_MDGenSamples_4AA_datasets(
            data_path,
            filter_codes=filter_codes,
        )
    elif trajectory_name == "MDGenSamples_5AA":
        datasets = get_MDGenSamples_5AA_datasets(
            data_path,
            filter_codes=filter_codes,
        )
    elif trajectory_name == "BoltzSamples":
        datasets = get_BoltzSamples_datasets(
            data_path,
            filter_codes=filter_codes,
        )
    elif trajectory_name == "BioEmuSamples":
        datasets = get_BioEmuSamples_datasets(
            data_path,
            filter_codes=filter_codes,
        )
    elif trajectory_name == "ChignolinReference":
        datasets = get_ChignolinReference_dataset(
            data_path,
        )
    else:
        raise ValueError(
            f"Trajectory type {trajectory_name} not supported. Available options: JAMUN, MDGenReference, TimewarpReference, JAMUNReference_2AA, JAMUNReference_5AA, Chignolin"
        )

    return {
        key: (dataset.trajectory, {"trajectory_files": dataset.trajectory_files, "topology_file": dataset.topology_file})
        for key, dataset in datasets.items()
    }


def load_trajectory_with_info(
    trajectory_name: str,
    peptide: str,
    data_path: Optional[str],
    run_path: Optional[str] = None,
    wandb_run: Optional[str] = None,
) -> Tuple[md.Trajectory, Dict[str, Any]]:
    """Returns the trajectory, trajectory files, and topology file for this model and peptide."""
    return load_all_trajectories_with_info(
        trajectory_name=trajectory_name,
        data_path=data_path,
        run_path=run_path,
        wandb_run=wandb_run,
        filter_codes=[peptide],
    )[peptide]
