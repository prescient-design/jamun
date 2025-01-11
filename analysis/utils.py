from typing import Dict, Optional, Sequence, Tuple, List
import os
import logging

import mdtraj as md
import pyemma

from jamun.data import parse_datasets_from_directory
from jamun.utils import convert_to_one_letter_code, convert_to_three_letter_code, get_wandb_run_config

logging.basicConfig(
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    level=logging.INFO
)
py_logger = logging.getLogger("analysis")


def find_project_root() -> str:
    """Returns the path to the root of the project."""
    current_dir = os.path.dirname(os.path.abspath(''))
    while not os.path.exists(os.path.join(current_dir, "pyproject.toml")):
        current_dir = os.path.dirname(current_dir)
    return current_dir


def get_run_path_for_wandb_run(wandb_run_path: str) -> str:
    """Returns the path to the run directory given a wandb run path."""
    cfg = get_wandb_run_config(wandb_run_path)
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


def get_JAMUN_trajectory_files(run_paths: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Returns a dictionary mapping peptide names to the path of the PDB file containing the predicted structure."""

    trajectory_files = {}
    for run_path in run_paths:

        # Get the list of peptides sampled in the run and the output directory where they are stored.
        peptides_in_run = get_peptides_in_JAMUN_run(run_path)

        # Check that there are no duplicates amongst runs.
        for peptide in peptides_in_run:
            if peptide in trajectory_files:
                raise ValueError(f"Peptide {peptide} found in multiple runs: {run_path} and {trajectory_files[peptide]['dcd']}")

            trajectory_files[peptide] = {
                "dcd": f"{run_path}/sampler/{peptide}/predicted_samples/dcd/joined.dcd",
            }

            if not os.path.exists(trajectory_files[peptide]["dcd"]):
                raise ValueError(f"DCD file {trajectory_files[peptide]['dcd']} not found.")

            for pdb_file in [
                f"{run_path}/sampler/{peptide}/topology.pdb",
                f"{run_path}/sampler/{peptide}/predicted_samples/pdb/0.pdb",
                f"{run_path}/pdbs/{peptide}-modified.pdb"
            ]:
                if os.path.exists(pdb_file):
                    trajectory_files[peptide]["pdb"] = pdb_file
                    break

            if "pdb" not in trajectory_files[peptide]:
                raise ValueError(f"No PDB file found for peptide {peptide} in run {run_path}")

    return trajectory_files



def get_JAMUN_trajectories(run_paths: Sequence[str], filter_codes: Optional[Sequence[str]] = None) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to the sampled MDTraj trajectory."""
    trajectory_files = get_JAMUN_trajectory_files(run_paths)
    trajectories = {}
    for peptide, peptide_files in trajectory_files.items():
        if filter_codes and peptide not in filter_codes:
            continue
        trajectories[peptide] = md.load_dcd(peptide_files["dcd"], top=peptide_files["pdb"])
    return trajectories


def get_Timewarp_trajectories(data_path: str, peptide_type: str, filter_codes: Optional[Sequence[str]] = None) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to the Timewarp MDTraj trajectory."""
    if peptide_type == "2AA":
        peptide_type_dir = "2AA-1-large"
    elif peptide_type == "4AA":
        peptide_type_dir = "4AA-large"
    else:
        raise ValueError(f"Invalid peptide type: {peptide_type}")

    one_letter_filter_codes = [''.join([convert_to_one_letter_code(aa) for aa in code]) for code in filter_codes]
    assert len(set(one_letter_filter_codes)) == len(one_letter_filter_codes), "Filter codes must be unique"

    datasets = parse_datasets_from_directory(
        root=f"{data_path}/timewarp/{peptide_type_dir}/test/",
        traj_pattern="^(.*)-traj-arrays.npz",
        pdb_pattern="^(.*)-traj-state0.pdb",
        filter_codes=one_letter_filter_codes,
    )

    # Remap keys.
    one_letter_filter_codes_map = dict(zip(one_letter_filter_codes, filter_codes))
    return {one_letter_filter_codes_map[dataset.label()]: dataset.trajectory for dataset in datasets}


def get_OpenMM_trajectories(data_path: str, filter_codes: Optional[Sequence[str]] = None) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to (our) OpenMM MDTraj trajectory."""
    three_letter_filter_codes = ['_'.join([convert_to_three_letter_code(aa) for aa in code]) for code in filter_codes]
    assert len(set(three_letter_filter_codes)) == len(three_letter_filter_codes), "Filter codes must be unique"

    datasets = parse_datasets_from_directory(
        root=f"{data_path}/capped_diamines/timewarp_splits/test",
        traj_pattern="^(.*).xtc",
        pdb_pattern="^(.*).pdb",
        filter_codes=three_letter_filter_codes,
    )

    # Remap keys.
    three_letter_filter_codes_map = dict(zip(three_letter_filter_codes, filter_codes))
    return {three_letter_filter_codes_map[dataset.label()]: dataset.trajectory for dataset in datasets}


def get_TBG_trajectories(root: str) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to the TBG MDTraj trajectory."""
    raise NotImplementedError("TBG trajectories not implemented yet.")


def get_pyEMMA_featurized_trajectory(traj: md.Trajectory, cossin: bool = False):
    """Featurize an MDTraj trajectory with backbone and sidechain torsion angles using pyEMMA.

    Args:
        traj (mdtraj.Trajectory): Input trajectory to featurize
        cossin (bool): Whether to transform angles to cosine/sine pairs

    Returns:
        tuple: (feat, featurized_traj) where feat is the PyEMMA featurizer
        and featurized_traj is the transformed trajectory data
    """
    feat = pyemma.coordinates.featurizer(traj.topology)
    feat.add_backbone_torsions(cossin=cossin)
    feat.add_sidechain_torsions(cossin=cossin)
    featurized_traj = feat.transform(traj)
    return feat, featurized_traj
