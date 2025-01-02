import os
from typing import Dict, Optional, Sequence, Tuple

import mdtraj as md
import pyemma

from jamun.data import parse_datasets_from_directory
from jamun.utils import convert_to_one_letter_code, convert_to_three_letter_code, get_wandb_run_config


def find_project_root() -> str:
    """Returns the path to the root of the project."""
    current_dir = os.path.dirname(os.path.abspath(''))
    while not os.path.exists(os.path.join(current_dir, "pyproject.toml")):
        current_dir = os.path.dirname(current_dir)
    return current_dir

def get_peptides_in_run(wandb_sample_run_path: str) -> Tuple[Sequence[str], str]:
    """Returns the list of peptides sampled in a run and the output directory where they are stored."""

    cfg = get_wandb_run_config(wandb_sample_run_path)
    output_dir = os.path.join(cfg["paths"]["run_path"])
    if output_dir.startswith("."):
        # Path is relative to the project root.
        output_dir = os.path.join(find_project_root(), output_dir)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} not found.")

    peptides_in_run = sorted(os.listdir(os.path.join(output_dir, "sampler")))
    if len(peptides_in_run) == 0:
        raise ValueError(f"No sampled peptides found in {wandb_sample_run_path}")

    return peptides_in_run, output_dir


def get_JAMUN_trajectory_files(wandb_sample_run_paths: Sequence[str]) -> Dict[str, Dict[str, str]]:
    """Returns a dictionary mapping peptide names to the path of the PDB file containing the predicted structure."""

    files = {}
    for wandb_sample_run_path in wandb_sample_run_paths:

        # Get the list of peptides sampled in the run and the output directory where they are stored.
        peptides_in_run, output_dir = get_peptides_in_run(wandb_sample_run_path)

        # Check that there are no duplicates amongst runs.
        for peptide in peptides_in_run:
            if peptide in files:
                raise ValueError(f"Peptide {peptide} found in multiple runs")

            files[peptide] = {
                "dcd": f"{output_dir}/sampler/{peptide}/predicted_samples/dcd/joined.dcd",
                "pdb": f"{output_dir}/pdbs/{peptide}-modified.pdb",
            }

    return files


def get_JAMUN_trajectories(wandb_sample_run_paths: Sequence[str]) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to the sampled MDTraj trajectory."""
    files = get_JAMUN_trajectory_files(wandb_sample_run_paths)
    trajs = {}
    for peptide, peptide_files in files.items():
        trajs[peptide] = md.load_dcd(peptide_files["dcd"], top=peptide_files["pdb"])
    return trajs

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
    """Returns a dictionary mapping peptide names to the Timewarp MDTraj trajectory."""
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


def get_featurized_trajectory(traj: md.Trajectory, cossin: bool = False):
    """Featurize an MDTraj trajectory with backbone and sidechain torsion angles.

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
