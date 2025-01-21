from typing import Dict, Optional, Sequence, Tuple, List
import os

import pyemma.coordinates.data
import tqdm
import numpy as np
import mdtraj as md
import pyemma
import pyemma.coordinates.clustering
import pandas as pd
from scipy.spatial import distance
from statsmodels.tsa import stattools
import warnings

from jamun import data
from jamun import utils

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pyemma.util.exceptions.PyEMMA_DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def find_project_root() -> str:
    """Returns the path to the root of the project."""
    current_dir = os.path.dirname(os.path.abspath(""))
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
            ]:
                if os.path.exists(pdb_file):
                    trajectory_files[peptide]["pdb"] = pdb_file
                    break

            if "pdb" not in trajectory_files[peptide]:
                raise ValueError(f"No PDB file found for peptide {peptide} in run {run_path}")

    # Remove the prefix "uncapped_" and "capped_" from the peptide names.
    for peptide in list(trajectory_files.keys()):
        for prefix in ["uncapped_", "capped_"]:
            if not peptide.startswith(prefix):
                continue
            trajectory_files[peptide[len(prefix) :]] = trajectory_files.pop(peptide)
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
            root=f"{data_path}/mdgen/data/4AA_sims_partitioned/{split}/",
            traj_pattern="^(.*).xtc",
            pdb_pattern="^(.*).pdb",
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


def get_2AA_JAMUNReference_trajectories(
    data_path: str, filter_codes: Optional[Sequence[str]] = None, split: str = "all"
) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to our reference 2AA MDTraj trajectory."""
    three_letter_filter_codes = [
        "_".join([utils.convert_to_three_letter_code(aa) for aa in code]) for code in filter_codes
    ]
    assert len(set(three_letter_filter_codes)) == len(three_letter_filter_codes), "Filter codes must be unique"

    def get_datasets_for_split(split: str):
        """Helper function to get datasets for a given split."""
        return data.parse_datasets_from_directory(
            root=f"{data_path}/capped_diamines/timewarp_splits/{split}",
            traj_pattern="^(.*).xtc",
            pdb_pattern="^(.*).pdb",
            filter_codes=three_letter_filter_codes,
        )

    if split in ["train", "val", "test"]:
        datasets = get_datasets_for_split(split)
    elif split == "all":
        datasets = get_datasets_for_split("train") + get_datasets_for_split("val") + get_datasets_for_split("test")

    # Remap keys.
    filter_codes_map = dict(zip(three_letter_filter_codes, filter_codes))
    return {filter_codes_map[dataset.label()]: dataset.trajectory for dataset in datasets}


def get_5AA_JAMUNReference_trajectories(
    data_path: str, filter_codes: Optional[Sequence[str]] = None
) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to our reference 5AA MDTraj trajectories."""
    three_letter_filter_codes = [
        "_".join([utils.convert_to_three_letter_code(aa) for aa in code]) for code in filter_codes
    ]
    assert len(set(three_letter_filter_codes)) == len(three_letter_filter_codes), "Filter codes must be unique"

    datasets = data.parse_datasets_from_directory(
        root=f"{data_path}/5AA/",
        traj_pattern="^(.*).xtc",
        pdb_pattern="^(.*).pdb",
        filter_codes=three_letter_filter_codes,
    )

    # Remap keys.
    filter_codes_map = dict(zip(three_letter_filter_codes, filter_codes))
    return {filter_codes_map[dataset.label()]: dataset.trajectory for dataset in datasets}


def get_TBG_trajectories(root: str) -> Dict[str, md.Trajectory]:
    """Returns a dictionary mapping peptide names to the TBG MDTraj trajectory."""
    raise NotImplementedError("TBG trajectories not implemented yet.")


def compute_PMF(traj: md.Trajectory, num_bins: int = 50) -> Dict[str, np.ndarray]:
    """Compute the potential of mean force (PMF) for a trajectory along a dihedral angle."""
    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    num_dihedrals = phi.shape[1]
    pmf = np.zeros((num_dihedrals, num_bins - 1, num_bins - 1))
    xedges = np.linspace(-np.pi, np.pi, num_bins)
    yedges = np.linspace(-np.pi, np.pi, num_bins)

    for dihedral_index in range(num_dihedrals):
        H, _, _ = np.histogram2d(phi[:, dihedral_index], psi[:, dihedral_index], bins=np.linspace(-np.pi, np.pi, num_bins))
        pmf[dihedral_index] = -np.log(H.T) + np.max(np.log(H.T))

    return {
        "pmf": pmf,
        "xedges": xedges,
        "yedges": yedges,
    }


def compute_PMFs(traj: md.Trajectory, ref_traj: md.Trajectory) -> Dict[str, np.ndarray]:
    """Compute the potential of mean force (PMF) for a trajectory along a dihedral angle."""
    return {
        "traj_pmf": compute_PMF(traj),
        "ref_traj_pmf": compute_PMF(ref_traj),
    }


def featurize_trajectory(traj: md.Trajectory, cossin: bool) -> Tuple[pyemma.coordinates.featurizer, np.ndarray]:
    """Featurize an MDTraj trajectory with backbone and sidechain torsion angles using pyEMMA.
    Adapted from MDGen.

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


def get_bond_lengths_for_trajectory(traj: md.Trajectory) -> Tuple[pyemma.coordinates.data.MDFeaturizer, np.ndarray]:
    """Compute bond lengths for a trajectory."""
    feat = pyemma.coordinates.featurizer(traj.topology)
    heavy_atom_distance_pairs = feat.pairs(feat.select_Heavy())
    feat.add_distances(heavy_atom_distance_pairs, periodic=False)
    featurized_traj = feat.transform(traj)
    return {
        "feat": feat,
        "featurized_traj": featurized_traj,
    }


def compute_bond_lengths(traj: md.Trajectory, ref_traj: md.Trajectory) -> Dict[str, np.ndarray]:
    """Compute bond lengths for a trajectory."""
    return {
        "traj_bond_lengths": get_bond_lengths_for_trajectory(traj),
        "ref_traj_bond_lengths": get_bond_lengths_for_trajectory(ref_traj),
    }

def get_KMeans(
    traj_featurized: np.ndarray, k: int = 100
) -> Tuple[pyemma.coordinates.clustering.KmeansClustering, np.ndarray]:
    """Cluster a featurized trajectory using k-means clustering. Taken from MDGen."""
    kmeans = pyemma.coordinates.cluster_kmeans(traj_featurized, k=k, max_iter=100, fixed_seed=137)
    return kmeans, kmeans.transform(traj_featurized)[:, 0]


def get_MSM(traj_featurized: np.ndarray, lag: int, num_states: int):
    """Estimate an Markov State Model (MSM), PCCA (clustering of MSM states), and coarse-grained MSM from a trajectory. Taken from MDGen."""
    msm = pyemma.msm.estimate_markov_model(traj_featurized, lag=lag)
    pcca = msm.pcca(num_states)
    assert len(msm.metastable_assignments) == lag // num_states
    cmsm = pyemma.msm.estimate_markov_model(msm.metastable_assignments[traj_featurized], lag=lag)
    return msm, pcca, cmsm


def discretize(
    traj_featurized: np.ndarray, kmeans: pyemma.coordinates.clustering.KmeansClustering, msm: pyemma.msm.MSM
) -> np.ndarray:
    """Returns the metastable state assignments for a trajectory, after clustering. Taken from MDGen."""
    return msm.metastable_assignments[kmeans.transform(traj_featurized)[:, 0]]


def compute_JSD_stats(
    traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer
) -> Dict[str, float]:
    """Compute Jenson-Shannon distances for a trajectory and reference trajectory. Taken from MDGen."""
    results = {}
    for i, feat in enumerate(feats.describe()):
        ref_p = np.histogram(ref_traj_featurized[:, i], range=(-np.pi, np.pi), bins=100)[0]
        traj_p = np.histogram(traj_featurized[:, i], range=(-np.pi, np.pi), bins=100)[0]
        results[feat] = distance.jensenshannon(ref_p, traj_p)

    # Compute JSDs for backbone, sidechain, and all torsions.
    results["backbone"] = np.mean([results[feat] for feat in feats.describe() if feat.startswith("PHI") or feat.startswith("PSI")])
    results["sidechain"] = np.mean([results[feat] for feat in feats.describe() if feat.startswith("CHI")])
    results["all_torsions"] = np.mean([results[feat] for feat in feats.describe() if feat.startswith("PHI") or feat.startswith("PSI") or feat.startswith("CHI")])            

    # Remove the first psi angle and last phi angle.
    # The first psi angle is for the N-terminal and the last phi angle is for the C-terminal.
    psi_indices = [i for i, feat in enumerate(feats.describe()) if feat.startswith("PSI")]
    phi_indices = [i for i, feat in enumerate(feats.describe()) if feat.startswith("PHI")]
    psi_indices = psi_indices[1:]
    phi_indices = phi_indices[:-1]

    for phi_index, psi_index in zip(phi_indices, psi_indices):
        ref_features = np.stack([ref_traj_featurized[:, phi_index], ref_traj_featurized[:, psi_index]], axis=1)
        ref_p = np.histogram2d(*ref_features.T, range=((-np.pi, np.pi), (-np.pi, np.pi)), bins=50)[0]

        traj_features = np.stack([traj_featurized[:, phi_index], traj_featurized[:, psi_index]], axis=1)
        traj_p = np.histogram2d(*traj_features.T, range=((-np.pi, np.pi), (-np.pi, np.pi)), bins=50)[0]

        phi_psi_feats = [feats.describe()[phi_index], feats.describe()[psi_index]]
        results["|".join(phi_psi_feats)] = distance.jensenshannon(ref_p.flatten(), traj_p.flatten())

    return results


def compute_JSDs_stats_against_time(
    traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer
) -> np.ndarray:
    """Computes the Jenson-Shannon distance between the Ramachandran distributions of a trajectory and a reference trajectory at different time points."""
    steps = np.linspace(0, len(traj_featurized), 11).astype(int)[1:]

    return {
        step: compute_JSD_stats(
            traj_featurized[:step],
            ref_traj_featurized,
            feats,
        )
        for step in steps
    }


def compute_TICA(
    traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, pyemma.coordinates.tica]:
    """Compute TICA projections of trajectories."""
    tica = pyemma.coordinates.tica(ref_traj_featurized, lag=1000, kinetic_map=True)
    ref_tica = tica.transform(ref_traj_featurized)
    traj_tica = tica.transform(traj_featurized)
    return traj_tica, ref_tica, tica


def compute_TICA_stats(traj_tica: np.ndarray, ref_tica: np.ndarray) -> Dict[str, float]:
    """Compute Jenson-Shannon distances on TICA projections of trajectories."""
    tica_0_min = min(ref_tica[:, 0].min(), traj_tica[:, 0].min())
    tica_0_max = max(ref_tica[:, 0].max(), traj_tica[:, 0].max())

    tica_1_min = min(ref_tica[:, 1].min(), traj_tica[:, 1].min())
    tica_1_max = max(ref_tica[:, 1].max(), traj_tica[:, 1].max())

    ref_p = np.histogram(ref_tica[:, 0], range=(tica_0_min, tica_0_max), bins=100)[0]
    traj_p = np.histogram(traj_tica[:, 0], range=(tica_0_min, tica_0_max), bins=100)[0]
    tica_0_jsd = distance.jensenshannon(ref_p, traj_p)

    ref_p = np.histogram2d(*ref_tica[:, :2].T, range=((tica_0_min, tica_0_max), (tica_1_min, tica_1_max)), bins=50)[0]
    traj_p = np.histogram2d(*traj_tica[:, :2].T, range=((tica_0_min, tica_0_max), (tica_1_min, tica_1_max)), bins=50)[0]
    tica_01_jsd = distance.jensenshannon(ref_p.flatten(), traj_p.flatten())

    return {
        "TICA-0 JSD": tica_0_jsd,
        "TICA-0,1 JSD": tica_01_jsd,
    }


def compute_autocorrelation_stats(traj_tica: np.ndarray, ref_tica: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute autocorrelation functions for TICA projections of trajectories."""
    nlag = 1000
    ref_autocorr = stattools.acovf(ref_tica[:, 0], nlag=nlag, adjusted=True, demean=False)
    traj_autocorr = stattools.acovf(traj_tica[:, 0], nlag=nlag, adjusted=True, demean=False)
    return {"ref_autocorr": ref_autocorr, "traj_autocorr": traj_autocorr}


def compute_MSM_stats(
    traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray, tica: pyemma.coordinates.tica
) -> Dict[str, np.ndarray]:
    """Compute MSM statistics for a trajectory and reference trajectory."""
    # MSM analysis
    kmeans, ref_kmeans = get_KMeans(tica.transform(ref_traj_featurized))
    msm, pcca, cmsm = get_MSM(ref_kmeans, lag=1000, num_states=10)

    ref_discrete = discretize(tica.transform(ref_traj_featurized), kmeans, msm)
    traj_discrete = discretize(tica.transform(traj_featurized), kmeans, msm)

    # Compute metastable probabilities
    ref_metastable_probs = (ref_discrete == np.arange(10)[:, None]).mean(1)
    traj_metastable_probs = (traj_discrete == np.arange(10)[:, None]).mean(1)

    # Compute transition matrices
    msm_transition_matrix = np.eye(10)
    for a, i in enumerate(cmsm.active_set):
        for b, j in enumerate(cmsm.active_set):
            msm_transition_matrix[i, j] = cmsm.transition_matrix[a, b]

    msm_pi = np.zeros(10)
    msm_pi[cmsm.active_set] = cmsm.pi

    # Compute trajectory MSM
    traj_msm = pyemma.msm.estimate_markov_model(traj_discrete, lag=10)
    traj_transition_matrix = np.eye(10)
    for a, i in enumerate(traj_msm.active_set):
        for b, j in enumerate(traj_msm.active_set):
            traj_transition_matrix[i, j] = traj_msm.transition_matrix[a, b]

    traj_pi = np.zeros(10)
    traj_pi[traj_msm.active_set] = traj_msm.pi

    # Store MSM results
    return {
        "ref_metastable_probs": ref_metastable_probs,
        "traj_metastable_probs": traj_metastable_probs,
        "msm_transition_matrix": msm_transition_matrix,
        "msm_pi": msm_pi,
        "traj_transition_matrix": traj_transition_matrix,
        "traj_pi": traj_pi,
        "pcca_pi": pcca._pi_coarse,
    }
