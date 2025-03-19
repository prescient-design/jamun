from typing import Dict, Any, Tuple, Optional
import os
import logging
import argparse
import sys
import pickle
import warnings
import collections

import mdtraj as md
import pyemma.coordinates.data
import numpy as np
import mdtraj as md
import pyemma
import pyemma.coordinates.clustering
import pandas as pd
from scipy.spatial import distance
from statsmodels.tsa import stattools

# TODO: Fix imports
sys.path.append("./")
import load_trajectory
import pyemma_helper

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("analysis")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pyemma.util.exceptions.PyEMMA_DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def load_trajectories(
    trajectory: str,
    reference: str,
    peptide: str,
    data_path: str,
    run_path: Optional[str],
    wandb_run: Optional[str],
) -> Tuple[md.Trajectory, md.Trajectory]:
    """Load trajectories based on command line arguments."""

    trajs_md = load_trajectory.load_trajectories_by_name(
        trajectory,
        peptide,
        data_path,
        run_path,
        wandb_run,
    )
    if not trajs_md:
        raise ValueError(f"No {trajectory} trajectories found for peptide {peptide}")

    ref_trajs_md = load_trajectory.load_trajectories_by_name(
        reference,
        peptide,
        data_path,
        run_path,
        wandb_run,
    )
    if not ref_trajs_md:
        raise ValueError(f"No {reference} trajectories found for peptide {peptide}")

    traj_md = trajs_md[peptide]
    ref_traj_md = ref_trajs_md[peptide]

    return traj_md, ref_traj_md


def subset_reference_trajectory(
    traj_md: md.Trajectory,
    ref_traj_md: md.Trajectory,
    traj_seconds_per_sample: float,
    ref_traj_seconds_per_sample: float,
    base_factor: float = 1.0,
) -> md.Trajectory:
    """Subset reference trajectory to match base_factor x length of the trajectory in actual sampling time."""
    traj_time = traj_seconds_per_sample * traj_md.n_frames
    ref_traj_time = ref_traj_seconds_per_sample * ref_traj_md.n_frames
    factor = min(traj_time / ref_traj_time, 1) * base_factor
    ref_traj_subset_md = ref_traj_md[: int(factor * ref_traj_md.n_frames)]
    return ref_traj_subset_md


def featurize_trajectory_with_torsions(
    traj: md.Trajectory, cossin: bool
) -> Tuple[pyemma.coordinates.featurizer, np.ndarray]:
    """Featurize an MDTraj trajectory with backbone and sidechain torsion angles using pyEMMA.
    Adapted from MDGen.

    Args:
        traj (mdtraj.Trajectory): Input trajectory to featurize
        cossin (bool): Whether to transform angles to cosine/sine pairs

    Returns:
        tuple: (feats, traj_featurized) where feats is the PyEMMA featurizer
        and traj_featurized is the transformed trajectory data
    """
    feats = pyemma.coordinates.featurizer(traj.topology)
    feats.add_backbone_torsions(cossin=cossin)
    feats.add_sidechain_torsions(cossin=cossin)
    traj_featurized = feats.transform(traj)
    return feats, traj_featurized


def featurize_trajectory_with_distances(traj: md.Trajectory) -> Tuple[pyemma.coordinates.featurizer, np.ndarray]:
    """Featurize an MDTraj trajectory with pairwise distances using pyEMMA."""
    feats = pyemma.coordinates.featurizer(traj.topology)
    alpha_carbon_distance_pairs = feats.pairs(feats.select_Ca())
    feats.add_distances(alpha_carbon_distance_pairs, periodic=False)
    traj_featurized = feats.transform(traj)
    return feats, traj_featurized


def featurize_trajectory(traj: md.Trajectory) -> Dict[str, np.ndarray]:
    """Featurize an MDTraj trajectory with backbone, and sidechain torsion angles and distances using pyEMMA."""

    feats, traj_featurized = featurize_trajectory_with_torsions(traj, cossin=False)
    feats_cossin, traj_featurized_cossin = featurize_trajectory_with_torsions(traj, cossin=True)
    feats_dists, traj_featurized_dists = featurize_trajectory_with_distances(traj)

    return {
        "feats": {
            "torsions": feats,
            "torsions_cossin": feats_cossin,
            "distances": feats_dists,
        },
        "traj_featurized": {
            "torsions": traj_featurized,
            "torsions_cossin": traj_featurized_cossin,
            "distances": traj_featurized_dists,
        },
    }


def featurize_trajectories(traj_md: md.Trajectory, ref_traj_md: md.Trajectory) -> Dict[str, Dict[str, np.ndarray]]:
    """Featurize MDTraj trajectories with backbone, and sidechain torsion angles and distances using pyEMMA."""
    return {
        "traj": featurize_trajectory(traj_md),
        "ref_traj": featurize_trajectory(ref_traj_md),
    }


def compute_feature_histograms_for_trajectory(traj_featurized_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute histograms of features for a trajectory."""
    return {
        key: pyemma_helper.compute_1D_histogram(traj_featurized)
        for key, traj_featurized in traj_featurized_dict.items()
    }


def compute_feature_histograms(
    traj_featurized_dict: Dict[str, np.ndarray], ref_traj_featurized_dict: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Compute histograms of features for a trajectory."""
    return {
        "traj": compute_feature_histograms_for_trajectory(traj_featurized_dict),
        "ref_traj": compute_feature_histograms_for_trajectory(ref_traj_featurized_dict),
    }


def compute_PMF(
    traj_featurized: np.ndarray,
    feats: pyemma.coordinates.data.MDFeaturizer,
    num_bins: int = 50,
    internal_angles: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute the potential of mean force (PMF) for a trajectory along a dihedral angle."""
    psi_indices = [i for i, feat in enumerate(feats.describe()) if feat.startswith("PSI")]
    phi_indices = [i for i, feat in enumerate(feats.describe()) if feat.startswith("PHI")]

    if internal_angles:
        # Remove the first psi angle and last phi angle.
        # The first psi angle is for the N-terminal and the last phi angle is for the C-terminal.
        psi_indices = psi_indices[1:]
        phi_indices = phi_indices[:-1]

    phi = traj_featurized[:, phi_indices]
    psi = traj_featurized[:, psi_indices]
    num_dihedrals = phi.shape[1]

    pmf = np.zeros((num_dihedrals, num_bins - 1, num_bins - 1))
    xedges = np.linspace(-np.pi, np.pi, num_bins)
    yedges = np.linspace(-np.pi, np.pi, num_bins)

    for dihedral_index in range(num_dihedrals):
        H, _, _ = np.histogram2d(
            phi[:, dihedral_index], psi[:, dihedral_index], bins=np.linspace(-np.pi, np.pi, num_bins)
        )
        pmf[dihedral_index] = -np.log(H.T) + np.max(np.log(H.T))

    return {
        "pmf": pmf,
        "xedges": xedges,
        "yedges": yedges,
    }


def compute_dihedral_PMFs(
    traj: np.ndarray, ref_traj: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer
) -> Dict[str, np.ndarray]:
    """Compute the potential of mean force (PMF) for a trajectory along a dihedral angle."""
    return {
        "traj": {
            "pmf_all": compute_PMF(traj, feats, internal_angles=False),
            "pmf_internal": compute_PMF(traj, feats, internal_angles=True),
        },
        "ref_traj": {
            "pmf_all": compute_PMF(ref_traj, feats, internal_angles=False),
            "pmf_internal": compute_PMF(ref_traj, feats, internal_angles=True),
        },
    }


def get_KMeans(
    traj_featurized: np.ndarray, K: int
) -> Tuple[pyemma.coordinates.clustering.KmeansClustering, np.ndarray]:
    """Cluster a featurized trajectory using k-means clustering. Taken from MDGen."""
    kmeans = pyemma.coordinates.cluster_kmeans(traj_featurized, k=K, max_iter=100, fixed_seed=137)
    return kmeans, kmeans.transform(traj_featurized)[:, 0]


def get_MSM(traj_featurized: np.ndarray, lag: int, num_states: int):
    """Estimate an Markov State Model (MSM), PCCA (clustering of MSM states), and coarse-grained MSM from a trajectory. Taken from MDGen."""
    msm = pyemma.msm.estimate_markov_model(traj_featurized, lag=lag)
    pcca = msm.pcca(num_states)
    cmsm = pyemma.msm.estimate_markov_model(msm.metastable_assignments[traj_featurized], lag=lag)
    return msm, pcca, cmsm


def get_MSM_after_KMeans(ref_traj_tica: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute MSM after KMeans clustering."""
    kmeans, ref_kmeans = get_KMeans(ref_traj_tica, K=100)
    msm, pcca, cmsm = get_MSM(ref_kmeans, lag=1000, num_states=10)
    return {
        "kmeans": kmeans,
        "msm": msm,
        "pcca": pcca,
        "cmsm": cmsm,
    }


def discretize(
    traj_featurized: np.ndarray, kmeans: pyemma.coordinates.clustering.KmeansClustering, msm: pyemma.msm.MSM
) -> np.ndarray:
    """Returns the metastable state assignments for a trajectory, after clustering. Taken from MDGen."""
    return msm.metastable_assignments[kmeans.transform(traj_featurized)[:, 0]]


def compute_JSD_torsion_stats(
    traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer
) -> Dict[str, float]:
    """Compute Jenson-Shannon distances for a trajectory and reference trajectory. Taken from MDGen."""
    results = {}
    for i, feat in enumerate(feats.describe()):
        ref_p = np.histogram(ref_traj_featurized[:, i], range=(-np.pi, np.pi), bins=100)[0]
        traj_p = np.histogram(traj_featurized[:, i], range=(-np.pi, np.pi), bins=100)[0]
        results[feat] = distance.jensenshannon(ref_p, traj_p)

    # Compute JSDs for backbone, sidechain, and all torsions.
    results["backbone_torsions"] = np.mean(
        [
            results[feat] for feat in feats.describe()
            if feat.startswith("PHI") or feat.startswith("PSI")
        ]
    )
    results["sidechain_torsions"] = np.mean(
        [
            results[feat] for feat in feats.describe()
            if feat.startswith("CHI")
        ]
    )
    results["all_torsions"] = np.mean(
        [
            results[feat]
            for feat in feats.describe()
            if feat.startswith("PHI") or feat.startswith("PSI") or feat.startswith("CHI")
        ]
    )

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


def compute_JSD_torsion_stats_against_time_for_trajectory(
    traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer
) -> Dict[int, Dict[str, float]]:
    """Computes the Jenson-Shannon distance between the Ramachandran distributions of a trajectory and a reference trajectory at different time points."""
    steps = np.logspace(0, np.log10(len(traj_featurized)), num=10, dtype=int)
    return {
        step: compute_JSD_torsion_stats(
            traj_featurized[:step],
            ref_traj_featurized,
            feats,
        )
        for step in steps
    }


def compute_JSD_torsion_stats_against_time(
    traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Computes the Jenson-Shannon distance between the Ramachandran distributions of a trajectory and a reference trajectory at different time points."""
    return {
        "traj": compute_JSD_torsion_stats_against_time_for_trajectory(traj_featurized, ref_traj_featurized, feats),
        "ref_traj": compute_JSD_torsion_stats_against_time_for_trajectory(ref_traj_featurized, ref_traj_featurized, feats),
    }


def compute_torsion_decorrelations(traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer):
    """Computes decorrelations of torsion angles."""

    ref_traj = ref_traj_featurized
    traj = traj_featurized
    
    torsion_decorrelations = collections.defaultdict(dict)
    for i, feat in enumerate(feats.describe()):
        baseline = np.sin(ref_traj[:,i]).mean()**2 + np.cos(ref_traj[:,i]).mean()**2
        
        autocorr = stattools.acovf(np.sin(ref_traj[:,i]), demean=False, adjusted=True, nlag=100000)
        autocorr += stattools.acovf(np.cos(ref_traj[:,i]), demean=False, adjusted=True, nlag=100000)

        ref_traj_autocorrelations = (autocorr - baseline) / (1 - baseline)
        torsion_decorrelations[feat]["ref_traj_autocorrelations"] = ref_traj_autocorrelations
        torsion_decorrelations[feat]["ref_traj_decorrelation_time"] = np.where(ref_traj_autocorrelations < 1 / np.e)[0][0]

        autocorr = stattools.acovf(np.sin(traj[:,i]), demean=False, adjusted=True, nlag=100000)
        autocorr += stattools.acovf(np.cos(traj[:,i]), demean=False, adjusted=True, nlag=100000)

        traj_autocorrelations = (autocorr - baseline) / (1 - baseline)
        torsion_decorrelations[feat]["traj_autocorrelations"] = traj_autocorrelations
        try:
            torsion_decorrelations[feat]["traj_decorrelation_time"] = np.where(traj_autocorrelations < 1 / np.e)[0][0]
        except IndexError:
            torsion_decorrelations[feat]["traj_decorrelation_time"] = np.nan
    return torsion_decorrelations


def compute_TICA(traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute TICA projections of trajectories."""
    tica = pyemma.coordinates.tica(ref_traj_featurized, lag=1000, kinetic_map=True)
    ref_traj_tica = tica.transform(ref_traj_featurized)
    traj_tica = tica.transform(traj_featurized)
    return {
        "traj_tica": traj_tica,
        "ref_traj_tica": ref_traj_tica,
        "tica": tica,
    }


def compute_TICA_stats(traj_tica: np.ndarray, ref_traj_tica: np.ndarray) -> Dict[str, float]:
    """Compute Jenson-Shannon distances on TICA projections of trajectories."""
    tica_0_min = min(ref_traj_tica[:, 0].min(), traj_tica[:, 0].min())
    tica_0_max = max(ref_traj_tica[:, 0].max(), traj_tica[:, 0].max())

    tica_1_min = min(ref_traj_tica[:, 1].min(), traj_tica[:, 1].min())
    tica_1_max = max(ref_traj_tica[:, 1].max(), traj_tica[:, 1].max())

    ref_p = np.histogram(ref_traj_tica[:, 0], range=(tica_0_min, tica_0_max), bins=100)[0]
    traj_p = np.histogram(traj_tica[:, 0], range=(tica_0_min, tica_0_max), bins=100)[0]
    tica_0_jsd = distance.jensenshannon(ref_p, traj_p)

    ref_p = np.histogram2d(
        *ref_traj_tica[:, :2].T, range=((tica_0_min, tica_0_max), (tica_1_min, tica_1_max)), bins=50
    )[0]
    traj_p = np.histogram2d(*traj_tica[:, :2].T, range=((tica_0_min, tica_0_max), (tica_1_min, tica_1_max)), bins=50)[0]
    tica_01_jsd = distance.jensenshannon(ref_p.flatten(), traj_p.flatten())

    # Compute TICA projections for plot.
    return {
        "TICA-0 JSD": tica_0_jsd,
        "TICA-0,1 JSD": tica_01_jsd,
        "TICA-0,1 histograms": {
            "ref_traj": pyemma_helper.compute_2D_histogram(ref_traj_tica[:, 0], ref_traj_tica[:, 1]),
            "traj": pyemma_helper.compute_2D_histogram(traj_tica[:, 0], traj_tica[:, 1]),
        },
    }


def compute_TICA_decorrelations(traj_tica: np.ndarray, ref_traj_tica: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute autocorrelation and decorrelation times for TICA projections of trajectories."""
    mu = ref_traj_tica[:,0].mean()
    sigma = ref_traj_tica[:,0].std()

    ref_traj_tica_0_normalized = (ref_traj_tica[:,0] - mu) / sigma
    traj_tica_0_normalized = (traj_tica[:,0] - mu) / sigma

    ref_autocorr = stattools.acovf(ref_traj_tica_0_normalized, nlag=100000, adjusted=True, demean=False)
    ref_traj_decorrelation_time = np.where(ref_autocorr < 1/np.e)[0][0]

    traj_autocorr = stattools.acovf(traj_tica_0_normalized, nlag=100000, adjusted=True, demean=False)
    if traj_autocorr[0] > 0.5:
        try:
            traj_decorrelation_time = np.where(traj_autocorr <= 0.5)[0][0]
        except IndexError:
            traj_decorrelation_time = np.nan
    else:
        traj_decorrelation_time = np.nan

    return {
        "ref_autocorr": ref_autocorr,
        "ref_traj_decorrelation_time": ref_traj_decorrelation_time,
        "traj_autocorr": traj_autocorr,
        "traj_decorrelation_time": traj_decorrelation_time
    }


def compute_MSM_stats(
    traj_tica: np.ndarray, ref_traj_tica: np.ndarray,
    precomputed_MSM_data: Optional[Dict[str, Any]] = None, JSD_only: bool = False
) -> Dict[str, np.ndarray]:
    """Compute MSM statistics for a trajectory and reference trajectory."""
    
    # Compute MSM after KMeans clustering.
    if precomputed_MSM_data is None:
        precomputed_MSM_data = get_MSM_after_KMeans(ref_traj_tica)

    msm = precomputed_MSM_data["msm"]
    pcca = precomputed_MSM_data["pcca"]
    cmsm = precomputed_MSM_data["cmsm"]
    kmeans = precomputed_MSM_data["kmeans"]

    # Assign metastable states.
    ref_discrete = discretize(ref_traj_tica, kmeans, msm)
    traj_discrete = discretize(traj_tica, kmeans, msm)

    # Compute metastable state probabilities.
    ref_metastable_probs = (ref_discrete == np.arange(10)[:, None]).mean(1)
    traj_metastable_probs = (traj_discrete == np.arange(10)[:, None]).mean(1)
    JSD_metastable_probs = distance.jensenshannon(ref_metastable_probs, traj_metastable_probs)

    if JSD_only:
        return {
            "JSD_metastable_probs": JSD_metastable_probs,
        }

    # Compute transition matrices.
    msm_transition_matrix = np.eye(10)
    for a, i in enumerate(cmsm.active_set):
        for b, j in enumerate(cmsm.active_set):
            msm_transition_matrix[i, j] = cmsm.transition_matrix[a, b]

    msm_pi = np.zeros(10)
    msm_pi[cmsm.active_set] = cmsm.pi

    # Compute trajectory MSM.
    traj_msm = pyemma.msm.estimate_markov_model(traj_discrete, lag=10)
    traj_transition_matrix = np.eye(10)
    for a, i in enumerate(traj_msm.active_set):
        for b, j in enumerate(traj_msm.active_set):
            traj_transition_matrix[i, j] = traj_msm.transition_matrix[a, b]

    traj_pi = np.zeros(10)
    traj_pi[traj_msm.active_set] = traj_msm.pi

    # Store MSM results.
    return {
        "ref_metastable_probs": ref_metastable_probs,
        "traj_metastable_probs": traj_metastable_probs,
        "JSD_metastable_probs": JSD_metastable_probs,
        "msm_transition_matrix": msm_transition_matrix,
        "msm_pi": msm_pi,
        "traj_transition_matrix": traj_transition_matrix,
        "traj_pi": traj_pi,
        "pcca_pi": pcca._pi_coarse,
    }


def compute_JSD_MSM_stats_against_time_for_trajectory(
    traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray, precomputed_MSM_data: Dict[str, Any]
) -> Dict[int, Dict[str, float]]:
    """Compute Jenson-Shannon distances for a trajectory and reference trajectory."""
    steps = np.logspace(0, np.log10(len(traj_featurized)), num=50, dtype=int)
    return {
        step: compute_MSM_stats(
            traj_featurized[:step],
            ref_traj_featurized,
            precomputed_MSM_data,
            JSD_only=True,
        )["JSD_metastable_probs"]
        for step in steps
    }


def compute_JSD_MSM_stats_against_time(
    traj_tica: np.ndarray,
    ref_traj_tica: np.ndarray,
) -> Dict[str, float]:
    """Compute Jenson-Shannon distances for a trajectory and reference trajectory."""
    precomputed_MSM_data = get_MSM_after_KMeans(ref_traj_tica)
    return {
        "traj": compute_JSD_MSM_stats_against_time_for_trajectory(traj_tica, ref_traj_tica, precomputed_MSM_data),
        "ref_traj": compute_JSD_MSM_stats_against_time_for_trajectory(ref_traj_tica, ref_traj_tica, precomputed_MSM_data),
    }


def analyze_trajectories(traj_md: md.Trajectory, ref_traj_md: md.Trajectory) -> Dict[str, Any]:
    """Run analysis on the trajectories and return results dictionary."""

    # Featurize trajectories.
    results = {}
    results["featurization"] = featurize_trajectories(traj_md, ref_traj_md)

    py_logger.info(f"Featurization complete.")
    traj_results = results["featurization"]["traj"]
    traj_feats = traj_results["feats"]["torsions"]
    traj_featurized_dict = traj_results["traj_featurized"]
    traj_featurized = traj_featurized_dict["torsions"]
    traj_featurized_cossin = traj_featurized_dict["torsions_cossin"]

    ref_traj_results = results["featurization"]["ref_traj"]
    ref_traj_featurized_dict = ref_traj_results["traj_featurized"]
    ref_traj_featurized = ref_traj_featurized_dict["torsions"]
    ref_traj_featurized_cossin = ref_traj_featurized_dict["torsions_cossin"]
    py_logger.info(f"Featurization complete.")

    # Compute feature histograms.
    results["feature_histograms"] = compute_feature_histograms(
        traj_featurized_dict,
        ref_traj_featurized_dict,
    )
    py_logger.info(f"Feature histograms computed.")

    # Compute PMFs.
    results["PMFs"] = compute_dihedral_PMFs(
        traj_featurized,
        ref_traj_featurized,
        traj_feats,
    )
    py_logger.info(f"PMFs computed.")

    # Compute JSDs.
    results["JSD_torsion_stats"] = compute_JSD_torsion_stats(
        traj_featurized,
        ref_traj_featurized,
        traj_feats,
    )
    py_logger.info(f"JSD torsion stats computed.")

    # Compute JSDs of torsions against time.
    results["JSD_torsion_stats_against_time"] = compute_JSD_torsion_stats_against_time(
        traj_featurized,
        ref_traj_featurized,
        traj_feats,
    )
    py_logger.info(f"JSD torsion stats as a function of time computed.")

    # TICA analysis.
    results["TICA"] = compute_TICA(
        traj_featurized_cossin,
        ref_traj_featurized_cossin,
    )
    py_logger.info(f"TICA computed.")

    traj_tica = results["TICA"]["traj_tica"]
    ref_traj_tica = results["TICA"]["ref_traj_tica"]

    # Compute TICA stats.
    results["TICA_stats"] = compute_TICA_stats(
        traj_tica,
        ref_traj_tica,
    )
    py_logger.info(f"TICA stats computed.")

    # Compute autocorrelation stats.
    results["autocorrelation_stats"] = compute_autocorrelation_stats(
        traj_tica,
        ref_traj_tica,
    )
    py_logger.info(f"Autocorrelation stats computed.")

    # Compute MSM stats.
    # Sometimes, this fails because the reference trajectory is too short.
    try:
        results["MSM_stats"] = compute_MSM_stats(
            traj_tica,
            ref_traj_tica,
        )
        py_logger.info(f"MSM stats computed.")

        # Compute JSDs against time.
        results["JSD_MSM_stats_against_time"] = compute_JSD_MSM_stats_against_time(
            traj_tica,
            ref_traj_tica,
        )
        py_logger.info(f"JSD MSM stats as a function of time computed.")
   
    except IndexError:
        py_logger.warning(f"MSM stats could not be computed.")

    return results


def save_results(results: Dict[str, Any], args: argparse.Namespace, output_path_suffix: Optional[str] = None) -> None:
    """Save analysis results to pickle file."""

    # Delete intermediate results, to reduce memory usage.
    if not args.no_delete_intermediates:
        del results["featurization"]["traj"]["traj_featurized"]
        del results["featurization"]["ref_traj"]["traj_featurized"]
        del results["TICA"]["traj_tica"]
        del results["TICA"]["ref_traj_tica"]

    output_dir = os.path.join(args.output_dir, args.experiment, args.trajectory, f"ref={args.reference}")
    os.makedirs(output_dir, exist_ok=True)

    if output_path_suffix:
        output_path = os.path.join(output_dir, f"{args.peptide}_{output_path_suffix}.pkl")
    else:
        output_path = os.path.join(output_dir, f"{args.peptide}.pkl")

    with open(output_path, "wb") as f:
        pickle.dump({"results": results, "args": vars(args)}, f)

    py_logger.info(f"Results saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze molecular dynamics trajectories for peptide sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--peptide", type=str, required=True, help="Peptide sequence to analyze (e.g., FAFG)")
    parser.add_argument(
        "--trajectory",
        type=str,
        choices=["JAMUN", "JAMUNReference_2AA", "JAMUNReference_5AA", "MDGenReference", "TimewarpReference"],
        help="Type of trajectory to analyze",
    )
    parser.add_argument(
        "--reference",
        type=str,
        choices=["JAMUNReference_2AA", "JAMUNReference_5AA", "MDGenReference", "TimewarpReference"],
        help="Type of reference trajectory to compare against",
    )
    parser.add_argument(
        "--run-path",
        type=str,
        help="Path to JAMUN run directory containing trajectory files",
    )
    parser.add_argument(
        "--wandb-run",
        type=str,
        help="Weights & Biases run paths for JAMUN sampling run. Can be used instead of --run-path",
    )
    parser.add_argument(
        "--data-path", type=str, help="Path to JAMUN data directory. Defaults to JAMUN_DATA_PATH environment variable."
    )
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name for saving results")
    parser.add_argument("--output-dir", type=str, default="analysis_results", help="Directory to save analysis results")
    parser.add_argument(
        "--no-delete-intermediates",
        action="store_true",
        default=False,
        help="Don't delete intermediate results to reduce memory usage",
    )
    args = parser.parse_args()

    # Load trajectories.
    traj, ref_traj = load_trajectories(
        args.trajectory,
        args.reference,
        args.peptide,
        args.data_path,
        args.run_path,
        args.wandb_run,
    )
    py_logger.info(f"Successfully loaded trajectories for {args.peptide}:")
    py_logger.info(f"{args.trajectory} trajectory loaded: {traj}")
    py_logger.info(f"{args.reference} reference trajectory loaded: {ref_traj}")

    # Run analysis.
    results = analyze_trajectories(traj, ref_traj)

    # Save results.
    save_results(results, args, output_path_suffix=None)

    # Compute sampling rates.
    traj_seconds_per_sample = load_trajectory.get_sampling_rate(args.trajectory, args.peptide, args.experiment)
    ref_traj_seconds_per_sample = load_trajectory.get_sampling_rate(args.reference, args.peptide, args.experiment)

    if traj_seconds_per_sample is not None and ref_traj_seconds_per_sample is not None:
        py_logger.info(f"Running analysis on subsetted reference trajectory.")

        # Run analysis again, this time with the subsetted reference trajectory.
        ref_traj_subset = subset_reference_trajectory(traj, ref_traj, traj_seconds_per_sample, ref_traj_seconds_per_sample, base_factor=1.0)
        results = analyze_trajectories(ref_traj_subset, ref_traj)
        save_results(results, args, output_path_suffix="benchmark")

        # Run analysis again, this time with the subsetted reference trajectory, but 10x longer.
        # ref_traj_subset_10x = subset_reference_trajectory(traj, ref_traj, traj_seconds_per_sample, ref_traj_seconds_per_sample, base_factor=10.0)
        # results = analyze_trajectories(ref_traj_subset_10x, ref_traj)
        # save_results(results, args, output_path_suffix="benchmark_10x")
