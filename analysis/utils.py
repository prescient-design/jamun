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


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pyemma.util.exceptions.PyEMMA_DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import sys

sys.path.append("./")

import pyemma_helper as pyemma_helper


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
    heavy_atom_distance_pairs = feats.pairs(feats.select_Heavy())
    feats.add_distances(heavy_atom_distance_pairs, periodic=False)
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


def featurize(traj_md: md.Trajectory, ref_traj_md: md.Trajectory) -> Dict[str, Dict[str, np.ndarray]]:
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


def compute_PMFs(
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
        "ref_traj_upto_T": {
            "pmf_all": compute_PMF(ref_traj[:T], feats, internal_angles=False),
            "pmf_internal": compute_PMF(ref_traj[:T], feats, internal_angles=True),
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
    results["backbone_torsions"] = np.mean(
        [results[feat] for feat in feats.describe() if feat.startswith("PHI") or feat.startswith("PSI")]
    )
    results["sidechain_torsions"] = np.mean([results[feat] for feat in feats.describe() if feat.startswith("CHI")])
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


def compute_JSDs_stats_against_time_for_trajectory(
    traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer
) -> Dict[int, Dict[str, float]]:
    """Computes the Jenson-Shannon distance between the Ramachandran distributions of a trajectory and a reference trajectory at different time points."""
    steps = np.logspace(0, np.log10(len(traj_featurized)), num=10, dtype=int)
    return {
        step: compute_JSD_stats(
            traj_featurized[:step],
            ref_traj_featurized,
            feats,
        )
        for step in steps
    }


def compute_JSDs_stats_against_time(
    traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Computes the Jenson-Shannon distance between the Ramachandran distributions of a trajectory and a reference trajectory at different time points."""
    return {
        "traj": compute_JSDs_stats_against_time_for_trajectory(traj_featurized, ref_traj_featurized, feats),
        "ref_traj": compute_JSDs_stats_against_time_for_trajectory(ref_traj_featurized, ref_traj_featurized, feats),
    }


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


def compute_autocorrelation_stats(traj_tica: np.ndarray, ref_traj_tica: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute autocorrelation functions for TICA projections of trajectories."""
    nlag = 1000
    ref_autocorr = stattools.acovf(ref_traj_tica[:, 0], nlag=nlag, adjusted=True, demean=False)
    traj_autocorr = stattools.acovf(traj_tica[:, 0], nlag=nlag, adjusted=True, demean=False)
    return {"ref_autocorr": ref_autocorr, "traj_autocorr": traj_autocorr}


def compute_MSM_stats(traj_tica: np.ndarray, ref_traj_tica: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute MSM statistics for a trajectory and reference trajectory."""
    # MSM analysis.
    kmeans, ref_kmeans = get_KMeans(ref_traj_tica, K=100)
    msm, pcca, cmsm = get_MSM(ref_kmeans, lag=1000, num_states=10)

    ref_discrete = discretize(ref_traj_tica, kmeans, msm)
    traj_discrete = discretize(traj_tica, kmeans, msm)

    # Compute metastable probabilities.
    ref_metastable_probs = (ref_discrete == np.arange(10)[:, None]).mean(1)
    traj_metastable_probs = (traj_discrete == np.arange(10)[:, None]).mean(1)
    JSD_metastable_probs = distance.jensenshannon(ref_metastable_probs, traj_metastable_probs)

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
