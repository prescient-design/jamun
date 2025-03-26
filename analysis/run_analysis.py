from typing import Dict, Any, Tuple, Optional
import os
import logging
import argparse
import sys
import pickle
import warnings
import collections

import scipy.stats
import mdtraj as md
import pyemma.coordinates.data
import numpy as np
import mdtraj as md
import pyemma
import pyemma.coordinates.clustering
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


def compute_feature_histograms(traj_featurized_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute histograms of features for a trajectory."""
    return {
        key: pyemma_helper.compute_1D_histogram(traj_featurized)
        for key, traj_featurized in traj_featurized_dict.items()
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
    traj: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer
) -> Dict[str, np.ndarray]:
    """Compute the potential of mean force (PMF) for a trajectory along a dihedral angle."""
    return {
        "pmf_all": compute_PMF(traj, feats, internal_angles=False),
        "pmf_internal": compute_PMF(traj, feats, internal_angles=True),
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


def compute_JSD_torsions(
    traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer
) -> Dict[str, float]:
    """Compute Jenson-Shannon distances for a trajectory and reference trajectory. Taken from MDGen."""
    results = {}
    for i, feat in enumerate(feats.describe()):
        ref_p = np.histogram(ref_traj_featurized[:, i], range=(-np.pi, np.pi), bins=100)[0]
        traj_p = np.histogram(traj_featurized[:, i], range=(-np.pi, np.pi), bins=100)[0]
        results["JSD_" + feat] = distance.jensenshannon(ref_p, traj_p)

    # Compute JSDs for backbone, sidechain, and all torsions.
    results["JSD_backbone_torsions"] = np.mean(
        [
            results["JSD_" + feat] for feat in feats.describe()
            if feat.startswith("PHI") or feat.startswith("PSI")
        ]
    )
    results["JSD_sidechain_torsions"] = np.mean(
        [
            results["JSD_" + feat] for feat in feats.describe()
            if feat.startswith("CHI")
        ]
    )
    results["JSD_all_torsions"] = np.mean(
        [
            results["JSD_" + feat]
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
        results["JSD_" + "|".join(phi_psi_feats)] = distance.jensenshannon(ref_p.flatten(), traj_p.flatten())

    return results


def compute_JSD_torsions_against_time(
    traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Computes the Jenson-Shannon distance between the Ramachandran distributions of a trajectory and a reference trajectory at different time points."""
    steps = np.logspace(0, np.log10(len(traj_featurized)), num=10, dtype=int)
    return {
        step: compute_JSD_torsions(
            traj_featurized[:step],
            ref_traj_featurized,
            feats,
        )
        for step in steps
    }


def compute_torsion_decorrelations(traj_featurized: np.ndarray, ref_traj_featurized: np.ndarray, feats: pyemma.coordinates.data.MDFeaturizer):
    """Computes decorrelations of torsion angles."""

    def autocorrelation_and_decorrelation_time(traj: np.ndarray, baseline: float) -> np.ndarray:
        """Compute autocorrelation and decorrelation time for a trajectory."""
        nlag = min(len(traj) - 1, 100000)
        autocorr = stattools.acovf(np.sin(traj), nlag=nlag, demean=False, adjusted=True)
        autocorr += stattools.acovf(np.cos(traj), nlag=nlag, demean=False, adjusted=True)
        autocorr = (autocorr - baseline) / (1 - baseline)

        if np.any(autocorr < 1 / np.e):
            decorrelation_time = np.where(autocorr < 1 / np.e)[0][0]
        else:
            decorrelation_time = np.nan

        return autocorr, decorrelation_time

    ref_traj = ref_traj_featurized
    traj = traj_featurized
    
    torsion_decorrelations = collections.defaultdict(dict)
    for i, feat in enumerate(feats.describe()):
        baseline = np.sin(ref_traj[:,i]).mean()**2 + np.cos(ref_traj[:,i]).mean()**2
        
        ref_traj_autocorrelations, ref_traj_decorrelation_time = autocorrelation_and_decorrelation_time(
            ref_traj[:, i], baseline
        )
        torsion_decorrelations[feat]["ref_traj_autocorrelations"] = ref_traj_autocorrelations
        torsion_decorrelations[feat]["ref_traj_decorrelation_time"] = ref_traj_decorrelation_time

        traj_autocorrelations, traj_decorrelation_time = autocorrelation_and_decorrelation_time(
            traj[:, i], baseline
        )
        torsion_decorrelations[feat]["traj_autocorrelations"] = traj_autocorrelations
        torsion_decorrelations[feat]["traj_decorrelation_time"] = traj_decorrelation_time
    
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


def compute_TICA_histogram_for_plotting(traj_tica: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute histograms of TICA projections for plotting."""
    return pyemma_helper.compute_2D_histogram(traj_tica[:, 0], traj_tica[:, 1])


def compute_JSD_TICA(traj_tica: np.ndarray, ref_traj_tica: np.ndarray) -> Dict[str, float]:
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
    traj_p = np.histogram2d(
        *traj_tica[:, :2].T, range=((tica_0_min, tica_0_max), (tica_1_min, tica_1_max)), bins=50
    )[0]    
    tica_01_jsd = distance.jensenshannon(ref_p.flatten(), traj_p.flatten())

    return {
        "JSD_TICA-0": tica_0_jsd,
        "JSD_TICA-0,1": tica_01_jsd,
    }


def compute_JSD_TICA_against_time(
    traj_tica: np.ndarray,
    ref_traj_tica: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Compute Jenson-Shannon distances for a trajectory and reference trajectory."""
    steps = np.logspace(0, np.log10(len(traj_tica)), num=10, dtype=int)
    return {
        step: compute_JSD_TICA(
            traj_tica[:step],
            ref_traj_tica,
        )
        for step in steps
    }


def compute_TICA_decorrelations(traj_tica: np.ndarray, ref_traj_tica: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute autocorrelation and decorrelation times for TICA projections of trajectories."""
    mu = ref_traj_tica[:,0].mean()
    sigma = ref_traj_tica[:,0].std()

    ref_traj_tica_0_normalized = (ref_traj_tica[:,0] - mu) / sigma
    traj_tica_0_normalized = (traj_tica[:,0] - mu) / sigma

    nlag = min(len(ref_traj_tica_0_normalized) - 1, 100000)
    ref_autocorr = stattools.acovf(ref_traj_tica_0_normalized, nlag=nlag, adjusted=True, demean=False)
    if np.any(ref_autocorr < 1/np.e):
        ref_traj_decorrelation_time = np.where(ref_autocorr < 1/np.e)[0][0]
    else:
        ref_traj_decorrelation_time = np.nan

    nlag = min(len(traj_tica_0_normalized) - 1, 100000)
    traj_autocorr = stattools.acovf(traj_tica_0_normalized, nlag=nlag, adjusted=True, demean=False)
    if traj_autocorr[0] > 0.5 and np.any(traj_autocorr <= 0.5):
        traj_decorrelation_time = np.where(traj_autocorr <= 0.5)[0][0]
    else:
        traj_decorrelation_time = np.nan

    return {
        "ref_autocorr": ref_autocorr,
        "ref_traj_decorrelation_time": ref_traj_decorrelation_time,
        "traj_autocorr": traj_autocorr,
        "traj_decorrelation_time": traj_decorrelation_time
    }


def compute_flux_matrix(transition_matrix: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Compute the flux matrix from a transition matrix and stationary distribution."""
    flux_matrix = np.multiply(transition_matrix, pi[:, None])
    np.fill_diagonal(flux_matrix, 0)

    row_sums = flux_matrix.sum(axis=1)
    column_sums = flux_matrix.sum(axis=0)
    assert np.allclose(row_sums, column_sums)

    return flux_matrix


def compute_JSD_MSM(
    traj_tica: np.ndarray,
    ref_traj_tica: np.ndarray,
    MSM_info: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    msm = MSM_info["msm"]
    kmeans = MSM_info["kmeans"]

    # Assign metastable states.
    ref_discrete = discretize(ref_traj_tica, kmeans, msm)
    traj_discrete = discretize(traj_tica, kmeans, msm)

    # Compute metastable state probabilities.
    ref_metastable_probs = (ref_discrete == np.arange(10)[:, None]).mean(1)
    traj_metastable_probs = (traj_discrete == np.arange(10)[:, None]).mean(1)
    JSD_metastable_probs = distance.jensenshannon(ref_metastable_probs, traj_metastable_probs)

    return {
        "ref_metastable_probs": ref_metastable_probs,
        "traj_metastable_probs": traj_metastable_probs,
        "JSD_metastable_probs": JSD_metastable_probs,
    }


def compute_JSD_MSM_against_time(
    traj_tica: np.ndarray,
    ref_traj_tica: np.ndarray,
    MSM_info: Dict[str, Any],
) -> Dict[int, Dict[str, float]]:
    """Compute Jenson-Shannon distances for a trajectory and reference trajectory."""
    steps = np.logspace(0, np.log10(len(traj_tica)), num=10, dtype=int)
    return {
        step: compute_JSD_MSM(
            traj_tica[:step],
            ref_traj_tica,
            MSM_info,
        )
        for step in steps
    }


def compute_MSM_transition_and_flux_matrices(
    traj_tica: np.ndarray,
    ref_traj_tica: np.ndarray,
    MSM_info: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Compute transition and flux matrices for a trajectory and reference trajectory according to a MSM."""
    
    msm = MSM_info["msm"]
    pcca = MSM_info["pcca"]
    cmsm = MSM_info["cmsm"]
    kmeans = MSM_info["kmeans"]

    # Assign metastable states.
    ref_discrete = discretize(ref_traj_tica, kmeans, msm)
    traj_discrete = discretize(traj_tica, kmeans, msm)

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

    # Compute flux matrices.
    msm_flux_matrix = compute_flux_matrix(msm_transition_matrix, msm_pi)
    traj_flux_matrix = compute_flux_matrix(traj_transition_matrix, traj_pi)

    # Compute Spearman correlation between corresponding flux matrices and transition matrices.
    flux_spearman_correlation = scipy.stats.spearmanr(
        msm_flux_matrix, traj_flux_matrix, axis=None
    ).statistic
    transition_spearman_correlation = scipy.stats.spearmanr(
        msm_transition_matrix, traj_transition_matrix, axis=None
    ).statistic

    # Store MSM results.
    return {
        "msm_transition_matrix": msm_transition_matrix,
        "msm_pi": msm_pi,
        "msm_flux_matrix": msm_flux_matrix,
        "traj_transition_matrix": traj_transition_matrix,
        "traj_pi": traj_pi,
        "traj_flux_matrix": traj_flux_matrix,
        "pcca_pi": pcca._pi_coarse,
        "flux_spearman_correlation": flux_spearman_correlation,
        "transition_spearman_correlation": transition_spearman_correlation,
    }


def analyze_trajectories(traj_md: md.Trajectory, ref_traj_md: md.Trajectory) -> Dict[str, Any]:
    """Run analysis on the trajectories and return results dictionary."""

    # Featurize trajectories.
    results = {}
    results["featurization"] = {
        "traj": featurize_trajectory(traj_md),
        "ref_traj": featurize_trajectory(ref_traj_md),
    }
    py_logger.info(f"Featurization complete.")

    traj_results = results["featurization"]["traj"]
    traj_feats = traj_results["feats"]["torsions"]
    traj_featurized_dict = traj_results["traj_featurized"]
    traj_featurized = traj_featurized_dict["torsions"]
    traj_featurized_cossin = traj_featurized_dict["torsions_cossin"]

    ref_traj_results = results["featurization"]["ref_traj"]
    ref_traj_feats = ref_traj_results["feats"]["torsions"]
    ref_traj_featurized_dict = ref_traj_results["traj_featurized"]
    ref_traj_featurized = ref_traj_featurized_dict["torsions"]
    ref_traj_featurized_cossin = ref_traj_featurized_dict["torsions_cossin"]

    assert traj_feats.describe() == ref_traj_feats.describe(), "Featurization of trajectories does not match."
    feats = traj_feats

    # Compute feature histograms.
    results["feature_histograms"] = {
        "traj": compute_feature_histograms(traj_featurized_dict),
        "ref_traj": compute_feature_histograms(ref_traj_featurized_dict),
    }
    py_logger.info(f"Feature histograms computed.")

    # We will compare the trajectory as well as the (shortened) reference trajectories.
    trajs_to_compare = {
        "traj": traj_featurized,
        "ref_traj": ref_traj_featurized,
        "ref_traj_10x": ref_traj_featurized[:len(ref_traj_featurized) // 10],
        "ref_traj_100x": ref_traj_featurized[:len(ref_traj_featurized) // 100],
        "ref_traj_1000x": ref_traj_featurized[:len(ref_traj_featurized) // 1000],
    }

    # Compute PMFs.
    results["PMFs"] = {}
    for key, traj in trajs_to_compare.items():
        results["PMFs"][key] = compute_dihedral_PMFs(traj, feats)
    py_logger.info(f"PMFs computed.")

    # Compute JSDs.
    results["JSD_torsions"] = {}
    for key, traj in trajs_to_compare.items():
        results["JSD_torsions"][key] = compute_JSD_torsions(
            traj,
            ref_traj_featurized,
            feats,
        )
    py_logger.info(f"JSD of torsion distributions computed.")

    # Compute JSDs of torsions against time.
    results["JSD_torsions_against_time"] = {}
    for key, traj in trajs_to_compare.items():
        results["JSD_torsions_against_time"][key] = compute_JSD_torsions_against_time(
            traj,
            ref_traj_featurized,
            feats,
        )
    py_logger.info(f"JSD of torsion distributions as a function of time computed.")

    # Compute torsion decorrelations.
    results["torsion_decorrelations"] = compute_torsion_decorrelations(
        traj_featurized,
        ref_traj_featurized,
        feats,
    )
    py_logger.info(f"Torsion decorrelations computed.")

    # TICA analysis.
    results["TICA"] = compute_TICA(
        traj_featurized_cossin,
        ref_traj_featurized_cossin,
    )
    py_logger.info(f"TICA computed.")

    traj_tica = results["TICA"]["traj_tica"]
    ref_traj_tica = results["TICA"]["ref_traj_tica"]

    traj_ticas_to_compare = {
        "traj": traj_tica,
        "ref_traj": ref_traj_tica,
        "ref_traj_10x": ref_traj_tica[:len(ref_traj_tica) // 10],
        "ref_traj_100x": ref_traj_tica[:len(ref_traj_tica) // 100],
        "ref_traj_1000x": ref_traj_tica[:len(ref_traj_tica) // 1000],
    }

    # Compute TICA stats.
    results["TICA_histograms"] = {}
    for key, tica in traj_ticas_to_compare.items():
        results["TICA_histograms"][key] = compute_TICA_histogram_for_plotting(
            tica,
        )
    py_logger.info(f"Histograms of TICA projections computed.")

    results["JSD_TICA"] = {}
    for key, tica in traj_ticas_to_compare.items():
        results["JSD_TICA"][key] = compute_JSD_TICA(
            tica,
            ref_traj_tica,
        )
    py_logger.info(f"JSD of TICA projections computed.")

    results["JSD_TICA_against_time"] = {}
    for key, tica in traj_ticas_to_compare.items():
        results["JSD_TICA_against_time"][key] = compute_JSD_TICA_against_time(
            tica,
            ref_traj_tica,
        )
    py_logger.info(f"JSD of TICA projections as a function of time computed.")

    # Compute autocorrelation stats.
    results["TICA_decorrelations"] = compute_TICA_decorrelations(
        traj_tica,
        ref_traj_tica,
    )
    py_logger.info(f"TICA decorrelations computed.")

    # Compute MSM.
    # Sometimes, this fails because the reference trajectory is too short.
    try:
        MSM_info = get_MSM_after_KMeans(ref_traj_tica)
        results["MSM"] = MSM_info
    except IndexError:
        py_logger.warning(f"MSM information could not be computed.")
        return results

    results["JSD_MSM"] = {}
    for key, tica in traj_ticas_to_compare.items():
        results["JSD_MSM"][key] = compute_JSD_MSM(
            tica,
            ref_traj_tica,
            MSM_info,
        )
    py_logger.info(f"JSD of MSM state probabilities computed.")
        
    results["JSD_MSM_against_time"] = {}
    for key, tica in traj_ticas_to_compare.items():
        results["JSD_MSM_against_time"][key] = compute_JSD_MSM_against_time(
            tica,
            ref_traj_tica,
            MSM_info,
        )
    py_logger.info(f"JSD of MSM state probabilities as a function of time computed.")

    results["MSM_matrices"] = {}
    for key, tica in traj_ticas_to_compare.items():
        try:
            results["MSM_matrices"][key] = compute_MSM_transition_and_flux_matrices(
                tica,
                ref_traj_tica,
                MSM_info,
            )
        except RuntimeError:
            py_logger.warning(f"MSM matrices could not be computed for {key}.")
            continue
    py_logger.info(f"MSM matrices computed.")

    return results


def save_results(results: Dict[str, Any], args: argparse.Namespace) -> None:
    """Save analysis results to pickle file."""

    # Delete intermediate results, to reduce memory usage.
    if not args.no_delete_intermediates:
        del results["featurization"]["traj"]["traj_featurized"]
        del results["featurization"]["ref_traj"]["traj_featurized"]
        del results["TICA"]["traj_tica"]
        del results["TICA"]["ref_traj_tica"]

    output_dir = os.path.join(args.output_dir, args.experiment, args.trajectory, f"ref={args.reference}")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{args.peptide}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump({"results": results, "args": vars(args)}, f)

    py_logger.info(f"Results saved to: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze molecular dynamics trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--peptide", type=str, required=True, help="Peptide sequence to analyze (e.g., FAFG)")
    parser.add_argument(
        "--trajectory",
        type=str,
        help="Type of trajectory to analyze",
        required=True,
    )
    parser.add_argument(
        "--reference",
        type=str,
        help="Type of reference trajectory to compare against",
        required=True,
    )
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name for saving results")
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
    parser.add_argument("--output-dir", type=str, default="analysis_results", help="Directory to save analysis results")
    parser.add_argument(
        "--no-delete-intermediates",
        action="store_true",
        default=False,
        help="Don't delete intermediate results to reduce memory usage",
    )
    args = parser.parse_args()

    # Load trajectories.
    traj, traj_info = load_trajectory.load_trajectory_with_info(
        args.trajectory,
        args.peptide,
        args.data_path,
        args.run_path,
        args.wandb_run,
    )
    ref_traj, ref_traj_info = load_trajectory.load_trajectory_with_info(
        args.reference,
        args.peptide,
        args.data_path,
        args.run_path,
        args.wandb_run,
    )

    py_logger.info(f"Successfully loaded trajectories for {args.peptide}:")
    py_logger.info(f"{args.trajectory} trajectory loaded: {traj} with info: {traj_info}")
    py_logger.info(f"{args.reference} reference trajectory loaded: {ref_traj} with info: {ref_traj_info}")

    # Run analysis.
    results = analyze_trajectories(traj, ref_traj)

    # Add trajectory info to results.
    results["info"] = {
        "traj": traj_info,
        "ref_traj": ref_traj_info,
    }
    py_logger.info(f"Analysis complete.")

    # Save results.
    save_results(results, args)
