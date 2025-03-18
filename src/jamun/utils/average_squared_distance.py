from typing import Optional, Sequence
import collections

import numpy as np
import torch

from jamun import utils


def compute_distance_matrix(x: np.ndarray, cutoff: Optional[float] = None) -> np.ndarray:
    """Computes the distance matrix between points in x, ignoring self-distances."""
    if x.shape[-1] != 3:
        raise ValueError("Last dimension of x must be 3.")

    dist_x = np.linalg.norm(x[..., :, None, :] - x[..., None, :, :], axis=-1)

    # Select non-diagonal elements
    num_points = x.shape[-2]
    mask = np.tri(num_points, num_points, k=-1, dtype=bool)
    assert dist_x[..., mask].shape == (*x.shape[:-2], num_points * (num_points - 1) / 2)

    # If cutoff is specified, only select distances below the cutoff
    if cutoff is not None:
        mask = mask & (dist_x < cutoff)

    if not np.any(mask):
        raise ValueError(f"No distances below cutoff {cutoff} found in the distance matrix: min {dist_x[dist_x > 0].min()} and max {dist_x[dist_x > 0].max()}.")

    dist_x = dist_x[..., mask]
    return dist_x


def compute_average_squared_distance(x: np.ndarray, cutoff: Optional[float] = None):
    """Computes the average squared distance between points in x, ignoring self-distances."""
    dist_x = compute_distance_matrix(x, cutoff)
    return np.mean(dist_x**2)


def compute_average_squared_distance_from_datasets(
    datasets: Sequence[torch.utils.data.Dataset],
    cutoff: float,
    num_estimation_datasets: int = 50,
    num_estimation_graphs_per_dataset: int = 100,
    verbose: bool = False,
) -> float:
    """Computes the average squared distance for normalization."""
    avg_sq_dists = collections.defaultdict(list)
    
    for dataset in datasets[:num_estimation_datasets]:
        num_graphs = 0
        
        for graph in dataset:
            pos = np.asarray(graph.pos)
            avg_sq_dist = compute_average_squared_distance(pos, cutoff=cutoff)
            avg_sq_dists[graph.dataset_label].append(avg_sq_dist)
            num_graphs += 1

        if num_graphs >= num_estimation_graphs_per_dataset:
            break

    mean_avg_sq_dist = sum(np.sum(avg_sq_dists[label]) for label in avg_sq_dists) / num_graphs
    utils.dist_log(f"Mean average squared distance = {mean_avg_sq_dist:0.3f} nm^2")

    if verbose:
        utils.dist_log(f"For cutoff {cutoff} nm:")
        for label in sorted(avg_sq_dists):
            utils.dist_log(
                f"- Dataset {label}: Average squared distance = {np.mean(avg_sq_dists[label]):0.3f} +- {np.std(avg_sq_dists[label]):0.3f} nm^2"
            )

    return float(mean_avg_sq_dist)
