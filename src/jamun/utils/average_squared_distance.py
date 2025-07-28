import collections
from collections.abc import Sequence

import numpy as np
import torch
import torch_geometric

from jamun import utils


def compute_distance_matrix(x: np.ndarray, cutoff: float | None = None) -> np.ndarray:
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
        raise ValueError(
            f"No distances below cutoff {cutoff} found in the distance matrix: min {dist_x[dist_x > 0].min()} and max {dist_x[dist_x > 0].max()}."
        )

    dist_x = dist_x[..., mask]
    return dist_x


def compute_average_squared_distance(x: np.ndarray, cutoff: float | None = None):
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


def compute_temporal_average_squared_distance_from_dataset(
    dataset,
    num_samples: int = 100,
    verbose: bool = False
) -> float:
    """
    Compute average squared distance between neighboring vertices in temporal graphs.
    
    Args:
        dataset: Dataset containing spatial graphs with hidden states
        num_samples: Number of samples to use for estimation
        verbose: Whether to print verbose output
        
    Returns:
        float: Average squared distance between temporal neighbors
    """
    from jamun.model.arch.spatiotemporal import spatial_to_temporal_graphs
    
    avg_sq_dists = []
    num_graphs = 0
    
    # Follow pattern from existing functions in this module
    for item in dataset:
        if num_graphs >= num_samples:
            break
        for graph in item:
            if num_graphs >= num_samples:
                break
            # Convert to temporal graphs
            temporal_batch = spatial_to_temporal_graphs(graph)
            temporal_graphs = torch_geometric.data.Batch.to_data_list(temporal_batch)
            graph_mean = 0.0
            num_nodes = graph.pos.shape[0]
            for temporal_graph in temporal_graphs:
                avg_sq_dist = compute_average_squared_distance(temporal_graph.pos, cutoff=None)
                graph_mean += avg_sq_dist / num_nodes
            avg_sq_dists.append(graph_mean)
            num_graphs += 1
        mean_avg_sq_dist = sum(avg_sq_dists) / num_graphs

    
    if verbose:
        print(f"Total graphs processed: {num_graphs}")
        print(f"Total temporal graphs processed: {len(avg_sq_dists)}")
        print(f"Mean average squared distance between temporal nodes: {mean_avg_sq_dist:.6f}")
        print(f"Standard deviation: {np.std(avg_sq_dists):.6f}")
    
    return float(mean_avg_sq_dist)
