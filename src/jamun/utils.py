from typing import Optional, Union, Tuple, List, Sequence
import logging
import collections

import einops
import numpy as np
import torch
import torch_scatter
import torchvision
import lightning.pytorch as pl


def dist_log(msg: str, logger=None) -> None:
    """Helper for distributed logging."""

    if logger is None:
        logger = logging.getLogger("jamun")

    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        for r in range(world_size):
            if r == rank:
                logger.info(f"[rank {rank}/{world_size}]: {msg}")
            torch.distributed.barrier()
    else:
        logger.info(f"{msg}")


def unsqueeze_trailing(x, n):
    """
    adds n trailing singleton dimensions to x
    """
    return x.reshape(*x.shape, *((1,) * n))


def find_rigid_alignment(A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Taken from https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    See https://en.wikipedia.org/wiki/Kabsch_algorithm.

    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- point cloud to align (source)
        -    B: Torch tensor of shape (N,D) -- reference point cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix.
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix.
    R = V.mm(U.T)
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V.mm(U.T)
    # Translation vector.
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def align_A_to_B(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Aligns point cloud A to point cloud B using the Kabsch algorithm.
    Args:
    - A: Torch tensor of shape (N,D) -- point cloud to align (source)
    - B: Torch tensor of shape (N,D) -- the reference point cloud (target)

    Returns:
    -    A_aligned: Torch tensor of shape (N,D) -- aligned point cloud
    """
    R, t = find_rigid_alignment(A, B)
    A_aligned = (R.mm(A.T)).T + t
    return A_aligned


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

    dist_x = dist_x[..., mask]
    return dist_x


def compute_average_squared_distance(x: np.ndarray, cutoff: Optional[float] = None):
    """Computes the average squared distance between points in x, ignoring self-distances."""
    dist_x = compute_distance_matrix(x, cutoff)
    return np.mean(dist_x**2)


def mean_by_attribute(x: torch.Tensor, attributes: Sequence[int]) -> List[Tuple[int, float]]:
    """Computes the mean of x grouped by attributes."""
    assert len(x) == len(attributes)

    # Calculate mean value for each attribute
    attributes = np.array(attributes)
    unique_attrs = np.unique(attributes)
    mean_xs = []
    for attr in unique_attrs:
        mask = (attributes == attr)
        mean_x = x[mask].mean(dim=0)
        mean_xs.append((attr.item(), mean_x.item()))
    return mean_xs


def compute_average_squared_distance_from_data(dataloader: torch.utils.data.DataLoader, cutoff: float, num_estimation_graphs: int = 5000) -> float:
    """Computes the average squared distance for normalization."""
    avg_sq_dists = collections.defaultdict(list)
    num_graphs = 0
    for batch in dataloader:
        for graph in batch.to_data_list():
            pos = np.asarray(graph.pos)
            avg_sq_dist = compute_average_squared_distance(pos, cutoff=cutoff)
            avg_sq_dists[graph.dataset_label].append(avg_sq_dist)
            num_graphs += 1
        
        if num_graphs >= num_estimation_graphs:
            break
    
    logger = logging.getLogger("jamun")
    logger.info(f"For cutoff {cutoff} nm:")
    for label in sorted(avg_sq_dists):
        logger.info(f"- Dataset {label}: Average squared distance = {np.mean(avg_sq_dists[label]):0.3f} +- {np.std(avg_sq_dists[label]):0.3f} nm^2")

    mean_avg_sq_dist = sum(np.sum(avg_sq_dists[label]) for label in avg_sq_dists) / num_graphs
    logger.info(f"Mean average squared distance = {mean_avg_sq_dist:0.3f} nm^2")
    return mean_avg_sq_dist
