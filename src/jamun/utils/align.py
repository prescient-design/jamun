from typing import Tuple

import torch
import torch_geometric


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


def align_A_to_B_batched(A: torch_geometric.data.Batch, B: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
    """Aligns each batch of A to corresponding batch in B."""
    num_batches = A.batch.max().item() + 1
    for i in range(num_batches):
        mask = A.batch == i
        A.pos[mask] = align_A_to_B(A.pos[mask], B.pos[mask])
    return A
