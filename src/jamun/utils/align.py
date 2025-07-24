import e3tools
import torch
import torch_geometric


def C1(S: torch.Tensor) -> torch.Tensor:
    """
    Coefficient for the first order correction.

    Args:
        S: Tensor of shape [batch_size, 3] containing diagonal elements.

    Returns:
        C1: Tensor of shape [batch_size, 3, 3] with diagonal matrices.
    """
    # Extract individual diagonal elements
    s1, s2, s3 = S[:, 0], S[:, 1], S[:, 2]  # Each is [batch_size]

    # Compute coefficients for each batch
    c1 = 1 / (s1 + s2) + 1 / (s1 + s3)  # [batch_size]
    c2 = 1 / (s2 + s1) + 1 / (s2 + s3)  # [batch_size]
    c3 = 1 / (s3 + s1) + 1 / (s3 + s2)  # [batch_size]

    # Stack to create diagonal elements matrix
    diag_elements = torch.stack([c1, c2, c3], dim=-1)  # [batch_size, 3]

    # Create batch of diagonal matrices
    C1_batch = torch.diag_embed(diag_elements)  # [batch_size, 3, 3]

    return -C1_batch / 2


def C2(S: torch.Tensor) -> torch.Tensor:
    """
    Coefficient for the second order correction.

    Args:
        S: Tensor of shape [batch_size, 3] containing diagonal elements.

    Returns:
        C2: Tensor of shape [batch_size, 3, 3] with diagonal matrices.
    """
    # Extract individual diagonal elements
    s1, s2, s3 = S[:, 0], S[:, 1], S[:, 2]  # Each is [batch_size]

    # Compute coefficients for each batch
    c1 = 1 / (s1 + s2) ** 2 + 1 / (s1 + s3) ** 2  # [batch_size]
    c2 = 1 / (s2 + s1) ** 2 + 1 / (s2 + s3) ** 2  # [batch_size]
    c3 = 1 / (s3 + s1) ** 2 + 1 / (s3 + s2) ** 2  # [batch_size]

    # Stack to create diagonal elements matrix
    diag_elements = torch.stack([c1, c2, c3], dim=-1)  # [batch_size, 3]

    # Create batch of diagonal matrices
    C2_batch = torch.diag_embed(diag_elements)  # [batch_size, 3, 3]

    return -C2_batch / 8


def alignment_correction_upto_order(S: torch.Tensor, sigma: float, correction_order: int) -> torch.Tensor:
    """
    Compute correction for alignment up to a given order.

    Args:
        S: Tensor of shape [batch_size, 3] containing diagonal elements.
        sigma: Float scalar multiplier.
        order: Integer specifying the order of correction (0, 1, or 2).

    Returns:
        correction: Tensor of shape [batch_size, 3, 3] with correction matrices.
    """
    batch_size = S.shape[0]
    assert S.shape == (batch_size, 3)

    identity = torch.eye(3, device=S.device, dtype=S.dtype).unsqueeze(0).expand(batch_size, -1, -1)

    if correction_order == 0:
        return identity
    if correction_order == 1:
        return identity + (sigma**2) * C1(S)
    if correction_order == 2:
        return identity + (sigma**2) * C1(S) + (sigma**4) * C2(S)

    raise ValueError(f"Correction order {correction_order} not supported.")


def kabsch_algorithm(
    A: torch.Tensor,
    B: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    sigma: float | None = None,
    correction_order: int = 0,
) -> torch.Tensor:
    """Compute the optimal rigid transformation between two sets of points.

    Given tensors `A` and `B` find the rigid transformation `T = (t, R)` which minimizes the RMSD between B and T(A).
    Returns the aligned points A.
    See https://en.wikipedia.org/wiki/Kabsch_algorithm.

    Parameters
    ----------
    A : Tensor
        Shape (N, D)
    B : Tensor
        Shape (N, D)
    batch : Tensor | None
        Shape (N,)

    Returns
    -------
    Tensor
        Aligned points y.
    """
    # Mean centering.
    A_mu = e3tools.scatter(A, batch, dim=-2, dim_size=num_graphs, reduce="mean")
    B_mu = e3tools.scatter(B, batch, dim=-2, dim_size=num_graphs, reduce="mean")
    A_c = A - A_mu[batch]
    B_c = B - B_mu[batch]

    # Compute batch covariance matrix.
    batch_one_hot = torch.nn.functional.one_hot(batch, num_classes=num_graphs).float()
    H = torch.einsum("Ni,Nj,NG->Gij", A_c, B_c, batch_one_hot)

    # SVD to get rotation.
    U, S_orig, VH = torch.linalg.svd(H)
    S = alignment_correction_upto_order(S_orig, sigma=sigma, correction_order=correction_order)
    R_check = torch.einsum("Gki,Gkk,Gjk->Gij", VH, S, U)  # V U^T

    # Remove reflections.
    dets = torch.linalg.det(R_check)
    signs = torch.eye(3, device=dets.device).repeat(num_graphs, 1, 1)  # repeat the identity matrix
    signs[:, 2, 2] = dets
    R = torch.einsum("Gki,Gkk,Gkk,Gjk->Gij", VH, signs, S, U)  # V S U^T

    # Align y to x.
    RA_mu = torch.einsum("Gij,Gj->Gi", R, A_mu)
    t = B_mu - RA_mu

    A_aligned = torch.einsum("Nij,Nj->Ni", R[batch], A) + t[batch]
    return A_aligned


def find_rigid_alignment(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Taken from https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8

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


def align_A_to_B_batched(
    A: torch_geometric.data.Batch, B: torch_geometric.data.Batch, sigma: float, correction_order: int
) -> torch_geometric.data.Batch:
    """Aligns each graph of A to corresponding graph in B."""
    A.pos = kabsch_algorithm(A.pos, B.pos, A.batch, A.num_graphs, sigma=sigma, correction_order=correction_order)
    return A


def align_A_to_B_batched_f(
    A: torch.Tensor,
    B: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    sigma: float | None = None,
    correction_order: int = 0,
) -> torch.Tensor:
    """Aligns each graph of A to corresponding graph in B."""
    return kabsch_algorithm(A, B, batch, num_graphs, sigma=sigma, correction_order=correction_order)
