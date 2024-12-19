import einops
import torch
import torch.nn.functional as F


def scaled_rmsd(x: torch.Tensor, xhat: torch.Tensor, sigma: float) -> torch.Tensor:
    """Computes the scaled RMSD between x and xhat, both assumed mean-centered."""
    assert x.shape == xhat.shape
    raw_loss = F.mse_loss(xhat, x, reduction="none")
    raw_loss = einops.rearrange(raw_loss, "... D -> (...) D")
    D = torch.as_tensor(x.shape[-1])
    raw_loss = F.mse_loss(xhat, x, reduction="none")
    raw_loss = raw_loss.sum(dim=-1)
    scaled_rmsd = torch.sqrt(raw_loss) / (sigma * torch.sqrt(D))
    scaled_rmsd = scaled_rmsd.mean()
    return scaled_rmsd
