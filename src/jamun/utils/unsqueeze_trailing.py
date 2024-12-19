import torch


def unsqueeze_trailing(x: torch.Tensor, n: int) -> torch.Tensor:
    """Adds n trailing singleton dimensions to x."""
    return x.reshape(*x.shape, *((1,) * n))
