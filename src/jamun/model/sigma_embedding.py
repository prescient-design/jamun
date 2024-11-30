import torch


def linear_embedding_transform(sigma: torch.Tensor, sigma_max: float = 1.0, embedding_scale: float = 1000.0):
    return (sigma / sigma_max) * embedding_scale
