"""
Normalization utilities for jamun models.
"""

import torch


def normalization_factors(
    sigma: float,
    average_squared_distance: float,
    normalization_type: str = "JAMUN",
    sigma_data: float = None,
    D: int = 3,
) -> tuple[float, float, float, float]:
    """
    Compute normalization factors for the input and output.

    Args:
        sigma: Noise level
        average_squared_distance: Average squared distance from the dataset
        normalization_type: Type of normalization ("JAMUN", "EDM", or None)
        sigma_data: Sigma data parameter (only used for EDM normalization)
        D: Dimensionality (default: 3)

    Returns:
        Tuple of (c_in, c_skip, c_out, c_noise) normalization factors
    """
    sigma = torch.as_tensor(sigma)

    if normalization_type is None:
        return 1.0, 0.0, 1.0, sigma

    if normalization_type == "EDM":
        if sigma_data is None:
            raise ValueError("sigma_data must be provided when normalization_type is 'EDM'")
        c_skip = (sigma_data**2) / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
        c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
        c_noise = torch.log(sigma / sigma_data) * 0.25
        return c_in, c_skip, c_out, c_noise

    if normalization_type == "JAMUN":
        A = torch.as_tensor(average_squared_distance)
        B = torch.as_tensor(2 * D * sigma**2)

        c_in = 1.0 / torch.sqrt(A + B)
        c_skip = A / (A + B)
        c_out = torch.sqrt((A * B) / (A + B))
        c_noise = torch.log(sigma) / 4
        return c_in, c_skip, c_out, c_noise

    raise ValueError(f"Unknown normalization type: {normalization_type}")
