import numpy as np
from typing import Tuple, List


def get_subsampled_indices(
    N: int,
    subsample_rate: int,
    total_lag_time: int,
    lag_subsample_rate: int,
) -> List[np.ndarray]:
    """
    Generate subsampled indices and their corresponding lagged indices.
    
    Args:
        N: Total number of frames
        subsample_rate: Rate at which to subsample the frames
        total_lag_time: Number of lagged frames to generate for each subsampled frame
        lag_subsample_rate: Rate at which to subsample the lagged frames
        
    Returns:
        List of arrays, where each array contains the lagged indices for a subsampled frame
                         
    Raises:
        ValueError: If the input parameters don't satisfy the required constraints
    """
    # Check guardrails
    if N / subsample_rate < 1:
        raise ValueError(f"Number of samples (N/subsample_rate = {N/subsample_rate}) must be >= 1")
    
    # if total_lag_time * lag_subsample_rate > subsample_rate:
    #     raise ValueError(
    #         f"total_lag_time * lag_subsample_rate ({total_lag_time * lag_subsample_rate}) "
    #         f"must be <= subsample_rate ({subsample_rate})"
    #     )
    
    # Generate subsampled indices
    subsampled_indices = np.arange(0, N, subsample_rate)
    
    # Generate lagged indices for each subsampled index
    lagged_indices = []
    for idx in subsampled_indices:
        # Calculate lagged indices
        lagged = [int(idx - j * lag_subsample_rate) for j in range(total_lag_time)]
        
        # Check if we have enough lagged indices
        if len(lagged) == total_lag_time and all(x >= 0 for x in lagged):
            lagged_indices.append(lagged) 
    
    return lagged_indices


def get_subsampled_trajectory(
    positions: np.ndarray,
    subsample_rate: int,
    total_lag_time: int,
    lag_subsample_rate: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Subsample a trajectory and generate lagged states for each subsampled frame.
    
    Args:
        positions: Array of shape (N, ...) containing trajectory positions
        subsample_rate: Rate at which to subsample the frames
        total_lag_time: Number of lagged frames to generate for each subsampled frame
        lag_subsample_rate: Rate at which to subsample the lagged frames
        
    Returns:
        Tuple containing:
        - subsampled_positions: Array of subsampled positions
        - lagged_positions: List of arrays, where each array contains the lagged positions
                          for the corresponding subsampled frame
                          
    Raises:
        ValueError: If the input parameters don't satisfy the required constraints
    """
    N = len(positions)
    
    # Get the lagged indices
    lagged_indices = get_subsampled_indices(N, subsample_rate, total_lag_time, lag_subsample_rate)
    
    # Extract subsampled positions (first element of each lagged indices list)
    subsampled_positions = np.array([positions[indices[0]] for indices in lagged_indices])
    
    # Generate lagged positions for each subsampled frame
    lagged_positions = [ [positions[idx] for idx in indices[1:]] for indices in lagged_indices]
    
    return subsampled_positions, lagged_positions 