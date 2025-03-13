from typing import Optional, Sequence, List, Dict, Any
import collections

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities import rank_zero_only
import hydra

from jamun import utils


class ComputeNormalizationModule(pl.LightningModule):
    """Lightning module for computing normalization statistics in a distributed environment.
    
    This module leverages PyTorch Lightning's distributed capabilities to compute
    average squared distances across multiple devices/nodes.
    """
    
    def __init__(self, cutoff, num_estimation_graphs=5000, verbose=False):
        """Initialize the normalization module.
        
        Args:
            cutoff (float): The radius cutoff for distance calculations
            num_estimation_graphs (int): Maximum number of graphs to process across all devices
            verbose (bool): Whether to print detailed statistics
        """
        super().__init__()
        self.cutoff = cutoff
        self.num_estimation_graphs = num_estimation_graphs
        self.verbose = verbose
        
        # Initialize statistics storage
        self.register_buffer("all_sq_dists", torch.tensor([], device=self.device))
        self.register_buffer("all_counts", torch.tensor([], device=self.device))
        self.dataset_stats = {}
        
    def train_dataloader(self):
        # Return the dataloader directly from the datamodule
        return self.trainer.datamodule.train_dataloader()
    
    def on_train_start(self):
        # Initialize the counters
        self.all_sq_dists = torch.tensor([], device=self.device)
        self.all_counts = torch.tensor([], device=self.device)
        self.dataset_stats = {}
    
    def on_train_batch_start(self, batch, batch_idx):
        # Process each graph in the batch
        for graph in batch.to_data_list():
            pos = np.asarray(graph.pos)
            avg_sq_dist = compute_average_squared_distance(pos, cutoff=self.cutoff)
            
            # Track per-dataset statistics if verbose
            if self.verbose:
                dataset_label = graph.dataset_label
                if dataset_label not in self.dataset_stats:
                    self.dataset_stats[dataset_label] = []
                self.dataset_stats[dataset_label].append(avg_sq_dist)
            
            # Append to overall statistics
            new_dist = torch.tensor([avg_sq_dist], device=self.device)
            self.all_sq_dists = torch.cat([self.all_sq_dists, new_dist])
            self.all_counts = torch.cat([self.all_counts, torch.ones(1, device=self.device)])
        
        # Stop after processing enough graphs (considering all devices)
        if hasattr(self, 'trainer') and self.trainer is not None:
            # Get the total count across all processes
            total_count_tensor = torch.tensor([self.all_counts.sum()], device=self.device)
            gathered_counts = self.all_gather(total_count_tensor)
            total_count = gathered_counts.sum().item()
            
            if total_count >= self.num_estimation_graphs:
                self.trainer.should_stop = True
    
    def training_step(self, batch, batch_idx):
        # Dummy loss for training
        return {"loss": torch.tensor(0.0, requires_grad=True, device=self.device)}
    
    def configure_optimizers(self):
        # Dummy optimizer for Lightning
        return torch.optim.SGD([torch.nn.Parameter(torch.tensor([0.0], device=self.device))], lr=0.1)
    
    def compute_final_statistics(self):
        """Compute final statistics across all processes."""
        # Gather results from all processes
        local_sum = self.all_sq_dists.sum()
        local_count = self.all_counts.sum()
        
        # Use Lightning's built-in collective operations
        all_sums = self.all_gather(local_sum.view(1))
        all_counts = self.all_gather(local_count.view(1))
        
        # Calculate global mean
        global_sum = all_sums.sum().item()
        global_count = all_counts.sum().item()
        average_squared_distance = global_sum / global_count
        
        # Print verbose statistics if requested
        if self.verbose and rank_zero_only.rank == 0:
            # For each dataset, gather and print statistics
            for label in sorted(self.dataset_stats.keys()):
                values = self.dataset_stats[label]
                mean_value = np.mean(values)
                std_value = np.std(values)
                print(f"Dataset {label}: Average squared distance = {mean_value:0.3f} +- {std_value:0.3f} nm^2")
        
        # Log the result
        if rank_zero_only.rank == 0:
            print(f"Mean average squared distance = {average_squared_distance:0.3f} nm^2")
        
        return average_squared_distance


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
