from torch.utils.data import IterableDataset
from typing import List, Iterator, Any
import numpy as np


class StreamingRandomChainDataset(IterableDataset):
    """
    A streaming dataset that randomly chains multiple IterableDatasets together.
    Never materializes the full datasets into memory.
    """
    def __init__(
        self,
        datasets: List[IterableDataset],
        weights: List[float] = None,
        seed: int = None
    ):
        """
        Args:
            datasets: List of IterableDatasets to chain
            weights: Optional sampling weights for each dataset. 
                    If None, samples uniformly.
            seed: Random seed for reproducibility
        """
        self.datasets = datasets
        if weights is None:
            weights = [1.0] * len(datasets)
            
        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]
        
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def __iter__(self) -> Iterator[Any]:
        """
        Returns an iterator that yields items randomly from all datasets
        according to their weights.
        """
        # Create iterators for all datasets.
        streams = [iter(dataset) for dataset in self.datasets]

        while True:
            # Randomly select which dataset to sample from.
            dataset_idx = np.random.randint(len(streams))

            # Get next item from selected dataset.
            try:
                yield next(streams[dataset_idx])
            
            except StopIteration: 
                # Refresh stream.
                streams[dataset_idx] = iter(self.datasets[dataset_idx])
