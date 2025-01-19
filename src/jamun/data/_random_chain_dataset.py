from torch.utils.data import IterableDataset
from typing import List, Iterator, Any
import random
from itertools import cycle


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
            random.seed(seed)
            
    def get_stream(self, dataset: IterableDataset) -> Iterator:
        """Creates an infinite stream from a dataset."""
        return cycle(iter(dataset))
    
    def __iter__(self) -> Iterator[Any]:
        """
        Returns an iterator that yields items randomly from all datasets
        according to their weights.
        """
        # Create iterators for all datasets
        streams = [self.get_stream(dataset) for dataset in self.datasets]
        
        while True:
            # Randomly select which dataset to sample from
            dataset_idx = random.choices(
                range(len(self.datasets)),
                weights=self.weights,
                k=1
            )[0]
            
            # Get next item from selected dataset
            try:
                yield next(streams[dataset_idx])
            except StopIteration:
                # Refresh the exhausted stream
                streams[dataset_idx] = self.get_stream(self.datasets[dataset_idx])
                yield next(streams[dataset_idx])