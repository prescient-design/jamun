from typing import List, Iterator, Any, Dict, Sequence, Union

from torch.utils.data import IterableDataset, Dataset, ConcatDataset
import torch_geometric.loader
import numpy as np
import lightning.pytorch as pl

from jamun import utils


class StreamingRandomChainDataset(IterableDataset):
    """
    A streaming dataset that randomly chains multiple IterableDatasets together.
    Never materializes the full datasets into memory.
    """

    def __init__(self, datasets: List[IterableDataset], seed: int = None):
        """
        Args:
            datasets: List of IterableDatasets to chain
            weights: Optional sampling weights for each dataset.
                    If None, samples uniformly.
            seed: Random seed for reproducibility
        """
        self.datasets = datasets
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


class RandomChainDataset(IterableDataset):
    """
    A dataset that randomly chains multiple Datasets together.
    Never materializes the full datasets into memory.
    """

    def __init__(self, datasets: List[Dataset], seed: int = None):
        """
        Args:
            datasets: List of Datasets to chain
            weights: Optional sampling weights for each dataset.
                    If None, samples uniformly.
            seed: Random seed for reproducibility
        """
        self.datasets = datasets
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def __iter__(self) -> Iterator[Any]:
        """
        Returns an iterator that yields items randomly from all datasets
        according to their weights.
        """
        while True:
            # Randomly select which dataset to sample from.
            dataset_idx = np.random.randint(len(self.datasets))
            dataset = self.datasets[dataset_idx]

            sample_idx = np.random.randint(len(dataset))
            yield dataset[sample_idx]


class MDtrajDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for MDtraj datasets."""

    def __init__(
        self,
        datasets: Dict[str, Sequence[Union[IterableDataset, Dataset]]],
        batch_size: int,
        num_workers: int,
        persistent_workers: bool = True,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        self.datasets = datasets
        self.concatenated_datasets = {}
        self.shuffle = True

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        for split, datasets in self.datasets.items():
            if datasets is None:
                continue

            if isinstance(datasets[0], Dataset):
                self.concatenated_datasets[split] = ConcatDataset(datasets)
                self.shuffle = True

            elif isinstance(datasets[0], IterableDataset):
                self.concatenated_datasets[split] = StreamingRandomChainDataset(datasets)
                self.shuffle = False

            utils.dist_log(
                f"Split {split}: Loaded {len(datasets)} datasets: {[dataset.label() for dataset in datasets]}."
            )

    def train_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self.concatenated_datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self.concatenated_datasets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self.concatenated_datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
        )