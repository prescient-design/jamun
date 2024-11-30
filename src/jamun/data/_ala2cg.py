import pathlib
import random
from typing import Callable, Optional

import einops
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch_geometric

from ._utils import download_file


def mask_atom_type(data: torch_geometric.data.Data) -> torch_geometric.data.Data:
    data.update({"x": torch.zeros_like(data.x)})
    return data


class ALA2CGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        download: bool = False,
        mean_center: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.root = pathlib.Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.root / "ala2_cg_2fs_Hmass_2_HBonds.npz"
        self.transform = transform
        self.download = download
        self.mean_center = mean_center

        if self.download and not self.dataset_path.exists():
            download_file(
                "https://ftp.imp.fu-berlin.de/pub/cmb-data/ala2_cg_2fs_Hmass_2_HBonds.npz",
                self.dataset_path,
                verbose=True,
            )

        coords = torch.from_numpy(np.load(self.dataset_path)["coords"])
        zs = einops.repeat(torch.tensor([6, 7, 6, 6, 7], dtype=torch.int), "l->b l", b=coords.shape[0])
        self.dset = torch.utils.data.TensorDataset(zs, coords)

    def __getitem__(self, idx):
        Z, pos = self.dset[idx]
        y = None
        data = torch_geometric.data.Data(x=Z, y=y, pos=pos)

        if self.mean_center:
            data.pos -= data.pos.mean(0)

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.dset)


class ALA2CGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        download=False,
        seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.download = download
        self.num_workers = num_workers
        self.dset_kwargs = kwargs
        self.seed = seed

    def prepare_data(self):
        if self.download:
            ALA2_CG_Dataset(self.root, download=True, **self.dset_kwargs)

    def setup(self, stage: str):
        dset = ALA2_CG_Dataset(root=self.root, **self.dset_kwargs)

        generator = torch.Generator().manual_seed(42)
        self.dset_train, self.dset_val, self.dset_test = torch.utils.data.random_split(
            dset, lengths=[0.8, 0.1, 0.1], generator=generator
        )

    def train_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self.dset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self.dset_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch_geometric.loader.DataLoader(
            self.dset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
