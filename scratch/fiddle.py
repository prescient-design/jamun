import functools
import logging
import os

import dotenv
import tqdm

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("jamun")

import torch

torch.set_float32_matmul_precision("high")

import e3nn
import e3tools.nn

e3nn.set_optimization_defaults(jit_script_fx=False)

import jamun
import jamun.data
import jamun.distributions
import jamun.model
import jamun.model.arch

dotenv.load_dotenv("../.env", verbose=True)
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")

# Device.
device = torch.device("cuda:0")

datasets = {
    "train": jamun.data.parse_datasets_from_directory(
        root=f"{JAMUN_DATA_PATH}/timewarp/2AA-1-large/train/",
        traj_pattern="^(.*)-traj-arrays.npz",
        pdb_file="AA-traj-state0.pdb",
        filter_codes=['AA'],
        as_iterable=False,
        subsample=100,
        max_datasets=1,
    )
}

datamodule = jamun.data.MDtrajDataModule(
    datasets=datasets,
    batch_size=32,
    num_workers=2,
)
datamodule.setup(None)

print("hello world")

