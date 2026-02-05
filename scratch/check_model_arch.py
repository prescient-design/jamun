# %%
import functools
import logging
import os

import dotenv

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("jamun")

import torch

torch.cuda.is_available()
torch.set_float32_matmul_precision("high")

import e3nn
import e3tools.nn

e3nn.set_optimization_defaults(jit_script_fx=False)

import jamun
import jamun.data
import jamun.distributions
import jamun.model
import jamun.model.arch

# %%
# dataset
dotenv.load_dotenv("../.env", verbose=True)
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")
JAMUN_ROOT_PATH = os.getenv("JAMUN_ROOT_PATH")

# %%
datasets = {
    "test": jamun.data.parse_datasets_from_directory(
        root=f"{JAMUN_DATA_PATH}/capped_diamines/timewarp_splits/train/",
        traj_pattern="^(.*).xtc",
        pdb_file="ALA_ALA.pdb",
        filter_codes=["ALA_ALA"],
        as_iterable=False,
        subsample=100,
        total_lag_time=10,
        lag_subsample_rate=100,
        max_datasets=1,
    )
}

# %%
datamodule = jamun.data.MDtrajDataModule(
    datasets=datasets,
    batch_size=5,
    num_workers=2,
)
datamodule.setup("test")
_, data_batch = next(enumerate(datamodule.test_dataloader()))
print(f"Number of hidden states: {len(data_batch.hidden_state)}")
print(f"Size of one hidden state: {data_batch.hidden_state[0].shape}")
# %% test the new e3conv_test class
import torch_geometric
from e3conv_test import E3Conv
from e3tools import radius_graph

trial_model = E3Conv(
    irreps_out="1x1e",
    irreps_hidden="120x0e + 32x1e",
    irreps_sh="1x0e + 1x1e",
    n_layers=1,
    edge_attr_dim=8,
    atom_type_embedding_dim=8,
    atom_code_embedding_dim=8,
    residue_code_embedding_dim=32,
    residue_index_embedding_dim=8,
    use_residue_information=False,
    use_residue_sequence_index=False,
    hidden_layer_factory=functools.partial(
        e3tools.nn.ConvBlock,
        conv=e3tools.nn.Conv,
    ),
    output_head_factory=functools.partial(e3tools.nn.EquivariantMLP, irreps_hidden_list=["120x0e + 32x1e"]),
    N_structures=2,
)


# %% postprocess data for plugging into model
def add_bond_mask(y: torch_geometric.data.Batch, cutoff: float = 1.0) -> torch_geometric.data.Batch:
    radial_edge_index = radius_graph(y.pos, cutoff, batch=y["batch"])
    bonded_edge_index = y.edge_index
    edge_index = torch.cat((radial_edge_index, bonded_edge_index), dim=-1)
    bond_mask = torch.cat(
        (
            torch.zeros(radial_edge_index.shape[1], dtype=torch.long, device=y.edge_index.device),
            torch.ones(bonded_edge_index.shape[1], dtype=torch.long, device=y.edge_index.device),
        ),
        dim=0,
    )
    y.edge_index = edge_index
    y.bond_mask = bond_mask
    return y


# add bond mask--do this only once!
bond_mask_exists = hasattr(data_batch, "bond_mask") and data_batch.bond_mask is not None
if not bond_mask_exists:
    py_logger.info("Adding bond mask to data_batch...")
    # Ensure data_batch is a torch_geometric.data.Batch
    if not isinstance(data_batch, torch_geometric.data.Batch):
        raise TypeError(f"Expected data_batch to be a torch_geometric.data.Batch, got {type(data_batch)}")

    # Add bond mask
    data_batch = add_bond_mask(data_batch)
else:
    py_logger.info("Bond mask already exists in data_batch, skipping addition.")
    # If bond mask already exists, we can still use it
    # but we should ensure it's in the correct format
    if not isinstance(data_batch.bond_mask, torch.Tensor):
        raise TypeError(f"Expected data_batch.bond_mask to be a torch.Tensor, got {type(data_batch.bond_mask)}")
    if data_batch.bond_mask.dtype != torch.long:
        raise ValueError(f"Expected data_batch.bond_mask to be of dtype torch.long, got {data_batch.bond_mask.dtype}")

    # Ensure edge_index is set correctly
    if not hasattr(data_batch, "edge_index") or data_batch.edge_index is None:
        raise ValueError("data_batch.edge_index is not set. Please ensure it is initialized before adding bond mask.")

    # If everything is fine, we can proceed with the existing bond mask
py_logger.info(f"data_batch has {data_batch.num_graphs} graphs and {data_batch.num_nodes} nodes.")
# Ensure data_batch is on the same device as the model
if hasattr(trial_model, "device"):
    data_batch = data_batch.to(trial_model.device)
elif next(trial_model.parameters()).is_cuda:
    data_batch = data_batch.to(next(trial_model.parameters()).device)

# %% Test the E3Conv model with the data_batch
py_logger.info("Testing E3Conv model with data_batch...")
# Ensure data_batch is on the same device as the model
y = data_batch
if hasattr(trial_model, "device"):
    y = y.to(trial_model.device)
elif next(trial_model.parameters()).is_cuda:
    y = y.to(next(trial_model.parameters()).device)
# Run a forward pass through the model
try:
    py_logger.info("Running forward pass through E3Conv model...")
    output = trial_model(torch.cat([y.pos, *y.hidden_state], dim=-1), y, torch.Tensor([1e-5]), 100.0)
    py_logger.info(f"Output shape: {output.shape}")
except Exception as e:
    py_logger.error(f"Error during forward pass: {e}")
    import traceback

    traceback.print_exc()
