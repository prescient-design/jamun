# %%
import functools
import logging
import os

import dotenv
import tqdm
from typing import Union
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
    batch_size=3,
    num_workers=2,
)
datamodule.setup('test')
_, data_batch = next(enumerate(datamodule.test_dataloader()))

# %%
import hydra 
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

with initialize(config_path=f"../outputs/train/dev/runs/2025-06-04_20-07-32/.hydra", job_name="my_script_job", version_base=None):
        # 2. Compose the configuration
        # `config_name` refers to the base YAML file (e.g., "config.yaml" -> "config")
        # You can also pass overrides programmatically, e.g.,
        # cfg = compose(config_name="config", overrides=["model.irreps_out='2x0e'"])
        cfg: DictConfig = compose(config_name="config")

        print("Full configuration loaded by Hydra:")
        print(OmegaConf.to_yaml(cfg))
        print("-" * 50)
        print(torch.cuda.is_available())
        # 3. Instantiate the model
        try:
            print("Attempting to instantiate model...")
            # hydra.utils.instantiate is still the way to go
            from jamun.utils import compute_average_squared_distance_from_datasets
            average_squared_distance = compute_average_squared_distance_from_datasets(datasets['test'], cfg.model.max_radius)
            cfg.model.average_squared_distance = average_squared_distance
            model: jamun.model.Denoiser = hydra.utils.instantiate(cfg.model)
        #     print("\nSuccessfully instantiated model:")
        #     print(model)
        #     print("-" * 50)

        #     # Example: Create some dummy input data
        #     batch_s = 2
        #     dummy_input = torch.randn(batch_s, model.irreps_in.dim)
        #     print(f"Created dummy input with shape: {dummy_input.shape}")

        #     # Perform a forward pass
        #     output = model(dummy_input)
        #     print(f"Model output shape: {output.shape}")
        #     assert output.shape == (batch_s, model.irreps_out.dim)
        #     print("Dummy forward pass successful!")

        except Exception as e:
            print(f"Error during model instantiation or processing: {e}")
            import traceback
            traceback.print_exc()


# %% play w/ spherical harmonics 
from e3nn import o3 
sh = o3.SphericalHarmonics(irreps_out="1x0e + 1x1e", normalize=True, normalization="component")
irreps_sh = e3nn.o3.Irreps("1x0e + 1x1e")
# %% 
full_pos = torch.cat([molecule_pos, molecule_pos_extra], dim=-1)
blocks = torch.split(full_pos, 3, dim=-1)
src, dst = molecule.edge_index
edge_sh = []
for block in blocks:
    edge_vec = block[src] - block[dst]
    edge_sh.append(sh(edge_vec))
edge_sh = torch.cat(edge_sh, dim=-1)


# %% test the new e3conv_test class 
from e3conv_test import E3Conv
import torch_geometric
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
        N_structures=2
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
y = add_bond_mask(data_batch)

# %% 
from jamun.utils.align import kabsch_algorithm 
from jamun.utils import mean_center 
from e3tools import scatter 

# y = mean_center(y)
def conditioner(y: torch_geometric.data.Batch) -> torch.Tensor:
    conditioned_structures = []
    for positions in y.hidden_state: 
        aligned_positions = kabsch_algorithm(positions, y.pos, y.batch, y.num_graphs)
        conditioned_structures.append(aligned_positions)
    return conditioned_structures

print('hello world')

# %%
