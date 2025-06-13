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

# %% check paths
import sys
project_root = "/homefs/home/sules/jamun" # Or use os.path.abspath("..") if your notebook is in a subdir of jamun

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    py_logger.info(f"Added '{project_root}' to sys.path for module discovery.")
else:
    py_logger.info(f"'{project_root}' is already in sys.path.")

# %% get configs 
from hydra import compose, initialize
import hydra
from omegaconf import OmegaConf
# Load the configuration file
with initialize(config_path="", job_name="jamun_test"):
    cfg = compose(
        config_name="config",
        overrides=[
            "model.arch._target_=scratch.e3conv_test.E3Conv",  # Override to use E3Conv from e3conv_test.py
            "model._target_=scratch.denoiser_test.Denoiser",  # Override to use Denoiser from denoiser_test.py
            "+model.arch.N_structures=2",  # Ensure N_structures is set, defaulting to 2
        ]
    )
# Log the configuration
py_logger.info("Loaded configuration:")
py_logger.info(OmegaConf.to_yaml(cfg))


# %% Re-instantiate the model with the updated configuration
try:
    py_logger.info("Attempting to re-instantiate model with updated arch...")
    # Ensure average_squared_distance is still set correctly
    if not hasattr(cfg.model, "average_squared_distance") or cfg.model.average_squared_distance is None:
        from jamun.utils import compute_average_squared_distance_from_datasets # Ensure import
        average_squared_distance = compute_average_squared_distance_from_datasets(datasets['test'], cfg.model.max_radius)
        cfg.model.average_squared_distance = average_squared_distance
        py_logger.info(f"Set cfg.model.average_squared_distance to {cfg.model.average_squared_distance}")
    # Provide conditioner if needed
    if not hasattr(cfg.model, "conditioner"):
        OmegaConf.set_struct(cfg.model, False)  # Allow modification
        cfg.model.conditioner = OmegaConf.create({})
        cfg.model.conditioner._target_ = 'scratch.conditioners.PositionConditioner'  # Use the PositionConditioner from scratch.conditioners
        OmegaConf.set_struct(cfg.model, True)  # Lock structure again
        py_logger.info("Set cfg.model.conditioner to instantiate 'scratch.conditioners.PositionConditioner'")
    model = hydra.utils.instantiate(cfg.model)
    py_logger.info("Successfully re-instantiated model with E3Conv from e3conv_test.py:")
    print(model)
    # You can inspect model.arch to confirm it's an instance of E3Conv
    py_logger.info(f"Instantiated model architecture type: {type(model.g)}")

except Exception as e:
    py_logger.error(f"Error during model re-instantiation: {e}")
    import traceback
    traceback.print_exc()

# %% Tests for Denoiser.noise_and_denoise

# Ensure 'model' (your Denoiser instance) and 'data_batch' are available from previous cells.
# If 'model' is not the correct Denoiser instance, re-instantiate it as needed.
# For example, if you were using the custom_denoiser_model:
# denoiser_to_test = custom_denoiser_model 
# Or if you are using the one from the cfg re-instantiation:
denoiser_to_test = model 

# Make sure data_batch is on the same device as the model
if hasattr(denoiser_to_test, 'device'):
    data_batch = data_batch.to(denoiser_to_test.device)
elif next(denoiser_to_test.parameters()).is_cuda:
    data_batch = data_batch.to(next(denoiser_to_test.parameters()).device)


py_logger.info(f"Testing Denoiser instance of type: {type(denoiser_to_test)}")
py_logger.info(f"Data batch has {data_batch.num_graphs} graphs and {data_batch.num_nodes} nodes.")

# %% Tests for Denoiser object (denoiser_to_test)

import torch_geometric.data # For isinstance checks

py_logger.info("Starting tests for Denoiser object...")
_, data_batch = next(enumerate(datamodule.test_dataloader()))
# Ensure data_batch has hidden_state for the tests, matching model's N_structures
if not hasattr(data_batch, 'hidden_state') or \
   not isinstance(data_batch.hidden_state, list) or \
   len(data_batch.hidden_state) != denoiser_to_test.g._orig_mod.N_structures - 1: # Use _orig_mod to access N_structures if g is compiled
    
    n_structures = denoiser_to_test.g._orig_mod.N_structures
    py_logger.info(f"data_batch.hidden_state is missing or incorrect. Re-creating with {n_structures} structures.")
    data_batch.hidden_state = [torch.randn_like(data_batch.pos) for _ in range(n_structures)]
    data_batch.hidden_state = [hs.to(data_batch.pos.device) for hs in data_batch.hidden_state]
else:
    py_logger.info(f"data_batch.hidden_state already exists with {len(data_batch.hidden_state)} structures.")


# %% Test 1: Denoiser.noise_and_denoise (align_noisy_input=False)

try:
    py_logger.info("Test 1: Denoiser.noise_and_denoise (align_noisy_input=False)")
    original_x = data_batch.clone()
    # sigma_test1 = torch.tensor(0.5, device=denoiser_to_test.device)
    sigma = denoiser_to_test.sigma_distribution.sample()*1e-5
    xhat1, y_processed1 = denoiser_to_test.noise_and_denoise(original_x.clone(), sigma, \
                                                             align_noisy_input=True)

    # assert isinstance(xhat1, torch_geometric.data.Batch), "xhat1 is not a PyG Batch object"
    # assert isinstance(y_processed1, torch_geometric.data.Batch), "y_processed1 is not a PyG Batch object"
    
    assert xhat1.pos.shape == original_x.pos.shape, "xhat1.pos shape mismatch"
    assert y_processed1.pos.shape == original_x.pos.shape, "y_processed1.pos shape mismatch"
    
    assert not torch.allclose(y_processed1.pos, original_x.pos), "y_processed1.pos should be different from original x.pos"
    
    assert xhat1.num_graphs == original_x.num_graphs, "xhat1 num_graphs mismatch"
    assert y_processed1.num_graphs == original_x.num_graphs, "y_processed1 num_graphs mismatch"
    assert xhat1.num_nodes == original_x.num_nodes, "xhat1 num_nodes mismatch"
    assert y_processed1.num_nodes == original_x.num_nodes, "y_processed1 num_nodes mismatch"
    assert torch.allclose(xhat1.batch, original_x.batch), "xhat1.batch mismatch"
    assert torch.allclose(y_processed1.batch, original_x.batch), "y_processed1.batch mismatch"

    # Check hidden_state in y_processed1 (noisy input)
    if hasattr(original_x, 'hidden_state') and original_x.hidden_state:
        assert hasattr(y_processed1, 'hidden_state') and len(y_processed1.hidden_state) == len(original_x.hidden_state), "y_processed1.hidden_state length mismatch"
        for i in range(len(original_x.hidden_state)):
            assert not torch.allclose(y_processed1.hidden_state[i], original_x.hidden_state[i]), f"y_processed1.hidden_state[{i}] should be different"
    
    # xhat inherits attributes from the input to xhat_normalized, which is the noisy graph (y_processed1)
    # So, xhat1 should also have hidden_state if y_processed1 does.
    if hasattr(y_processed1, 'hidden_state') and y_processed1.hidden_state:
         assert hasattr(xhat1, 'hidden_state') and len(xhat1.hidden_state) == len(y_processed1.hidden_state), "xhat1.hidden_state length mismatch with y_processed1"

    py_logger.info("Test 1 PASSED.")
except Exception as e:
    py_logger.error(f"Test 1 FAILED: {e}")
    import traceback
    traceback.print_exc()

# %% Test 3: Denoiser.training_step
try:
    py_logger.info("Test 3: Denoiser.training_step")
    # Get a fresh batch for training_step to avoid issues with modified data_batch from other tests
    _, train_batch = next(enumerate(datamodule.test_dataloader())) # Using test_dataloader for convenience
    
    # Ensure train_batch has hidden_state
    if not hasattr(train_batch, 'hidden_state') or \
       not isinstance(train_batch.hidden_state, list) or \
       len(train_batch.hidden_state) != denoiser_to_test.g._orig_mod.N_structures-1:
        n_structures = denoiser_to_test.g._orig_mod.N_structures
        train_batch.hidden_state = [torch.randn_like(train_batch.pos) for _ in range(n_structures)]
    else: 
        py_logger.info(f"train_batch.hidden_state already exists with {len(train_batch.hidden_state)} structures.")
    train_batch.hidden_state = [hs.to(denoiser_to_test.device) for hs in train_batch.hidden_state]
    train_batch = train_batch.to(denoiser_to_test.device)


    # Manually set align_noisy_input_during_training if not set (it's a param of Denoiser)
    if not hasattr(denoiser_to_test, 'align_noisy_input_during_training'):
        py_logger.warning("Denoiser missing 'align_noisy_input_during_training', defaulting to False for this test.")
        denoiser_to_test.align_noisy_input_during_training = False # Or True, as needed

    logs_dict = denoiser_to_test.training_step(train_batch, 0)
    
    assert isinstance(logs_dict, dict), "Logs is not a dictionary"
    expected_keys = ["mse", "rmsd", "scaled_rmsd", "loss"]
    for key in expected_keys:
        assert key in logs_dict, f"Key '{key}' missing in logs"
        assert isinstance(logs_dict[key], torch.Tensor), f"Log value for '{key}' is not a tensor"
        if key == 'loss':
             assert logs_dict[key].ndim == 0, f"logs_dict['loss'] is not scalar, shape: {logs_dict[key].shape}"
        else: # mse, rmsd, scaled_rmsd are averaged in training_step's aux_mean
             assert logs_dict[key].ndim == 0, f"logs_dict['{key}'] is not scalar, shape: {logs_dict[key].shape}"


    py_logger.info(f"Test 3 PASSED. Loss: {logs_dict['mse'].item()}")
except Exception as e:
    py_logger.error(f"Test 3 FAILED: {e}")
    import traceback
    traceback.print_exc()
# %%
