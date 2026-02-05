import e3nn

e3nn.set_optimization_defaults(jit_script_fx=False)
import logging
import os
import sys

import dotenv
import torch
from denoiser_test import Denoiser
from hydra import compose, initialize

import jamun.data
from jamun.utils import find_checkpoint

# Setup logging
logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("check_hidden_state")

# Add project root to path
project_root = "/homefs/home/sules/jamun"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load configuration
with initialize(config_path="", job_name="check_hidden_state"):
    cfg = compose(
        config_name="config",
        overrides=[
            "model.arch._target_=scratch.e3conv_test.E3Conv",
            "model._target_=scratch.denoiser_test.Denoiser",
            "+model.arch.N_structures=2",  # We need at least 2 structures to test hidden state
            "model.use_torch_compile=false",  # Disable torch.compile to avoid ScriptModule issues
            "+model.conditioner._target_=scratch.conditioners.SelfConditioner",
        ],
    )

# Load checkpoint
checkpoint_path = find_checkpoint(wandb_train_run_path="sule-shashank/jamun/y4rm5488", checkpoint_type="last")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

# Modify hyperparameters to disable torch.compile
if "hyper_parameters" in checkpoint:
    checkpoint["hyper_parameters"]["use_torch_compile"] = False
    checkpoint["hyper_parameters"]["torch_compile_kwargs"] = None

# Load model with modified hyperparameters
breakpoint()
model = Denoiser.load_from_checkpoint(checkpoint_path)
model.eval()

# Get test data
dotenv.load_dotenv("../.env", verbose=True)
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")
JAMUN_ROOT_PATH = os.getenv("JAMUN_ROOT_PATH")

datasets = {
    "test": jamun.data.parse_datasets_from_directory(
        root=f"{JAMUN_DATA_PATH}/timewarp/2AA-1-large/train/",
        traj_pattern="^(.*)-traj-arrays.npz",
        pdb_file="AA-traj-state0.pdb",
        filter_codes=["AA"],
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
datamodule.setup("test")
_, test_data = next(enumerate(datamodule.test_dataloader()))
test_data = test_data.to(model.device)

# Ensure test_data has hidden_state
if not hasattr(test_data, "hidden_state") or not test_data.hidden_state:
    py_logger.info("Adding hidden state to test data")
    test_data.hidden_state = [torch.randn_like(test_data.pos) for _ in range(model.g.N_structures - 1)]

breakpoint()
# Add noise and denoise
sigma = torch.tensor(0.04)  # Same sigma as in config
with torch.no_grad():
    xhat, y = model.noise_and_denoise(test_data, sigma, align_noisy_input=False)

# Check if hidden state is preserved
print("\nChecking hidden state preservation:")
print(f"Original hidden state shapes: {[hs.shape for hs in test_data.hidden_state]}")
print(f"Noisy hidden state shapes: {[hs.shape for hs in y.hidden_state]}")
print(f"Denoised hidden state shapes: {[hs.shape for hs in xhat.hidden_state]}")

# Check if hidden state values are preserved
for i in range(len(test_data.hidden_state)):
    hidden_state_diff = torch.abs(xhat.hidden_state[i] - test_data.hidden_state[i]).mean()
    print(f"\nMean absolute difference between original and denoised hidden state {i}: {hidden_state_diff.item()}")

# Check if positions are actually denoised
pos_diff = torch.abs(xhat.pos - test_data.pos).mean()
print(f"Mean absolute difference between original and denoised positions: {pos_diff.item()}")

# Check if noisy positions are different from original
noisy_pos_diff = torch.abs(y.pos - test_data.pos).mean()
print(f"Mean absolute difference between original and noisy positions: {noisy_pos_diff.item()}")
