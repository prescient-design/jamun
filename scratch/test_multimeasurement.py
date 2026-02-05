import e3nn

e3nn.set_optimization_defaults(jit_script_fx=False)
import os
import sys

import dotenv
import hydra
import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf

from jamun.utils import compute_average_squared_distance_from_datasets

# Fix PyTorch Geometric backend issues
try:
    import torch_cluster
    import torch_scatter
    import torch_sparse
except ImportError:
    print("Warning: Some PyTorch Geometric extensions not available")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dotenv.load_dotenv("../.env", verbose=True)  # Adjust path if script is not in scratch/
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")
JAMUN_ROOT_PATH = os.getenv("JAMUN_ROOT_PATH")

project_root = "/homefs/home/sules/jamun"  # Adjust if necessary
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added '{project_root}' to sys.path for module discovery.")
else:
    print(f"'{project_root}' is already in sys.path.")


def compute_average_squared_distance_from_config(cfg: OmegaConf) -> float:
    """Computes the average squared distance for normalization from the data."""
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("compute_normalization")
    train_datasets = datamodule.datasets["train"]
    cutoff = cfg.model.max_radius
    average_squared_distance = compute_average_squared_distance_from_datasets(train_datasets, cutoff)
    return average_squared_distance


def test_training_mode(cfg, mode_name, max_graphs_per_batch):
    """Test training with specified optimization mode."""
    print(f"\n{'=' * 50}")
    print(f"Testing {mode_name} mode (max_graphs_per_batch={max_graphs_per_batch})")
    print(f"{'=' * 50}")

    # # Configure DenoiserMultimeasurement
    # cfg.model._target_ = "jamun.model.denoiser_multimeasurement.DenoiserMultimeasurement"
    # cfg.model.sigma_distribution._target_ = "jamun.distributions.ConstantSigma"
    # cfg.model.sigma_distribution.sigma = 0.04

    # Set multimeasurement parameters
    # cfg.model.multimeasurement = True
    cfg.model.N_measurements_hidden = 2
    cfg.model.N_measurements = 2
    cfg.model.max_graphs_per_batch = max_graphs_per_batch

    # Compute normalization
    average_squared_distance = compute_average_squared_distance_from_config(cfg)
    cfg.model.average_squared_distance = average_squared_distance
    breakpoint()
    print("Loading datamodule...")
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("test")
    breakpoint()
    print("Loading model...")
    model = hydra.utils.instantiate(cfg.model)
    print(f"Model loaded: {type(model)}")
    print(f"Multimeasurement: {model.multimeasurement}")
    print(f"N_measurements_hidden: {model.N_measurements_hidden}")
    print(f"N_measurements: {model.N_measurements}")
    print(f"Automatic optimization: {model.automatic_optimization}")
    print(f"Sigma: {model.sigma_distribution.sigma}")
    breakpoint()
    # Get a single batch
    print("Getting a batch of data...")
    train_loader = datamodule.train_dataloader()
    _, batch = next(enumerate(train_loader))
    breakpoint()
    print(f"Batch shape: {batch.pos.shape}")
    print(f"Batch num_graphs: {batch.num_graphs}")
    if hasattr(batch, "hidden_state") and batch.hidden_state is not None:
        print(f"Hidden state shapes: {[h.shape for h in batch.hidden_state]}")
    else:
        print("No hidden states in batch")
    breakpoint()
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        sigma = model.sigma_distribution.sample()
        x_target, xhat, y = model.noise_and_denoise(batch, sigma, align_noisy_input=True)
    breakpoint()
    print(f"Input shape: {batch.pos.shape}")
    print(f"Noisy shape: {y.pos.shape}")
    print(f"Output shape: {xhat.pos.shape}")
    print(f"Target shape: {x_target.pos.shape}")

    # Verify multimeasurement expansion
    expected_graphs = batch.num_graphs * model.N_measurements_hidden * model.N_measurements
    actual_graphs = y.num_graphs
    print(f"Expected graphs after multimeasurement: {expected_graphs}")
    print(f"Actual graphs: {actual_graphs}")
    assert actual_graphs == expected_graphs, f"Graph count mismatch: expected {expected_graphs}, got {actual_graphs}"

    # Test actual training with fast_dev_run
    print("Testing training with fast_dev_run...")

    # Configure trainer to use only 1 GPU
    if torch.cuda.is_available():
        print(f"CUDA available with {torch.cuda.device_count()} GPUs - using GPU 0 only")
        trainer = pl.Trainer(
            fast_dev_run=1,  # Run 1 train, 1 val batch and stop
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            accelerator="gpu",
            devices=[0],  # Explicitly use only GPU 0
            strategy="auto",  # Single device strategy
        )
    else:
        print("CUDA not available - using CPU")
        trainer = pl.Trainer(
            fast_dev_run=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
        )

    try:
        trainer.fit(model, datamodule)
        print(f"{mode_name} mode training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        raise

    print(f"{mode_name} mode test completed successfully!")
    return model


@hydra.main(version_base=None, config_path="../src/jamun/hydra_config", config_name="train")
def main(cfg):
    # # Override data config to use only ALA_ALA
    # cfg.data.datamodule.filter_codes = ['ALA_ALA']
    # cfg.data.datamodule.subsample = 10  # Use fewer samples for faster testing
    # cfg.data.datamodule.batch_size = 4   # Small batch size for testing

    print("Testing DenoiserMultimeasurement training modes")
    print(f"Using ALA_ALA data from: {JAMUN_DATA_PATH}")

    # Test automatic optimization mode
    test_training_mode(cfg.copy(), "AUTOMATIC", None)

    # Test manual optimization mode
    test_training_mode(cfg.copy(), "MANUAL", 2)  # Process 2 graphs at a time

    print(f"\n{'=' * 50}")
    print("ALL TESTS PASSED!")
    print("Both automatic and manual optimization modes work correctly.")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
