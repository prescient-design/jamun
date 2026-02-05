import e3nn

e3nn.set_optimization_defaults(jit_script_fx=False)
import math
import os
import sys

import dotenv
import hydra
import torch
from omegaconf import OmegaConf

from jamun.utils import compute_average_squared_distance_from_datasets
from jamun.utils._normalizations import normalization_factors
from jamun.utils.average_squared_distance import compute_temporal_average_squared_distance_from_datasets


def compute_radial_cutoff(max_radius: float, average_squared_distance: float, sigma: float, D: int = 3) -> float:
    """
    Compute radial cutoff using the same formula as the denoiser.

    This replicates the computation from denoiser_conditional.py:
    radial_cutoff = effective_radial_cutoff(sigma) / c_in
    where:
    - effective_radial_cutoff = sqrt(max_radius² + 6σ²)
    - c_in = 1.0 / sqrt(average_squared_distance + 2Dσ²)

    Args:
        max_radius: Maximum radius parameter
        average_squared_distance: Average squared distance from dataset
        sigma: Noise level
        D: Dimensionality (default 3 for 3D coordinates)

    Returns:
        Computed radial cutoff
    """
    # Effective radial cutoff based on noise level
    effective_radial_cutoff = math.sqrt(max_radius**2 + 6 * sigma**2)

    # JAMUN normalization factor c_in
    A = average_squared_distance
    B = 2 * D * sigma**2
    c_in = 1.0 / math.sqrt(A + B)

    # Final radial cutoff
    radial_cutoff = effective_radial_cutoff / c_in

    print("Radial cutoff computation:")
    print(f"  max_radius: {max_radius}")
    print(f"  average_squared_distance: {average_squared_distance}")
    print(f"  sigma: {sigma}")
    print(f"  D: {D}")
    print(f"  effective_radial_cutoff: {effective_radial_cutoff}")
    print(f"  c_in: {c_in}")
    print(f"  final radial_cutoff: {radial_cutoff}")

    return radial_cutoff


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


def compute_temporal_average_squared_distance_from_config(cfg: OmegaConf) -> float:
    """Computes the temporal average squared distance for normalization from the data."""
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("compute_normalization")
    train_datasets = datamodule.datasets["train"]

    average_squared_distance = compute_temporal_average_squared_distance_from_datasets(
        train_datasets,
        num_samples=100,  # Use reasonable number of samples
        verbose=True,
    )
    return average_squared_distance


@hydra.main(version_base=None, config_path="../src/jamun/hydra_config", config_name="train")
def main(cfg):
    # Override configuration to use denoiser_conditional with DenoisingConditioner
    # cfg.model._target_ = "jamun.model.denoiser_conditional.Denoiser"
    # cfg.model.sigma_distribution._target_ = "jamun.distributions.ConstantSigma"
    # cfg.model.sigma_distribution.sigma = 0.04
    breakpoint()
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("test")
    breakpoint()
    # Load the test config
    average_squared_distance = compute_average_squared_distance_from_config(cfg)
    temporal_average_squared_distance = compute_temporal_average_squared_distance_from_config(cfg)
    cfg.model.average_squared_distance = average_squared_distance

    # Compute radial cutoff for spatiotemporal model using the same formula as denoiser
    sigma = cfg.model.sigma_distribution.sigma
    max_radius = cfg.model.max_radius
    spatial_radial_cutoff = compute_radial_cutoff(
        max_radius=max_radius,
        average_squared_distance=average_squared_distance,  # Use temporal for spatiotemporal model
        sigma=sigma,
        D=3,
    )
    temporal_radial_cutoff = compute_radial_cutoff(
        max_radius=max_radius,
        average_squared_distance=temporal_average_squared_distance,  # Use temporal for spatiotemporal model
        sigma=sigma,
        D=3,
    )
    cfg.model.conditioner.spatiotemporal_model.radial_cutoff = spatial_radial_cutoff
    cfg.model.conditioner.spatiotemporal_model.temporal_cutoff = temporal_radial_cutoff
    # Compute c_in using the utility function
    sigma = cfg.model.sigma_distribution.sigma
    c_in, c_skip, c_out, c_noise = normalization_factors(sigma, average_squared_distance)
    c_in_float = float(c_in)
    c_noise_float = float(c_noise)
    print(f"Computed normalization factors with sigma={sigma}:")
    print(f"  c_in: {c_in_float}")
    print(f"  c_skip: {c_skip}")
    print(f"  c_out: {c_out}")
    print(f"  c_noise: {c_noise}")
    breakpoint()
    # Configure DenoisingConditioner with computed c_in
    if cfg.model.conditioner._target_ == "jamun.model.conditioners.DenoisedConditioner":
        # cfg.model.conditioner.N_structures = 2  # Must match architecture N_structures
        cfg.model.conditioner.pretrained_model_path = "sule-shashank/jamun/88i7qkj2"
        cfg.model.conditioner.c_in = c_in_float

    if cfg.model.conditioner._target_ == "jamun.model.conditioners.conditioners.SpatioTemporalConditioner":
        cfg.model.conditioner.spatiotemporal_model.radial_cutoff = average_squared_distance
        max_radius = cfg.model.max_radius
        temporal_average_squared_distance = compute_temporal_average_squared_distance_from_config(cfg)
        temporal_radial_cutoff = compute_radial_cutoff(
            max_radius=max_radius,
            average_squared_distance=temporal_average_squared_distance,  # Use temporal for spatiotemporal model
            sigma=sigma,
            D=3,
        )
        cfg.model.conditioner.spatiotemporal_model.temporal_cutoff = temporal_radial_cutoff
        cfg.model.conditioner.c_noise = c_noise_float
        cfg.model.conditioner.c_in = c_in_float

    print("Loading model...")
    model = hydra.utils.instantiate(cfg.model)
    model.conditioning_module.c_noise = c_noise
    print(f"Model loaded: {type(model)}")
    print(f"Conditioner: {type(model.conditioning_module)}")
    print(f"Sigma: {model.sigma_distribution.sigma}")
    # print(f"Conditioner c_in: {model.conditioning_module.c_in}")
    breakpoint()

    # Get a single batch
    print("Getting a batch of data...")
    train_loader = datamodule.train_dataloader()
    _, batch = next(enumerate(train_loader))

    print(f"Batch shape: {batch.pos.shape}")
    print(f"Hidden state shape: {[h.shape for h in batch.hidden_state]}")
    breakpoint()

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        sigma = model.sigma_distribution.sample()
        x_target, xhat, y = model.noise_and_denoise(batch, sigma, align_noisy_input=True)

    print(f"Input shape: {batch.pos.shape}")
    print(f"Noisy shape: {y.pos.shape}")
    print(f"Output shape: {xhat.pos.shape}")
    breakpoint()

    # Test single training step
    print("Testing training step...")
    loss_output = model.training_step(batch, 0)
    print(f"Loss: {loss_output['loss']:.4f}")
    breakpoint()


if __name__ == "__main__":
    main()
