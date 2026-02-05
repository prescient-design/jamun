"""
Test gradient equivalence between automatic and manual optimization in DenoiserMultimeasurement.

To run this test with proper hydra configuration:

python3 scratch/test_gradient_equivalence.py --config-dir=configs experiment=train_test_single_shape_conditional ++model._target_=jamun.model.denoiser_multimeasurement.DenoiserMultimeasurement ++model.multimeasurement=True ++model.N_measurements_hidden=2 ++model.N_measurements=2 ++model.max_graphs_per_batch=1

This will:
- Use the train_test_single_shape_conditional experiment config
- Override model to use DenoiserMultimeasurement
- Enable multimeasurement with 2 hidden measurements and 2 measurements
- Set max_graphs_per_batch=1 for manual optimization testing
"""

import e3nn

e3nn.set_optimization_defaults(jit_script_fx=False)
import os
import sys

import dotenv
import hydra
import numpy as np
import torch
import torch_geometric

dotenv.load_dotenv("../.env", verbose=True)
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")

project_root = "/homefs/home/sules/jamun"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jamun.utils import compute_average_squared_distance_from_datasets


def create_model_and_data(cfg, max_graphs_per_batch=None):
    """Create a model and datamodule with specified optimization mode."""
    # Configure DenoiserMultimeasurement
    cfg.model._target_ = "jamun.model.denoiser_multimeasurement.DenoiserMultimeasurement"
    cfg.model.sigma_distribution._target_ = "jamun.distributions.ConstantSigma"
    cfg.model.sigma_distribution.sigma = 0.04
    cfg.model.multimeasurement = True
    cfg.model.N_measurements_hidden = 2
    cfg.model.N_measurements = 2
    cfg.model.max_graphs_per_batch = max_graphs_per_batch

    # # Set up data - use correct attribute name filter_codes
    # cfg.data.datamodule.datasets.train.filter_codes = ['ALA_ALA']
    # cfg.data.datamodule.datasets.val.filter_codes = ['ALA_ALA']
    # cfg.data.datamodule.datasets.test.filter_codes = ['ALA_ALA']
    # cfg.data.datamodule.batch_size = 8  # Larger batch for meaningful chunking

    # Compute normalization
    datamodule = hydra.utils.instantiate(cfg.data.datamodule)
    datamodule.setup("compute_normalization")
    train_datasets = datamodule.datasets["train"]
    cutoff = cfg.model.max_radius
    average_squared_distance = compute_average_squared_distance_from_datasets(train_datasets, cutoff)
    cfg.model.average_squared_distance = average_squared_distance

    # Create model and data
    model = hydra.utils.instantiate(cfg.model)
    datamodule.setup("test")

    return model, datamodule


def get_model_gradients(model):
    """Extract gradients from model parameters."""
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone().detach()
    return gradients


def compute_gradient_norm(gradients):
    """Compute the total gradient norm across all parameters."""
    total_norm = 0.0
    for grad in gradients.values():
        total_norm += grad.norm().item() ** 2
    return total_norm**0.5


def compare_gradients(grad1, grad2, tolerance=1e-3):
    """Compare two gradient dictionaries."""
    if set(grad1.keys()) != set(grad2.keys()):
        print("ERROR: Different parameter names!")
        return False

    max_relative_diff = 0.0
    for name in grad1.keys():
        g1, g2 = grad1[name], grad2[name]

        # Compute relative difference
        diff = torch.abs(g1 - g2)
        max_val = torch.max(torch.abs(g1), torch.abs(g2))
        relative_diff = torch.where(max_val > 1e-8, diff / (max_val + 1e-8), diff)
        max_rel_diff_param = relative_diff.max().item()
        max_relative_diff = max(max_relative_diff, max_rel_diff_param)

        print(
            f"{name:30s}: max_rel_diff = {max_rel_diff_param:.6f}, norm_ratio = {g1.norm().item() / g2.norm().item():.6f}"
        )

    print(f"\nOverall max relative difference: {max_relative_diff:.6f}")
    return max_relative_diff < tolerance


@hydra.main(version_base=None, config_path="../src/jamun/hydra_config", config_name="train")
def test_gradient_equivalence(cfg):
    """Test that automatic and manual optimization produce equivalent gradients."""
    print("=" * 80)
    print("TESTING GRADIENT EQUIVALENCE: AUTOMATIC vs MANUAL OPTIMIZATION")
    print("=" * 80)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create automatic optimization model
    print("\n1. Creating AUTOMATIC optimization model...")
    model_auto, datamodule_auto = create_model_and_data(cfg.copy(), max_graphs_per_batch=None)
    print(f"   Automatic optimization: {model_auto.automatic_optimization}")

    # Create manual optimization model with same architecture
    print("2. Creating MANUAL optimization model...")
    model_manual, datamodule_manual = create_model_and_data(cfg.copy(), max_graphs_per_batch=2)  # 2 graphs per chunk
    print(f"   Automatic optimization: {model_manual.automatic_optimization}")

    # Ensure models are in training mode and parameters require gradients
    print("3. Setting up models for gradient computation...")
    model_auto.train()
    model_manual.train()

    # Ensure all parameters require gradients
    for param in model_auto.parameters():
        param.requires_grad_(True)
    for param in model_manual.parameters():
        param.requires_grad_(True)

    # Copy weights from auto to manual model to ensure identical starting point
    print("4. Synchronizing model weights...")
    model_manual.load_state_dict(model_auto.state_dict())

    # Get the same batch of data
    print("5. Getting identical batch...")
    torch.manual_seed(42)  # Reset seed to get same batch
    train_loader_auto = datamodule_auto.train_dataloader()
    batch_auto = next(iter(train_loader_auto))

    torch.manual_seed(42)  # Reset seed to get same batch
    train_loader_manual = datamodule_manual.train_dataloader()
    batch_manual = next(iter(train_loader_manual))

    print(f"   Batch shapes: auto={batch_auto.pos.shape}, manual={batch_manual.pos.shape}")
    print(f"   Batch equality: {torch.allclose(batch_auto.pos, batch_manual.pos)}")

    # Use the same sigma for both (important!)
    print("\n6. Setting identical sigma values...")
    sigma_value = 0.04  # Fixed sigma instead of sampling
    sigma_auto = torch.tensor(sigma_value)
    sigma_manual = torch.tensor(sigma_value)

    print(f"   Using fixed sigma: {sigma_value}")

    # Test forward pass equivalence (without multimeasurement first)
    print("\n7. Testing forward pass equivalence...")

    # Disable multimeasurement temporarily for cleaner testing
    model_auto.multimeasurement = False
    model_manual.multimeasurement = False

    with torch.no_grad():
        # Reset random seeds before each forward pass
        torch.manual_seed(123)
        x_target_auto, xhat_auto, y_auto = model_auto.noise_and_denoise(batch_auto, sigma_auto, align_noisy_input=True)

        torch.manual_seed(123)  # Same seed for manual
        x_target_manual, xhat_manual, y_manual = model_manual.noise_and_denoise(
            batch_manual, sigma_manual, align_noisy_input=True
        )

    forward_equal = torch.allclose(xhat_auto.pos, xhat_manual.pos, atol=1e-5)
    print(f"   Forward pass output equality: {forward_equal}")

    if not forward_equal:
        print(f"   Max difference: {(xhat_auto.pos - xhat_manual.pos).abs().max().item():.8f}")
        print("   Continuing with test anyway...")

    # Test gradient computation
    print("\n8. Computing gradients...")

    # AUTOMATIC OPTIMIZATION
    print("   Computing automatic optimization gradients...")
    model_auto.zero_grad()

    # Use same random seed for loss computation
    torch.manual_seed(456)
    x_target_auto, xhat_auto, y_auto = model_auto.noise_and_denoise(batch_auto, sigma_auto, align_noisy_input=True)
    loss_auto, aux_auto = model_auto.compute_loss(x_target_auto, xhat_auto, sigma_auto)
    loss_auto_mean = loss_auto.mean()

    print(f"   Auto loss: {loss_auto_mean.item():.6f}")
    print(f"   Auto loss requires_grad: {loss_auto_mean.requires_grad}")

    loss_auto_mean.backward()

    gradients_auto = get_model_gradients(model_auto)
    grad_norm_auto = compute_gradient_norm(gradients_auto)

    print(f"   Auto grad_norm: {grad_norm_auto:.6f}")

    # MANUAL OPTIMIZATION (simulate _manual_step)
    print("   Computing manual optimization gradients...")
    model_manual.zero_grad()

    # Use same random seed for noise generation
    torch.manual_seed(456)
    y_manual, x_target_manual_prep = model_manual._prepare_noisy_batch(
        batch_manual, sigma_manual, align_noisy_input=True
    )

    # Split into chunks
    y_list = y_manual.to_data_list()
    x_target_list = x_target_manual_prep.to_data_list()
    chunk_size = model_manual.max_graphs_per_batch
    num_chunks = (len(y_list) + chunk_size - 1) // chunk_size

    print(f"   Manual: {len(y_list)} graphs → {num_chunks} chunks of size {chunk_size}")

    # Process chunks and accumulate gradients (simulate manual_step)
    total_loss_manual = 0.0
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(y_list))

        y_chunk = torch_geometric.data.Batch.from_data_list(y_list[start_idx:end_idx])
        x_target_chunk = torch_geometric.data.Batch.from_data_list(x_target_list[start_idx:end_idx])

        xhat_chunk = model_manual.xhat(y_chunk, sigma_manual)
        loss_chunk, aux_chunk = model_manual.compute_loss(x_target_chunk, xhat_chunk, sigma_manual)
        loss_chunk_mean = loss_chunk.mean()

        # Scale loss by number of chunks (the fix we implemented)
        scaled_loss = loss_chunk_mean / num_chunks
        scaled_loss.backward()

        total_loss_manual += loss_chunk_mean.item()

    gradients_manual = get_model_gradients(model_manual)
    grad_norm_manual = compute_gradient_norm(gradients_manual)

    print(f"   Manual total loss: {total_loss_manual:.6f}, grad_norm: {grad_norm_manual:.6f}")

    # Compare gradients
    print("\n9. COMPARING GRADIENTS:")
    print(f"   Gradient norm ratio (manual/auto): {grad_norm_manual / grad_norm_auto:.6f}")
    print(f"   Loss ratio (manual/auto): {total_loss_manual / loss_auto_mean.item():.6f}")

    print("\n   Parameter-wise comparison:")
    gradients_match = compare_gradients(gradients_auto, gradients_manual, tolerance=1e-2)

    print("\n" + "=" * 80)
    if gradients_match:
        print("✅ SUCCESS: Manual and automatic optimization produce equivalent gradients!")
        print(f"   Gradient norms: auto={grad_norm_auto:.6f}, manual={grad_norm_manual:.6f}")
        print(f"   Relative difference: {abs(grad_norm_manual - grad_norm_auto) / grad_norm_auto * 100:.3f}%")
    else:
        print("❌ FAILURE: Gradients do not match within tolerance!")
        print("   This indicates an issue with the manual optimization implementation.")
    print("=" * 80)

    return gradients_match


if __name__ == "__main__":
    test_gradient_equivalence()
