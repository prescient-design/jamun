import e3nn

e3nn.set_optimization_defaults(jit_script_fx=False)
import os
import sys

import dotenv
import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from jamun.data import MDtrajDataModule
from jamun.utils import ModelSamplingWrapperMemory, find_checkpoint

dotenv.load_dotenv("../.env", verbose=True)  # Adjust path if script is not in scratch/
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")
JAMUN_ROOT_PATH = os.getenv("JAMUN_ROOT_PATH")

project_root = "/homefs/home/sules/jamun"  # Adjust if necessary
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added '{project_root}' to sys.path for module discovery.")
else:
    print(f"'{project_root}' is already in sys.path.")


@hydra.main(version_base=None, config_path="../src/jamun/hydra_config", config_name="sample_memory")
def main(cfg):
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(cfg))
    breakpoint()

    # Load checkpoint using find_checkpoint function
    print("Finding checkpoint...")
    checkpoint_path = find_checkpoint(
        wandb_train_run_path=cfg.get("wandb_train_run_path"),
        checkpoint_dir=cfg.get("checkpoint_dir"),
        checkpoint_type=cfg.get("checkpoint_type"),
    )
    print(f"Checkpoint found at: {checkpoint_path}")
    # cfg.M = 1/6.0
    # cfg.delta = float(cfg.sigma)
    # cfg.friction = float(-np.log(np.sqrt(1-4*cfg.M)))
    # u = 1/cfg.M
    # cfg.inverse_temperature = float(4/(u*(1- np.sqrt(1 - 4/u))))
    print(f"Sampler params: {cfg.M}, {cfg.delta}, {cfg.friction}, {cfg.inverse_temperature}")
    breakpoint()

    # Load the model from checkpoint by instantiating it with the checkpoint path
    print("Loading model from checkpoint...")
    cfg.model.checkpoint_path = checkpoint_path
    model = hydra.utils.instantiate(cfg.model)
    from e3tools.nn import LayerNorm

    model.conditioning_module.spatiotemporal_model.temporal_to_spatial_pooler.layer_norm = LayerNorm(
        model.conditioning_module.spatiotemporal_model.temporal_module.irreps_out
    )
    model.conditioning_module.spatiotemporal_model.spatial_to_temporal_pooler.layer_norm = LayerNorm(
        model.conditioning_module.spatiotemporal_model.spatial_module.irreps_out
    )
    print(f"Model loaded: {type(model)}")
    breakpoint()

    # Set up initial datasets for sampling
    print("Setting up initial datasets...")
    init_datasets = hydra.utils.instantiate(cfg.init_datasets)
    print(f"Initial datasets loaded: {len(init_datasets)} datasets")
    print(f"Dataset types: {[type(ds) for ds in init_datasets]}")
    breakpoint()

    # Manually construct the DataModule
    print("Creating datamodule for testing...")
    datamodule = MDtrajDataModule(
        datasets={"train": init_datasets, "val": init_datasets, "test": init_datasets}, batch_size=1, num_workers=1
    )

    datamodule.setup("test")
    print("Datamodule setup complete")
    breakpoint()

    # Get a sample batch
    print("Getting a sample batch...")
    test_loader = datamodule.test_dataloader()
    batch_idx, batch = next(enumerate(test_loader))
    print(f"Batch shape: {batch.pos.shape}")
    print(f"Batch keys: {batch.keys}")
    # if hasattr(batch, 'hidden_state') and len(batch.hidden_state) > 0:
    #     print(f"Hidden state shapes: {[h.shape for h in batch.hidden_state]}")
    breakpoint()

    # Set up sampler
    print("Setting up sampler...")
    sampler = hydra.utils.instantiate(cfg.sampler)
    print(f"Sampler created: {type(sampler)}")
    breakpoint()

    # set up batch sampler
    batch_sampler = hydra.utils.instantiate(cfg.batch_sampler)
    print(f"Batch sampler created: {type(batch_sampler)}")
    print(f"Batch sampler mcmc: {batch_sampler.mcmc}")
    breakpoint()

    # Write test for score
    print("Testing score function...")
    with torch.no_grad():
        init_graphs = batch
        init_graphs = init_graphs.to(sampler.fabric.device)
        model_wrapped = ModelSamplingWrapperMemory(
            model=model, init_graphs=init_graphs, sigma=batch_sampler.sigma, recenter_on_init=True
        )
        y_init = model_wrapped.sample_initial_noisy_positions()
        y_hist_init = model_wrapped.sample_initial_noisy_history()
        init_score = model_wrapped.score(y_init, y_hist_init, batch_sampler.sigma)
        print(f"Initial score: {init_score}")
        breakpoint()

    # Test walk
    with torch.no_grad():
        y, v, y_hist, y_traj, score_traj, y_hist_traj = batch_sampler.mcmc(
            y_init,
            y_hist_init,
            lambda y, y_hist: model_wrapped.score(y, y_hist, batch_sampler.sigma),
            v_init="zero",
            steps=5,
        )
        print(f"Score trajectory: {score_traj}")
        breakpoint()

    # Test jump
    with torch.no_grad():
        xhat_traj = torch.stack(
            [
                model_wrapped.xhat(y_traj[i, :], y_hist_traj[i], sigma=batch_sampler.sigma)
                for i in tqdm(range(y_traj.size(0)), leave=False, desc="Jump")
            ],
            dim=0,
        )
        print(f"Xhat trajectory: {xhat_traj}")
        breakpoint()

    # Test walkjump
    with torch.no_grad():
        out = batch_sampler.sample(model_wrapped, y_init=y_init, v_init="zero", y_hist_init=y_hist_init)
        print(f"Out: {out}")
        breakpoint()

    # Test unbatching
    with torch.no_grad():
        samples = model_wrapped.unbatch_samples(out)
        print(f"Samples: {samples}")
        breakpoint()

    # Test sampling parameters
    print("Testing sampling setup...")
    print(f"Sigma: {cfg.sigma}")
    print(f"M: {cfg.M}")
    print(f"Delta: {cfg.delta}")
    print(f"Friction: {cfg.friction}")
    print(f"Number of sampling steps per batch: {cfg.num_sampling_steps_per_batch}")
    print(f"Number of batches: {cfg.num_batches}")
    breakpoint()

    # Test a forward pass with the model
    print("Testing model forward pass...")
    model.eval()
    with torch.no_grad():
        # Test if the model can process the batch
        if hasattr(model, "noise_and_denoise"):
            sigma_tensor = torch.tensor([cfg.sigma])
            x_target, xhat, y = model.noise_and_denoise(batch, sigma_tensor, align_noisy_input=True)
            print("Forward pass successful!")
            print(f"Input shape: {batch.pos.shape}")
            print(f"Noisy shape: {y.pos.shape}")
            print(f"Denoised shape: {xhat.pos.shape}")
        else:
            print("Model doesn't have noise_and_denoise method, testing direct forward pass")
            output = model(batch)
            print(f"Model output: {type(output)}")
    breakpoint()

    print("Script completed successfully!")


if __name__ == "__main__":
    main()
