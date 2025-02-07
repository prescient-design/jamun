import dotenv
import os
import functools
import tqdm

import logging
logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("jamun")

import torch
torch.set_float32_matmul_precision('high')

import e3nn
e3nn.set_optimization_defaults(jit_script_fx=False)

import jamun
import jamun.data
import jamun.model
import jamun.model.arch
import jamun.e3tools
import jamun.distributions


dotenv.load_dotenv(".env", verbose=True)
JAMUN_DATA_PATH = os.getenv("JAMUN_DATA_PATH")

datasets = {
   "train": jamun.data.parse_datasets_from_directory(
        root=f"{JAMUN_DATA_PATH}/mdgen/data/4AA_sims_partitioned_chunked/train/",
        traj_pattern="^(....)_.*.xtc",
        pdb_pattern="^(....).pdb",
        as_iterable=True,
        subsample=5,
        max_datasets=10,
    )
}

datamodule = jamun.data.MDtrajDataModule(
    datasets=datasets,
    batch_size=64,
    num_workers=4,
)
datamodule.setup(None)

arch = functools.partial(
    jamun.model.arch.E3Conv,
    irreps_out="1x1e",
    irreps_hidden="120x0e + 32x1e",
    irreps_sh="1x0e + 1x1e",
    n_layers=5,
    edge_attr_dim=64,
    atom_type_embedding_dim=8,
    atom_code_embedding_dim=8,
    residue_code_embedding_dim=32,
    residue_index_embedding_dim=8,
    use_residue_information=True,
    use_residue_sequence_index=False,
    hidden_layer_factory=functools.partial(
        jamun.e3tools.nn.ConvBlock,
        conv=jamun.e3tools.nn.Conv,
    ),
    output_head_factory=functools.partial(
        jamun.e3tools.nn.EquivariantMLP,
        irreps_hidden_list=["120x0e + 32x1e"]
    )
)
py_logger.info(f"Number of params: {sum(p.numel() for p in arch().parameters())}")

optim = functools.partial(
    torch.optim.Adam,
    lr=1e-2,
    weight_decay=0.0
)

sigma_distribution = jamun.distributions.ConstantSigma(
    sigma=0.04
)

denoiser = jamun.model.Denoiser(
    arch=arch,
    optim=optim,
    sigma_distribution=sigma_distribution,
    lr_scheduler_config=None,
    max_radius=1.0,
    average_squared_distance=0.332,
    add_fixed_noise=False,
    add_fixed_ones=False,
    align_noisy_input_during_training=True,
    align_noisy_input_during_evaluation=True,
    mean_center_input=True,
    mean_center_output=True,
    mirror_augmentation_rate=0.0,
    use_torch_compile=True,
    torch_compile_kwargs=dict(
        fullgraph=True,
        dynamic=True,
        mode="max-autotune-no-cudagraphs",
    ),
)

# Transfer to device.
device = torch.device("cuda:0")
denoiser = denoiser.to(device)
opt = denoiser.configure_optimizers()["optimizer"]

# Warmup.
n_warmup = 10
for i, batch in tqdm.tqdm(enumerate(datamodule.train_dataloader()), total=n_warmup, desc="Warmup"):
    if i == n_warmup:
        break

    batch = batch.to(device)
    out = denoiser.training_step(batch, i)
    loss = out["loss"]
    loss.backward()
    opt.step()
    opt.zero_grad()

        
# Actual training.
n_actual = 20
torch.cuda.cudart().cudaProfilerStart()

for i, batch in tqdm.tqdm(enumerate(datamodule.train_dataloader()), total=n_actual, desc="Training"):
    if i == n_actual:
        break

    batch = batch.to(device)

    torch.cuda.nvtx.range_push(f"iter_{i}")

    torch.cuda.nvtx.range_push("forward")
    out = denoiser.training_step(batch, i)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("backward")
    loss = out["loss"]
    loss.backward()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("step")
    opt.step()
    opt.zero_grad()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()

torch.cuda.cudart().cudaProfilerStop()






