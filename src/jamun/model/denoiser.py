import logging
from collections.abc import Callable

import e3nn
import e3tools
import lightning.pytorch as pl
import numpy as np
import torch
import torch_geometric

from jamun.utils import align_A_to_B_batched_f, mean_center_f, to_atom_graphs, unsqueeze_trailing


def compute_normalization_factors(
    sigma: float | torch.Tensor,
    *,
    average_squared_distance: float,
    normalization_type: str | None,
    sigma_data: float | None = None,
    D: int = 3,
    device: torch.device | None = None,
) -> tuple[float, float, float, float]:
    """Compute the normalization factors for the input, skip connection, output, and noise."""
    sigma = torch.as_tensor(sigma, device=device)

    if normalization_type is None:
        c_in = torch.as_tensor(1.0, device=device)
        c_skip = torch.as_tensor(0.0, device=device)
        c_out = torch.as_tensor(1.0, device=device)
        c_noise = torch.as_tensor(sigma, device=device)
        return c_in, c_skip, c_out, c_noise

    if normalization_type == "EDM":
        c_skip = (sigma_data**2) / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / torch.sqrt(sigma_data**2 + sigma**2)
        c_in = 1 / torch.sqrt(sigma**2 + sigma_data**2)
        c_noise = torch.log(sigma / sigma_data) * 0.25
        return c_in, c_skip, c_out, c_noise

    if normalization_type == "JAMUN":
        A = torch.as_tensor(average_squared_distance, device=device)
        B = torch.as_tensor(2 * D * sigma**2, device=device)

        c_in = 1.0 / torch.sqrt(A + B)
        c_skip = A / (A + B)
        c_out = torch.sqrt((A * B) / (A + B))
        c_noise = torch.log(sigma) / 4
        return c_in, c_skip, c_out, c_noise

    raise ValueError(f"Unknown normalization type: {normalization_type}")


def add_edges(
    y: torch.Tensor,
    topology: torch_geometric.data.Batch,
    batch: torch.Tensor,
    radial_cutoff: float,
) -> torch_geometric.data.Batch:
    """Add edges to the graph based on the effective radial cutoff."""
    if topology.get("edge_index") is not None:
        return topology

    topology = topology.clone()
    with torch.cuda.nvtx.range("radial_graph"):
        radial_edge_index = e3tools.radius_graph(y, radial_cutoff, batch)

    with torch.cuda.nvtx.range("concatenate_edges"):
        edge_index = torch.cat((radial_edge_index, topology.bonded_edge_index), dim=-1)
        if topology.bonded_edge_index.numel() == 0:
            bond_mask = torch.zeros(radial_edge_index.shape[1], dtype=torch.long, device=y.device)
        else:
            bond_mask = torch.cat(
                (
                    torch.zeros(radial_edge_index.shape[1], dtype=torch.long, device=y.device),
                    torch.ones(topology.bonded_edge_index.shape[1], dtype=torch.long, device=y.device),
                ),
                dim=0,
            )

    topology.edge_index = edge_index
    topology.bond_mask = bond_mask
    return topology


class Denoiser(pl.LightningModule):
    """The main denoiser model."""

    def __init__(
        self,
        arch: Callable[..., torch.nn.Module],
        optim: Callable[..., torch.optim.Optimizer],
        sigma_distribution: torch.distributions.Distribution,
        max_radius: float,
        average_squared_distance: float,
        add_fixed_noise: bool,
        add_fixed_ones: bool,
        align_noisy_input_during_training: bool,
        align_noisy_input_during_evaluation: bool,
        mean_center: bool,
        mirror_augmentation_rate: float,
        bond_loss_coefficient: float = 1.0,
        normalization_type: str | None = "JAMUN",
        sigma_data: float | None = None,  # Only used if normalization_type is "EDM"
        lr_scheduler_config: dict | None = None,
        use_torch_compile: bool = True,
        torch_compile_kwargs: dict | None = None,
        rotational_augmentation: bool = False,
        alignment_correction_order: int = 0,
        pass_topology_as_atom_graphs: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.g = arch()
        if use_torch_compile:
            if torch_compile_kwargs is None:
                torch_compile_kwargs = {}

            self.g = torch.compile(self.g, **torch_compile_kwargs)

        py_logger = logging.getLogger("jamun")
        py_logger.info(self.g)

        self.optim_factory = optim
        self.lr_scheduler_config = lr_scheduler_config
        self.sigma_distribution = sigma_distribution
        self.max_radius = max_radius
        if self.max_radius is None:
            raise ValueError("max_radius must be provided")

        self.add_fixed_noise = add_fixed_noise
        self.add_fixed_ones = add_fixed_ones
        if self.add_fixed_noise and self.add_fixed_ones:
            raise ValueError("Can't add fixed noise and fixed ones at the same time")
        if self.add_fixed_noise:
            py_logger.info("Adding fixed noise")
        if self.add_fixed_ones:
            py_logger.info("Adding fixed ones")

        self.average_squared_distance = average_squared_distance
        py_logger.info(f"Average squared distance = {self.average_squared_distance}")

        self.align_noisy_input_during_training = align_noisy_input_during_training
        if self.align_noisy_input_during_training:
            py_logger.info("Aligning noisy input during training.")
        else:
            py_logger.info("Not aligning noisy input during training.")

        self.align_noisy_input_during_evaluation = align_noisy_input_during_evaluation
        if self.align_noisy_input_during_evaluation:
            py_logger.info("Aligning noisy input during evaluation.")
        else:
            py_logger.info("Not aligning noisy input during evaluation.")

        self.mean_center = mean_center
        if self.mean_center:
            py_logger.info("Mean centering input and output.")
        else:
            py_logger.info("Not mean centering input and output.")

        self.mirror_augmentation_rate = mirror_augmentation_rate
        py_logger.info(f"Mirror augmentation rate: {self.mirror_augmentation_rate}")

        self.normalization_type = normalization_type
        if self.normalization_type is not None:
            py_logger.info(f"Normalization type: {self.normalization_type}")
        else:
            py_logger.info("No normalization")

        self.sigma_data = sigma_data
        if self.normalization_type == "EDM" and self.sigma_data is None:
            raise ValueError("sigma_data must be provided when normalization_type is 'EDM'")
        elif self.normalization_type != "EDM" and self.sigma_data is not None:
            raise ValueError("sigma_data can only be used when normalization_type is 'EDM'")

        self.bond_loss_coefficient = bond_loss_coefficient
        self.alignment_correction_order = alignment_correction_order
        py_logger.info(f"Alignment correction order: {self.alignment_correction_order}")

        self.pass_topology_as_atom_graphs = pass_topology_as_atom_graphs
        self.rotational_augmentation = rotational_augmentation
        if self.rotational_augmentation:
            py_logger.info("Rotational augmentation is enabled.")

    def add_noise(self, x: torch.Tensor, sigma: float | torch.Tensor, num_graphs: int) -> torch.Tensor:
        # pos [B, ...]
        sigma = unsqueeze_trailing(sigma, x.ndim)

        if self.add_fixed_ones:
            noise = torch.ones_like(x)
        elif self.add_fixed_noise:
            torch.manual_seed(0)
            if len(x.shape) == 2:
                num_nodes_per_graph = x.shape[0] // num_graphs
                noise = torch.randn_like(x[:num_nodes_per_graph]).repeat(num_graphs, 1)
            if len(x.shape) == 3:
                num_nodes_per_graph = x.shape[1]
                noise = torch.randn_like(x[0]).repeat(num_graphs, 1, 1)
        else:
            noise = torch.randn_like(x)

        y = x + sigma * noise
        if torch.rand(()) < self.mirror_augmentation_rate:
            y = -y
        return y

    def score(self, data: torch_geometric.data.Batch, sigma: float | torch.Tensor) -> torch.Tensor:
        """Compute the score function."""
        y, topology, batch, num_graphs = data.pos, data.clone(), data.batch, data.num_graphs
        del topology.batch, topology.num_graphs
        sigma = torch.as_tensor(sigma).to(y)
        return (self.xhat(y, topology, batch, num_graphs, sigma) - y) / (unsqueeze_trailing(sigma, y.ndim - 1) ** 2)

    def effective_radial_cutoff(self, sigma: float | torch.Tensor) -> torch.Tensor:
        """Compute the effective radial cutoff for the noise level."""
        return torch.sqrt((self.max_radius**2) + 6 * (sigma**2))

    def xhat_normalized(
        self,
        y: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:
        """Compute the denoised prediction using the normalization factors from JAMUN."""
        sigma = torch.as_tensor(sigma).to(y)

        # Compute the normalization factors.
        with torch.cuda.nvtx.range("normalization_factors"):
            c_in, c_skip, c_out, c_noise = compute_normalization_factors(
                sigma,
                average_squared_distance=self.average_squared_distance,
                normalization_type=self.normalization_type,
                sigma_data=self.sigma_data,
                D=y.shape[-1],
                device=y.device,
            )

        radial_cutoff = self.effective_radial_cutoff(sigma) / c_in

        # Adjust dimensions.
        c_in = unsqueeze_trailing(c_in, y.ndim - 1)
        c_skip = unsqueeze_trailing(c_skip, y.ndim - 1)
        c_out = unsqueeze_trailing(c_out, y.ndim - 1)
        c_noise = c_noise.unsqueeze(0)

        # Add edges to the graph.
        with torch.cuda.nvtx.range("add_edges"):
            topology = add_edges(y, topology, batch, radial_cutoff)

        with torch.cuda.nvtx.range("scale_y"):
            y_scaled = y * c_in

        with torch.cuda.nvtx.range("g"):
            g_pred = self.g(
                pos=y_scaled,
                topology=topology,
                c_noise=c_noise,
                effective_radial_cutoff=radial_cutoff,
                batch=batch,
                num_graphs=num_graphs,
            )

        return c_skip * y + c_out * g_pred

    def xhat(
        self,
        y: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:
        """Compute the denoised prediction."""
        if self.mean_center:
            with torch.cuda.nvtx.range("mean_center_y"):
                y = mean_center_f(y, batch, num_graphs)

        with torch.cuda.nvtx.range("xhat_normalized"):
            xhat = self.xhat_normalized(y, topology, batch, num_graphs, sigma)

        # Mean center the prediction.
        if self.mean_center:
            with torch.cuda.nvtx.range("mean_center_xhat"):
                xhat = mean_center_f(xhat, batch, num_graphs)

        return xhat

    def noise_and_denoise(
        self,
        x: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
        align_noisy_input: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add noise to the input and denoise it."""
        with torch.no_grad():
            if self.mean_center:
                with torch.cuda.nvtx.range("mean_center_x"):
                    x = mean_center_f(x, batch, num_graphs)

            sigma = torch.as_tensor(sigma).to(x)

            with torch.cuda.nvtx.range("add_noise"):
                y = self.add_noise(x, sigma, num_graphs)

            if self.mean_center:
                with torch.cuda.nvtx.range("mean_center_y"):
                    y = mean_center_f(y, batch, num_graphs)

            # Aligning each batch.
            if align_noisy_input:
                with torch.cuda.nvtx.range("align_A_to_B_batched"):
                    x = align_A_to_B_batched_f(
                        x,
                        y,
                        topology.batch,
                        topology.num_graphs,
                        sigma=sigma,
                        correction_order=self.alignment_correction_order,
                    )

        with torch.cuda.nvtx.range("xhat"):
            xhat = self.xhat(y, topology, batch, num_graphs, sigma)

        return xhat, x, y

    def compute_loss(
        self,
        x: torch.Tensor,
        xhat: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the loss."""
        if self.mean_center:
            with torch.cuda.nvtx.range("mean_center_x"):
                x = mean_center_f(x, batch, num_graphs)

        D = xhat.shape[-1]

        # Compute the raw loss.
        with torch.cuda.nvtx.range("raw_coordinate_loss"):
            raw_coordinate_loss = (xhat - x).pow(2).sum(dim=-1)

        # Take the mean over each graph.
        with torch.cuda.nvtx.range("mean_over_graphs"):
            mse = e3tools.scatter(raw_coordinate_loss, batch, dim=0, dim_size=num_graphs, reduce="mean")

        # Compute the scaled RMSD.
        with torch.cuda.nvtx.range("scaled_rmsd"):
            rmsd = torch.sqrt(mse)
            scaled_rmsd = rmsd / (sigma * np.sqrt(D))

        # Account for the loss weight across graphs and noise levels.
        with torch.cuda.nvtx.range("loss_weight"):
            loss = mse * topology.loss_weight
            _, _, c_out, _ = compute_normalization_factors(
                sigma,
                average_squared_distance=self.average_squared_distance,
                normalization_type=self.normalization_type,
                sigma_data=self.sigma_data,
                D=D,
                device=x.device,
            )
            loss = loss / (c_out**2)

        return loss, {
            "mse": mse,
            "rmsd": rmsd,
            "scaled_rmsd": scaled_rmsd,
        }

    def noise_and_compute_loss(
        self,
        x: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
        align_noisy_input: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Add noise to the input and compute the loss."""
        xhat, x, _ = self.noise_and_denoise(x, topology, sigma, align_noisy_input=align_noisy_input)
        return self.compute_loss(x, xhat, topology, sigma)

    def training_step(self, data: torch_geometric.data.Batch, data_idx: int):
        """Called during training."""
        with torch.cuda.nvtx.range("sample_sigma"):
            sigma = self.sigma_distribution.sample().to(self.device)

        topology = data.clone()
        del topology.pos, topology.batch, topology.num_graphs

        if self.pass_topology_as_atom_graphs:
            topology = to_atom_graphs(topology)

        x, batch, num_graphs = data.pos, data.batch, data.num_graphs
        with torch.cuda.nvtx.range("rotational_augmentation"):
            if self.rotational_augmentation:
                R = e3nn.o3.rand_rotation_matrix(device=self.device, dtype=x.dtype)
                x = torch.einsum("ni,ij->nj", x, R.T)

        loss, aux = self.noise_and_compute_loss(
            x,
            topology,
            batch,
            num_graphs,
            sigma,
            align_noisy_input=self.align_noisy_input_during_training,
        )

        # Average the loss and other metrics over all graphs.
        with torch.cuda.nvtx.range("mean_over_graphs"):
            aux["loss"] = loss
            for key in aux:
                aux[key] = aux[key].mean()

                self.log(f"train/{key}", aux[key], prog_bar=False, batch_size=num_graphs, sync_dist=False)

        return {
            "sigma": sigma,
            **aux,
        }

    def validation_step(self, data: torch_geometric.data.Batch, data_idx: int):
        """Called during validation."""
        sigma = self.sigma_distribution.sample().to(self.device)

        topology = data.clone()
        del topology.pos, topology.batch, topology.num_graphs

        if self.pass_topology_as_atom_graphs:
            topology = to_atom_graphs(topology)

        x, batch, num_graphs = data.pos, data.batch, data.num_graphs
        if self.rotational_augmentation:
            R = e3nn.o3.rand_rotation_matrix(device=self.device, dtype=x.dtype)
            x = torch.einsum("ni,ji->nj", x, R)

        loss, aux = self.noise_and_compute_loss(
            x, topology, batch, num_graphs, sigma, align_noisy_input=self.align_noisy_input_during_training
        )

        # Average the loss and other metrics over all graphs.
        aux["loss"] = loss
        for key in aux:
            aux[key] = aux[key].mean()
            self.log(f"val/{key}", aux[key], prog_bar=(key == "scaled_rmsd"), batch_size=num_graphs, sync_dist=True)

        return {
            "sigma": sigma,
            **aux,
        }

    def configure_optimizers(self):
        """Set up the optimizer and learning rate scheduler."""
        optimizer = self.optim_factory(params=self.parameters())

        out = {"optimizer": optimizer}
        if self.lr_scheduler_config:
            scheduler = self.lr_scheduler_config.pop("scheduler")
            out["lr_scheduler"] = {
                "scheduler": scheduler(optimizer),
                **self.lr_scheduler_config,
            }

        return out
