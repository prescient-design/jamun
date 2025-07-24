import functools
import logging
from collections.abc import Callable

import e3nn
import e3tools
import lightning.pytorch as pl
import numpy as np
import torch
import torch_geometric

from jamun.model.denoiser import add_edges, compute_normalization_factors
from jamun.utils import align_A_to_B_batched_f, mean_center_f, to_atom_graphs, unsqueeze_trailing


def energy_direct(
    y: torch.Tensor, batch: torch.Tensor, num_graphs: int, sigma: torch.Tensor, g_y: torch.Tensor
) -> torch.Tensor:
    energies = (g_y - y).pow(2).sum(dim=-1) / (2 * (sigma**2))
    return e3tools.scatter(
        energies,
        batch,
        dim=0,
        dim_size=num_graphs,
        reduce="sum",
    )


def model_predictions_f(
    y: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    sigma: torch.Tensor,
    g: Callable,
    energy_only: bool,
) -> torch.Tensor:
    """Returns the model predictions: xhat, energy, and score."""
    if energy_only:
        # If we only need the energy, we can skip the VJP computation.
        g_y = g(y, batch=batch, num_graphs=num_graphs)
        energy = energy_direct(y, batch, num_graphs, sigma, g_y)
        return None, energy, None

    # NOTE g must be torch.Tensor to torch.Tensor
    g = functools.partial(g, batch=batch, num_graphs=num_graphs)
    g_y, vjp_func = torch.func.vjp(g, y)
    xhat = g_y - vjp_func(g_y - y, create_graph=True, retain_graph=True)[0]
    energy = energy_direct(y, batch, num_graphs, sigma, g_y)
    score = (xhat - y) / (sigma**2)
    return xhat, energy, score


def norm_wrapper(
    y: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    g: Callable,
    c_in: torch.Tensor,
    c_skip: torch.Tensor,
    c_out: torch.Tensor,
) -> torch.Tensor:
    return c_skip * y + c_out * g(c_in * y, batch=batch, num_graphs=num_graphs)


class EnergyModel(pl.LightningModule):
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
        normalization_type: str | None = "JAMUN",
        sigma_data: float | None = None,
        lr_scheduler_config: dict | None = None,
        use_torch_compile: bool = True,
        torch_compile_kwargs: dict | None = None,
        alignment_correction_order: int = 0,
        rotational_augmentation: bool = False,
        pass_topology_as_atom_graphs: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.g = arch()
        if use_torch_compile:
            if torch_compile_kwargs is None:
                torch_compile_kwargs = {}

            self.g = torch.compile(self.g, **torch_compile_kwargs)

        self.use_torch_compile = use_torch_compile
        self.torch_compile_kwargs = torch_compile_kwargs or {}

        py_logger = logging.getLogger("jamun")
        py_logger.info(self.g)

        self.optim_factory = optim
        self.lr_scheduler_config = lr_scheduler_config
        self.sigma_distribution = sigma_distribution
        self.max_radius = max_radius

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

        self.mirror_augmentation_rate = mirror_augmentation_rate
        py_logger.info(f"Mirror augmentation rate: {self.mirror_augmentation_rate}")

        self.alignment_correction_order = alignment_correction_order
        py_logger.info(f"Alignment correction order: {self.alignment_correction_order}")

        self.rotational_augmentation = rotational_augmentation
        if self.rotational_augmentation:
            py_logger.info("Rotational augmentation is enabled.")

        self.pass_topology_as_atom_graphs = pass_topology_as_atom_graphs

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

    def effective_radial_cutoff(self, sigma: float | torch.Tensor) -> torch.Tensor:
        """Compute the effective radial cutoff for the noise level."""
        return torch.sqrt((self.max_radius**2) + 6 * (sigma**2))

    def get_model_predictions(
        self,
        pos: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
        energy_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the denoised prediction, energy, and score."""
        if self.mean_center:
            with torch.cuda.nvtx.range("mean_center_y"):
                pos = mean_center_f(pos, batch, num_graphs)

        sigma = torch.as_tensor(sigma).to(pos)

        # Compute the normalization factors.
        with torch.cuda.nvtx.range("normalization_factors"):
            c_in, c_skip, c_out, c_noise = compute_normalization_factors(
                sigma,
                average_squared_distance=self.average_squared_distance,
                normalization_type=self.normalization_type,
                sigma_data=self.sigma_data,
                D=pos.shape[-1],
                device=pos.device,
            )
        radial_cutoff = self.effective_radial_cutoff(sigma) / c_in

        # Adjust dimensions.
        c_in = unsqueeze_trailing(c_in, pos.ndim - 1)
        c_skip = unsqueeze_trailing(c_skip, pos.ndim - 1)
        c_out = unsqueeze_trailing(c_out, pos.ndim - 1)
        c_noise = c_noise.unsqueeze(0)

        # Add edges to the graph.
        with torch.cuda.nvtx.range("add_edges"):
            topology = add_edges(pos, topology, batch, radial_cutoff)

        g = functools.partial(self.g, topology=topology, c_noise=c_noise, effective_radial_cutoff=radial_cutoff)
        h = functools.partial(norm_wrapper, g=g, c_in=c_in, c_skip=c_skip, c_out=c_out)

        with torch.cuda.nvtx.range("g"):
            model_predictions_fn = torch.compile(
                model_predictions_f,
                disable=not self.use_torch_compile,
                **self.torch_compile_kwargs,
            )
            xhat, energy, score = model_predictions_fn(pos, batch, num_graphs, sigma, h, energy_only=energy_only)

        return xhat, energy, score

    def xhat(
        self,
        y: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:
        """Compute the denoised prediction."""
        with torch.cuda.nvtx.range("get_model_predictions"):
            xhat, _, _ = self.get_model_predictions(y, topology, batch, num_graphs, sigma)

        # Mean center the prediction.
        if self.mean_center:
            with torch.cuda.nvtx.range("mean_center_xhat"):
                xhat = mean_center_f(xhat, batch, num_graphs)

        return xhat

    def score(self, data: torch_geometric.data.Batch, sigma: float | torch.Tensor) -> torch.Tensor:
        """Compute the score function."""
        y, topology, batch, num_graphs = data.pos, data.clone(), data.batch, data.num_graphs
        del topology.pos, topology.batch
        sigma = torch.as_tensor(sigma).to(y)
        _, _, score = self.get_model_predictions(y, topology, batch, num_graphs, sigma)
        return score

    def energy(
        self,
        pos: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:
        """Compute the energy and score for the given positions."""
        _, energy, _ = self.get_model_predictions(pos, topology, batch, num_graphs, sigma, energy_only=True)
        return energy

    def energy_and_score(
        self,
        pos: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the energy and score for the given positions."""
        _, energy, score = self.get_model_predictions(pos, topology, batch, num_graphs, sigma)
        return energy, score

    def noise_and_denoise(
        self,
        x: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
        align_noisy_input: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

            # Aligning each data.
            if align_noisy_input:
                with torch.cuda.nvtx.range("align_A_to_B_batched"):
                    x = align_A_to_B_batched_f(
                        x,
                        y,
                        batch,
                        num_graphs,
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
        xhat, x, _ = self.noise_and_denoise(x, topology, batch, num_graphs, sigma, align_noisy_input=align_noisy_input)
        return self.compute_loss(x, xhat, topology, batch, num_graphs, sigma)

    def training_step(self, data: torch_geometric.data.Batch, batch_idx: int):
        """Called during training."""
        with torch.cuda.nvtx.range("sample_sigma"):
            sigma = self.sigma_distribution.sample().to(self.device)

        topology = data.clone()
        del topology.pos, topology.batch, topology.num_graphs

        if self.pass_topology_as_atom_graphs:
            topology = to_atom_graphs(topology)

        x, batch, num_graphs = data.pos, data.batch, data.num_graphs
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

    def validation_step(self, data: torch_geometric.data.Batch, batch_idx: int):
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
