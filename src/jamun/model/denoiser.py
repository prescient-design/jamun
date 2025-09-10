import logging
from collections.abc import Callable

import e3nn
import lightning.pytorch as pl
import torch
import torch_geometric

from jamun.model.utils import add_edges, compute_normalization_factors, compute_rmsd_metrics
from jamun.utils import align_A_to_B_batched_f, mean_center_f, to_atom_graphs, unsqueeze_trailing


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
        use_alignment_estimators: bool,
        mean_center: bool,
        mirror_augmentation_rate: float,
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

        py_logger = logging.getLogger(__name__)
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

        self.alignment_correction_order = alignment_correction_order
        self.use_alignment_estimators = use_alignment_estimators
        if self.use_alignment_estimators:
            py_logger.info(f"Using alignment estimators for x with correction order {self.alignment_correction_order}.")
        else:
            py_logger.info("Using standard estimators for x.")

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

    def score(
        self,
        y: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:
        """Compute the score function."""
        sigma = torch.as_tensor(sigma, device=y.device, dtype=y.dtype)
        xhat = self.xhat(y, topology, batch, num_graphs, sigma)
        return (xhat - y) / (unsqueeze_trailing(sigma, y.ndim - 1) ** 2)

    def xhat_normalized(
        self,
        y: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:
        """Compute the denoised prediction using the normalization factors from JAMUN."""
        sigma = torch.as_tensor(sigma, device=y.device, dtype=y.dtype)

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

        # Adjust dimensions.
        c_in = unsqueeze_trailing(c_in, y.ndim - 1)
        c_skip = unsqueeze_trailing(c_skip, y.ndim - 1)
        c_out = unsqueeze_trailing(c_out, y.ndim - 1)
        c_noise = c_noise.unsqueeze(0)

        with torch.cuda.nvtx.range("scale_y"):
            y_scaled = y * c_in

        # Add edges to the graph.
        with torch.cuda.nvtx.range("add_edges"):
            topology = add_edges(y_scaled, topology, batch, radial_cutoff=self.max_radius)

        if self.pass_topology_as_atom_graphs:
            topology = to_atom_graphs(topology, batch, num_graphs)

        with torch.cuda.nvtx.range("g"):
            g_pred = self.g(
                pos=y_scaled,
                topology=topology,
                c_noise=c_noise,
                c_in=c_in,
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
        use_alignment_estimators: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add noise to the input and denoise it, also returning the estimator of x."""
        with torch.no_grad():
            if self.mean_center:
                with torch.cuda.nvtx.range("mean_center_x"):
                    x = mean_center_f(x, batch, num_graphs)

            sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)

            with torch.cuda.nvtx.range("add_noise"):
                y = self.add_noise(x, sigma, num_graphs)

            if self.mean_center:
                with torch.cuda.nvtx.range("mean_center_y"):
                    y = mean_center_f(y, batch, num_graphs)

            if use_alignment_estimators:
                with torch.cuda.nvtx.range("align_x"):
                    xtarget = align_A_to_B_batched_f(
                        x,
                        y,
                        batch,
                        num_graphs,
                        sigma=sigma,
                        correction_order=self.alignment_correction_order,
                    )
            else:
                xtarget = x

        xtarget_second_order = align_A_to_B_batched_f(
            x,
            y,
            batch,
            num_graphs,
            sigma=sigma,
            correction_order=2,
        )

        with torch.cuda.nvtx.range("xhat"):
            xhat = self.xhat(y, topology, batch, num_graphs, sigma)

        return xhat, {
            "xtarget": xtarget,
            "xtarget_second_order": xtarget_second_order,
            "y": y,
        }

    def compute_loss(
        self,
        *,
        x: torch.Tensor,
        xtarget: torch.Tensor,
        xtarget_second_order: torch.Tensor,
        xhat: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the loss."""
        aux = {}
        aux_xtarget = compute_rmsd_metrics(
            x=xtarget, xhat=xhat, batch=batch, num_graphs=num_graphs, sigma=sigma, mean_center=self.mean_center
        )
        for key in aux_xtarget:
            aux[f"xtarget/{key}"] = aux_xtarget[key]

        aux_xtarget_second_order = compute_rmsd_metrics(
            x=xtarget_second_order,
            xhat=xhat,
            batch=batch,
            num_graphs=num_graphs,
            sigma=sigma,
            mean_center=self.mean_center,
        )
        for key in aux_xtarget_second_order:
            aux[f"xtarget_second_order/{key}"] = aux_xtarget_second_order[key]

        aux_x = compute_rmsd_metrics(
            x=x, xhat=xhat, batch=batch, num_graphs=num_graphs, sigma=sigma, mean_center=self.mean_center
        )
        for key in aux_x:
            aux[f"x/{key}"] = aux_x[key]

        mse = aux["xtarget/mse"]
        D = xhat.shape[-1]

        # Account for the loss weight across graphs and noise levels.
        with torch.cuda.nvtx.range("loss_weight"):
            loss = mse * topology.loss_weight
            _, _, c_out, _ = compute_normalization_factors(
                sigma,
                average_squared_distance=self.average_squared_distance,
                normalization_type=self.normalization_type,
                sigma_data=self.sigma_data,
                D=D,
                device=xtarget.device,
            )
            loss = loss / (c_out**2)

        return loss, aux

    def noise_and_compute_loss(
        self,
        *,
        x: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        sigma: float | torch.Tensor,
        use_alignment_estimators: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Add noise to the input and compute the loss."""
        xhat, aux = self.noise_and_denoise(
            x=x,
            topology=topology,
            batch=batch,
            num_graphs=num_graphs,
            sigma=sigma,
            use_alignment_estimators=use_alignment_estimators,
        )
        return self.compute_loss(
            x=x,
            xtarget=aux["xtarget"],
            xtarget_second_order=aux["xtarget_second_order"],
            xhat=xhat,
            topology=topology,
            batch=batch,
            num_graphs=num_graphs,
            sigma=sigma,
        )

    def training_step(self, data: torch_geometric.data.Batch, data_idx: int):
        """Called during training."""
        with torch.cuda.nvtx.range("sample_sigma"):
            sigma = self.sigma_distribution.sample().to(self.device)

        with torch.cuda.nvtx.range("clone_data"):
            topology = data.clone()

        with torch.cuda.nvtx.range("clear_topology"):
            del topology.pos, topology.batch, topology.num_graphs

        x, batch, num_graphs = data.pos, data.batch, data.num_graphs
        if self.rotational_augmentation:
            with torch.cuda.nvtx.range("rotational_augmentation"):
                R = e3nn.o3.rand_matrix(device=self.device, dtype=x.dtype)
                x = torch.einsum("Ni,ij->Nj", x, R.T)

        loss, aux = self.noise_and_compute_loss(
            x=x,
            topology=topology,
            batch=batch,
            num_graphs=num_graphs,
            sigma=sigma,
            use_alignment_estimators=self.use_alignment_estimators,
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

        x, batch, num_graphs = data.pos, data.batch, data.num_graphs
        if self.rotational_augmentation:
            R = e3nn.o3.rand_matrix(device=self.device, dtype=x.dtype)
            x = torch.einsum("Ni,ij->Nj", x, R.T)

        loss, aux = self.noise_and_compute_loss(
            x=x,
            topology=topology,
            batch=batch,
            num_graphs=num_graphs,
            sigma=sigma,
            use_alignment_estimators=self.use_alignment_estimators,
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
