import logging
from typing import Callable, Optional, Tuple, Union, Dict

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch_geometric
import torch_scatter

from jamun.utils import align_A_to_B_batched, mean_center, unsqueeze_trailing

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
        mean_center_input: bool,
        mean_center_output: bool,
        mirror_augmentation_rate: float,
        bond_loss_coefficient: float = 1.0,
        lr_scheduler_config: Optional[Dict] = None,
        use_torch_compile: bool = True,
        torch_compile_kwargs: Optional[Dict] = None,
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

        self.mean_center_input = mean_center_input
        if self.mean_center_input:
            py_logger.info("Mean centering input.")
        else:
            py_logger.info("Not mean centering input.")

        self.mean_center_output = mean_center_output
        if self.mean_center_output:
            py_logger.info("Mean centering output.")
        else:
            py_logger.info("Not mean centering output.")

        self.mirror_augmentation_rate = mirror_augmentation_rate
        py_logger.info(f"Mirror augmentation rate: {self.mirror_augmentation_rate}")

        self.bond_loss_coefficient = bond_loss_coefficient

    def add_noise(self, x: torch_geometric.data.Batch, sigma: Union[float, torch.Tensor]) -> torch_geometric.data.Batch:
        # pos [B, ...]
        sigma = torch.as_tensor(sigma, device=x.pos.device, dtype=x.pos.dtype)
        sigma = unsqueeze_trailing(sigma, x.pos.ndim)

        y = x.clone()
        if self.add_fixed_ones:
            noise = torch.ones_like(x.pos)
        elif self.add_fixed_noise:
            torch.manual_seed(0)
            num_batches = x.batch.max().item() + 1
            if len(x.pos.shape) == 2:
                num_nodes_per_batch = x.pos.shape[0] // num_batches
                noise = torch.randn_like((x.pos[:num_nodes_per_batch])).repeat(num_batches, 1)
            if len(x.pos.shape) == 3:
                num_nodes_per_batch = x.pos.shape[1]
                noise = torch.randn_like((x.pos[0])).repeat(num_batches, 1, 1)
        else:
            noise = torch.randn_like(x.pos)

        y.pos = x.pos + sigma * noise
        if torch.rand(()) < self.mirror_augmentation_rate:
            y.pos = -y.pos
        return y

    def score(self, y: torch_geometric.data.Batch, sigma: Union[float, torch.Tensor]) -> torch_geometric.data.Batch:
        sigma = torch.as_tensor(sigma, device=y.pos.device, dtype=y.pos.dtype)
        return (self.xhat(y, sigma).pos - y.pos) / (unsqueeze_trailing(sigma, y.pos.ndim - 1) ** 2)

    @classmethod
    def normalization_factors(cls, sigma: float, average_squared_distance: float, D: int = 3) -> Tuple[float, float]:
        """Normalization factors for the input and output."""
        A = torch.as_tensor(average_squared_distance)
        B = torch.as_tensor(2 * D * sigma**2)

        c_in = 1.0 / torch.sqrt(A + B)
        c_skip = A / (A + B)
        c_out = torch.sqrt((A * B) / (A + B))
        c_noise = torch.log(sigma) / 4
        return c_in, c_skip, c_out, c_noise

    @classmethod
    def loss_weight(cls, sigma: float, average_squared_distance: float, D: int = 3) -> float:
        """Loss weight for this graph."""
        _, _, c_out, _ = cls.normalization_factors(sigma, average_squared_distance, D)
        return 1 / (c_out**2)

    def effective_radial_cutoff(self, sigma: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute the effective radial cutoff for the noise level."""
        return torch.sqrt((self.max_radius**2) + 6 * (sigma**2))

    def add_edges(self, y: torch_geometric.data.Batch, radial_cutoff: float) -> torch_geometric.data.Batch:
        """Add edges to the graph based on the effective radial cutoff."""
        pos = y.pos
        if "batch" in y:
            batch = y["batch"]
        else:
            batch = torch.zeros(y.num_nodes, dtype=torch.long, device=pos.device)

        # Our dataloader already adds the bonded edges.
        bonded_edge_index = y.edge_index
        radial_edge_index = torch_geometric.nn.radius_graph(pos, radial_cutoff, batch)

        edge_index = torch.cat((radial_edge_index, bonded_edge_index), dim=-1)
        bond_mask = torch.cat(
            (
                torch.zeros(radial_edge_index.shape[1], dtype=torch.long, device=pos.device),
                torch.ones(bonded_edge_index.shape[1], dtype=torch.long, device=pos.device),
            ),
            dim=0,
        )
        y.edge_index = edge_index
        y.bond_mask = bond_mask
        return y

    def pad(self, y: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
        """Add padding to the graph."""
        return y

    def unpad(self, y: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
        """Remove padding from the graph."""
        return y

    def xhat_normalized(
        self, y: torch_geometric.data.Batch, sigma: Union[float, torch.Tensor]
    ) -> torch_geometric.data.Batch:
        """Compute the denoised prediction using the normalization factors from JAMUN."""
        # Output, noise and skip scale
        D = y.pos.shape[-1]
        assert D == 3
        sigma = torch.as_tensor(sigma, device=y.pos.device, dtype=y.pos.dtype)

        # Compute the normalization factors.
        c_in, c_skip, c_out, c_noise = self.normalization_factors(sigma, self.average_squared_distance, D)
        radial_cutoff = self.effective_radial_cutoff(sigma) / c_in
        # print("radial_cutoff", radial_cutoff, self.effective_radial_cutoff(sigma), c_in)

        # Adjust dimensions.
        c_in = unsqueeze_trailing(c_in, y.pos.ndim - 1)
        c_skip = unsqueeze_trailing(c_skip, y.pos.ndim - 1)
        c_out = unsqueeze_trailing(c_out, y.pos.ndim - 1)
        c_noise = c_noise.unsqueeze(0)

        # Add edges to the graph.
        y = self.add_edges(y, radial_cutoff)

        # Pad here.
        y = self.pad(y)
        # print("shapes", y.pos.shape, y.edge_index.shape)

        y_scaled = y.clone()
        y_scaled.pos = y.pos * c_in

        xhat = y.clone()
        g_pred = self.g(y_scaled, c_noise, radial_cutoff)
        xhat.pos = c_skip * y.pos + c_out * g_pred.pos

        # Unpad here.
        xhat = self.unpad(xhat)

        return xhat

    def xhat(self, y: torch.Tensor, sigma: Union[float, torch.Tensor]):
        """Compute the denoised prediction."""
        if self.mean_center_input:
            y = mean_center(y)

        xhat = self.xhat_normalized(y, sigma)

        # Mean center the prediction.
        if self.mean_center_output:
            xhat = mean_center(xhat)

        return xhat

    def noise_and_denoise(
        self,
        x: torch_geometric.data.Batch,
        sigma: Union[float, torch.Tensor],
        align_noisy_input: bool,
    ) -> Tuple[torch_geometric.data.Batch, torch_geometric.data.Batch]:
        """Add noise to the input and denoise it."""
        with torch.no_grad():
            y = self.add_noise(x, sigma)
            if self.mean_center_input:
                y = mean_center(y)

            # Aligning each batch.
            if align_noisy_input:
                y = align_A_to_B_batched(y, x)

        xhat = self.xhat(y, sigma)
        return xhat, y

    def compute_loss(
        self,
        x: torch_geometric.data.Batch,
        xhat: torch.Tensor,
        sigma: Union[float, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the loss."""
        # If we mean center the output, we should also mean center the target.
        if self.mean_center_output:
            x = mean_center(x)

        device = xhat.pos.device
        D = xhat.pos.shape[-1]
        assert D == 3
        D = torch.tensor(D, device=device)

        def coordinate_loss():
            """Standard l2 loss on atom coordinates."""
            # Compute the raw loss.
            raw_loss = F.mse_loss(xhat.pos, x.pos, reduction="none")
            raw_loss = raw_loss.sum(dim=-1)

            # Compute the scaled RMSD.
            scaled_rmsd = torch.sqrt(raw_loss) / (sigma * torch.sqrt(D))

            # Take the mean over each graph.
            raw_loss = torch_scatter.scatter_mean(raw_loss, x.batch)
            scaled_rmsd = torch_scatter.scatter_mean(scaled_rmsd, x.batch)

            # Account for the loss weight across graphs and noise levels.
            scaled_loss = raw_loss * x.loss_weight
            scaled_loss *= self.loss_weight(sigma, self.average_squared_distance, D).to(device)

            return scaled_loss, raw_loss, scaled_rmsd

        scaled_coordinate_loss, raw_coordinate_loss, scaled_rmsd = coordinate_loss()
        total_loss = scaled_coordinate_loss
        return total_loss, {
            "coordinate_loss": scaled_coordinate_loss,
            "raw_coordinate_loss": raw_coordinate_loss,
            "scaled_rmsd": scaled_rmsd,
        }

    def noise_and_compute_loss(
        self,
        x: torch_geometric.data.Batch,
        sigma: Union[float, torch.Tensor],
        align_noisy_input: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add noise to the input and compute the loss."""
        if self.mean_center_input:
            x = mean_center(x)

        xhat, _ = self.noise_and_denoise(x, sigma, align_noisy_input=align_noisy_input)
        return self.compute_loss(x, xhat, sigma)

    def training_step(self, batch: torch_geometric.data.Batch, batch_idx: int):
        """Called during training."""
        sigma = self.sigma_distribution.sample().to(batch.pos.device)
        loss, aux = self.noise_and_compute_loss(
            batch, sigma, align_noisy_input=self.align_noisy_input_during_training,
        )

        # Average the loss and other metrics over all graphs.
        aux["loss"] = loss
        for key in aux:
            aux[key] = aux[key].mean()
            self.log(f"train/{key}", aux[key], prog_bar=False, batch_size=batch.num_graphs, sync_dist=False)

        return {
            "sigma": sigma,
            **aux,
        }

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int):
        """Called during validation."""
        sigma = self.sigma_distribution.sample().to(batch.pos.device)
        loss, aux = self.noise_and_compute_loss(
            batch, sigma, align_noisy_input=self.align_noisy_input_during_training
        )

        # Average the loss and other metrics over all graphs.
        aux["loss"] = loss
        for key in aux:
            aux[key] = aux[key].mean()
            self.log(
                f"val/{key}", aux[key], prog_bar=(key == "scaled_rmsd"), batch_size=batch.num_graphs, sync_dist=False
            )

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
