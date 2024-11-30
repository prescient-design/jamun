import logging
import os
import shutil
from typing import Callable, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torch_scatter
import torch_geometric

from jamun.utils import unsqueeze_trailing
from jamun.utils_md import align_A_to_B_batched, mean_center


class e3NoiseConditionedScoreModel(pl.LightningModule):
    def __init__(
        self,
        arch: Callable[..., torch.nn.Module],
        optim: Callable[..., torch.optim.Optimizer],
        sigma_distribution: torch.distributions.Distribution,
        max_radius: float,
        average_squared_distance: float,
        add_fixed_noise: bool,
        add_fixed_ones: bool,
        save_tensors: bool,
        align_noisy_input_during_training: bool,
        align_noisy_input_during_evaluation: bool,
        mean_center_input: bool,
        mean_center_output: bool,
        mirror_augmentation_rate: float,
        bond_loss_coefficient: float = 1.0,
        lr_scheduler_config: Optional[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.g = arch()

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

        # Remove the directory if it already exists
        self.save_tensors = save_tensors
        if self.save_tensors:
            shutil.rmtree(os.pwd, ignore_errors=True)

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
        if torch.rand(1) < self.mirror_augmentation_rate:
            y.pos = -y.pos
        return y

    def score(self, y: torch_geometric.data.Batch, sigma: Union[float, torch.Tensor]) -> torch_geometric.data.Batch:
        sigma = torch.as_tensor(sigma, device=y.pos.device, dtype=y.pos.dtype)
        return (self.xhat(y, sigma).pos - y.pos) / (unsqueeze_trailing(sigma, y.pos.ndim - 1) ** 2)

    @classmethod
    def _A(cls, sigma: float, average_squared_distance: float, D: int = 3) -> float:
        return torch.as_tensor(average_squared_distance)

    @classmethod
    def _B(cls, sigma: float, average_squared_distance: float, D: int = 3) -> float:
        return torch.as_tensor(2 * D * sigma**2)

    @classmethod
    def c_in(cls, sigma: float, average_squared_distance: float, D: int = 3) -> float:
        """Input scaling factor."""
        # return torch.as_tensor(1., device=sigma.device)
        A = cls._A(sigma, average_squared_distance, D)
        B = cls._B(sigma, average_squared_distance, D)
        return 1.0 / torch.sqrt(A + B)

    @classmethod
    def c_skip(cls, sigma: float, average_squared_distance: float, D: int = 3) -> float:
        """Skip connection weight."""
        A = cls._A(sigma, average_squared_distance, D)
        B = cls._B(sigma, average_squared_distance, D)
        return A / (A + B)

    @classmethod
    def c_out(cls, sigma: float, average_squared_distance: float, D: int = 3) -> float:
        """Output scaling factor."""
        A = cls._A(sigma, average_squared_distance, D)
        B = cls._B(sigma, average_squared_distance, D)
        return torch.sqrt((A * B) / (A + B))

    @classmethod
    def c_noise(cls, sigma: float, average_squared_distance: float, D: int = 3) -> float:
        """Noise scaling factor."""
        return torch.log(sigma) / 4

    @classmethod
    def loss_weight(cls, sigma: float, average_squared_distance: float, D: int = 3) -> float:
        return 1 / (cls.c_out(sigma, average_squared_distance, D) ** 2)

    def effective_radial_cutoff(self, sigma: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute the effective radial cutoff for the noise level."""
        return torch.sqrt((self.max_radius ** 2) + 6 * (sigma ** 2))

    def xhat_normalized(
        self, y: torch_geometric.data.Batch, sigma: Union[float, torch.Tensor], save_prefix: str = ""
    ) -> torch_geometric.data.Batch:
        """Compute the denoised prediction using the normalized parametrization."""
        # Output, noise and skip scale
        assert y.pos.shape[-1] == 3
        D = y.pos.shape[-1]
        sigma = torch.as_tensor(sigma, device=y.pos.device, dtype=y.pos.dtype)
        c_in = self.c_in(sigma, self.average_squared_distance, D)
        c_out = self.c_out(sigma, self.average_squared_distance, D)
        c_skip = self.c_skip(sigma, self.average_squared_distance, D)
        c_noise = self.c_noise(sigma, self.average_squared_distance, D)

        c_in = unsqueeze_trailing(c_in, y.pos.ndim - 1)
        c_out = unsqueeze_trailing(c_out, y.pos.ndim - 1)
        c_skip = unsqueeze_trailing(c_skip, y.pos.ndim - 1)

        y_scaled = y.clone()
        y_scaled.pos = y.pos * c_in

        if self.save_tensors and self.global_step % 100 == 0:
            torch.save(y_scaled.pos, f"{self.save_dir}/{save_prefix}_y_pos_scaled_{self.global_step}.pt")
            torch.save(y_scaled.edge_index, f"{self.save_dir}/{save_prefix}_y_edges_{self.global_step}.pt")

        xhat = y.clone()
        xhat.pos = c_skip * y.pos + c_out * self.g(y_scaled, c_noise, self.effective_radial_cutoff(sigma)).pos
        return xhat

    def xhat(self, y: torch.Tensor, sigma: Union[float, torch.Tensor], save_prefix: str = ""):
        if self.mean_center_input:
            y = mean_center(y)

        xhat = self.xhat_normalized(y, sigma, save_prefix=save_prefix)

        # Mean center the prediction.
        if self.mean_center_output:
            xhat = mean_center(xhat)

        if self.save_tensors and self.global_step % 100 == 0:
            torch.save(xhat.pos, f"{self.save_dir}/{save_prefix}_xhat_pos_{self.global_step}.pt")

        return xhat

    def noise_and_denoise(
        self,
        x: torch_geometric.data.Batch,
        sigma: Union[float, torch.Tensor],
        align_noisy_input: bool,
        save_prefix: str = "",
    ) -> Tuple[torch_geometric.data.Batch, torch_geometric.data.Batch]:

        with torch.no_grad():
            y = self.add_noise(x, sigma)
            if self.mean_center_input:
                y = mean_center(y)

            if self.save_tensors and self.global_step % 100 == 0:
                torch.save(x.pos, f"{self.save_dir}/{save_prefix}_x_pos_{self.global_step}.pt")
                torch.save(y.pos, f"{self.save_dir}/{save_prefix}_y_pos_{self.global_step}.pt")

            # Aligning each batch.
            if align_noisy_input:
                y = align_A_to_B_batched(y, x)
                if self.save_tensors and self.global_step % 100 == 0:
                    torch.save(y.pos, f"{self.save_dir}/{save_prefix}_y_pos_aligned_{self.global_step}.pt")

        xhat = self.xhat(y, sigma, save_prefix=save_prefix)
        return xhat, y

    def compute_loss(
        self,
        x: torch_geometric.data.Batch,
        xhat: torch.Tensor,
        sigma: Union[float, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        save_prefix: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mean_center_input:
            x = mean_center(x)

        xhat, _ = self.noise_and_denoise(x, sigma, align_noisy_input=align_noisy_input, save_prefix=save_prefix)
        return self.compute_loss(x, xhat, sigma)

    def training_step(self, batch: torch_geometric.data.Batch, batch_idx: int):
        sigma = self.sigma_distribution.sample().to(batch.pos.device)
        loss, aux = self.noise_and_compute_loss(
            batch, sigma, align_noisy_input=self.align_noisy_input_during_training, save_prefix="train"
        )

        # Average the loss over all graphs.
        loss = loss.mean()
        self.log("train/loss", loss, prog_bar=False, batch_size=batch.num_graphs, sync_dist=True)
    
        for key, value in aux.items():
            value = value.mean()
            self.log(f"train/{key}", value, prog_bar=False, batch_size=batch.num_graphs, sync_dist=True)

        return {
            "loss": loss,
            "sigma": sigma,
            **aux,
        }

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int):
        sigma = self.sigma_distribution.sample().to(batch.pos.device)
        loss, aux = self.noise_and_compute_loss(
            batch, sigma, align_noisy_input=self.align_noisy_input_during_training, save_prefix="val"
        )

        # Average the loss over all graphs.
        loss = loss.mean()
        self.log("val/loss", loss, prog_bar=False, batch_size=batch.num_graphs, sync_dist=True)

        for key, value in aux.items():
            value = value.mean()
            self.log(f"val/{key}", value, prog_bar=False, batch_size=batch.num_graphs, sync_dist=True)

        return {
            "loss": loss,
            "sigma": sigma,
            **aux,
        }

    def configure_optimizers(self):
        optimizer = self.optim_factory(params=self.parameters())

        out = {"optimizer": optimizer}
        if self.lr_scheduler_config:
            scheduler = self.lr_scheduler_config.pop("scheduler")
            out["lr_scheduler"] = {
                "scheduler": scheduler(optimizer),
                **self.lr_scheduler_config,
            }

        return out
