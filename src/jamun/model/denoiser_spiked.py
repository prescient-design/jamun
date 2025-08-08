import logging
from typing import Callable, Dict, Optional, Tuple, Union

import e3tools
import lightning.pytorch as pl
import numpy as np
import torch
import torch_geometric
from e3tools import scatter
from tqdm import tqdm

from jamun.utils import align_A_to_B_batched, mean_center, unsqueeze_trailing
from jamun.utils.align import kabsch_algorithm


class DenoiserSpiked(pl.LightningModule):
    """The main denoiser model with conditional architecture that includes clean sample conditioning."""

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
        normalization_type: Optional[str] = "JAMUN",
        sigma_data: Optional[float] = None,  # Only used if normalization_type is "EDM"
        lr_scheduler_config: Optional[Dict] = None,
        use_torch_compile: bool = True,
        torch_compile_kwargs: Optional[Dict] = None,
        conditioner: Callable[..., list[torch.Tensor]] = None,
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
        self.conditioning_module = conditioner
        if self.conditioning_module is not None and not callable(self.conditioning_module):
            raise ValueError("Conditioner must be a callable or None")
        py_logger.info(f"Conditioner: {self.conditioning_module}")

    def on_before_optimizer_step(self, optimizer):
        # Log gradients and parameters.
        for name, param in self.named_parameters():
            self.log(f"parameter_norms/{name}", param.norm(), sync_dist=True)
            if param.grad is not None:
                self.log(f"gradient_norms/{name}", param.grad.norm(), sync_dist=True)

    def conditioner_default(self, y: torch_geometric.data.Batch, x_clean: torch_geometric.data.Batch = None) -> list[torch.Tensor]:
        conditioned_structures = [y.pos]  # Return complete list starting with current position
        if x_clean is not None:
            conditioned_structures.append(x_clean.pos)  # Add clean sample positions
        return conditioned_structures

    def conditioner(self, y: torch_geometric.data.Batch, x_clean: torch_geometric.data.Batch = None) -> list[torch.Tensor]:
        if self.conditioning_module is None:
            return self.conditioner_default(y, x_clean)
        elif callable(self.conditioning_module):
            return self.conditioning_module(y, x_clean)
        else:
            raise ValueError("Conditioner must be a callable or None")
    
    def _align_A_to_B_batched_with_hidden_states(
        self, A: torch_geometric.data.Batch, B: torch_geometric.data.Batch
    ) -> torch_geometric.data.Batch:
        """Aligns each graph of A to the corresponding graph in B, including hidden states."""
        A_aligned = A.clone()

        # Align positions
        A_aligned.pos = kabsch_algorithm(A.pos, B.pos, A.batch, A.num_graphs)

            # Align hidden states
        if hasattr(A, "hidden_state") and A.hidden_state is not None:
            A_aligned.hidden_state = []
            for i in range(len(A.hidden_state)):
                A_aligned.hidden_state.append(kabsch_algorithm(
                    A.hidden_state[i], B.pos, A.batch, A.num_graphs
                ))
        return A_aligned

    def _mean_center_hidden_states(self, data: torch_geometric.data.Batch):
        if hasattr(data, "hidden_state") and data.hidden_state is not None:
            for i in range(len(data.hidden_state)):
                mean = scatter(data.hidden_state[i], data.batch, dim=0, reduce="mean")
                data.hidden_state[i] = data.hidden_state[i] - mean[data.batch]
        return data
    
    def add_noise(self, x: torch_geometric.data.Batch, sigma: Union[float, torch.Tensor]) -> torch_geometric.data.Batch:
        # pos [B, ...]
        sigma = unsqueeze_trailing(sigma, x.pos.ndim)

        y = x.clone()
        if self.add_fixed_ones:
            noise = torch.ones_like(x.pos)
            hidden_noise = [torch.randn_like(x.hidden_state[i]) for i in range(len(x.hidden_state))]
        elif self.add_fixed_noise:
            torch.manual_seed(0)
            num_batches = x.batch.max().item() + 1
            if len(x.pos.shape) == 2:
                num_nodes_per_batch = x.pos.shape[0] // num_batches
                noise = torch.randn_like((x.pos[:num_nodes_per_batch])).repeat(num_batches, 1)
                hidden_noise = [torch.randn_like((x.hidden_state[i][:num_nodes_per_batch])).repeat(num_batches, 1) for i in range(len(x.hidden_state))]
            if len(x.pos.shape) == 3:
                num_nodes_per_batch = x.pos.shape[1]
                noise = torch.randn_like((x.pos[0])).repeat(num_batches, 1, 1)
                hidden_noise = [torch.randn_like((x.hidden_state[i][0])).repeat(num_batches, 1, 1) for i in range(len(x.hidden_state))]
        else:
            noise = torch.randn_like(x.pos)
            hidden_noise = [torch.randn_like(x.hidden_state[i]) for i in range(len(x.hidden_state))]
        y.pos = x.pos + sigma * noise
        for i in range(len(y.hidden_state)):
            y.hidden_state[i] = x.hidden_state[i] + sigma * hidden_noise[i]
        if torch.rand(()) < self.mirror_augmentation_rate:
            y.pos = -y.pos
        return y

    def score(self, y: torch_geometric.data.Batch, sigma: Union[float, torch.Tensor], x_clean: torch_geometric.data.Batch) -> torch_geometric.data.Batch:
        """Compute the score function."""
        sigma = torch.as_tensor(sigma).to(y.pos)
        return (self.xhat(y, sigma, x_clean).pos - y.pos) / (unsqueeze_trailing(sigma, y.pos.ndim - 1) ** 2)

    def normalization_factors(self, sigma: float, D: int = 3) -> Tuple[float, float, float, float]:
        """Normalization factors for the input and output."""
        sigma = torch.as_tensor(sigma)

        if self.normalization_type is None:
            return 1.0, 0.0, 1.0, sigma

        if self.normalization_type == "EDM":
            c_skip = (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)
            c_out = sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)
            c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)
            c_noise = torch.log(sigma / self.sigma_data) * 0.25
            return c_in, c_skip, c_out, c_noise

        if self.normalization_type == "JAMUN":
            A = torch.as_tensor(self.average_squared_distance)
            B = torch.as_tensor(2 * D * sigma**2)

            c_in = 1.0 / torch.sqrt(A + B)
            c_skip = A / (A + B)
            c_out = torch.sqrt((A * B) / (A + B))
            c_noise = torch.log(sigma) / 4
            return c_in, c_skip, c_out, c_noise

        raise ValueError(f"Unknown normalization type: {self.normalization_type}")

    def loss_weight(self, sigma: float, D: int = 3) -> float:
        """Loss weight for this graph."""
        _, _, c_out, _ = self.normalization_factors(sigma, D)
        return 1 / (c_out**2)

    def effective_radial_cutoff(self, sigma: Union[float, torch.Tensor]) -> torch.Tensor:
        """Compute the effective radial cutoff for the noise level."""
        return torch.sqrt((self.max_radius**2) + 6 * (sigma**2))

    def add_edges(self, y: torch_geometric.data.Batch, radial_cutoff: float) -> torch_geometric.data.Batch:
        """Add edges to the graph based on the effective radial cutoff."""
        if y.get("edge_index") is not None:
            return y

        y = y.clone()
        if "batch" in y:
            batch = y["batch"]
        else:
            batch = torch.zeros(y.num_nodes, dtype=torch.long, device=self.device)

        with torch.cuda.nvtx.range("radial_graph"):
            radial_edge_index = e3tools.radius_graph(y.pos, radial_cutoff, batch)

        with torch.cuda.nvtx.range("concatenate_edges"):
            edge_index = torch.cat((radial_edge_index, y.bonded_edge_index), dim=-1)
            if y.bonded_edge_index.numel() == 0:
                bond_mask = torch.zeros(radial_edge_index.shape[1], dtype=torch.long, device=y.pos.device)
            else:
                bond_mask = torch.cat(
                    (
                        torch.zeros(radial_edge_index.shape[1], dtype=torch.long, device=y.pos.device),
                        torch.ones(y.bonded_edge_index.shape[1], dtype=torch.long, device=y.pos.device),
                    ),
                    dim=0,
                )

        y.edge_index = edge_index
        y.bond_mask = bond_mask
        return y

    def xhat_normalized(
        self, y: torch_geometric.data.Batch, sigma: Union[float, torch.Tensor], x_clean: torch_geometric.data.Batch
    ) -> torch_geometric.data.Batch:
        """Compute the denoised prediction using the normalization factors from JAMUN."""
        sigma = torch.as_tensor(sigma).to(y.pos)
        D = y.pos.shape[-1]

        # Compute the normalization factors.
        with torch.cuda.nvtx.range("normalization_factors"):
            c_in, c_skip, c_out, c_noise = self.normalization_factors(sigma, D)
        radial_cutoff = self.effective_radial_cutoff(sigma) / c_in

        # Adjust dimensions.
        c_in = unsqueeze_trailing(c_in, y.pos.ndim - 1)
        c_skip = unsqueeze_trailing(c_skip, y.pos.ndim - 1)
        c_out = unsqueeze_trailing(c_out, y.pos.ndim - 1)
        c_noise = c_noise.unsqueeze(0)

        # Add edges to the graph.
        with torch.cuda.nvtx.range("add_edges"):
            y = self.add_edges(y, radial_cutoff)

        with torch.cuda.nvtx.range("scale_y"):
            y_scaled = y.clone()
            y_scaled.pos = y.pos * c_in
            # Manually copy hidden state
            if hasattr(y, "hidden_state") and y.hidden_state is not None:
                y_scaled.hidden_state = []
                for positions in y.hidden_state:
                    y_scaled.hidden_state.append(positions * c_in)

        # Keep clean sample unscaled
        with torch.cuda.nvtx.range("clone_y"):
            xhat = y.clone()
            # Manually copy hidden state
            if hasattr(y, "hidden_state") and y.hidden_state is not None:
                xhat.hidden_state = [h.clone() for h in y.hidden_state]

        with torch.cuda.nvtx.range("conditioning"): 
            conditioned_structures = self.conditioner(y_scaled, x_clean)
            # print(f"Conditioner is working, number of conditioned structures: {len(conditioned_structures)}")
        with torch.cuda.nvtx.range("g"):    
            g_pred = self.g(torch.cat([*conditioned_structures], dim=-1), topology=y_scaled, \
                            c_noise=c_noise, effective_radial_cutoff=radial_cutoff)

        xhat.pos = c_skip * y.pos + c_out * g_pred
        if hasattr(y, "hidden_state") and y.hidden_state is not None:
            xhat.hidden_state = [y.pos, *y.hidden_state[:-1]]
        return xhat

    def xhat(self, y: torch.Tensor, sigma: Union[float, torch.Tensor], x_clean: torch_geometric.data.Batch):
        """Compute the denoised prediction."""
        if self.mean_center:
            with torch.cuda.nvtx.range("mean_center_y"):
                y = mean_center(y)
                y = self._mean_center_hidden_states(y)
            with torch.cuda.nvtx.range("mean_center_x_clean"):
                x_clean = mean_center(x_clean)
                x_clean = self._mean_center_hidden_states(x_clean)

        with torch.cuda.nvtx.range("xhat_normalized"):
            xhat = self.xhat_normalized(y, sigma, x_clean)

        # Mean center the prediction.
        if self.mean_center:
            with torch.cuda.nvtx.range("mean_center_xhat"):
                xhat = mean_center(xhat)

        return xhat

    def noise_and_denoise(
        self,
        x: torch_geometric.data.Batch,
        sigma: Union[float, torch.Tensor],
        align_noisy_input: bool,
    ) -> Tuple[torch_geometric.data.Batch, torch_geometric.data.Batch, torch_geometric.data.Batch]:
        """
        Add noise to the input and denoise it.
        Returns the target for the loss, the prediction, and the noisy input.
        """
        with torch.no_grad():
            if self.mean_center:
                # Operate on a clone to avoid side effects on the original batch object.
                x_processed = mean_center(x)
                x_processed = self._mean_center_hidden_states(x_processed)
            else:
                x_processed = x

            sigma = torch.as_tensor(sigma).to(x_processed.pos)

            with torch.cuda.nvtx.range("add_noise"):
                y = self.add_noise(x_processed, sigma)
            x_target = x_processed.clone()
            # Manually copy hidden state
            if hasattr(x_processed, "hidden_state") and x_processed.hidden_state is not None:
                x_target.hidden_state = [h.clone() for h in x_processed.hidden_state]

            if self.mean_center:
                with torch.cuda.nvtx.range("mean_center_y"):
                    y = mean_center(y)
                    y = self._mean_center_hidden_states(y)

            # Aligning each batch.
            if align_noisy_input:
                with torch.cuda.nvtx.range("align_A_to_B_batched"):
                    y = self._align_A_to_B_batched_with_hidden_states(y, x_target)

        # KEY CHANGE: Pass both noisy sample (y) AND clean sample (x_target) to xhat
        with torch.cuda.nvtx.range("xhat"):
            xhat = self.xhat(y, sigma, x_target)

        return x_target, xhat, y

    def compute_loss(
        self,
        x: torch_geometric.data.Batch,
        xhat: torch.Tensor,
        sigma: Union[float, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the loss."""
        if self.mean_center:
            with torch.cuda.nvtx.range("mean_center_x"):
                x = mean_center(x)

        D = xhat.pos.shape[-1]

        # Compute the raw loss.
        with torch.cuda.nvtx.range("raw_coordinate_loss"):
            raw_coordinate_loss = (xhat.pos - x.pos).pow(2).sum(dim=-1)

        # Take the mean over each graph.
        with torch.cuda.nvtx.range("mean_over_graphs"):
            mse = scatter(raw_coordinate_loss, x.batch, dim=0, dim_size=x.num_graphs, reduce="mean")

        # Compute the scaled RMSD.
        with torch.cuda.nvtx.range("scaled_rmsd"):
            rmsd = torch.sqrt(mse)
            scaled_rmsd = rmsd / (sigma * np.sqrt(D))

        # Account for the loss weight across graphs and noise levels.
        with torch.cuda.nvtx.range("loss_weight"):
            loss = mse * x.loss_weight
            loss = loss * self.loss_weight(sigma, D)

        return loss, {
            "mse": mse,
            "rmsd": rmsd,
            "scaled_rmsd": scaled_rmsd,
        }

    def noise_and_compute_loss(
        self,
        x: torch_geometric.data.Batch,
        sigma: Union[float, torch.Tensor],
        align_noisy_input: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add noise to the input and compute the loss."""
        x_target, xhat, _ = self.noise_and_denoise(x, sigma, align_noisy_input=align_noisy_input)
        return self.compute_loss(x_target, xhat, sigma)

    def _automatic_step(self, batch: torch_geometric.data.Batch, stage: str):
        """The standard step for automatic optimization."""
        align_noisy_input = self.align_noisy_input_during_training if stage == "train" else self.align_noisy_input_during_evaluation
        sigma = self.sigma_distribution.sample().to(self.device)
        
        loss, aux = self.noise_and_compute_loss(
            batch,
            sigma,
            align_noisy_input=align_noisy_input,
        )

        # Average the loss and other metrics over all graphs.
        with torch.cuda.nvtx.range("mean_over_graphs"):
            aux["loss"] = loss
            for key in aux:
                aux[key] = aux[key].mean()
                if stage == "train":
                    self.log(f"train/{key}", aux[key], prog_bar=False, batch_size=batch.num_graphs, sync_dist=False)
                elif stage == "val":
                    self.log(
                f"val/{key}", aux[key], prog_bar=(key == "scaled_rmsd"), batch_size=batch.num_graphs, sync_dist=True
            )
                else:
                    continue


        return {
            "sigma": sigma,
            **aux,
        }


    def training_step(self, batch: torch_geometric.data.Batch, batch_idx: int):
        """Called during training."""
        return self._automatic_step(batch, "train")

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int):
        """Called during validation."""
        self._automatic_step(batch, "val")

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