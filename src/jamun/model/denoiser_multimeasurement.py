import logging
from collections.abc import Callable

import e3tools
import lightning.pytorch as pl
import numpy as np
import torch
import torch_geometric
from e3tools import scatter

from jamun.utils import mean_center, unsqueeze_trailing
from jamun.utils.align import kabsch_algorithm


class DenoiserMultimeasurement(pl.LightningModule):
    """The main denoiser mode with conditional architecture."""

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
        conditioner: Callable[..., list[torch.Tensor]] = None,
        multimeasurement: bool = False,
        N_measurements_hidden: int = 1,
        N_measurements: int = 1,
        max_graphs_per_batch: int = None,
        rotational_augmentation: bool = False,
        alignment_correction_order: int = 0,
        pass_topology_as_atom_graphs: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Let us control the optimization process only if we need to chunk batches.
        self.automatic_optimization = max_graphs_per_batch is None

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

        self.multimeasurement = multimeasurement
        self.N_measurements_hidden = N_measurements_hidden
        self.N_measurements = N_measurements
        self.max_graphs_per_batch = max_graphs_per_batch
        if not self.automatic_optimization:
            py_logger.info(f"Manual optimization enabled with micro-batch size of {self.max_graphs_per_batch} graphs.")

    def on_before_optimizer_step(self, optimizer):
        # Log gradients and parameters.
        for name, param in self.named_parameters():
            self.log(f"parameter_norms/{name}", param.norm(), sync_dist=True)
            if param.grad is not None:
                self.log(f"gradient_norms/{name}", param.grad.norm(), sync_dist=True)

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
                A_aligned.hidden_state.append(kabsch_algorithm(A.hidden_state[i], B.pos, A.batch, A.num_graphs))
        return A_aligned

    def _mean_center_hidden_states(self, data: torch_geometric.data.Batch):
        if hasattr(data, "hidden_state") and data.hidden_state is not None:
            for i in range(len(data.hidden_state)):
                mean = scatter(data.hidden_state[i], data.batch, dim=0, reduce="mean")
                data.hidden_state[i] = data.hidden_state[i] - mean[data.batch]
        return data

    def _prepare_noisy_batch(
        self,
        x: torch_geometric.data.Batch,
        sigma: float | torch.Tensor,
        align_noisy_input: bool,
    ):
        """Prepare a batch of noisy graphs and their targets."""
        with torch.no_grad():
            if self.mean_center:
                x_processed = mean_center(x)
                x_processed = self._mean_center_hidden_states(x_processed)
            else:
                x_processed = x

            sigma_tensor = torch.as_tensor(sigma).to(x_processed.pos.device)

            y = self.add_noise_hiddens(x_processed, self.N_measurements_hidden, self.N_measurements, sigma_tensor)

            x_list = x_processed.to_data_list()
            repeated_x_list = [
                graph.clone() for graph in x_list for _ in range(self.N_measurements_hidden * self.N_measurements)
            ]
            x_target = torch_geometric.data.Batch.from_data_list(repeated_x_list).to(x_processed.pos.device)

            if self.mean_center:
                y = mean_center(y)
                y = self._mean_center_hidden_states(y)

            if align_noisy_input:
                y = self._align_A_to_B_batched_with_hidden_states(y, x_target)

        return y, x_target

    def conditioner_default(self, y: torch_geometric.data.Batch) -> list[torch.Tensor]:
        conditioned_structures = [y.pos]  # Return complete list starting with current position
        return conditioned_structures

    def conditioner(self, y: torch_geometric.data.Batch) -> list[torch.Tensor]:
        if self.conditioning_module is None:
            return self.conditioner_default(y)
        elif callable(self.conditioning_module):
            return self.conditioning_module(y)
        else:
            raise ValueError("Conditioner must be a callable or None")

    def add_noise(self, x: torch_geometric.data.Batch, sigma: float | torch.Tensor) -> torch_geometric.data.Batch:
        # pos [B, ...]
        sigma = unsqueeze_trailing(sigma, x.pos.ndim)

        y = x.clone()
        if self.add_fixed_ones:
            noise = torch.ones_like(x.pos)
            if hasattr(x, "hidden_state") and x.hidden_state is not None:
                hidden_noise = [torch.randn_like(x.hidden_state[i]) for i in range(len(x.hidden_state))]
            else:
                hidden_noise = []
        elif self.add_fixed_noise:
            torch.manual_seed(0)
            num_batches = x.batch.max().item() + 1
            if len(x.pos.shape) == 2:
                num_nodes_per_batch = x.pos.shape[0] // num_batches
                noise = torch.randn_like(x.pos[:num_nodes_per_batch]).repeat(num_batches, 1)
                if hasattr(x, "hidden_state") and x.hidden_state is not None:
                    hidden_noise = [
                        torch.randn_like(x.hidden_state[i][:num_nodes_per_batch]).repeat(num_batches, 1)
                        for i in range(len(x.hidden_state))
                    ]
                else:
                    hidden_noise = []
            if len(x.pos.shape) == 3:
                num_nodes_per_batch = x.pos.shape[1]
                noise = torch.randn_like(x.pos[0]).repeat(num_batches, 1, 1)
                if hasattr(x, "hidden_state") and x.hidden_state is not None:
                    hidden_noise = [
                        torch.randn_like(x.hidden_state[i][0]).repeat(num_batches, 1, 1)
                        for i in range(len(x.hidden_state))
                    ]
                else:
                    hidden_noise = []
        else:
            noise = torch.randn_like(x.pos)
            if hasattr(x, "hidden_state") and x.hidden_state is not None:
                hidden_noise = [torch.randn_like(x.hidden_state[i]) for i in range(len(x.hidden_state))]
            else:
                hidden_noise = []
        y.pos = x.pos + sigma * noise
        if hasattr(y, "hidden_state") and y.hidden_state is not None and hidden_noise:
            for i in range(len(y.hidden_state)):
                y.hidden_state[i] = x.hidden_state[i] + sigma * hidden_noise[i]
        if torch.rand(()) < self.mirror_augmentation_rate:
            y.pos = -y.pos
        return y

    def add_noise_hiddens(
        self,
        x: torch_geometric.data.Batch,
        N_measurements_hidden: int,
        N_measurements: int,
        sigma: float | torch.Tensor,
    ) -> torch_geometric.data.Batch:
        """
        Makes N_measurements_hidden number of noisy copies of the hidden states of x
        and then for every noisy copy, makes N_measurements number of noisy copies of the positions of x.

        Args:
            x (Batch): A torch_geometric Batch object. Must have `pos` and `hidden_state` attributes.
                        `hidden_state` is expected to be a list of tensors.
            N_measurements_hidden (int): Number of noisy copies of hidden states.
            N_measurements (int): Number of noisy copies of positions for each noisy hidden state.
            sigma (float or torch.Tensor): The standard deviation of the Gaussian noise to add.

        Returns:
            Batch: A new Batch object containing all the noisy copies.
        """
        x_list = x.to_data_list()
        noisy_y_list = []

        for graph in x_list:
            for _ in range(N_measurements_hidden):
                # Create a noisy version of the hidden state
                noisy_hidden_state = []
                if hasattr(graph, "hidden_state") and graph.hidden_state is not None:
                    for hs_tensor in graph.hidden_state:
                        noise = torch.randn_like(hs_tensor) * sigma
                        noisy_hidden_state.append(hs_tensor + noise)

                for _ in range(N_measurements):
                    noisy_graph = graph.clone()

                    # Add noise to positions
                    pos_noise = torch.randn_like(graph.pos) * sigma
                    noisy_graph.pos = graph.pos + pos_noise

                    # Assign the noisy hidden state
                    if hasattr(graph, "hidden_state") and graph.hidden_state is not None:
                        noisy_graph.hidden_state = [hs.clone() for hs in noisy_hidden_state]

                    noisy_y_list.append(noisy_graph)

        return torch_geometric.data.Batch.from_data_list(noisy_y_list)

    def score(self, y: torch_geometric.data.Batch, sigma: float | torch.Tensor) -> torch_geometric.data.Batch:
        """Compute the score function."""
        sigma = torch.as_tensor(sigma).to(y.pos)
        return (self.xhat(y, sigma).pos - y.pos) / (unsqueeze_trailing(sigma, y.pos.ndim - 1) ** 2)

    def normalization_factors(self, sigma: float, D: int = 3) -> tuple[float, float, float, float]:
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

    def effective_radial_cutoff(self, sigma: float | torch.Tensor) -> torch.Tensor:
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

    def xhat_normalized(self, y: torch_geometric.data.Batch, sigma: float | torch.Tensor) -> torch_geometric.data.Batch:
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
            if hasattr(y, "hidden_state") and y.hidden_state is not None:
                scaled_hidden_state = []
                for positions in y.hidden_state:
                    scaled_hidden_state.append(positions * c_in)
                y_scaled.hidden_state = scaled_hidden_state

        with torch.cuda.nvtx.range("clone_y"):
            xhat = y.clone()

        with torch.cuda.nvtx.range("conditioning"):
            conditioned_structures = self.conditioner(y_scaled)

        with torch.cuda.nvtx.range("g"):
            g_pred = self.g(
                torch.cat([*conditioned_structures], dim=-1),
                topology=y_scaled,
                c_noise=c_noise,
                effective_radial_cutoff=radial_cutoff,
            )

        xhat.pos = c_skip * y.pos + c_out * g_pred
        if hasattr(y, "hidden_state") and y.hidden_state is not None:
            xhat.hidden_state = [y.pos, *y.hidden_state[:-1]]  # the hidden state updates!
        return xhat

    def xhat(self, y: torch.Tensor, sigma: float | torch.Tensor):
        """Compute the denoised prediction."""
        if self.mean_center:
            with torch.cuda.nvtx.range("mean_center_y"):
                y = mean_center(y)
                y = self._mean_center_hidden_states(y)

        with torch.cuda.nvtx.range("xhat_normalized"):
            xhat = self.xhat_normalized(y, sigma)

        # Mean center the prediction.
        if self.mean_center:
            with torch.cuda.nvtx.range("mean_center_xhat"):
                xhat = mean_center(xhat)
                xhat = self._mean_center_hidden_states(xhat)

        return xhat

    def noise_and_denoise(
        self,
        x: torch_geometric.data.Batch,
        sigma: float | torch.Tensor,
        align_noisy_input: bool,
    ) -> tuple[torch_geometric.data.Batch, torch_geometric.data.Batch, torch_geometric.data.Batch]:
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

            if self.multimeasurement:
                with torch.cuda.nvtx.range("add_noise_hiddens"):
                    y = self.add_noise_hiddens(x_processed, self.N_measurements_hidden, self.N_measurements, sigma)

                # Repeat x_processed to match y's batch size for alignment and loss calculation.
                x_list = x_processed.to_data_list()
                repeated_x_list = [
                    graph.clone() for graph in x_list for _ in range(self.N_measurements_hidden * self.N_measurements)
                ]
                x_target = torch_geometric.data.Batch.from_data_list(repeated_x_list).to(x_processed.pos.device)

            else:
                with torch.cuda.nvtx.range("add_noise"):
                    y = self.add_noise(x_processed, sigma)
                x_target = x_processed.clone()

            if self.mean_center:
                with torch.cuda.nvtx.range("mean_center_y"):
                    y = mean_center(y)
                    y = self._mean_center_hidden_states(y)

            # Aligning each batch.
            if align_noisy_input:
                with torch.cuda.nvtx.range("align_A_to_B_batched"):
                    y = self._align_A_to_B_batched_with_hidden_states(y, x_target)

        with torch.cuda.nvtx.range("xhat"):
            xhat = self.xhat(y, sigma)

        return x_target, xhat, y

    def compute_loss(
        self,
        x: torch_geometric.data.Batch,
        xhat: torch.Tensor,
        sigma: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the loss."""
        if self.mean_center:
            with torch.cuda.nvtx.range("mean_center_x"):
                x = mean_center(x)
                x = self._mean_center_hidden_states(x)

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
        sigma: float | torch.Tensor,
        align_noisy_input: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add noise to the input and compute the loss."""
        x_target, xhat, _ = self.noise_and_denoise(x, sigma, align_noisy_input=align_noisy_input)
        return self.compute_loss(x_target, xhat, sigma)

    def _automatic_step(self, batch: torch_geometric.data.Batch, stage: str):
        """The standard step for automatic optimization."""
        align_noisy_input = (
            self.align_noisy_input_during_training if stage == "train" else self.align_noisy_input_during_evaluation
        )
        sigma = self.sigma_distribution.sample().to(self.device)

        loss, aux = self.noise_and_compute_loss(
            batch,
            sigma,
            align_noisy_input=align_noisy_input,
        )  # check if the loss is nan. if nan then save the model, and the batch and see what went on.
        # if torch.isnan(loss.sum()):
        #     print(f"Loss is nan at step {self.global_step}")
        #     print(f"Batch: {batch}")
        #     print(f"Sigma: {sigma}")
        #     print(f"Align noisy input: {align_noisy_input}")
        #     print(f"Loss: {loss}")
        #     print(f"Aux: {aux}")
        #     # Create debug directory if it doesn't exist
        #     debug_dir = f"/homefs/home/sules/jamun/debug_nan_loss_step_{self.global_step}"
        #     os.makedirs(debug_dir, exist_ok=True)
        #
        #     # Save model checkpoint
        #     checkpoint_path = os.path.join(debug_dir, "model_nan_loss.ckpt")
        #     self.trainer.save_checkpoint(checkpoint_path)
        #     print(f"Model saved to {checkpoint_path}")
        #
        #     torch.save(batch, debug_dir + "/batch_nan_loss.pt")
        #
        #     # Optionally raise an exception to stop training
        #     raise RuntimeError(f"NaN loss detected at step {self.global_step}. Debug files saved to {debug_dir}")

        # Average the loss and other metrics over all graphs.
        with torch.cuda.nvtx.range("mean_over_graphs"):
            aux["loss"] = loss
            for key in aux:
                aux[key] = aux[key].mean()
                if stage == "train":
                    self.log(
                        f"train/{key}",
                        aux[key],
                        prog_bar=False,
                        batch_size=batch.num_graphs,
                        sync_dist=False,
                    )
                elif stage == "val":
                    self.log(
                        f"val/{key}",
                        aux[key],
                        prog_bar=(key == "scaled_rmsd"),
                        batch_size=batch.num_graphs,
                        sync_dist=True,
                    )
                else:
                    continue

        return {
            "sigma": sigma,
            **aux,
        }

    def _manual_step(self, batch: torch_geometric.data.Batch, stage: str):
        """A shared step for training and validation with manual optimization."""
        sigma = self.sigma_distribution.sample().to(self.device)
        align_noisy_input = (
            self.align_noisy_input_during_training if stage == "train" else self.align_noisy_input_during_evaluation
        )

        y, x_target = self._prepare_noisy_batch(batch, sigma, align_noisy_input)

        y_list = y.to_data_list()
        x_target_list = x_target.to_data_list()

        chunk_size = self.max_graphs_per_batch
        num_chunks = (len(y_list) + chunk_size - 1) // chunk_size

        all_aux = []
        opt = self.optimizers() if stage == "train" else None

        # print(f"Processing {num_chunks} chunks of size {chunk_size} for {stage}...")
        for i in range(num_chunks):
            start_index = i * chunk_size
            end_index = min(start_index + chunk_size, len(y_list))

            y_micro_batch_list = y_list[start_index:end_index]
            x_target_micro_batch_list = x_target_list[start_index:end_index]

            if not y_micro_batch_list:
                continue

            y_micro_batch = torch_geometric.data.Batch.from_data_list(y_micro_batch_list)
            x_target_micro_batch = torch_geometric.data.Batch.from_data_list(x_target_micro_batch_list)

            xhat_micro_batch = self.xhat(y_micro_batch, sigma)

            loss, aux = self.compute_loss(x_target_micro_batch, xhat_micro_batch, sigma)

            with torch.cuda.nvtx.range("mean_over_graphs"):
                aux["loss"] = loss
                for key in aux:
                    aux[key] = aux[key].mean()
            if stage == "train":
                # Scale loss by number of chunks to match automatic optimization gradients
                scaled_loss = aux["loss"] / num_chunks
                opt.zero_grad()
                self.manual_backward(scaled_loss)
                opt.step()
            all_aux.append(aux)

        avg_aux = {}
        with torch.no_grad():
            if all_aux:
                for key in all_aux[0]:
                    avg_aux[key] = torch.tensor([d[key] for d in all_aux]).mean()
                log_opts = {
                    "prog_bar": (stage == "val" and "scaled_rmsd" in avg_aux),
                    "batch_size": len(y_list),
                    "sync_dist": (stage == "val"),  # Only sync for validation
                }

                # Ensure training metrics are always logged
                if stage == "train":
                    log_opts["on_step"] = True
                    log_opts["on_epoch"] = True

                for key, value in avg_aux.items():
                    self.log(f"{stage}/{key}", value, **log_opts)

        return {"sigma": sigma, **avg_aux}

    def training_step(self, batch: torch_geometric.data.Batch, batch_idx: int):
        """Called during training."""
        if self.automatic_optimization:
            return self._automatic_step(batch, "train")
        else:
            # print(f"Manual optimization enabled for training step {batch_idx}.")
            return self._manual_step(batch, "train")

    def validation_step(self, batch: torch_geometric.data.Batch, batch_idx: int):
        """Called during validation."""
        if self.automatic_optimization:
            return self._automatic_step(batch, "val")
        else:
            return self._manual_step(batch, "val")

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
