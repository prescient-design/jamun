from typing import Dict, List

import einops
import torch
import torch.nn as nn
import torch_geometric

from jamun.utils import mean_center


class ModelSamplingWrapper:
    """Wrapper to sample positions from a model."""

    def __init__(self, model: nn.Module, init_graphs: torch_geometric.data.Data, sigma: float, recenter_on_init: bool = True):
        self._model = model
        self.init_graphs = init_graphs
        self.sigma = sigma
        
        # Apply mean centering if requested
        if recenter_on_init:
            self.init_graphs = mean_center(self.init_graphs)

    @property
    def device(self) -> torch.device:
        return self._model.device

    def sample_initial_noisy_positions(self) -> torch.Tensor:
        pos = self.init_graphs.pos
        pos = pos + torch.randn_like(pos) * self.sigma
        return pos

    def __getattr__(self, name):
        return getattr(self._model, name)

    def score(self, y, sigma, *args, **kwargs):
        return self._model.score(self.positions_to_graph(y), sigma)

    def xhat(self, y, sigma, *args, **kwargs):
        xhat_graph = self._model.xhat(self.positions_to_graph(y), sigma)
        return xhat_graph.pos

    def positions_to_graph(self, positions: torch.Tensor) -> torch_geometric.data.Data:
        """Wraps a tensor of positions to a graph with these positions as an attribute."""
        # Check input validity
        assert len(positions) == self.init_graphs.num_nodes, "The number of positions and nodes should be the same"
        assert positions.shape[1] == 3, "Positions tensor should have a shape of (n, 3)"

        input_graphs = self.init_graphs.clone()
        input_graphs.pos = positions

        # Save for debugging.
        self.input_graphs = input_graphs
        return input_graphs.to(positions.device)

    def unbatch_samples(self, samples: Dict[str, torch.Tensor]) -> List[torch_geometric.data.Data]:
        """Unbatch samples."""
        if "batch" not in self.init_graphs:
            raise ValueError("The initial graph does not have a batch attribute.")

        # Copy off the input graphs, to update attributes later.
        output_graphs = self.init_graphs.clone()
        output_graphs = torch_geometric.data.Batch.to_data_list(output_graphs)

        for key, value in samples.items():
            if value.ndim not in [2, 3]:
                # py_logger = logging.getLogger("jamun")
                # py_logger.info(f"Skipping unbatching of key {key} with shape {value.shape} as it is not 2D or 3D.")
                continue

            if value.ndim == 3:
                value = einops.rearrange(
                    value,
                    "num_frames atoms coords -> atoms num_frames coords",
                )

            unbatched_values = torch_geometric.utils.unbatch(value, self.init_graphs.batch)
            for output_graph, unbatched_value in zip(output_graphs, unbatched_values, strict=True):
                if key in output_graph:
                    raise ValueError(f"Key {key} already exists in the output graph.")

                if unbatched_value.shape[0] != output_graph.num_nodes:
                    raise ValueError(
                        f"Number of nodes in unbatched value ({unbatched_value.shape[0]}) for key {key} does not match "
                        f"number of nodes in output graph ({output_graph.num_nodes})."
                    )

                output_graph[key] = unbatched_value

        return output_graphs


class ModelSamplingWrapperMemory:
    """Wrapper for models that depend on a memory of states."""

    def __init__(self, model: nn.Module, init_graphs: torch_geometric.data.Data, sigma: float, recenter_on_init: bool = True):
        self._model = model
        self.init_graphs = init_graphs
        self.sigma = sigma
        
        # Apply mean centering if requested
        if recenter_on_init:
            # Mean center positions
            self.init_graphs = mean_center(self.init_graphs)
            
            # Mean center hidden states if they exist and aren't empty
            if hasattr(self.init_graphs, 'hidden_state') and self.init_graphs.hidden_state:
                for i in range(len(self.init_graphs.hidden_state)):
                    # Mean center each hidden state in-place
                    self.init_graphs.hidden_state[i] = self.init_graphs.hidden_state[i] - self.init_graphs.hidden_state[i].mean(dim=0, keepdim=True)

    @property
    def device(self) -> torch.device:
        return next(self._model.parameters()).device
    def sample_initial_noisy_positions(self) -> torch.Tensor:
        pos = self.init_graphs.pos
        pos = pos + torch.randn_like(pos) * self.sigma
        return pos
    
    def sample_initial_noisy_history(self) -> list:
        noisy_history = []
        for hidden_state in self.init_graphs.hidden_state:
            noisy_history.append(hidden_state + torch.randn_like(hidden_state) * self.sigma)
        return noisy_history
    
    def __getattr__(self, name):
        return getattr(self._model, name)

    def score(self, y, y_hist, sigma):
        graph = self.positions_to_graph(y, y_hist).to(self.device)
        return self._model.score(graph, sigma)

    def xhat(self, y, y_hist, sigma):
        graph = self.positions_to_graph(y, y_hist).to(self.device)
        xhat_graph = self._model.xhat(graph, sigma)
        return xhat_graph.pos

    def positions_to_graph(self, positions: torch.Tensor, y_hist: list) -> torch_geometric.data.Data:
        """Wraps positions to a graph and attaches the historical states."""
        assert len(positions) == self.init_graphs.num_nodes
        assert positions.shape[1] == 3
        input_graph = self.init_graphs.clone()
        input_graph.pos = positions
        input_graph.hidden_state = y_hist
        return input_graph.to(positions.device)

    def unbatch_samples(self, samples: Dict[str, torch.Tensor]) -> List[torch_geometric.data.Data]:
        """Unbatch samples."""
        if "batch" not in self.init_graphs:
            raise ValueError("The initial graph does not have a batch attribute.")

        # Copy off the input graphs, to update attributes later.
        output_graphs = self.init_graphs.clone()
        output_graphs = torch_geometric.data.Batch.to_data_list(output_graphs)

        for key, value in samples.items():
            if key == "y_hist" or key == "y_hist_traj":
                if key == "y_hist":
                    value = [value]
                value = torch.stack([torch.stack(traj, dim=1) for traj in value], dim=1)
            else:
                if hasattr(value, "ndim") and value.ndim not in [2, 3]:
                    # py_logger = logging.getLogger("jamun")
                    # py_logger.info(f"Skipping unbatching of key {key} with shape {value.shape} as it is not 2D or 3D.")
                    continue
                if hasattr(value, "ndim") and value.ndim == 3:
                    value = einops.rearrange(
                        value,
                        "num_frames atoms coords -> atoms num_frames coords",
                    )

            unbatched_values = torch_geometric.utils.unbatch(value, self.init_graphs.batch)
            for output_graph, unbatched_value in zip(output_graphs, unbatched_values, strict=True):
                if key in output_graph:
                    raise ValueError(f"Key {key} already exists in the output graph.")

                if unbatched_value.shape[0] != output_graph.num_nodes:
                    raise ValueError(
                        f"Number of nodes in unbatched value ({unbatched_value.shape[0]}) for key {key} does not match "
                        f"number of nodes in output graph ({output_graph.num_nodes})."
                    )
                if key == "y_hist":
                    unbatched_value = [t.squeeze(-2).squeeze(1) for t in torch.split(unbatched_value, 1, dim=-2)]
                if key == "y_hist_traj":
                    unbatched_value = [t.squeeze(-2) for t in torch.split(unbatched_value, 1, dim=-2)]
                output_graph[key] = unbatched_value

        return output_graphs