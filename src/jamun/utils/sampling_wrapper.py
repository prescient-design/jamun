from typing import Dict, List

import einops
import torch
import torch.nn as nn
import torch_geometric


class ModelSamplingWrapper:
    """Wrapper to sample positions from a model."""

    def __init__(self, model: nn.Module, init_graphs: torch_geometric.data.Data, sigma: float):
        self._model = model
        self.init_graphs = init_graphs
        self.sigma = sigma

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
