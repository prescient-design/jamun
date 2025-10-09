from collections.abc import Callable

import torch
import torch.nn as nn
import torch_geometric.data


class MLP(nn.Module):
    """A simple MLP architecture with noise conditioning."""

    def __init__(
        self,
        atom_embedder_factory: Callable[..., torch.nn.Module],
        n_layers: int,
        input_dims: int,
        embed_dims: int,
        noise_input_dims: int,
        num_nodes: int,
    ):
        super().__init__()
        self.atom_embedder = atom_embedder_factory()
        self.input_dims = input_dims
        self.embed_dims = embed_dims
        self.num_nodes = num_nodes

        # Since we concatenate atom embeddings to input features, adjust input dimensions.
        input_dims += self.atom_embedder.irreps_out.dim

        self.input_projection = nn.Linear(input_dims, embed_dims)
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(num_nodes * (embed_dims + noise_input_dims), num_nodes * embed_dims),
                    nn.GELU(),
                )
            )
            self.layer_norms.append(nn.LayerNorm(num_nodes * embed_dims))
        self.output_projection = nn.Linear(num_nodes * embed_dims, num_nodes * self.input_dims)

    def forward(
        self,
        pos: torch.Tensor,
        topology: torch_geometric.data.Batch,
        batch: torch.Tensor,
        num_graphs: int,
        c_noise: torch.Tensor,
        c_in: torch.Tensor,
    ) -> torch.Tensor:
        del c_in
        c_noise = c_noise.unsqueeze(0).expand(pos.shape[0], -1)

        # Concatenate position and atom embeddings
        x = torch.cat([pos, self.atom_embedder(topology)], dim=-1)

        # Project to embedding dimension and flatten per graph
        x = self.input_projection(x)
        x = x.view(num_graphs, self.num_nodes * self.embed_dims)
        c_noise = c_noise.view(num_graphs, self.num_nodes * c_noise.shape[-1])

        # Process through MLP layers
        for layer, layer_norm in zip(self.layers, self.layer_norms):
            x_with_noise = torch.cat([x, c_noise], dim=-1)
            x = layer(x_with_noise) + x
            x = layer_norm(x)

        # Output projection
        x = self.output_projection(x)
        x = x.view(num_graphs * self.num_nodes, self.input_dims)

        return x
