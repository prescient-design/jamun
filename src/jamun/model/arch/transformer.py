from collections.abc import Callable

import torch
import torch.nn as nn
import torch_geometric.data


class NoiseConditionalSelfAttention(nn.Module):
    """Self-attention layer with noise conditioning."""

    def __init__(self, input_dims: int, embed_dims: int, n_heads: int, noise_input_dims: int):
        super().__init__()
        self.embed_dims = embed_dims
        self.n_heads = n_heads
        self.head_dim = embed_dims // n_heads

        assert embed_dims % n_heads == 0, "embed_dims must be divisible by n_heads"

        self.Q = nn.Linear(input_dims + noise_input_dims, embed_dims)
        self.K = nn.Linear(input_dims + noise_input_dims, embed_dims)
        self.V = nn.Linear(input_dims + noise_input_dims, embed_dims)
        self.output_projection = nn.Linear(embed_dims, embed_dims)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        num_nodes, _ = x.shape

        # Concatenate input with noise conditioning
        x_with_noise = torch.cat([x, c_noise], dim=-1)

        # Compute Q, K, V
        Q = self.Q(x_with_noise)  # [num_nodes, embed_dims]
        K = self.K(x_with_noise)  # [num_nodes, embed_dims]
        V = self.V(x_with_noise)  # [num_nodes, embed_dims]

        # Reshape for multi-head attention
        Q = Q.view(num_nodes, self.n_heads, self.head_dim).transpose(-3, -2)
        K = K.view(num_nodes, self.n_heads, self.head_dim).transpose(-3, -2)
        V = V.view(num_nodes, self.n_heads, self.head_dim).transpose(-3, -2)

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        logits = (Q @ K.transpose(-2, -1)) * scale

        # Mask attention scores for different graphs in the batch
        mask = batch.unsqueeze(0) != batch.unsqueeze(1)
        mask = mask.unsqueeze(0).expand(self.n_heads, -1, -1)  # [n_heads, num_nodes, num_nodes]
        # print("Mask", mask.shape)
        # print("Logits before mask", logits.shape)
        logits = logits.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(logits, dim=-1)
        attn_output = attn_weights @ V

        # Concatenate heads
        attn_output = attn_output.transpose(-3, -2).contiguous().view(num_nodes, self.embed_dims)

        # Output projection
        output = self.output_projection(attn_output)
        return output


class Transformer(nn.Module):
    """A simple Transformer architecture with noise conditioning."""

    def __init__(
        self,
        atom_embedder_factory: Callable[..., torch.nn.Module],
        n_layers: int,
        input_dims: int,
        embed_dims: int,
        noise_input_dims: int,
        n_heads: int,
        num_nodes: int,
    ):
        super().__init__()
        self.atom_embedder = atom_embedder_factory()
        self.input_dims = input_dims
        self.embed_dims = embed_dims
        self.num_nodes = num_nodes

        # Since we concatenate atom embeddings to input features, adjust input dimensions.
        input_dims += self.atom_embedder.irreps_out.dim

        # Input projection to match embedding dimension
        self.input_projection = nn.Linear(input_dims, embed_dims)

        self.self_attention_layers = nn.ModuleList()
        self.self_attention_layer_norms = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_layer_norms = nn.ModuleList()

        for _ in range(n_layers):
            self.self_attention_layers.append(
                NoiseConditionalSelfAttention(
                    input_dims=embed_dims,
                    embed_dims=embed_dims,
                    n_heads=n_heads,
                    noise_input_dims=noise_input_dims,
                )
            )
            self.self_attention_layer_norms.append(nn.LayerNorm(embed_dims))
            self.ffn_layer_norms.append(nn.LayerNorm(embed_dims))

            # Feed-forward network
            self.ffn_layers.append(
                nn.Sequential(nn.Linear(embed_dims, embed_dims * 4), nn.ReLU(), nn.Linear(embed_dims * 4, embed_dims))
            )

        self.output_projection = nn.Linear(embed_dims, self.input_dims)

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

        # c_noise: [num_nodes, noise_input_dims]
        # pos: [num_nodes, input_dims]
        x = torch.cat([pos, self.atom_embedder(topology)], dim=-1)

        # Project input to embedding dimension
        x = self.input_projection(x)

        for self_attention, self_attention_layer_norm, ffn, ffn_layer_norm in zip(
            self.self_attention_layers, self.self_attention_layer_norms, self.ffn_layers, self.ffn_layer_norms
        ):
            # Self-attention with residual connection and layer norm
            attn_output = self_attention(x, c_noise, batch, num_graphs)
            x = x + attn_output
            x = self_attention_layer_norm(x)

            # Feed-forward with residual connection and layer norm
            ffn_output = ffn(x)
            x = x + ffn_output
            x = ffn_layer_norm(x)

        # Final projection back to input dimensions
        x = self.output_projection(x)

        return x
