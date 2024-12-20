from typing import Callable, Optional, Union

import e3nn
import torch
import torch_geometric
from e3nn import o3

from jamun.model.atom_embedding import AtomEmbeddingWithResidueInformation, SimpleAtomEmbedding
from jamun.model.noise_conditioning import NoiseConditionalScaling
from jamun.model.skip_connection import NoiseConditionalSkipConnection


class E3Conv(torch.nn.Module):
    """A simple E(3)-equivariant convolutional neural network, similar to NequIP."""

    def __init__(
        self,
        irreps_out: Union[str, e3nn.o3.Irreps],
        irreps_hidden: Union[str, e3nn.o3.Irreps],
        irreps_sh: Union[str, e3nn.o3.Irreps],
        hidden_layer_factory: Callable[..., torch.nn.Module],
        output_head_factory: Callable[..., torch.nn.Module],
        use_residue_information: bool,
        n_layers: int,
        edge_attr_dim: int,
        atom_type_embedding_dim: int,
        atom_code_embedding_dim: int,
        residue_code_embedding_dim: int,
        residue_index_embedding_dim: int,
        use_residue_sequence_index: bool,
        test_equivariance: bool = False,
    ):
        super().__init__()

        self.test_equivariance = test_equivariance
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.n_layers = n_layers
        self.edge_attr_dim = edge_attr_dim

        self.bonded_edge_attr_dim, self.radial_edge_attr_dim = self.edge_attr_dim // 2, (self.edge_attr_dim + 1) // 2
        self.embed_bondedness = torch.nn.Embedding(2, self.bonded_edge_attr_dim)

        if use_residue_information:
            self.atom_embedder = AtomEmbeddingWithResidueInformation(
                atom_type_embedding_dim,
                atom_code_embedding_dim,
                residue_code_embedding_dim,
                residue_index_embedding_dim,
                use_residue_sequence_index,
            )
        else:
            self.atom_embedder = SimpleAtomEmbedding(
                embedding_dim=atom_type_embedding_dim
                + atom_code_embedding_dim
                + residue_code_embedding_dim
                + residue_index_embedding_dim
            )

        self.initial_noise_scaling = NoiseConditionalScaling(self.atom_embedder.irreps_out)
        self.initial_projector = hidden_layer_factory(
            irreps_in=self.initial_noise_scaling.irreps_out,
            irreps_out=self.irreps_hidden,
            irreps_sh=self.irreps_sh,
            edge_attr_dim=edge_attr_dim,
        )

        self.layers = torch.nn.ModuleList()
        self.noise_scalings = torch.nn.ModuleList()
        self.skip_connections = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                hidden_layer_factory(
                    irreps_in=self.irreps_hidden,
                    irreps_out=self.irreps_hidden,
                    irreps_sh=self.irreps_sh,
                    edge_attr_dim=self.edge_attr_dim,
                )
            )
            self.noise_scalings.append(NoiseConditionalScaling(self.irreps_hidden))
            self.skip_connections.append(NoiseConditionalSkipConnection(self.irreps_hidden))

        self.output_head = output_head_factory(irreps_in=self.irreps_hidden, irreps_out=self.irreps_out)
        self.output_gain = torch.nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        data: torch_geometric.data.Batch,
        c_noise: torch.Tensor,
        effective_radial_cutoff: Optional[torch.Tensor] = None,
    ) -> torch_geometric.data.Batch:
        # Test equivariance on the first forward pass.
        if self.test_equivariance:

            def forward_wrapped(pos: torch.Tensor):
                data_copy = data.clone()
                data_copy.pos = pos
                return self.forward(data_copy, c_noise, effective_radial_cutoff).pos

            self.test_equivariance = False
            e3nn.util.test.assert_equivariant(
                forward_wrapped,
                args_in=[data.pos],
                irreps_in=[self.irreps_out],
                irreps_out=[self.irreps_out],
            )

        # Add edges to the input, based on the radial cutoff.
        pos = data.pos
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=pos.device)
        radial_edge_index = torch_geometric.nn.radius_graph(pos, effective_radial_cutoff, batch)

        bonded_edge_index = data.edge_index
        edge_index = torch.cat((radial_edge_index, bonded_edge_index), dim=-1)
        bond_mask = torch.cat(
            (
                torch.zeros(radial_edge_index.shape[1], dtype=torch.long, device=pos.device),
                torch.ones(bonded_edge_index.shape[1], dtype=torch.long, device=pos.device),
            ),
            dim=0,
        )

        src, dst = edge_index
        edge_vec = pos[src] - pos[dst]
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization="component")

        bonded_edge_attr = self.embed_bondedness(bond_mask)
        radial_edge_attr = e3nn.math.soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            0.0,
            effective_radial_cutoff,
            self.radial_edge_attr_dim,
            basis="gaussian",
            cutoff=True,
        )
        edge_attr = torch.cat((bonded_edge_attr, radial_edge_attr), dim=-1)

        c_noise = c_noise.unsqueeze(0)
        node_attr = self.atom_embedder(data)
        node_attr = self.initial_noise_scaling(node_attr, c_noise)
        node_attr = self.initial_projector(node_attr, edge_index, edge_attr, edge_sh)
        for scaling, skip, layer in zip(self.noise_scalings, self.skip_connections, self.layers):
            node_attr = skip(node_attr, layer(scaling(node_attr, c_noise), edge_index, edge_attr, edge_sh), c_noise)
        node_attr = self.output_head(node_attr)
        node_attr = node_attr * self.output_gain

        data.pos = node_attr
        return data
