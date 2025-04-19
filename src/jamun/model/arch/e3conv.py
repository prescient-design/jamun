from functools import partial
from typing import Callable

import e3nn
import torch
import torch_geometric
from e3nn import o3
from e3nn.o3 import Irreps
from torch import Tensor
from e3tools import scatter

from jamun.model.atom_embedding import AtomEmbeddingWithResidueInformation, SimpleAtomEmbedding
from jamun.model.noise_conditioning import NoiseConditionalScaling, NoiseConditionalSkipConnection


class E3Conv(torch.nn.Module):
    """A simple E(3)-equivariant convolutional neural network, similar to NequIP."""

    def __init__(
        self,
        irreps_out: str | Irreps,
        irreps_hidden: str | Irreps,
        irreps_sh: str | Irreps,
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
        num_atom_types: int = 20,
        max_sequence_length: int = 10,
        num_atom_codes: int = 10,
        num_residue_types: int = 25,
        test_equivariance: bool = False,
        reduce: str | None = None,
    ):
        super().__init__()

        self.test_equivariance = test_equivariance
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.n_layers = n_layers
        self.edge_attr_dim = edge_attr_dim

        self.sh = o3.SphericalHarmonics(irreps_out=self.irreps_sh, normalize=True, normalization="component")
        self.bonded_edge_attr_dim, self.radial_edge_attr_dim = self.edge_attr_dim // 2, (self.edge_attr_dim + 1) // 2
        self.embed_bondedness = torch.nn.Embedding(2, self.bonded_edge_attr_dim)

        if use_residue_information:
            self.atom_embedder = AtomEmbeddingWithResidueInformation(
                atom_type_embedding_dim=atom_type_embedding_dim,
                atom_code_embedding_dim=atom_code_embedding_dim,
                residue_code_embedding_dim=residue_code_embedding_dim,
                residue_index_embedding_dim=residue_index_embedding_dim,
                use_residue_sequence_index=use_residue_sequence_index,
                num_atom_types=num_atom_types,
                max_sequence_length=max_sequence_length,
                num_atom_codes=num_atom_codes,
                num_residue_types=num_residue_types,
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
        self.reduce = reduce

    def forward(
        self,
        pos: Tensor,
        topology: torch_geometric.data.Batch,
        c_noise: Tensor,
        effective_radial_cutoff: float,
    ) -> torch_geometric.data.Batch:
        # Extract edge attributes.
        edge_index = topology["edge_index"]
        bond_mask = topology["bond_mask"]

        src, dst = edge_index
        edge_vec = pos[src] - pos[dst]
        edge_sh = self.sh(edge_vec)

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

        node_attr = self.atom_embedder(topology)
        node_attr = self.initial_noise_scaling(node_attr, c_noise)
        node_attr = self.initial_projector(node_attr, edge_index, edge_attr, edge_sh)
        for scaling, skip, layer in zip(self.noise_scalings, self.skip_connections, self.layers):
            node_attr = skip(node_attr, layer(scaling(node_attr, c_noise), edge_index, edge_attr, edge_sh), c_noise)
        node_attr = self.output_head(node_attr)
        node_attr = node_attr * self.output_gain

        if self.reduce is not None:
            node_attr = scatter(node_attr, topology.batch, dim=0, reduce=self.reduce)

        return node_attr
