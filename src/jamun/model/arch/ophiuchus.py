from collections.abc import Callable
from typing import NamedTuple

import e3nn
import e3nn.util.test
import einops
import torch
import torch.nn.functional as F
import torch_geometric
from e3tools import radius_graph
from e3tools.nn import AxisToMul, Gate, MulToAxis
from torch import nn

from jamun import utils
from jamun.model.noise_conditioning import NoiseConditionalScaling, NoiseConditionalSkipConnection


def dprint(*args):
    print(*args)


class ResidueState(NamedTuple):
    """State of a residue."""

    coords: torch.Tensor
    features: torch.Tensor


class ResidueData(torch_geometric.data.Data):
    """Data for a single residue."""

    batch: torch.Tensor
    residue_base_coords: torch.Tensor
    residue_relative_coords: torch.Tensor
    residue_code_index: torch.Tensor
    residue_sequence_index: torch.Tensor
    residue_batch: torch.Tensor
    residue_index: torch.Tensor
    residue_index_atomwise: torch.Tensor
    atom_code_index: torch.Tensor
    atom_type_index: torch.Tensor


def to_residue_data(data: utils.DataWithResidueInformation) -> ResidueData:
    """Convert atom-based data to residue-based data."""
    if data.batch is None:
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

    ALPHA_CARBON_INDEX = utils.ResidueMetadata.ATOM_CODES.index("CA")
    alpha_carbon_indices = data.atom_code_index == ALPHA_CARBON_INDEX
    base_coords = data.pos[alpha_carbon_indices]
    relative_coords = data.pos - base_coords[data.residue_index]

    return ResidueData(
        atom_code_index=data.atom_code_index,
        atom_type_index=data.atom_type_index,
        batch=data.batch,
        residue_base_coords=base_coords,
        residue_relative_coords=relative_coords,
        residue_code_index=data.residue_code_index[alpha_carbon_indices],
        residue_sequence_index=data.residue_sequence_index[alpha_carbon_indices],
        residue_batch=data.batch[alpha_carbon_indices],
        residue_index=data.residue_index[alpha_carbon_indices],
        residue_index_atomwise=data.residue_index,
    )


def to_atom_data(residue_data: ResidueData) -> utils.DataWithResidueInformation:
    """Convert residue-based data to atom-based data."""
    if residue_data.batch is None:
        residue_data.batch = torch.zeros(residue_data.num_nodes, dtype=torch.long)

    base_coords = residue_data.residue_base_coords[residue_data.residue_index_atomwise]
    relative_coords = residue_data.residue_relative_coords

    ALPHA_CARBON_INDEX = utils.ResidueMetadata.ATOM_CODES.index("CA")
    alpha_carbon_indices = residue_data.atom_code_index == ALPHA_CARBON_INDEX
    relative_coords[alpha_carbon_indices] = 0.0
    coords = base_coords + relative_coords

    return utils.DataWithResidueInformation(
        pos=coords,
        atom_code_index=residue_data.atom_code_index,
        atom_type_index=residue_data.atom_type_index,
        residue_code_index=residue_data.residue_code_index[residue_data.residue_index_atomwise],
        residue_index=residue_data.residue_index_atomwise,
        batch=residue_data.batch,
    )


class TestEquivariance(nn.Module):
    """Test equivariance of a module."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.test_equivariance = True
        self.module = module
        self.irreps_in = module.irreps_in
        self.irreps_out = module.irreps_out

    def forward(self, x):
        if self.test_equivariance:

            def forward_wrapped(x):
                return self.module(x)

            self.test_equivariance = False
            e3nn.util.test.equivariance_error(
                forward_wrapped, args_in=[x], irreps_in=[self.irreps_in], irreps_out=[self.irreps_out]
            )

        return self.module(x)


class ApplyToResidueFeatures(nn.Module):
    """Apply a function to residue features."""

    def __init__(self, f: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.f = f

    def forward(self, residue_state: ResidueState) -> ResidueState:
        features = self.f(residue_state.features)
        return residue_state._replace(features=features)


class InitialResidueEmbedding(nn.Module):
    """Initial embedding for residues."""

    def __init__(
        self,
        irreps_out: e3nn.o3.Irreps,
        atom_code_embedding_dim: int,
        atom_type_embedding_dim: int,
        residue_code_embedding_dim: int,
        residue_index_embedding_dim: int,
        use_residue_sequence_index: bool,
        max_sequence_length: int,
        residue_pad_length: int,
    ):
        super().__init__()
        self.irreps_out = e3nn.o3.Irreps(irreps_out)
        self.use_residue_sequence_index = use_residue_sequence_index
        self.residue_pad_length = residue_pad_length

        self.atom_code_embedder = torch.nn.Embedding(
            num_embeddings=len(utils.ResidueMetadata.ATOM_CODES) + 1,
            embedding_dim=atom_code_embedding_dim,
        )
        self.atom_type_embedder = torch.nn.Embedding(
            num_embeddings=len(utils.ResidueMetadata.ATOM_TYPES) + 1,
            embedding_dim=atom_type_embedding_dim,
        )

        self.residue_code_embedder = torch.nn.Embedding(
            num_embeddings=len(utils.ResidueMetadata.RESIDUE_CODES) + 1,
            embedding_dim=residue_code_embedding_dim,
        )
        irreps_embed = e3nn.o3.Irreps(
            f"{residue_pad_length}x1e + "
            f"{residue_pad_length * atom_code_embedding_dim}x0e + "
            f"{residue_pad_length * atom_type_embedding_dim}x0e + "
            f"{residue_code_embedding_dim}x0e"
        )

        if use_residue_sequence_index:
            self.residue_index_embedder = torch.nn.Embedding(
                num_embeddings=max_sequence_length,
                embedding_dim=residue_index_embedding_dim,
            )
            irreps_embed += e3nn.o3.Irreps(f"{residue_index_embedding_dim}x0e")

        self.post_linear = e3nn.o3.Linear(
            irreps_in=irreps_embed,
            irreps_out=self.irreps_out,
        )

    def forward(self, residue_data: torch_geometric.data.Data) -> ResidueState:
        base_coords = residue_data.residue_base_coords
        device = base_coords.device

        mask = []
        relative_coords = []
        atom_codes = []
        atom_types = []
        for index in residue_data.residue_index:
            assert (
                residue_data.residue_relative_coords.shape[0]
                == residue_data.atom_code_index.shape[0]
                == residue_data.atom_type_index.shape[0]
            )

            residue_mask = residue_data.residue_index_atomwise == index
            num_atoms = (residue_mask).sum()
            mask.append(
                torch.cat(
                    [
                        torch.ones(num_atoms, dtype=torch.bool, device=device),
                        torch.zeros(self.residue_pad_length - num_atoms, dtype=torch.bool, device=device),
                    ]
                )
            )

            residue_relative_coords = residue_data.residue_relative_coords[residue_mask]
            residue_relative_coords = F.pad(residue_relative_coords, (0, 0, 0, self.residue_pad_length - num_atoms))
            residue_relative_coords = einops.rearrange(
                residue_relative_coords, "... pad_length coords -> ... (pad_length coords)"
            )
            relative_coords.append(residue_relative_coords)

            residue_atom_codes = residue_data.atom_code_index[residue_mask]
            residue_atom_codes = F.pad(residue_atom_codes, (0, self.residue_pad_length - residue_atom_codes.shape[0]))
            atom_codes.append(residue_atom_codes)

            residue_atom_types = residue_data.atom_type_index[residue_mask]
            residue_atom_types = F.pad(residue_atom_types, (0, self.residue_pad_length - residue_atom_types.shape[0]))
            atom_types.append(residue_atom_types)

        mask = torch.stack(mask)
        relative_coords = torch.stack(relative_coords)
        atom_codes = torch.stack(atom_codes)
        atom_types = torch.stack(atom_types)

        atom_codes_embedded = self.atom_code_embedder(atom_codes)
        atom_codes_embedded *= mask.unsqueeze(-1)
        atom_codes_embedded = atom_codes_embedded.reshape(atom_codes_embedded.shape[0], -1)

        atom_types_embedded = self.atom_type_embedder(atom_types)
        atom_types_embedded *= mask.unsqueeze(-1)
        atom_types_embedded = atom_types_embedded.reshape(atom_types_embedded.shape[0], -1)

        residue_codes_embedded = self.residue_code_embedder(residue_data.residue_code_index)
        features = [
            relative_coords,
            atom_codes_embedded,
            atom_types_embedded,
            residue_codes_embedded,
        ]
        if self.use_residue_sequence_index:
            residue_index_embedded = self.residue_index_embedder(residue_data.residue_sequence_index)
            features.append(residue_index_embedded)

        features = torch.cat(features, dim=-1)
        # print("features.shape", features.shape)
        features = self.post_linear(features)
        # print("post features.shape", features.shape)
        assert base_coords.shape[0] == features.shape[0]

        return ResidueState(
            coords=base_coords,
            features=features,
        )


class TensorSquare(nn.Module):
    """Tensor square layer after factoring multiplicity."""

    def __init__(self, irreps_in: e3nn.o3.Irreps, mul_factor: int):
        super().__init__()

        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.mul_to_axis = MulToAxis(
            irreps_in=irreps_in,
            factor=mul_factor,
        )

        self.tensor_square = e3nn.o3.TensorSquare(
            irreps_in=self.mul_to_axis.irreps_out,
        )

        self.axis_to_mul = AxisToMul(
            irreps_in=self.tensor_square.irreps_out,
            factor=mul_factor,
        )
        self.irreps_out = self.axis_to_mul.irreps_out

    def forward(self, x):
        x = self.mul_to_axis(x)
        x_sq = self.tensor_square(x)
        x_sq = self.axis_to_mul(x_sq)
        return x_sq


class SelfInteraction(nn.Module):
    """Self-interaction layer for residues."""

    def __init__(self, irreps_in: e3nn.o3.Irreps, mul_factor: int):
        super().__init__()
        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.tensor_square = TensorSquare(
            irreps_in=self.irreps_in,
            mul_factor=mul_factor,
        )
        self.gate = Gate(
            irreps_out=self.irreps_in,
        )
        self.gate_linear = e3nn.o3.Linear(
            irreps_in=self.irreps_in + self.tensor_square.irreps_out,
            irreps_out=self.gate.irreps_in,
        )
        self.noise_scaling = NoiseConditionalScaling(
            self.gate.irreps_out,
        )

    def forward(self, residue_state: ResidueState, c_noise: torch.Tensor) -> ResidueState:
        features = residue_state.features
        features_squared = self.tensor_square(features)
        features = torch.concat([features, features_squared], dim=-1)
        features = self.gate_linear(features)
        features = self.gate(features)
        features = self.noise_scaling(features, c_noise)
        return residue_state._replace(features=features)


class SpatialConvolution(nn.Module):
    """Spatial convolution layer for residues."""

    def __init__(
        self,
        irreps_in: e3nn.o3.Irreps,
        irreps_sh: e3nn.o3.Irreps,
        edge_attr_dim: int,
        conv_factory: Callable[..., torch.nn.Module],
    ):
        super().__init__()
        self.conv = conv_factory(
            irreps_in=irreps_in,
            irreps_out=irreps_in,
            irreps_sh=irreps_sh,
            edge_attr_dim=edge_attr_dim,
        )
        self.noise_scaling = NoiseConditionalScaling(
            self.conv.irreps_out,
        )

    def forward(
        self,
        residue_state: ResidueState,
        residue_edge_index: torch.Tensor,
        residue_edge_attr: torch.Tensor,
        residue_edge_sh: torch.Tensor,
        c_noise: torch.Tensor,
    ) -> ResidueState:
        features = self.conv(residue_state.features, residue_edge_index, residue_edge_attr, residue_edge_sh)
        features = self.noise_scaling(features, c_noise)
        return residue_state._replace(features=features)


class OutputHead(nn.Module):
    """Output head for coordinates of all atoms."""

    def __init__(self, irreps_in: e3nn.o3.Irreps, irreps_out: e3nn.o3.Irreps, residue_pad_length: int):
        super().__init__()
        self.base_coords_linear = e3nn.o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)
        self.relative_coords_linear = e3nn.o3.Linear(
            irreps_in=irreps_in,
            irreps_out=residue_pad_length * irreps_out,
        )

    def forward(self, residue_state: ResidueState, residue_data: ResidueData) -> ResidueData:
        features = residue_state.features
        base_coords = self.base_coords_linear(features)
        relative_coords = self.relative_coords_linear(features)
        relative_coords = einops.rearrange(
            relative_coords, "... (pad_length coords) -> ... pad_length coords", coords=3
        )
        unpadded_relative_coords = []
        for index in residue_data.residue_index:
            residue_mask = residue_data.residue_index_atomwise == index
            num_residue_atoms = (residue_mask).sum()
            residue_relative_coords = relative_coords[index, :num_residue_atoms]
            unpadded_relative_coords.append(residue_relative_coords)
        relative_coords = torch.cat(unpadded_relative_coords)

        residue_data = residue_data.clone()
        residue_data.residue_base_coords = base_coords
        residue_data.residue_relative_coords = relative_coords
        return residue_data


class OphiuchusBlock(nn.Module):
    """A single block of the Ophiuchus model."""

    def __init__(
        self,
        irreps_in: e3nn.o3.Irreps,
        irreps_sh: e3nn.o3.Irreps,
        edge_attr_dim: int,
        conv_factory: Callable[..., torch.nn.Module],
        mul_factor: int,
    ):
        super().__init__()
        self.self_interaction = SelfInteraction(
            irreps_in=irreps_in,
            mul_factor=mul_factor,
        )
        self.spatial_convolution = SpatialConvolution(
            irreps_in=irreps_in,
            irreps_sh=irreps_sh,
            edge_attr_dim=edge_attr_dim,
            conv_factory=conv_factory,
        )

    def forward(
        self,
        residue_state: ResidueState,
        residue_edge_index: torch.Tensor,
        residue_edge_attr: torch.Tensor,
        residue_edge_sh: torch.Tensor,
        c_noise: torch.Tensor,
    ) -> ResidueState:
        residue_state = self.self_interaction(residue_state, c_noise)
        residue_state = self.spatial_convolution(
            residue_state, residue_edge_index, residue_edge_attr, residue_edge_sh, c_noise
        )
        return residue_state


class ResidueNoiseConditionalSkipConnection(nn.Module):
    """Skip connection for residues with noise conditioning."""

    def __init__(self, irreps_in: e3nn.o3.Irreps):
        super().__init__()
        self.irreps_in = e3nn.o3.Irreps(irreps_in)
        self.skip_connection = NoiseConditionalSkipConnection(self.irreps_in)

    def forward(
        self, residue_state: ResidueState, new_residue_state: ResidueState, c_noise: torch.Tensor
    ) -> ResidueState:
        features = self.skip_connection(residue_state.features, new_residue_state.features, c_noise)
        return residue_state._replace(features=features)


class Ophiuchus(nn.Module):
    """Ophiuchus model for hierarchical protein structure modeling.

    ref: https://arxiv.org/abs/2310.02508
    ref: https://arxiv.org/abs/2410.09667
    """

    MAX_ATOMS_IN_RESIDUE: int = 16
    MAX_SEQUENCE_LENGTH: int = 20

    def __init__(
        self,
        irreps_out: e3nn.o3.Irreps,
        irreps_hidden: e3nn.o3.Irreps,
        irreps_sh: e3nn.o3.Irreps,
        conv_factory: Callable[..., torch.nn.Module],
        edge_attr_dim: int,
        mul_factor: int,
        n_layers: int,
        atom_type_embedding_dim: int,
        atom_code_embedding_dim: int,
        residue_code_embedding_dim: int,
        residue_index_embedding_dim: int,
        use_residue_sequence_index: bool,
        test_equivariance: bool = True,
    ):
        super().__init__()
        self.test_equivariance = test_equivariance
        self.irreps_out = e3nn.o3.Irreps(irreps_out)
        self.irreps_hidden = e3nn.o3.Irreps(irreps_hidden)
        self.irreps_sh = e3nn.o3.Irreps(irreps_sh)
        self.edge_attr_dim = edge_attr_dim

        self.initial_residue_embedding = InitialResidueEmbedding(
            irreps_out=irreps_hidden,
            atom_code_embedding_dim=atom_code_embedding_dim,
            atom_type_embedding_dim=atom_type_embedding_dim,
            residue_code_embedding_dim=residue_code_embedding_dim,
            residue_index_embedding_dim=residue_index_embedding_dim,
            use_residue_sequence_index=use_residue_sequence_index,
            residue_pad_length=self.MAX_ATOMS_IN_RESIDUE,
            max_sequence_length=self.MAX_SEQUENCE_LENGTH,
        )

        self.blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for _ in range(n_layers):
            self.blocks.append(
                OphiuchusBlock(
                    irreps_in=irreps_hidden,
                    irreps_sh=irreps_sh,
                    edge_attr_dim=edge_attr_dim,
                    conv_factory=conv_factory,
                    mul_factor=mul_factor,
                )
            )
            self.skip_connections.append(
                ResidueNoiseConditionalSkipConnection(
                    irreps_in=irreps_hidden,
                )
            )

        self.output_head = OutputHead(
            irreps_in=irreps_hidden,
            irreps_out=self.irreps_out,
            residue_pad_length=self.MAX_ATOMS_IN_RESIDUE,
        )

    def compute_edge_attributes(
        self, edge_index: torch.Tensor, coords: torch.Tensor, effective_radial_cutoff: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_vec = coords[edge_index[1]] - coords[edge_index[0]]
        edge_sh = e3nn.o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization="component")
        edge_attr = e3nn.math.soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            0.0,
            effective_radial_cutoff,
            self.edge_attr_dim,
            basis="gaussian",
            cutoff=True,
        )
        return edge_attr, edge_sh

    def forward(
        self,
        data: torch_geometric.data.Batch,
        c_noise: torch.Tensor,
        effective_radial_cutoff: torch.Tensor | None,
    ) -> torch.Tensor:
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

        c_noise = c_noise.unsqueeze(0)

        # Convert data to residue-based representation.
        residue_data = to_residue_data(data)

        # Compute residue edges and corresponding attributes.
        residue_edge_index = radius_graph(
            residue_data.residue_base_coords, effective_radial_cutoff, batch=residue_data.residue_batch
        )
        residue_edge_attr, residue_edge_sh = self.compute_edge_attributes(
            residue_edge_index, residue_data.residue_base_coords, effective_radial_cutoff
        )
        # print("edge_attr: ", residue_edge_attr.shape)
        # print("edge_sh: ", residue_edge_sh.shape)
        # print("edge_index: ", residue_edge_index.shape)

        # Create initial residue state.
        residue_state = self.initial_residue_embedding(residue_data)
        # print("initial embedding done")

        # Message-passing at the residue-level.
        for block, skip in zip(self.blocks, self.skip_connections):
            # print("message-passing")
            residue_state = skip(
                residue_state,
                block(residue_state, residue_edge_index, residue_edge_attr, residue_edge_sh, c_noise),
                c_noise,
            )
            # print("done")

        # Update base and relative coordinates.
        # print("starting output head")
        residue_data = self.output_head(residue_state, residue_data)
        # print("output head done")
        # Convert data back to atom-based representation.
        return to_atom_data(residue_data)
