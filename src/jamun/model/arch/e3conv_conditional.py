from typing import Callable

import e3nn
import torch
import torch_geometric
from e3nn import o3
from e3nn.o3 import Irreps
from e3tools import scatter
from torch import Tensor
from jamun.model.atom_embedding import AtomEmbeddingWithResidueInformation, SimpleAtomEmbedding
from jamun.model.noise_conditioning import NoiseConditionalScaling, NoiseConditionalSkipConnection
import e3tools.nn


class E3ConvConditional(torch.nn.Module):
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
        N_structures: int = 1
    ):
        super().__init__()

        self.test_equivariance = test_equivariance
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.n_layers = n_layers
        self.edge_attr_dim = edge_attr_dim
        self.N_structures = N_structures
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
            irreps_sh=N_structures*self.irreps_sh,
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
                    irreps_sh=N_structures*self.irreps_sh,
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
        pos: Tensor, # should be [batch_size*N, 3T], T is the number of previous time-steps
        topology: torch_geometric.data.Batch,
        c_noise: Tensor,
        effective_radial_cutoff: float,
    ) -> torch_geometric.data.Batch:
        # Extract edge attributes.
        edge_index = topology["edge_index"]
        bond_mask = topology["bond_mask"]

        src, dst = edge_index                    # compute edge spherical harmonics over concat structures
        positions = torch.split(pos, 3, dim=-1)
        edge_sh = []
        for block in positions: 
            edge_vec = block[src] - block[dst]
            edge_sh.append(self.sh(edge_vec))
        edge_sh = torch.cat(edge_sh, dim=-1) 

        # print(f"Edge spherical harmonics: {type(edge_sh)}")
        bonded_edge_attr = self.embed_bondedness(bond_mask)
        edge_vec_main = positions[0][src] - positions[0][dst]
        radial_edge_attr = e3nn.math.soft_one_hot_linspace(
            edge_vec_main.norm(dim=1),
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


class E3ConvConditionalWithInputAttr(E3ConvConditional):
    """
    Extension of E3ConvConditional that can accept additional input attributes
    and combine them with the computed node attributes.
    """
    
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
        N_structures: int = 1,
        input_attr_irreps: str | Irreps | None = None,
    ):
        """
        Initialize E3ConvConditionalWithInputAttr.
        
        Args:
            input_attr_irreps: Irreps of the input attributes that will be combined with node_attr.
                              If None, the model behaves like the parent class.
            All other args: Same as parent E3ConvConditional class.
        """
        super().__init__(
            irreps_out=irreps_out,
            irreps_hidden=irreps_hidden,
            irreps_sh=irreps_sh,
            hidden_layer_factory=hidden_layer_factory,
            output_head_factory=output_head_factory,
            use_residue_information=use_residue_information,
            n_layers=n_layers,
            edge_attr_dim=edge_attr_dim,
            atom_type_embedding_dim=atom_type_embedding_dim,
            atom_code_embedding_dim=atom_code_embedding_dim,
            residue_code_embedding_dim=residue_code_embedding_dim,
            residue_index_embedding_dim=residue_index_embedding_dim,
            use_residue_sequence_index=use_residue_sequence_index,
            num_atom_types=num_atom_types,
            max_sequence_length=max_sequence_length,
            num_atom_codes=num_atom_codes,
            num_residue_types=num_residue_types,
            test_equivariance=test_equivariance,
            reduce=reduce,
            N_structures=N_structures,
        )
        
        self.input_attr_irreps = o3.Irreps(input_attr_irreps) if input_attr_irreps is not None else None
        
        # Create input irrep aggregator if input attributes are provided
        if self.input_attr_irreps is not None:
            # Combined irreps: node_attr irreps + input_attr irreps
            combined_irreps = self.irreps_hidden + self.input_attr_irreps
            
            # Create aggregator that takes combined input and outputs node_attr irreps
            self.input_irrep_aggregator = e3tools.nn.EquivariantMLP(
                irreps_in=combined_irreps,
                irreps_out=self.irreps_hidden,
                irreps_hidden_list=[self.irreps_hidden],  # Single hidden layer
            )
        else:
            self.input_irrep_aggregator = None
    
    def forward(
        self,
        pos: Tensor,
        topology: torch_geometric.data.Batch,
        c_noise: Tensor,
        effective_radial_cutoff: float,
        input_attr: Tensor | None = None,
    ) -> Tensor:
        """
        Forward pass with optional input attributes.
        
        Args:
            pos: Node positions
            topology: Graph topology
            c_noise: Noise conditioning
            effective_radial_cutoff: Radial cutoff for edges
            input_attr: Optional input attributes to combine with node_attr.
                       Should have shape [N, input_attr_irreps.dim] where N is number of nodes.
        
        Returns:
            Node attributes after processing
        """
        # Extract edge attributes.
        edge_index = topology["edge_index"]
        bond_mask = topology["bond_mask"]

        src, dst = edge_index                    # compute edge spherical harmonics over concat structures
        positions = torch.split(pos, 3, dim=-1)
        edge_sh = []
        for block in positions: 
            edge_vec = block[src] - block[dst]
            edge_sh.append(self.sh(edge_vec))
        edge_sh = torch.cat(edge_sh, dim=-1) 

        # print(f"Edge spherical harmonics: {type(edge_sh)}")
        bonded_edge_attr = self.embed_bondedness(bond_mask)
        edge_vec_main = positions[0][src] - positions[0][dst]
        radial_edge_attr = e3nn.math.soft_one_hot_linspace(
            edge_vec_main.norm(dim=1),
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
        
        # Combine with input attributes if provided
        if input_attr is not None and self.input_irrep_aggregator is not None:
            # Validate input_attr shape
            expected_dim = self.input_attr_irreps.dim
            if input_attr.shape[-1] != expected_dim:
                raise ValueError(
                    f"Expected input_attr to have dimension {expected_dim}, "
                    f"but got {input_attr.shape[-1]}"
                )
            if input_attr.shape[0] != node_attr.shape[0]:
                raise ValueError(
                    f"Expected input_attr to have {node_attr.shape[0]} nodes, "
                    f"but got {input_attr.shape[0]}"
                )
            
            # Concatenate node_attr with input_attr
            combined_attr = torch.cat([node_attr, input_attr], dim=-1)
            
            # Aggregate to get back to node_attr irreps
            node_attr = self.input_irrep_aggregator(combined_attr)
        elif input_attr is not None and self.input_irrep_aggregator is None:
            raise ValueError(
                "input_attr provided but input_attr_irreps was not specified during initialization"
            )
        
        # Continue with normal processing
        for scaling, skip, layer in zip(self.noise_scalings, self.skip_connections, self.layers):
            node_attr = skip(node_attr, layer(scaling(node_attr, c_noise), edge_index, edge_attr, edge_sh), c_noise)
        node_attr = self.output_head(node_attr)
        node_attr = node_attr * self.output_gain

        if self.reduce is not None:
            node_attr = scatter(node_attr, topology.batch, dim=0, reduce=self.reduce)

        return node_attr


class E3ConvConditionalSpatioTemporal(E3ConvConditional):
    """
    E3ConvConditional specifically designed for spatiotemporal conditioning.
    
    This class expects input positions to be concatenated as [y.pos, spatial_features]
    where y.pos are the physical 3D coordinates and spatial_features are additional
    attributes from the spatiotemporal model.
    
    Key differences from E3ConvConditional:
    - Edge spherical harmonics are only computed for the first 3 coordinates (y.pos)
    - Remaining coordinates are treated as per-node input attributes
    - Input attributes are combined with computed node attributes
    """
    
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
        N_structures: int = 1,  # Should be 2 for [y.pos, spatial_features]
        input_attr_irreps: str | Irreps = "3x1e",  # Default for spatial features
    ):
        """
        Initialize E3ConvConditionalSpatioTemporal.
        
        Args:
            input_attr_irreps: Irreps of the spatial features from spatiotemporal model.
                              Should match the irreps_out of the spatiotemporal model.
            N_structures: Should be 2 for [y.pos, spatial_features]
            All other args: Same as parent E3ConvConditional class.
        """
        super().__init__(
            irreps_out=irreps_out,
            irreps_hidden=irreps_hidden,
            irreps_sh=irreps_sh,
            hidden_layer_factory=hidden_layer_factory,
            output_head_factory=output_head_factory,
            use_residue_information=use_residue_information,
            n_layers=n_layers,
            edge_attr_dim=edge_attr_dim,
            atom_type_embedding_dim=atom_type_embedding_dim,
            atom_code_embedding_dim=atom_code_embedding_dim,
            residue_code_embedding_dim=residue_code_embedding_dim,
            residue_index_embedding_dim=residue_index_embedding_dim,
            use_residue_sequence_index=use_residue_sequence_index,
            num_atom_types=num_atom_types,
            max_sequence_length=max_sequence_length,
            num_atom_codes=num_atom_codes,
            num_residue_types=num_residue_types,
            test_equivariance=test_equivariance,
            reduce=reduce,
            N_structures=N_structures,
        )
        
        # Set up input attribute handling
        self.input_attr_irreps = o3.Irreps(input_attr_irreps)
        
        # Create input irrep aggregator to combine node_attr with input_attr
        # Combined irreps: node_attr irreps + input_attr irreps
        combined_irreps = self.irreps_hidden + self.input_attr_irreps
        
        # Create aggregator that takes combined input and outputs node_attr irreps
        self.input_irrep_aggregator = e3tools.nn.EquivariantMLP(
            irreps_in=combined_irreps,
            irreps_out=self.irreps_hidden,
            irreps_hidden_list=[self.irreps_hidden],  # Single hidden layer
        )
    
    def forward(
        self,
        pos: Tensor,  # should be [N, 3 + spatial_features_dim] from [y.pos, spatial_features]
        topology: torch_geometric.data.Batch,
        c_noise: Tensor,
        effective_radial_cutoff: float,
    ) -> Tensor:
        """
        Forward pass with spatiotemporal conditioning.
        
        Args:
            pos: Concatenated positions [y.pos, spatial_features] with shape [N, 3 + spatial_features_dim]
            topology: Graph topology
            c_noise: Noise conditioning
            effective_radial_cutoff: Radial cutoff for edges
        
        Returns:
            Node attributes after processing
        """
        # Extract edge attributes.
        edge_index = topology["edge_index"]
        bond_mask = topology["bond_mask"]

        src, dst = edge_index
        
        # Split positions: first 3 coords are physical positions, rest are spatial features
        pos_physical = pos[:, :3]  # [N, 3] - physical coordinates
        pos_features = pos[:, 3:]  # [N, spatial_features_dim] - spatial features
        
        # Compute edge spherical harmonics ONLY for physical positions
        edge_vec_physical = pos_physical[src] - pos_physical[dst]
        edge_sh = self.sh(edge_vec_physical)

        # Compute edge attributes using physical positions
        bonded_edge_attr = self.embed_bondedness(bond_mask)
        radial_edge_attr = e3nn.math.soft_one_hot_linspace(
            edge_vec_physical.norm(dim=1),
            0.0,
            effective_radial_cutoff,
            self.radial_edge_attr_dim,
            basis="gaussian",
            cutoff=True,
        )
        edge_attr = torch.cat((bonded_edge_attr, radial_edge_attr), dim=-1)

        # Compute initial node attributes
        node_attr = self.atom_embedder(topology)
        node_attr = self.initial_noise_scaling(node_attr, c_noise)
        node_attr = self.initial_projector(node_attr, edge_index, edge_attr, edge_sh)
        
        # Combine node_attr with spatial features (input_attr)
        # Validate spatial features shape
        expected_dim = self.input_attr_irreps.dim
        if pos_features.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected spatial features to have dimension {expected_dim}, "
                f"but got {pos_features.shape[-1]}"
            )
        
        # Concatenate node_attr with spatial features
        combined_attr = torch.cat([node_attr, pos_features], dim=-1)
        
        # Aggregate to get back to node_attr irreps
        node_attr = self.input_irrep_aggregator(combined_attr)
        
        # Continue with normal processing using only physical positions for edge computations
        for scaling, skip, layer in zip(self.noise_scalings, self.skip_connections, self.layers):
            node_attr = skip(node_attr, layer(scaling(node_attr, c_noise), edge_index, edge_attr, edge_sh), c_noise)
        node_attr = self.output_head(node_attr)
        node_attr = node_attr * self.output_gain

        if self.reduce is not None:
            node_attr = scatter(node_attr, topology.batch, dim=0, reduce=self.reduce)

        return node_attr
