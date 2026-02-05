import e3nn
import e3tools
import e3tools.nn
import torch
import torch.nn as nn
import torch_geometric.data
from e3nn import o3


class E3Transformer(nn.Module):
    """E(3)-equivariant transformer with temporal graph support."""

    def __init__(
        self,
        irreps_out: str | e3nn.o3.Irreps,
        irreps_hidden: str | e3nn.o3.Irreps,
        irreps_sh: str | e3nn.o3.Irreps,
        irreps_node_attr: str | e3nn.o3.Irreps,
        num_layers: int,
        edge_attr_dim: int,
        num_attention_heads: int,
        reduce: str | None = None,
    ):
        super().__init__()

        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)  # input irreps
        self.num_layers = num_layers
        self.edge_attr_dim = edge_attr_dim
        self.num_attention_heads = num_attention_heads
        self.reduce = reduce
        self.sh = o3.SphericalHarmonics(irreps_out=self.irreps_sh, normalize=True, normalization="component")
        # Split edge attribute dimensions: radial and temporal (bondedness is optional)
        self.radial_edge_attr_dim = self.edge_attr_dim // 2
        self.temporal_edge_attr_dim = self.edge_attr_dim - self.radial_edge_attr_dim

        # Optional bondedness embedding (only used if bond_mask exists in graph)
        self.embed_bondedness = nn.Embedding(2, self.edge_attr_dim // 3)

        # Gate for combining node attributes with temporal position
        # Input: node_attr (from data) + temporal_position (1x0e scalar)
        irreps_with_temporal = self.irreps_node_attr + o3.Irreps("1x0e")
        self.temporal_gate = e3tools.nn.GateWrapper(
            irreps_in=irreps_with_temporal,
            irreps_out=self.irreps_hidden,
            irreps_gate=irreps_with_temporal,
        )
        # self.initial_linear = o3.Linear(
        #     self.temporal_gate.irreps_out, self.irreps_hidden
        # )

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                e3tools.nn.TransformerBlock(
                    irreps_in=self.irreps_hidden,
                    irreps_out=self.irreps_hidden,
                    irreps_sh=self.irreps_sh,
                    edge_attr_dim=self.edge_attr_dim,
                    num_heads=self.num_attention_heads,
                )
            )
        self.output_head = e3tools.nn.EquivariantMLP(
            irreps_in=self.irreps_hidden,
            irreps_out=self.irreps_out,
            irreps_hidden_list=[self.irreps_hidden],
        )

    def forward(
        self,
        node_attr: torch.Tensor,
        temporal_graph: torch_geometric.data.Batch,
        effective_radial_cutoff: float,
        temporal_cutoff: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass of the E3Transformer model."""
        # Extract graph data
        pos = temporal_graph.pos
        edge_index = temporal_graph.edge_index
        temporal_position = temporal_graph.temporal_position
        batch = temporal_graph.batch
        num_graphs = temporal_graph.num_graphs

        src, dst = edge_index
        edge_vec = pos[src] - pos[dst]
        edge_sh = self.sh(edge_vec)

        # Compute edge attributes: radial and temporal
        radial_edge_attr = e3nn.math.soft_one_hot_linspace(
            edge_vec.norm(dim=1),
            0.0,
            effective_radial_cutoff,
            self.radial_edge_attr_dim,
            basis="gaussian",
            cutoff=True,
        )

        # Temporal edge attributes from temporal_position differences
        temporal_edge_vec = temporal_position[src] - temporal_position[dst]
        temporal_edge_attr = e3nn.math.soft_one_hot_linspace(
            temporal_edge_vec.abs(),  # Use absolute difference
            0.0,
            temporal_cutoff,
            self.temporal_edge_attr_dim,
            basis="gaussian",
            cutoff=True,
        )

        # Optional bondedness (if bond_mask exists in the temporal graph)
        if hasattr(temporal_graph, "bond_mask") and temporal_graph.bond_mask is not None:
            bonded_edge_attr = self.embed_bondedness(temporal_graph.bond_mask)
            edge_attr = torch.cat((bonded_edge_attr, radial_edge_attr, temporal_edge_attr), dim=-1)
        else:
            edge_attr = torch.cat((radial_edge_attr, temporal_edge_attr), dim=-1)

        # Process node attributes with temporal gating

        # Concatenate node_attr with temporal_position (scalar)
        temporal_position_expanded = temporal_position.unsqueeze(-1)  # [N, 1] for concatenation
        node_attr_with_temporal = torch.cat([node_attr, temporal_position_expanded], dim=-1)

        # Apply temporal gate
        node_attr_processed = self.temporal_gate(node_attr_with_temporal)
        # node_attr_processed = self.initial_linear(node_attr_gated)

        # Perform message passing with gated node attributes
        for layer in self.layers:
            node_attr_processed = layer(node_attr_processed, edge_index, edge_attr, edge_sh)
        node_attr_processed = self.output_head(node_attr_processed)

        # Pool over nodes.
        if self.reduce is not None:
            node_attr_processed = e3tools.scatter(
                node_attr_processed,
                index=batch,
                dim=0,
                dim_size=num_graphs,
                reduce=self.reduce,
            )

        return node_attr_processed


class E3SpatioTemporal(nn.Module):
    """
    E(3)-equivariant spatio-temporal model that combines spatial and temporal processing.

    This model implements the complete workflow:
    1. Process input spatial graph and hidden states through spatial module
    2. Pool spatial features to temporal graph representation
    3. Process temporal graph through temporal module
    4. Pool temporal features back to spatial representation
    5. Convert temporal graph back to spatial graph
    """

    def __init__(
        self,
        spatial_module: nn.Module,
        temporal_module: nn.Module,
        spatial_to_temporal_pooler: nn.Module,
        temporal_to_spatial_pooler: nn.Module,
        radial_cutoff: float,
        temporal_cutoff: float = 1.0,
    ):
        """
        Initialize the E3SpatioTemporal model.

        Args:
            spatial_module: Module for processing spatial positions (e.g., E3Conv)
            temporal_module: Module for processing temporal graphs (e.g., E3Transformer)
            spatial_to_temporal_pooler: Module to convert spatial-temporal features to temporal node attributes
            temporal_to_spatial_pooler: Module to convert temporal features back to spatial features
            radial_cutoff: Cutoff for spatial radial edge weights
            temporal_cutoff: Cutoff for temporal edge weights
        """
        super().__init__()

        self.spatial_module = spatial_module
        self.temporal_module = temporal_module
        self.spatial_to_temporal_pooler = spatial_to_temporal_pooler
        self.temporal_to_spatial_pooler = temporal_to_spatial_pooler
        self.radial_cutoff = radial_cutoff
        self.temporal_cutoff = temporal_cutoff

    def forward(
        self,
        batch: torch_geometric.data.Batch,
        c_noise: torch.Tensor,
        return_temporal_features: bool = False,
        return_temporal_graph: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass implementing the complete spatio-temporal workflow.

        Args:
            batch: Input spatial graph batch with pos, batch, num_graphs, and optionally hidden_state
            c_noise: Noise conditioning tensor
            return_temporal_features: Whether to return intermediate temporal features
            return_temporal_graph: Whether to return the temporal graph

        Returns:
            If return_temporal_features or return_temporal_graph is True, returns dict with:
                - 'spatial_features': Final spatial features
                - 'spatial_graph': Output spatial graph
                - 'temporal_features': Temporal features (if requested)
                - 'temporal_graph': Temporal graph (if requested)
            Otherwise returns just the final spatial features tensor
        """
        from convert_spatiotemporal import spatial_to_temporal_graphs, temporal_to_spatial_graphs

        # Store original device
        device = batch.pos.device

        # Step 1: Convert spatial graph to temporal graphs
        temporal_batch = spatial_to_temporal_graphs(batch)

        # Step 2: Process all positions (current + hidden states) with spatial module
        # Create topology for spatial processing (without positions)
        topology = batch.clone()
        # Remove position-dependent attributes but keep graph structure
        if hasattr(topology, "pos"):
            del topology.pos
        if hasattr(topology, "batch"):
            del topology.batch
        if hasattr(topology, "num_graphs"):
            del topology.num_graphs

        node_attr_list = []

        # Process current positions
        node_attr_current = self.spatial_module(
            batch.pos,
            topology,
            batch.batch,
            num_graphs=batch.num_graphs,
            c_noise=c_noise,
            effective_radial_cutoff=self.radial_cutoff,
        ).unsqueeze(1)  # [N, 1, features]
        node_attr_list.append(node_attr_current)

        # Process hidden state positions if they exist
        if hasattr(batch, "hidden_state") and batch.hidden_state is not None and len(batch.hidden_state) > 0:
            for hidden_pos in batch.hidden_state:
                node_attr_hidden = self.spatial_module(
                    hidden_pos,
                    topology,
                    batch.batch,
                    num_graphs=batch.num_graphs,
                    c_noise=c_noise,
                    effective_radial_cutoff=self.radial_cutoff,
                ).unsqueeze(1)  # [N, 1, features]
                node_attr_list.append(node_attr_hidden)

        # Step 3: Stack spatial-temporal features
        node_attr_spatial_temporal = torch.cat(node_attr_list, dim=1)  # [N, T, features]

        # Step 4: Convert spatial-temporal features to temporal node attributes
        temporal_node_attr = self.spatial_to_temporal_pooler(node_attr_spatial_temporal, temporal_batch)

        # Step 5: Process temporal graph through temporal module
        temporal_output = self.temporal_module(
            temporal_node_attr, temporal_batch, self.radial_cutoff, self.temporal_cutoff
        )

        # Step 6: Pool temporal features back to spatial features
        spatial_features = self.temporal_to_spatial_pooler(temporal_output, temporal_batch)

        # Step 7: Convert temporal graph back to spatial graph
        output_spatial_graph = temporal_to_spatial_graphs(temporal_batch)

        # Prepare return values
        if return_temporal_features or return_temporal_graph:
            result = {
                "spatial_features": spatial_features,
                "spatial_graph": output_spatial_graph,
            }
            if return_temporal_features:
                result["temporal_features"] = temporal_output
            if return_temporal_graph:
                result["temporal_graph"] = temporal_batch
            return result
        else:
            return spatial_features

    def get_spatial_output_irreps(self):
        """Get the irreps of the spatial module output."""
        if hasattr(self.spatial_module, "irreps_out"):
            return self.spatial_module.irreps_out
        else:
            raise AttributeError("Spatial module does not have irreps_out attribute")

    def get_temporal_output_irreps(self):
        """Get the irreps of the temporal module output."""
        if hasattr(self.temporal_module, "irreps_out"):
            return self.temporal_module.irreps_out
        else:
            raise AttributeError("Temporal module does not have irreps_out attribute")
