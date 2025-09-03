from collections.abc import Callable

import torch
from orb_models.forcefield.angular import SphericalHarmonics
from orb_models.forcefield.base import AtomGraphs
from orb_models.forcefield.rbf import BesselBasis

from jamun.model.arch.orb_gns import MoleculeGNS


class MoleculeGNSWrapper(torch.nn.Module):
    """A wrapper for the MoleculeGNS model."""

    def __init__(
        self,
        latent_dim: int,
        num_message_passing_steps: int,
        base_mlp_hidden_dim: int,
        base_mlp_depth: int,
        head_mlp_hidden_dim: int,
        head_mlp_depth: int,
        activation: str,
        sh_lmax: int,
        max_radius: float,
        bessel_num_bases: int,
        atom_embedder_factory: Callable[..., torch.nn.Module],
        bond_edge_embedder_factory: Callable[..., torch.nn.Module],
        use_torch_compile: bool = True,
    ):
        super().__init__()

        self.atom_embedder = atom_embedder_factory()
        self.bond_edge_embedder = bond_edge_embedder_factory()

        self.gns = MoleculeGNS(
            latent_dim=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            num_mlp_layers=base_mlp_depth,
            mlp_hidden_dim=base_mlp_hidden_dim,
            rbf_transform=BesselBasis(
                r_max=max_radius,
                num_bases=bessel_num_bases,
            ),
            angular_transform=SphericalHarmonics(
                lmax=sh_lmax,
                normalize=True,
                normalization="component",
            ),
            outer_product_with_cutoff=True,
            use_embedding=False,
            expects_atom_type_embedding=False,
            interaction_params={
                "distance_cutoff": True,
                "attention_gate": "sigmoid",
            },
            edge_feature_names=["bond_edge_embedding"],
            extra_embed_dims=(
                self.atom_embedder.irreps_out.dim - 118,
                self.bond_edge_embedder.irreps_out.dim,
            ),
            activation=activation,
            mlp_norm="rms_norm",
            max_radius=max_radius,
        )
        if use_torch_compile:
            self.gns.compile(fullgraph=True, dynamic=True)

    def update_features(self, pos: torch.Tensor, topology: AtomGraphs, c_in: torch.Tensor) -> AtomGraphs:
        atomic_numbers_embedding = self.atom_embedder(topology)
        bond_edge_embedding = self.bond_edge_embedder(topology)
        unscaled_pos = pos / c_in
        vectors = unscaled_pos[topology.senders] - unscaled_pos[topology.receivers]

        topology = topology._replace(
            node_features={
                **topology.node_features,
                "positions": pos,
                "atomic_numbers_embedding": atomic_numbers_embedding,
            },
            edge_features={
                **topology.edge_features,
                "vectors": vectors,
                "bond_edge_embedding": bond_edge_embedding,
            },
        )
        return topology

    def forward(
        self,
        pos: torch.Tensor,
        topology: AtomGraphs,
        batch: torch.Tensor,
        num_graphs: int,
        c_noise: torch.Tensor,
        c_in: torch.Tensor,
    ) -> torch.Tensor:
        del batch, num_graphs, c_noise
        topology = self.update_features(pos, topology, c_in)
        return self.gns(topology)["pred"]
