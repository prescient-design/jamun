import torch
from orb_models.forcefield.angular import SphericalHarmonics
from orb_models.forcefield.base import AtomGraphs
from orb_models.forcefield.gns import MoleculeGNS
from orb_models.forcefield.rbf import BesselBasis


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
    ):
        super().__init__()

        self.model = MoleculeGNS(
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
            use_embedding=True,
            interaction_params={
                "distance_cutoff": True,
                "attention_gate": "sigmoid",
            },
            node_feature_names=["feat"],
            edge_feature_names=["feat"],
            activation=activation,
            mlp_norm="rms_norm",
        )

    def forward(
        self,
        pos: torch.Tensor,
        topology: AtomGraphs,
        c_noise: torch.Tensor,
        effective_radial_cutoff: float,
    ) -> torch.Tensor:
        del pos, c_noise, effective_radial_cutoff
        return self.model(topology)["pred"]
