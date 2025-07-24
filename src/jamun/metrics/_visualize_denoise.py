import os
import tempfile

import einops
import matplotlib.pyplot as plt
import mdtraj as md
import torch
import torch_geometric
import torchmetrics
import wandb
from torchmetrics.utilities import dim_zero_cat

from jamun import utils
from jamun.data import MDtrajDataset
from jamun.metrics._ramachandran import plot_ramachandran
from jamun.metrics._utils import validate_sample


def plot_ramachandran_grid(trajs: dict[str, md.Trajectory], dataset_label: str):
    """Plot a grid of Ramachandran plots for each trajectory."""
    # Create the figure and subplots.
    num_dihedrals = md.compute_phi(trajs["x"], periodic=False)[1].shape[1]
    fig, axes = plt.subplots(nrows=3, ncols=num_dihedrals, figsize=(5 * num_dihedrals, 15), squeeze=False)
    fig.suptitle(f"Ramachandran Plots for Dataset {dataset_label}")

    # Iterate through the data and plot.
    for j in range(num_dihedrals):
        for i, key in enumerate(["x", "y", "xhat"]):
            ax = axes[i, j]
            _, im = plot_ramachandran(trajs[key], dihedral_index=j, fig=fig, ax=ax, colorbar=False)
            ax.set_title(f"Dihedral {j + 1}")

            # Only add labels for the first column
            if j == 0:
                fig.text(0.05, (3 - i - 0.5) / 3, key, va="center", ha="right", fontsize=12, fontweight="bold")

    # Add colorbar.
    plt.colorbar(im, ax=axes.ravel().tolist())
    return fig, axes


class VisualizeDenoiseMetrics(torchmetrics.Metric):
    """Plots and computes metrics for samples from a single dataset."""

    def __init__(self, dataset: MDtrajDataset, sigma_list: list[float]):
        # TODO: Understand why we need sync_on_compute=False.
        super().__init__(sync_on_compute=False)

        self.dataset = dataset
        self.sigma_list = sigma_list

        # torchmetrics doesn't support Dicts as state, so we store the coordinates as a list of tensors.
        self.add_state("has_samples", default=torch.tensor(False), dist_reduce_fx="sum")
        for sigma in sigma_list:
            for key in ["x", "y", "xhat"]:
                self.add_state(f"coordinates_{sigma}_{key}", default=[], dist_reduce_fx="cat")

    def update(
        self,
        xhat: torch_geometric.data.Batch,
        y: torch_geometric.data.Batch,
        x: torch_geometric.data.Batch,
        sigma: float,
    ) -> None:
        """Update the metric with a new sample."""
        samples = {
            "xhat": xhat,
            "y": y,
            "x": x,
        }
        for key, sample in samples.items():
            validate_sample(sample, self.dataset)

            key_coordinates = sample.pos
            if key_coordinates.ndim != 2:
                raise ValueError(f"Invalid sample shape: {key_coordinates.shape}, expected (num_atoms, 3).")

            # Reshape key_coordinates to be of shape (1, num_atoms, 3).
            coordinates_sigma_key = getattr(self, f"coordinates_{sigma}_{key}")
            coordinates_sigma_key.append(key_coordinates[None])
            setattr(self, f"coordinates_{sigma}_{key}", coordinates_sigma_key)

            self.has_samples = torch.tensor(True, device=self.device)

    def coordinates_to_trajectories(self) -> dict[float, dict[str, md.Trajectory]]:
        all_trajs = {}
        for sigma in self.sigma_list:
            sigma_trajs = {}
            for key in ["x", "y", "xhat"]:
                coords = getattr(self, f"coordinates_{sigma}_{key}")
                coords = dim_zero_cat(coords)
                coords = einops.rearrange(coords, "b n x -> n b x")
                traj = utils.coordinates_to_trajectories(coords, self.dataset.topology)[0]
                sigma_trajs[key] = traj
            all_trajs[sigma] = sigma_trajs
        return all_trajs

    def compute(self) -> tuple[dict[str, md.Trajectory] | None, dict[float, float] | None]:
        if not self.has_samples:
            return None, None

        # Convert the coordinates to MDtraj trajectories.
        # Note that these do not actually correspond to any actual trajectories, since the samples are iid from the dataset.
        trajectories = self.coordinates_to_trajectories()

        # Compute the scaled RMSD for each sigma
        scaled_rmsd_per_sigma = {}
        for sigma in self.sigma_list:
            xhat = dim_zero_cat(getattr(self, f"coordinates_{sigma}_xhat"))
            xhat = einops.rearrange(xhat, "b n x -> n b x")
            x = dim_zero_cat(getattr(self, f"coordinates_{sigma}_x"))
            x = einops.rearrange(x, "b n x -> n b x")

            assert xhat.ndim == x.ndim == 3, f"{xhat.shape=}"

            xhat -= xhat.mean(dim=0, keepdim=True)
            x -= x.mean(dim=0, keepdim=True)

            scaled_rmsd_per_sigma[sigma] = utils.scaled_rmsd(x, xhat, sigma)
        return trajectories, scaled_rmsd_per_sigma

    def log(
        self,
        trajectories: dict[str, md.Trajectory] | None = None,
        scaled_rmsd_per_sigma: dict[float, float] | None = None,
    ) -> None:
        if trajectories is None:
            trajectories, _ = self.compute()

        for sigma, sigma_trajs in trajectories.items():
            # Convert the trajectories to RDKit mols.
            mols = {key: utils.to_rdkit_mols(traj[:5]) for key, traj in sigma_trajs.items()}

            # Plot with py3Dmol.
            view = utils.plot_molecules_with_py3Dmol(mols)

            # Log the HTML file to Weights & Biases.
            temp_html = tempfile.NamedTemporaryFile(suffix=".html").name
            view.write_html(temp_html)
            with open(temp_html) as f:
                utils.wandb_dist_log({f"{self.dataset.label()}/visualize_denoise/3D_view/sigma={sigma}": wandb.Html(f)})
            os.remove(temp_html)

        try:
            for sigma, sigma_trajs in trajectories.items():
                fig, _ = plot_ramachandran_grid(sigma_trajs, self.dataset.label())

                utils.wandb_dist_log(
                    {
                        f"{self.dataset.label()}/visualize_denoise/ramachandran_plots_static/sigma={sigma}": wandb.Image(
                            fig
                        )
                    }
                )
                plt.close(fig)
        except ValueError:
            pass

        if scaled_rmsd_per_sigma is not None:
            for sigma, scaled_rmsd in scaled_rmsd_per_sigma.items():
                utils.wandb_dist_log({f"{self.dataset.label()}/scaled_rmsd_per_dataset/sigma={sigma}": scaled_rmsd})
