import logging
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib.cm
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import ot
import wandb
from lightning.pytorch.utilities import rank_zero_only
from matplotlib import animation, colors
from scipy.spatial import distance

from jamun import utils
from jamun.metrics._utils import TrajectoryMetric


def num_dihedrals(trajectory: md.Trajectory) -> int:
    """Get the number of dihedrals in a trajectory."""
    return md.compute_phi(trajectory)[1].shape[1]


def get_ramachandran_angles(
    trajectory: md.Trajectory, dihedral_index: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the phi and psi angles from a trajectory."""
    _, phi_angles = md.compute_phi(trajectory, periodic=False)
    _, psi_angles = md.compute_psi(trajectory, periodic=False)
    if dihedral_index is None:
        return phi_angles, psi_angles
    return phi_angles[:, dihedral_index, None], psi_angles[:, dihedral_index, None]


def plot_ramachandran(
    trajectory: md.Trajectory,
    dihedral_index: Optional[int] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    colorbar: bool = True,
) -> Tuple[plt.Figure, matplotlib.cm.ScalarMappable]:
    """Creates a Ramachandran plot from a trajectory."""
    phi_angles, psi_angles = get_ramachandran_angles(trajectory, dihedral_index)

    if fig is None or ax is None:
        fig, ax = plt.subplots()
    _, _, _, im = ax.hist2d(
        phi_angles.flatten(),
        psi_angles.flatten(),
        bins=100,
        range=((-np.pi, np.pi), (-np.pi, np.pi)),
        cmap="viridis",
        norm=colors.LogNorm(),
        density=True,
    )
    if colorbar:
        fig.colorbar(im)
    ax.grid(linestyle="--")
    ax.set_xticks([-np.pi, 0, np.pi], labels=[r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_yticks([-np.pi, 0, np.pi], labels=[r"$-\pi$", r"$0$", r"$\pi$"])
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")

    return fig, im


def plot_ramachandran_animation(
    trajectory: md.Trajectory,
    title: str,
    subsample_factor: int = 1,
    accumulation: bool = True,
    dihedral_index: Optional[int] = None,
) -> animation.FuncAnimation:
    """Creates a Ramachandran plot from a trajectory."""
    phi_angles, psi_angles = get_ramachandran_angles(trajectory, dihedral_index)

    fig, ax = plt.subplots()
    h, xedges, yedges = np.histogram2d(
        phi_angles[:subsample_factor].flatten(),
        psi_angles[:subsample_factor].flatten(),
        bins=100,
        range=((-np.pi, np.pi), (-np.pi, np.pi)),
        density=True,
    )
    im = ax.pcolormesh(xedges, yedges, h.T, cmap="viridis", norm=colors.LogNorm())
    plt.colorbar(im)
    plt.grid(linestyle="--")
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\psi$")
    plt.title(title)
    plt.tight_layout()

    # Animation update function
    def init():
        return (im,)

    def update(frame):
        frame = max(frame, 1)

        # Get the frame
        if accumulation:
            phi_angles_for_frame = phi_angles[frame * subsample_factor : (frame + 1) * subsample_factor]
            psi_angles_for_frame = psi_angles[frame * subsample_factor : (frame + 1) * subsample_factor]
        else:
            phi_angles_for_frame = phi_angles[frame * subsample_factor : (frame + 1) * subsample_factor]
            psi_angles_for_frame = psi_angles[frame * subsample_factor : (frame + 1) * subsample_factor]

        # Update the histogram
        h, _, _ = np.histogram2d(
            phi_angles_for_frame.flatten(),
            psi_angles_for_frame.flatten(),
            bins=100,
            range=((-np.pi, np.pi), (-np.pi, np.pi)),
            density=True,
        )
        im.set_array(h.T.ravel())

        return (im,)

    # Create the animation
    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(trajectory) // subsample_factor, interval=100, blit=True
    )
    return anim


def compute_ramachandran_histogram(trajectory: md.Trajectory, bins: int):
    """Computes the Ramachandran histogram from a trajectory."""
    phi_angles, psi_angles = get_ramachandran_angles(trajectory)
    hist, _, _ = np.histogram2d(
        phi_angles.flatten(), psi_angles.flatten(), bins=bins, range=((-np.pi, np.pi), (-np.pi, np.pi))
    )
    hist /= hist.sum()
    return hist


def compute_JS_divergence_of_ramachandran(
    trajectory: md.Trajectory, ref_trajectory: md.Trajectory, bins: int = 100
) -> float:
    """Computes the Jensen-Shannon divergence between the Ramachandran histograms from two trajectories."""

    hist = compute_ramachandran_histogram(trajectory, bins)
    ref_hist = compute_ramachandran_histogram(ref_trajectory, bins)

    return distance.jensenshannon(hist.flatten(), ref_hist.flatten()) ** 2


def compute_sliced_Wasserstein_distance_of_ramachandran(
    trajectory: md.Trajectory, ref_trajectory: md.Trajectory, n_projections: int = 20
) -> float:
    """Computes the sliced Wasserstein distance between the Ramachandran plots from two trajectories."""

    def compute_descriptors(phi_angles: np.ndarray, psi_angles: np.ndarray) -> np.ndarray:
        """Computes the descriptors for the sliced Wasserstein distance."""
        return np.concatenate([np.cos(phi_angles), np.sin(phi_angles), np.cos(psi_angles), np.sin(psi_angles)], axis=-1)

    phi_angles, psi_angles = get_ramachandran_angles(trajectory)
    descriptors = compute_descriptors(phi_angles, psi_angles)

    ref_phi_angles, ref_psi_angles = get_ramachandran_angles(ref_trajectory)
    ref_descriptors = compute_descriptors(ref_phi_angles, ref_psi_angles)

    assert descriptors.ndim == ref_descriptors.ndim == 2
    assert descriptors.shape[-1] == ref_descriptors.shape[-1]

    return ot.sliced_wasserstein_distance(
        descriptors,
        ref_descriptors,
        n_projections=n_projections,
    )


def _num_subsamples_sequence_for_trajectory(trajectory: md.Trajectory) -> List[int]:
    """Computes the sequence of subsamples for metrics for a trajectory."""
    num_samples_list = [100 * (2**i) for i in range(10) if 100 * (2**i) < len(trajectory)]
    num_samples_list.append(len(trajectory))
    return num_samples_list


def compute_JS_divergence_vs_num_samples(
    trajectory: md.Trajectory, ref_trajectory: md.Trajectory
) -> List[Tuple[int, float]]:
    """Compute the JS divergence versus the reference trajectory, as the number of samples is increased."""
    jsds = []
    for num_samples in _num_subsamples_sequence_for_trajectory(trajectory):
        jsd = compute_JS_divergence_of_ramachandran(trajectory[:num_samples], ref_trajectory)
        jsds.append([num_samples, jsd])
    return jsds


def compute_sliced_Wasserstein_distance_vs_num_samples(
    trajectory: md.Trajectory, ref_trajectory: md.Trajectory
) -> List[Tuple[int, float]]:
    """Compute the sliced Wasserstein distance versus the reference trajectory, as the number of samples is increased."""
    wsds = []
    for num_samples in _num_subsamples_sequence_for_trajectory(trajectory):
        ws_distance = compute_sliced_Wasserstein_distance_of_ramachandran(trajectory[:num_samples], ref_trajectory)
        wsds.append([num_samples, ws_distance])
    return wsds


def _log_js_divergence_vs_num_samples(
    trajectory: md.Trajectory, ref_trajectory: md.Trajectory, label: str, title: str
) -> None:
    # Log the JS divergence vs. number of samples plot.
    table = wandb.Table(
        data=compute_JS_divergence_vs_num_samples(trajectory, ref_trajectory),
        columns=["Number of Samples", "JS Divergence"],
    )
    if rank_zero_only.rank == 0:
        utils.wandb_dist_log(
            {
                label: wandb.plot.line(
                    table,
                    "Number of Samples",
                    "JS Divergence",
                    title=title,
                )
            }
        )


def _log_ws_distance_vs_num_samples(trajectory: md.Trajectory, ref_trajectory: md.Trajectory, label: str, title: str):
    # Log the WS distance vs. number of samples plot.
    table = wandb.Table(
        data=compute_sliced_Wasserstein_distance_vs_num_samples(trajectory, ref_trajectory),
        columns=["Number of Samples", "Sliced Wasserstein Distance"],
    )

    if rank_zero_only.rank == 0:
        utils.wandb_dist_log(
            {
                label: wandb.plot.line(
                    table,
                    "Number of Samples",
                    "Sliced Wasserstein Distance",
                    title=title,
                )
            }
        )


class RamachandranPlotMetrics(TrajectoryMetric):
    """Plots and computes metrics for samples from a single dataset."""

    def __init__(self, show_animation: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show_animation = show_animation

    def on_sample_start(self):
        # Hide the matplotlib logging.
        plt.set_loglevel("warning")
        py_logger = logging.getLogger("ramachandran")

        # Plot the Ramachandran for the true trajectory.
        true_trajectory = self.dataset.trajectory
        py_logger.info(f"{self.dataset.label()}: Loaded true trajectory {true_trajectory}.")
        for dihedral_index in range(num_dihedrals(true_trajectory)):
            fig, _ = plot_ramachandran(true_trajectory, dihedral_index=dihedral_index)
            utils.wandb_dist_log(
                {f"{self.dataset.label()}/ramachandran_static/dihedral_{dihedral_index}/true_traj": wandb.Image(fig)}
            )
            utils.wandb_dist_log(
                {f"{self.dataset.label()}/ramachandran_static/true_traj/dihedral_{dihedral_index}": wandb.Image(fig)}
            )
            plt.close(fig)

        # Plot the metrics for the true trajectory.
        _log_js_divergence_vs_num_samples(
            true_trajectory,
            true_trajectory,
            label=f"{self.dataset.label()}/js_divergence_vs_num_samples/true_traj",
            title="Jenson-Shannon Divergence vs. Number of Samples for True Trajectory",
        )
        _log_ws_distance_vs_num_samples(
            true_trajectory,
            true_trajectory,
            label=f"{self.dataset.label()}/ws_distance_vs_num_samples/true_traj",
            title="Sliced Wasserstein Distance vs. Number of Samples for True Trajectory",
        )

    def compute(self) -> Dict[str, float]:
        # Hide the matplotlib logging.
        plt.set_loglevel("warning")
        py_logger = logging.getLogger("jamun")

        # Convert the samples to trajectories.
        pred_trajectories = self.sample_trajectories(new=True)
        pred_trajectory_joined = self.joined_sample_trajectory()
        py_logger.info(
            f"{self.dataset.label()}: Obtained predicted trajectories {len(pred_trajectories)=} {pred_trajectories}."
        )

        # Make plots for each trajectory.
        for trajectory_index, pred_trajectory in enumerate(
            pred_trajectories + [pred_trajectory_joined], start=self.num_chains_seen
        ):
            if trajectory_index == len(pred_trajectories) + self.num_chains_seen:
                trajectory_index = "joined"

            # Compute the Ramachandran plot.
            for dihedral_index in range(num_dihedrals(pred_trajectory)):
                fig, _ = plot_ramachandran(pred_trajectory, dihedral_index=dihedral_index)
                utils.wandb_dist_log(
                    {
                        f"{self.dataset.label()}/ramachandran_static/dihedral_{dihedral_index}/pred_traj_{trajectory_index}": wandb.Image(
                            fig
                        )
                    }
                )
                utils.wandb_dist_log(
                    {
                        f"{self.dataset.label()}/ramachandran_static/pred_traj_{trajectory_index}/dihedral_{dihedral_index}": wandb.Image(
                            fig
                        )
                    }
                )
                plt.close(fig)

                # Make an animation of the Ramachandran plot.
                if self.show_animation:
                    subsample_factor = max(1, len(pred_trajectory) // 100)
                    anim = plot_ramachandran_animation(
                        pred_trajectory,
                        "Ramachandran Plot",
                        subsample_factor=subsample_factor,
                        dihedral_index=dihedral_index,
                    )
                    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_mp4:
                        anim.save(temp_mp4.name, writer="ffmpeg")
                        utils.wandb_dist_log(
                            {
                                f"{self.dataset.label()}/ramachandran_animation/dihedral_{dihedral_index}/pred_traj_{trajectory_index}": wandb.Video(
                                    temp_mp4.name
                                )
                            }
                        )
                    plt.close()

            # Log the metrics for the predicted trajectory.
            true_trajectory = self.dataset.trajectory
            _log_js_divergence_vs_num_samples(
                pred_trajectory,
                true_trajectory,
                label=f"{self.dataset.label()}/js_divergence_vs_num_samples/pred_traj_{trajectory_index}",
                title=f"Jenson-Shannon Divergence vs. Number of Samples for Predicted Trajectory {trajectory_index}",
            )
            _log_ws_distance_vs_num_samples(
                pred_trajectory,
                true_trajectory,
                label=f"{self.dataset.label()}/ws_distance_vs_num_samples/pred_traj_{trajectory_index}",
                title=f"Sliced Wasserstein Distance vs. Number of Samples for Predicted Trajectory {trajectory_index}",
            )

        return {}
