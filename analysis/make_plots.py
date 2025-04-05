from typing import Dict, Optional, Any
import os
import pickle
import argparse

import logging

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("numexpr.utils").setLevel(logging.ERROR)

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("analysis")

import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import lovelyplots
import matplotlib as mpl
import matplotlib.colors

mpl.rcParams["axes.formatter.useoffset"] = False
mpl.rcParams["axes.formatter.limits"] = (-10000, 10000)  # Controls range before scientific notation is used
plt.style.use("ipynb")

from jamun import utils

import sys

sys.path.append("./")

import pyemma_helper
import load_trajectory


def load_results(results_dir: str, experiment: str, traj_name: str, ref_traj_name: str) -> pd.DataFrame:
    """Loads the results as a pandas DataFrame."""

    results_path = os.path.join(results_dir, experiment, traj_name, f"ref={ref_traj_name}")
    py_logger.info(f"Searching for results in {results_path}")

    results = []
    for results_file in sorted(os.listdir(results_path)):
        peptide, ext = os.path.splitext(results_file)
        if ext != ".pkl":
            continue

        with open(os.path.join(results_path, results_file), "rb") as f:
            all_results = pickle.load(f)

        results.append(
            {
                "traj": traj_name,
                "ref_traj": ref_traj_name,
                "peptide": peptide,
                "results": all_results["results"],
                "args": all_results["args"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.reset_index(drop=True)

    results_df.attrs["experiment"] = experiment
    results_df.attrs["traj_name"] = traj_name
    results_df.attrs["ref_traj_name"] = ref_traj_name

    return results_df


def get_format_traj_name_fn(results_df: pd.DataFrame):
    def format_traj_name_fn(traj_name: str) -> str:
        """Format the trajectory name for plotting"""
        known_names = {
            "traj": results_df.attrs["traj_name"],
            "ref_traj": "Reference",
            "ref_traj_10x": "Reference\n(10x shorter)",
            "ref_traj_100x": "Reference\n(100x shorter)",
            "ref_traj_1000x": "Reference\n(1000x shorter)",
            "TBG": "TBG",
            "TBG_20x": "TBG\n(20x shorter)",
            "TBG_200x": "TBG\n(200x shorter)",
            "JAMUN_0.2A": "JAMUN\n" + r"($\sigma=0.2\AA$)",
            "JAMUN_0.8A": "JAMUN\n" + r"($\sigma=0.8\AA$)",
        }
        if traj_name in known_names:
            return known_names[traj_name]

        return traj_name

    return format_traj_name_fn


def get_short_format_traj_name_fn(results_df: pd.DataFrame):
    long_name = get_format_traj_name_fn(results_df)("traj")
    if "\n" in long_name:
        short_name = long_name.replace("\n", " ")
    else:
        short_name = long_name

    def format_traj_name_fn(traj_name: str) -> str:
        """Format the trajectory name for plotting"""
        known_names = {
            "traj": short_name,
            "ref_traj": "Ref.",
            "ref_traj_10x": "Ref. (10x shorter)",
            "ref_traj_100x": "Ref. (100x shorter)",
            "ref_traj_1000x": "Ref. (1000x shorter)",
            "TBG": "TBG",
            "TBG_20x": "TBG (20x shorter)",
            "TBG_200x": "TBG (200x shorter)",
            "JAMUN_0.2A": r"JAMUN ($\sigma=0.2\AA$)",
            "JAMUN_0.8A": r"JAMUN ($\sigma=0.8\AA$)",
        }
        if traj_name in known_names:
            return known_names[traj_name]

        return traj_name

    return format_traj_name_fn


def get_colors_fn(results_df: pd.DataFrame):
    colors = {
        "traj": "#FF7F0E",  # Orange
        "ref_traj": "#1F77B4",  # Dark blue
        "ref_traj_10x": "#6BAED6",  # Medium blue
        "ref_traj_100x": "#BDD7E7",  # Light blue
        "ref_traj_1000x": "#F7FBFF",  # Very light blue
        "TBG": "#2CA02C",  # Dark green
        "TBG_20x": "#74C476",  # Medium green
        "TBG_200x": "#BAE4B3",  # Light green
        "JAMUN_0.2A": "#FFB366",  # Dark red
        "JAMUN_0.8A": "#CC5500",  # Light red
        "MDGen": "#DDA0DD",  # Purple
    }
    return lambda traj_name: colors.get(traj_name, "#000000")  # Default to black if not found


def format_quantity(quantity: str) -> str:
    """Format the quantity for plotting."""
    if not quantity.startswith("JSD_"):
        return format_quantity("JSD_" + quantity)

    return {
        "JSD_backbone_torsions": "Backbone Torsions",
        "JSD_sidechain_torsions": "Sidechain Torsions",
        "JSD_all_torsions": "All Torsions",
        "JSD_TICA-0": "TICA-0 Projections",
        "JSD_TICA-0,1": "TICA-0,1 Projections",
        "JSD_metastable_probs": "Metastable State Probabilities",
    }[quantity]


def format_peptide_name(peptide: str) -> str:
    """Formats the peptide name for plotting."""
    # TODO: Return a function like get_format_traj_name_fn
    if peptide == "filtered":
        return "Chignolin"
    if peptide.startswith("uncapped_"):
        peptide = peptide[len("uncapped_") :]
    if peptide.startswith("capped_"):
        peptide = peptide[len("capped_") :]
    return utils.convert_to_one_letter_codes(peptide)



def plot_ramachandran_contour(results: Dict[str, Any], dihedral_index: int, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plots the Ramachandran contour plot of a trajectory."""

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    pmf, xedges, yedges = results["pmf"], results["xedges"], results["yedges"]
    im = ax.contourf(xedges[:-1], yedges[:-1], pmf[dihedral_index], cmap="viridis", levels=50)
    contour = ax.contour(
        xedges[:-1], yedges[:-1], pmf[dihedral_index], colors="white", linestyles="solid", levels=10, linewidths=0.25
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$\phi$")
    ax.set_ylabel("$\psi$")

    tick_eps = 0.1
    ticks = [-np.pi + tick_eps, -np.pi / 2, 0, np.pi / 2, np.pi - tick_eps]
    tick_labels = ["$-\pi$", "$-\pi/2$", "$0$", "$\pi/2$", "$\pi$"]
    ax.set_xticks(ticks, tick_labels)
    ax.set_yticks(ticks, tick_labels)
    return ax


def get_num_dihedrals(experiment: str, pmf_type: str) -> int:
    # "internal" for psi_2 - phi_2, psi_3 - phi_3, etc.
    # "all" for psi_1 - phi_2, psi_2 - phi_3, etc.
    if pmf_type not in ["internal", "all"]:
        raise ValueError(f"Invalid pmf_type: {pmf_type}")

    if experiment == "Our_2AA":
        num_dihedrals = 2
    elif "2AA" in experiment:
        num_dihedrals = 1
    elif "4AA" in experiment:
        num_dihedrals = 3
    elif "5AA" in experiment:
        num_dihedrals = 4
    elif "Chignolin" in experiment:
        num_dihedrals = 9

    if pmf_type == "internal":
        num_dihedrals -= 1

    return num_dihedrals


def get_JSD_results(results_df, quantity: str, name: str, key: str):
    """Helper to load final JSD results."""
    JSDs = []

    for i, row in results_df.iterrows():
        try:
            JSD = row["results"][key][name][quantity]
        except KeyError:
            continue
        JSDs.append(JSD)

    JSDs = np.asarray(JSDs)
    return JSDs


def get_all_JSD_results(results_df: pd.DataFrame):
    """Helper to load all final JSD results."""
    JSD_results = {
        "JSD_backbone_torsions": {},
        "JSD_sidechain_torsions": {},
        "JSD_all_torsions": {},
        "JSD_TICA-0": {},
        "JSD_TICA-0,1": {},
        "JSD_metastable_probs": {},
    }
    traj_names = ["traj", "ref_traj"]
    if results_df.attrs.get("extra_traj_names") is not None:
        traj_names.extend(results_df.attrs["extra_traj_names"])
    else:
        traj_names = ["traj", "ref_traj", "ref_traj_10x", "ref_traj_100x"]

    for quantity in ["JSD_backbone_torsions", "JSD_sidechain_torsions", "JSD_all_torsions"]:
        for name in traj_names:
            JSD_results[quantity][name] = get_JSD_results(results_df, quantity, name, "JSD_torsions")

    for quantity in ["JSD_TICA-0", "JSD_TICA-0,1"]:
        for name in traj_names:
            JSD_results[quantity][name] = get_JSD_results(results_df, quantity, name, "JSD_TICA")

    for quantity in ["JSD_metastable_probs"]:
        for name in traj_names:
            JSD_results[quantity][name] = get_JSD_results(results_df, quantity, name, "JSD_MSM")

    return JSD_results


def get_JSD_results_against_time(results_df, quantity: str, name: str, key: str) -> np.ndarray:
    """Helper to load JSD vs time results."""
    JSD_vs_time = {"progress": None, "JSDs": []}

    for i, row in results_df.iterrows():
        try:
            results = row["results"][key]
            progress = np.asarray(list(results[name].keys()))
        except KeyError:
            continue

        if JSD_vs_time["progress"] is None:
            JSD_vs_time["progress"] = progress

        assert np.allclose(JSD_vs_time["progress"], progress), (progress, JSD_vs_time["progress"])

        JSDs = np.asarray(list([v[quantity] for v in results[name].values()]))
        JSD_vs_time["JSDs"].append(JSDs)

    JSD_vs_time["JSDs"] = np.stack(JSD_vs_time["JSDs"])
    return JSD_vs_time


def get_all_JSD_results_against_time(results_df: pd.DataFrame):
    """Helper to load all JSD vs time results."""

    JSD_results = {
        "JSD_backbone_torsions": {},
        "JSD_sidechain_torsions": {},
        "JSD_all_torsions": {},
        "JSD_TICA-0": {},
        "JSD_TICA-0,1": {},
        "JSD_metastable_probs": {},
    }
    traj_names = ["traj", "ref_traj"]
    if results_df.attrs.get("extra_traj_names") is not None:
        traj_names.extend(results_df.attrs["extra_traj_names"])
    else:
        traj_names = ["traj", "ref_traj", "ref_traj_10x", "ref_traj_100x"]

    for quantity in ["JSD_backbone_torsions", "JSD_sidechain_torsions", "JSD_all_torsions"]:
        for name in traj_names:
            JSD_results[quantity][name] = get_JSD_results_against_time(
                results_df, quantity, name, "JSD_torsions_against_time"
            )

    for quantity in ["JSD_TICA-0", "JSD_TICA-0,1"]:
        for name in traj_names:
            JSD_results[quantity][name] = get_JSD_results_against_time(
                results_df, quantity, name, "JSD_TICA_against_time"
            )

    for quantity in ["JSD_metastable_probs"]:
        for name in traj_names:
            JSD_results[quantity][name] = get_JSD_results_against_time(
                results_df, quantity, name, "JSD_MSM_against_time"
            )

    return JSD_results


def plot_ramachandran_against_reference(results_df: pd.DataFrame) -> None:
    """Plots Ramachandran contours against the reference trajectory."""
    pmf_type = "all"
    experiment = results_df.attrs["experiment"]
    num_dihedrals = get_num_dihedrals(experiment, pmf_type)
    label_offset = 0.0 if num_dihedrals % 2 == 0 else 0.5
    format_traj_name_fn = get_format_traj_name_fn(results_df)

    ones = list(np.ones(num_dihedrals))
    fig, axs = plt.subplots(
        len(results_df),
        2 * num_dihedrals + 1,
        figsize=(8 * num_dihedrals, 4 * len(results_df)),
        gridspec_kw={"width_ratios": ones + [0.1] + ones, "hspace": 0.1},
        squeeze=False,
    )
    for i, row in results_df.iterrows():
        peptide = row["peptide"]

        for j in range(num_dihedrals):
            plot_ramachandran_contour(row["results"]["PMFs"]["ref_traj"][f"pmf_{pmf_type}"], j, axs[i, j])
            plot_ramachandran_contour(
                row["results"]["PMFs"]["traj"][f"pmf_{pmf_type}"], j, axs[i, j + num_dihedrals + 1]
            )

        # Add labels.
        ax_index = num_dihedrals // 2
        axs[0, ax_index].text(
            label_offset,
            1.1,
            format_traj_name_fn("ref_traj"),
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[0, ax_index].transAxes,
            fontsize=22,
        )

        ax_index = num_dihedrals // 2 + num_dihedrals + 1
        axs[0, ax_index].text(
            label_offset,
            1.1,
            format_traj_name_fn("traj"),
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[0, ax_index].transAxes,
            fontsize=22,
        )

        ax_index = -1
        axs[i, ax_index].text(
            1.1,
            0.5,
            format_peptide_name(peptide),
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            transform=axs[i, ax_index].transAxes,
            fontsize=18,
        )

        axs[i, num_dihedrals].axis("off")

        if i != len(axs) - 1:
            for j in range(len(axs[i])):
                axs[i, j].set_xticks([])
                axs[i, j].set_xlabel("")

        for j in range(1, len(axs[i])):
            axs[i, j].set_yticks([])
            axs[i, j].set_ylabel("")

    plt.subplots_adjust(hspace=0.06, wspace=0.04)


def plot_ramachandran_against_reference_shortened(results_df: pd.DataFrame) -> None:
    """Plots Ramachandran contours against a shortened reference trajectory."""
    pmf_type = "all"
    experiment = results_df.attrs["experiment"]
    num_dihedrals = get_num_dihedrals(experiment, pmf_type)
    label_offset = 0.0 if num_dihedrals % 2 == 0 else 0.5
    format_traj_name_fn = get_format_traj_name_fn(results_df)

    ones = list(np.ones(num_dihedrals))
    fig, axs = plt.subplots(
        len(results_df),
        3 * num_dihedrals + 2,
        figsize=(12 * num_dihedrals, 4 * len(results_df)),
        gridspec_kw={"width_ratios": ones + [0.1] + ones + [0.1] + ones, "hspace": 0.1},
        squeeze=False,
    )

    for i, row in results_df.iterrows():
        peptide = row["peptide"]

        for j in range(num_dihedrals):
            plot_ramachandran_contour(row["results"]["PMFs"]["ref_traj"][f"pmf_{pmf_type}"], j, axs[i, j])
            plot_ramachandran_contour(
                row["results"]["PMFs"]["traj"][f"pmf_{pmf_type}"], j, axs[i, j + num_dihedrals + 1]
            )
            plot_ramachandran_contour(
                row["results"]["PMFs"]["ref_traj_100x"][f"pmf_{pmf_type}"], j, axs[i, j + 2 * num_dihedrals + 2]
            )

        # Add labels.
        ax_index = num_dihedrals // 2
        axs[0, ax_index].text(
            label_offset,
            1.1,
            format_traj_name_fn("ref_traj"),
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[0, ax_index].transAxes,
            fontsize=22,
        )

        ax_index = num_dihedrals // 2 + num_dihedrals + 1
        axs[0, ax_index].text(
            label_offset,
            1.1,
            format_traj_name_fn("traj"),
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[0, ax_index].transAxes,
            fontsize=22,
        )

        ax_index = num_dihedrals // 2 + 2 * num_dihedrals + 2
        axs[0, ax_index].text(
            label_offset,
            1.1,
            format_traj_name_fn("ref_traj_100x"),
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[0, ax_index].transAxes,
            fontsize=22,
        )

        ax_index = -1
        axs[i, ax_index].text(
            1.1,
            0.5,
            format_peptide_name(peptide),
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            transform=axs[i, ax_index].transAxes,
            fontsize=18,
        )

        axs[i, num_dihedrals].axis("off")
        axs[i, 2 * num_dihedrals + 1].axis("off")

        if i != len(axs) - 1:
            for j in range(len(axs[i])):
                axs[i, j].set_xticks([])
                axs[i, j].set_xlabel("")

        for j in range(1, len(axs[i])):
            axs[i, j].set_yticks([])
            axs[i, j].set_ylabel("")

    plt.subplots_adjust(hspace=0.06, wspace=0.04)


def plot_ramachandran_against_all_trajectories(results_df: pd.DataFrame, peptide: str) -> None:
    """Plots Ramachandran contours for a single peptide against all trajectories."""
    pmf_type = "all"
    experiment = results_df.attrs["experiment"]
    num_dihedrals = get_num_dihedrals(experiment, pmf_type)
    format_traj_name_fn = get_format_traj_name_fn(results_df)

    row = results_df[results_df["peptide"] == peptide].iloc[0]
    traj_names = row["results"]["PMFs"].keys()
    traj_names = ["ref_traj"] + [
        traj
        for traj in traj_names
        if traj not in ["ref_traj", "ref_traj_10x", "ref_traj_100x", "ref_traj_1000x", "TBG_200x"]
    ]

    if len(traj_names) == 2:
        traj_names = ["ref_traj", "traj", "ref_traj_10x"]

    fig, axs = plt.subplots(len(traj_names), num_dihedrals, figsize=(max(3 * num_dihedrals, 12), 8), squeeze=False)
    for j in range(num_dihedrals):
        for k, traj in enumerate(traj_names):
            plot_ramachandran_contour(row["results"]["PMFs"][traj][f"pmf_{pmf_type}"], j, axs[k, j])

    for i in range(len(axs)):
        for j in range(1, len(axs[i])):
            axs[i, j].set_yticks([])
            axs[i, j].set_ylabel("")

    for i in range(len(axs) - 1):
        for j in range(len(axs[i])):
            axs[i, j].set_xticks([])
            axs[i, j].set_xlabel("")

    # Add labels.
    for k, traj in enumerate(traj_names):
        axs[k, -1].text(
            1.2,
            0.5,
            format_traj_name_fn(traj),
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            transform=axs[k, -1].transAxes,
        )
    fig.suptitle(format_peptide_name(peptide))
    plt.subplots_adjust(hspace=0.06, wspace=0.04)


def plot_torsion_histograms(results_df) -> None:
    """Plots torsion angle histograms for the sampled peptides."""
    format_traj_name_fn = get_format_traj_name_fn(results_df)

    fig, axs = plt.subplots(nrows=len(results_df), ncols=2, figsize=(14, 4 * len(results_df)), squeeze=False)
    for i, row in results_df.iterrows():
        peptide = row["peptide"]

        feats = row["results"]["featurization"]
        histograms = row["results"]["feature_histograms"]

        pyemma_helper.plot_feature_histograms(
            histograms["ref_traj"]["torsions"]["histograms"],
            histograms["ref_traj"]["torsions"]["edges"],
            feature_labels=feats["ref_traj"]["feats"]["torsions"].describe(),
            ax=axs[i, 0],
        )

        pyemma_helper.plot_feature_histograms(
            histograms["traj"]["torsions"]["histograms"],
            histograms["traj"]["torsions"]["edges"],
            feature_labels=feats["traj"]["feats"]["torsions"].describe(),
            ax=axs[i, 1],
        )

        axs[i, -1].text(
            1.1,
            0.5,
            format_peptide_name(peptide),
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            transform=axs[i, -1].transAxes,
        )

    axs[0, 0].set_title(format_traj_name_fn("ref_traj"))
    axs[0, 1].set_title(format_traj_name_fn("traj"))
    plt.tight_layout()


def plot_distance_histograms(results_df) -> None:
    """Plots distance histograms for the sampled peptides."""
    format_traj_name_fn = get_format_traj_name_fn(results_df)

    fig, axs = plt.subplots(nrows=len(results_df), ncols=2, figsize=(14, 4 * len(results_df)), squeeze=False)
    for i, row in results_df.iterrows():
        peptide = row["peptide"]

        feats = row["results"]["featurization"]
        histograms = row["results"]["feature_histograms"]

        num_hists = len(histograms["ref_traj"]["distances"]["histograms"])
        indices = np.random.choice(num_hists, replace=False, size=min(num_hists, 10))

        pyemma_helper.plot_feature_histograms(
            histograms["ref_traj"]["distances"]["histograms"][indices],
            histograms["ref_traj"]["distances"]["edges"][indices],
            feature_labels=[feats["ref_traj"]["feats"]["distances"].describe()[i] for i in indices],
            ax=axs[i, 0],
        )

        pyemma_helper.plot_feature_histograms(
            histograms["traj"]["distances"]["histograms"][indices],
            histograms["traj"]["distances"]["edges"][indices],
            feature_labels=[feats["traj"]["feats"]["distances"].describe()[i] for i in indices],
            ax=axs[i, 1],
        )

        axs[i, 1].set_xlim(axs[i, 0].get_xlim())  # Ensure both axes have the same x-limits
        axs[i, 1].set_ylim(axs[i, 0].get_ylim())  # Ensure both axes have the same y-limits

        axs[i, -1].text(
            1.1,
            0.5,
            format_peptide_name(peptide),
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            transform=axs[i, -1].transAxes,
        )

    axs[0, 0].set_title(format_traj_name_fn("ref_traj"))
    axs[0, 1].set_title(format_traj_name_fn("traj"))
    plt.tight_layout()


def collect_torsion_angle_decorrelation_times(results_df) -> None:
    """Collects the torsion angle decorrelation times from the results DataFrame."""

    torsion_decorrelation_times = {
        "ref_traj": {"backbone": [], "sidechain": []},
        "traj": {"backbone": [], "sidechain": []},
    }
    total_count = {"backbone": 0, "sidechain": 0}

    for i, row in results_df.iterrows():
        results = row["results"]["torsion_decorrelations"]

        for feat in results:
            ref_decorrelation_time = results[feat]["ref_traj"]["decorrelation_time"]
            traj_decorrelation_time = results[feat]["traj"]["decorrelation_time"]

            if "PHI" in feat or "PSI" in feat:
                torsion_type = "backbone"
            elif "CHI" in feat:
                torsion_type = "sidechain"
            else:
                raise ValueError(f"Unknown torsion type: {feat}")

            total_count[torsion_type] += 1

            if np.isnan(ref_decorrelation_time) or np.isnan(traj_decorrelation_time):
                continue

            torsion_decorrelation_times["ref_traj"][torsion_type].append(ref_decorrelation_time)
            torsion_decorrelation_times["traj"][torsion_type].append(traj_decorrelation_time)

    for traj in torsion_decorrelation_times:
        for key in torsion_decorrelation_times[traj]:
            torsion_decorrelation_times[traj][key] = np.asarray(torsion_decorrelation_times[traj][key])

    return torsion_decorrelation_times, total_count


def plot_backbone_decorrelation_times(
    results_df: pd.DataFrame, torsion_decorrelation_times: Dict[str, Dict[str, np.ndarray]]
) -> None:
    # Scatter plot of probabilities.
    format_traj_name_fn = get_format_traj_name_fn(results_df)
    plt.scatter(
        torsion_decorrelation_times["ref_traj"]["backbone"],
        torsion_decorrelation_times["traj"]["backbone"],
        alpha=0.3,
        edgecolors="none",
        color="tab:blue",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(format_traj_name_fn("ref_traj"))
    plt.ylabel(format_traj_name_fn("traj"))
    plt.title("Decorrelation Times of Backbone Torsions")

    # Fit line.
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        np.log(torsion_decorrelation_times["ref_traj"]["backbone"]),
        np.log(torsion_decorrelation_times["traj"]["backbone"]),
    )

    # # Create x points for line.
    # x_line = np.array([np.percentile(torsion_decorrelation_times["ref_traj"]["backbone"], 5), np.percentile(torsion_decorrelation_times["ref_traj"]["backbone"], 95)])
    # log_x_line = np.log(x_line)
    # log_y_line = slope * log_x_line + intercept

    # # Transform back to original scale for plotting
    # y_line = np.exp(log_y_line)

    # # Plot the fitted line with dashed style.
    # plt.plot(x_line, y_line, color='tab:blue', linestyle='--')

    plt.text(0.65, 0.90, f"R² = {r_value**2:.3f}", transform=plt.gca().transAxes, color="tab:blue")
    plt.tight_layout()


def plot_backbone_decorrelation_speedups(torsion_decorrelation_times):
    """Plots the speedups of backbone torsion decorrelation times."""
    backbone_torsion_speedups = (
        torsion_decorrelation_times["ref_traj"]["backbone"] / torsion_decorrelation_times["traj"]["backbone"]
    )

    bins = np.geomspace(np.min(backbone_torsion_speedups), np.max(backbone_torsion_speedups) + 1e-5, 21)
    plt.hist(backbone_torsion_speedups, bins=bins)
    plt.xscale("log")
    plt.xlabel("Speedup Factor")
    plt.xticks([1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
    plt.ylabel("Frequency")
    plt.suptitle(f"Speedups of Backbone Torsion Decorrelation Times")
    plt.tight_layout()


def plot_sidechain_decorrelation_times(
    results_df, torsion_decorrelation_times: Dict[str, Dict[str, np.ndarray]]
) -> None:
    # Scatter plot of probabilities.
    format_traj_name_fn = get_format_traj_name_fn(results_df)
    plt.scatter(
        torsion_decorrelation_times["ref_traj"]["sidechain"],
        torsion_decorrelation_times["traj"]["sidechain"],
        alpha=0.3,
        edgecolors="none",
        color="tab:orange",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(format_traj_name_fn("ref_traj"))
    plt.ylabel(format_traj_name_fn("traj"))
    plt.title("Decorrelation Times of Sidechain Torsions")

    # Fit line.
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        np.log(torsion_decorrelation_times["ref_traj"]["sidechain"]),
        np.log(torsion_decorrelation_times["traj"]["sidechain"]),
    )

    # # Create x points for line.
    # x_line = np.array([np.percentile(torsion_decorrelation_times["ref_traj"]["sidechain"], 5), np.percentile(torsion_decorrelation_times["ref_traj"]["sidechain"], 95)])
    # log_x_line = np.log(x_line)
    # log_y_line = slope * log_x_line + intercept

    # # Transform back to original scale for plotting
    # y_line = np.exp(log_y_line)

    # # Plot the fitted line with dashed style.
    # plt.plot(x_line, y_line, color='tab:orange', linestyle='--')
    plt.text(0.65, 0.90, f"R² = {r_value**2:.3f}", transform=plt.gca().transAxes, color="tab:orange")

    plt.tight_layout()


def plot_sidechain_decorrelation_speedups(torsion_decorrelation_times) -> None:
    sidechain_torsion_speedups = (
        torsion_decorrelation_times["ref_traj"]["sidechain"] / torsion_decorrelation_times["traj"]["sidechain"]
    )

    bins = np.geomspace(np.min(sidechain_torsion_speedups), np.max(sidechain_torsion_speedups) + 1e-5, 21)
    plt.hist(sidechain_torsion_speedups, bins=bins)
    plt.xscale("log")
    plt.xlabel("Speedup Factor")
    plt.ylabel("Frequency")
    plt.suptitle(f"Speedups of Sidechain Torsion Decorrelation Times")
    plt.tight_layout()


def plot_JSD_distribution(JSD_final_results, key: str):
    """Plot distribution of JSDs across systems."""
    JSD_MSM = JSD_final_results[key]["traj"]
    plt.hist(JSD_MSM)
    plt.title("Jenson-Shannon Distances of Metastable State Probabilities")
    plt.xlabel("JSD")
    plt.xticks(np.arange(0.1, JSD_MSM.max() + 0.1, 0.1))
    plt.ylabel("Frequency")
    plt.ticklabel_format(useOffset=False, style="plain")
    plt.tight_layout()


def plot_metastable_probs(results_df: pd.DataFrame):
    """Plot metastable state probabilities against the reference trajectory."""
    metastable_probs = collect_metastable_probs(results_df)
    format_traj_name_fn = get_format_traj_name_fn(results_df)

    # Scatter plot of probabilities.
    plt.scatter(metastable_probs["ref_traj"], metastable_probs["traj"], color="C2", alpha=0.3, edgecolors="none")

    # Fit line.
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        metastable_probs["ref_traj"], metastable_probs["traj"]
    )

    # Create x points for line.
    x_line = np.array([-0.5, 1.5])
    y_line = slope * x_line + intercept

    # Plot the fitted line with dashed style.
    plt.plot(x_line, y_line, color="C2", linestyle="--")
    plt.text(0.35, 0.90, f"R² = {r_value**2:.3f}", transform=plt.gca().transAxes, color="C2")

    plt.title("MSM State Probabilities", pad=10)
    plt.axis("square")
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel(format_traj_name_fn("ref_traj"))
    plt.ylabel(format_traj_name_fn("traj"))
    # plt.tight_layout()


def plot_JSD_against_time(results_df: pd.DataFrame):
    """Plot JSD against trajectory progress for all peptides."""
    JSD_results = get_all_JSD_results_against_time(results_df)
    format_traj_name_fn = get_format_traj_name_fn(results_df)
    colors_fn = get_colors_fn(results_df)

    figs = {}
    for quantity in JSD_results:
        for name in JSD_results[quantity]:
            mean = np.mean(JSD_results[quantity][name]["JSDs"], axis=0)
            std = np.std(JSD_results[quantity][name]["JSDs"], axis=0)
            progress = JSD_results[quantity][name]["progress"]

            # Plot mean line
            color = colors_fn(name)

            (line,) = plt.plot(progress, mean, label=format_traj_name_fn(name), color=color)

            # Add shaded region for standard deviation
            plt.fill_between(progress, mean - std, mean + std, alpha=0.2, color=color)

        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")
        plt.title(f"JSD vs Trajectory Progress\n{format_quantity(quantity)}")
        plt.xlabel("Trajectory Progress")
        plt.ylabel("JSD")
        plt.ticklabel_format(useOffset=False, style="plain")

        fig = plt.gcf()
        figs[quantity] = fig
        plt.close(fig)

    return figs


def plot_TICA_histograms(results_df: pd.DataFrame):
    """Plots TICA histograms for the sampled peptides."""
    format_traj_name_fn = get_format_traj_name_fn(results_df)

    traj_names = ["ref_traj", "traj"]
    if "TBG" in results_df.attrs.get("extra_traj_names", []):
        traj_names.append("TBG")
    if "MDGen" in results_df.attrs.get("extra_traj_names", []):
        traj_names.append("MDGen")

    fig, axs = plt.subplots(nrows=len(results_df), ncols=len(traj_names), figsize=(6 * len(traj_names), 3.5 * len(results_df)), squeeze=False)
    for i, row in results_df.iterrows():
        peptide = row["peptide"]
        results = row["results"]["TICA_histograms"]

        for j, traj in enumerate(traj_names):
            pyemma_helper.plot_free_energy(
                *results[traj], cmap="plasma", ax=axs[i, j]
            )
            
            axs[i, j].ticklabel_format(useOffset=False, style="plain")
            axs[i, j].set_xlim(axs[i, 0].get_xlim())
            axs[i, j].set_ylim(axs[i, 0].get_ylim())


        if i == 0:
            for j, traj in enumerate(traj_names):
                axs[i, j].set_title(format_traj_name_fn(traj))
        
        axs[i, -1].text(
            1.4,
            0.5,
            format_peptide_name(peptide),
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            transform=axs[i, -1].transAxes,
        )

    plt.suptitle("TICA-0,1 Projections", fontsize="x-large")
    plt.tight_layout()


def collect_TICA_0_speedups(results_df: pd.DataFrame) -> np.ndarray:
    """Collects the speedups of TICA-0 decorrelation times from the results DataFrame."""
    tica_0_speedups = []
    for i, row in results_df.iterrows():
        results = row["results"]["TICA_decorrelations"]

        speedup_factor = results["ref_traj"]["decorrelation_time"] / results["traj"]["decorrelation_time"]
        if np.isnan(speedup_factor):
            continue

        tica_0_speedups.append(speedup_factor)

    return np.asarray(tica_0_speedups)


def make_JSD_table(JSD_final_results) -> pd.DataFrame:
    """Create a DataFrame summarizing the JSD results."""
    JSD_final_results_df = pd.DataFrame.from_dict(JSD_final_results)

    means_series = JSD_final_results_df.map(lambda x: np.mean(x) if isinstance(x, np.ndarray) else None)
    stds_series = JSD_final_results_df.map(lambda x: np.std(x) if isinstance(x, np.ndarray) else None)

    means_series = means_series.rename(lambda x: f"{x}_mean", axis=1)
    stds_series = stds_series.rename(lambda x: f"{x}_std", axis=1)

    JSD_table = pd.concat([means_series, stds_series], axis=1)
    return JSD_table


def plot_JSD_table(results_df: pd.DataFrame, JSD_table: pd.DataFrame) -> None:
    """Plots the JSD table as bar plots with error bars for each metric."""

    # Remove row for ref_traj
    if "ref_traj" in JSD_table.index:
        JSD_table = JSD_table.drop("ref_traj")

    # Separate mean and std columns
    mean_cols = [col for col in JSD_table.columns if col.endswith("_mean")]
    std_cols = [col for col in JSD_table.columns if col.endswith("_std")]

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(10, 10))
    axes = axes.flatten()

    colors_fn = get_colors_fn(results_df)
    colors_list = [colors_fn(idx) for idx in JSD_table.index]

    format_traj_name_fn = get_short_format_traj_name_fn(results_df)
    traj_names = [format_traj_name_fn(name) for name in JSD_table.index]

    # Plot each metric (mean with std error bars)
    for i, (mean_col, std_col) in enumerate(zip(mean_cols, std_cols)):
        metric_name = mean_col.replace("_mean", "").replace("JSD_", "")

        # Create bar plot
        bars = axes[i].bar(
            range(len(traj_names)), JSD_table[mean_col], yerr=JSD_table[std_col], capsize=5, color=colors_list
        )

        # Set x-axis labels and rotate them
        axes[i].set_xticks(range(len(traj_names)))
        axes[i].set_xticklabels(traj_names, rotation=45, ha="right")

        # Set title and labels
        axes[i].set_title(format_quantity(metric_name))
        axes[i].set_ylabel("JSD Value")

        # Add a grid for better readability
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)

    # Add a legend with trajectory names
    # fig.legend(bars, traj_names, loc='lower center', bbox_to_anchor=(0.5, 0),
    #            ncol=len(traj_names), title="Trajectory")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for legend
    plt.suptitle("Comparison of Jensen-Shannon Divergence Metrics", fontsize=16, y=0.98)


def plot_TICA_0_speedups(results_df: pd.DataFrame) -> None:
    """Plots the speedups of TICA-0 decorrelation times."""
    tica_0_speedups = collect_TICA_0_speedups(results_df)
    py_logger.info(f"Number of systems with valid decorrelations: {len(tica_0_speedups)} out of {len(results_df)}")

    # Scatter plot of speedups.
    bins = np.geomspace(np.min(tica_0_speedups), np.max(tica_0_speedups), 21)
    plt.hist(tica_0_speedups, bins=bins)
    plt.xscale("log")
    plt.xlabel("Speedup Factor")
    plt.ylabel("Frequency")
    plt.suptitle(f"Speedups of TICA-0 Decorrelation Times")
    plt.tight_layout()


def collect_metastable_probs(results_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Collects the metastable probabilities from the results DataFrame."""
    metastable_probs = {
        "ref_traj": [],
        "traj": [],
    }
    for i, row in results_df.iterrows():
        try:
            results = row["results"]["JSD_MSM"]["traj"]
            ref_metastable_probs = results["ref_metastable_probs"]
            traj_metastable_probs = results["traj_metastable_probs"]
        except KeyError:
            py_logger.warning(
                f"Missing metastable probabilities for peptide {row['peptide']} in trajectory {results_df.attrs['traj_name']}."
            )
            continue

        metastable_probs["ref_traj"].append(ref_metastable_probs)
        metastable_probs["traj"].append(traj_metastable_probs)

    metastable_probs["ref_traj"] = np.concatenate(metastable_probs["ref_traj"])
    metastable_probs["traj"] = np.concatenate(metastable_probs["traj"])

    return metastable_probs


def plot_transition_matrices(results_df: pd.DataFrame):
    """Plots the transition matrices for the sampled peptides."""
    format_traj_name_fn = get_format_traj_name_fn(results_df)

    fig, axs = plt.subplots(2, len(results_df), figsize=(15, 5), squeeze=False)

    # Get min and max values for color normalization
    for i, row in results_df.iterrows():
        peptide = row["peptide"]
        results = row["results"]["MSM_matrices"]["traj"]

        msm_transition_matrix = results["msm_transition_matrix"]
        traj_transition_matrix = results["traj_transition_matrix"]
        correlation = results["transition_spearman_correlation"]

        im = axs[0][i].imshow(msm_transition_matrix, cmap="Blues", vmin=0, vmax=1)
        axs[1][i].imshow(traj_transition_matrix, cmap="Blues", vmin=0, vmax=1)
        axs[0][i].set_title(f"{format_peptide_name(peptide)}\nρ = {correlation:.2f}")

    axs[0][0].text(
        -0.4,
        0.5,
        format_traj_name_fn("ref_traj"),
        horizontalalignment="right",
        verticalalignment="center",
        transform=axs[0, 0].transAxes,
    )

    axs[1][0].text(
        -0.4,
        0.5,
        format_traj_name_fn("traj"),
        horizontalalignment="right",
        verticalalignment="center",
        transform=axs[1, 0].transAxes,
    )

    fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.022)


def plot_flux_matrices(results_df: pd.DataFrame):
    """Plots the flux matrices for the sampled peptides."""
    format_traj_name_fn = get_format_traj_name_fn(results_df)
    fig, axs = plt.subplots(2, len(results_df), figsize=(15, 5), squeeze=False)

    vmin = np.inf
    vmax = -np.inf

    for i, row in results_df.iterrows():
        peptide = row["peptide"]
        results = row["results"]["MSM_matrices"]["traj"]

        msm_flux_matrix = results["msm_flux_matrix"]
        traj_flux_matrix = results["traj_flux_matrix"]

        vmin = min(vmin, np.min(msm_flux_matrix), np.min(traj_flux_matrix))
        vmax = max(vmax, np.max(msm_flux_matrix), np.max(traj_flux_matrix))

    for i, row in results_df.iterrows():
        peptide = row["peptide"]
        results = row["results"]["MSM_matrices"]["traj"]

        msm_flux_matrix = results["msm_flux_matrix"]
        traj_flux_matrix = results["traj_flux_matrix"]
        correlation = results["flux_spearman_correlation"]

        im = axs[0][i].imshow(
            msm_flux_matrix, cmap="Blues", norm=matplotlib.colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        )
        axs[1][i].imshow(
            traj_flux_matrix, cmap="Blues", norm=matplotlib.colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        )
        axs[0][i].set_title(f"{format_peptide_name(peptide)}\nρ = {correlation:.2f}")

    axs[0][0].text(
        -0.4,
        0.5,
        format_traj_name_fn("ref_traj"),
        horizontalalignment="right",
        verticalalignment="center",
        transform=axs[0, 0].transAxes,
    )

    axs[1][0].text(
        -0.4,
        0.5,
        format_traj_name_fn("traj"),
        horizontalalignment="right",
        verticalalignment="center",
        transform=axs[1, 0].transAxes,
    )

    fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.022)


def get_transition_matrix_correlations(results_df: pd.DataFrame):
    """Collects the transition matrix correlations from the results DataFrame."""
    correlations = []
    for i, row in results_df.iterrows():
        try:
            correlation = row["results"]["MSM_matrices"]["traj"]["transition_spearman_correlation"]
        except KeyError:
            continue

        correlations.append(correlation)
    return np.asarray(correlations)


def get_flux_matrix_correlations(results_df: pd.DataFrame):
    """Collects the flux matrix correlations from the results DataFrame."""
    correlations = []
    for i, row in results_df.iterrows():
        try:
            correlation = row["results"]["MSM_matrices"]["traj"]["flux_spearman_correlation"]
        except KeyError:
            continue

        correlations.append(correlation)
    return np.asarray(correlations)


def add_recursively(dict1: Dict[str, Any], dict2: Dict[str, Any], new_key_name: str) -> None:
    """Recursively adds a key "traj" from dict2 to dict1 with the specified new_key_name."""
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return

    if new_key_name in dict1:
        raise ValueError(f"Key '{new_key_name}' already exists in dict1")

    if "traj" in dict1:
        try:
            dict1[new_key_name] = dict2["traj"]
        except KeyError:
            pass
        return

    for key in dict1:
        add_recursively(dict1[key], dict2[key], new_key_name)


def add_extra_trajectories(
    results_df: pd.DataFrame, results_dir: str, experiment: str, traj_name: str, ref_traj_name: str
) -> pd.DataFrame:
    """Adds extra trajectories to the results DataFrame."""
    extra_results_dfs = None
    if experiment == "Timewarp_2AA" and traj_name == "JAMUN":
        extra_results_dfs = {
            "TBG": load_results(results_dir, "Timewarp_2AA", "TBG", ref_traj_name),
            "TBG_20x": load_results(results_dir, "Timewarp_2AA", "TBG_20x", ref_traj_name),
            "TBG_200x": load_results(results_dir, "Timewarp_2AA", "TBG_200x", ref_traj_name),
        }

    if experiment == "Timewarp_4AA_0.4A" and traj_name == "JAMUN":
        extra_results_dfs = {
            "JAMUN_0.2A": load_results(results_dir, "Timewarp_4AA_0.2A", "JAMUN", ref_traj_name),
            "JAMUN_0.8A": load_results(results_dir, "Timewarp_4AA_0.8A", "JAMUN", ref_traj_name),
        }
        results_df.attrs["traj_name"] = "JAMUN\n"+ r"($\sigma=0.4\AA$)"
        results_df.attrs["compare_JSD_against_reference"] = False

    if experiment == "MDGen_4AA" and traj_name == "JAMUN":
        extra_results_dfs = {
            "MDGen": load_results(results_dir, experiment, "MDGenSamples_4AA", ref_traj_name),
        }

    if experiment == "Chignolin" and traj_name == "JAMUN_2x":
        results_df.attrs["traj_name"] = "JAMUN"

    if experiment == "Our_5AA" and traj_name == "JAMUN":
        boltz_results_df = load_results(results_dir, "Our_5AA", "BoltzSamples", ref_traj_name)
        boltz_results_df["peptide"] = boltz_results_df["peptide"].apply(lambda x: "uncapped_" + x)
        extra_results_dfs = {
            "Boltz-1": boltz_results_df,
        }

    if extra_results_dfs is None:
        return

    # Add extra results to the main results.
    for name, extra_results_df in extra_results_dfs.items():
        for i, row in results_df.iterrows():
            peptide = row["peptide"]

            try:
                other_row = extra_results_df[extra_results_df["peptide"] == peptide].iloc[0]
            except IndexError:
                py_logger.warning(f"No results found for peptide {peptide} in {name}.")
                continue

            our_results = row["results"]
            other_results = other_row["results"]

            add_recursively(our_results, other_results, new_key_name=name)

    results_df.attrs["extra_traj_names"] = list(extra_results_dfs.keys())


def make_plots(experiment: str, traj_name: str, ref_traj_name: str, results_dir: str, output_dir: str) -> None:
    """Makes plots for the given experiment, trajectory and reference."""

    output_dir = os.path.join(output_dir, experiment, traj_name, f"ref={ref_traj_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Save logs.
    file_handler = logging.FileHandler(os.path.join(output_dir, f"output.log"))
    py_logger.addHandler(file_handler)
    py_logger.info(f"Plots will be saved to {output_dir}")

    # Load all trajectories
    results_df = load_results(results_dir, experiment, traj_name, ref_traj_name)
    # Add any extra trajectories to compare.
    add_extra_trajectories(results_df, results_dir, experiment, traj_name, ref_traj_name)
    py_logger.info(f"Loaded {len(results_df)} results for {experiment}.")

    # Filter based on peptide names.
    if experiment == "Our_5AA":
        peptides = ["KTYDI", "NRLCQ", "VWSPF"]
        peptides = ["uncapped_" + peptide for peptide in peptides]
        results_df = results_df[results_df["peptide"].isin(peptides)]
        sampled_results_df = results_df.copy()

    else:
        # Sample 4 random peptides
        sampled_results_df = results_df.sample(n=min(len(results_df), 4), random_state=42)

    sampled_results_df = sampled_results_df.reset_index(drop=True)
    py_logger.info(f"Sampled {len(sampled_results_df)} results for {experiment}.")

    # Ramachandran Plots against Reference
    plot_ramachandran_against_reference(sampled_results_df)
    plt.savefig(os.path.join(output_dir, "ramachandran_contours.pdf"), dpi=300)
    plt.close()

    plot_ramachandran_against_reference_shortened(sampled_results_df)
    plt.savefig(os.path.join(output_dir, "ramachandran_contours_with_shortened_reference.pdf"), dpi=300)
    plt.close()

    # Ramachandran Plots for a Single Peptide
    for peptide in sampled_results_df["peptide"]:
        plot_ramachandran_against_all_trajectories(sampled_results_df, peptide)
        plt.savefig(os.path.join(output_dir, f"ramachandran_contours_{format_peptide_name(peptide)}.pdf"), dpi=300)
        plt.close()

    # Torsion Histograms
    plot_torsion_histograms(sampled_results_df)
    plt.savefig(os.path.join(output_dir, "torsion_histograms.pdf"), dpi=300)
    plt.close()

    # Distance Histograms
    plot_distance_histograms(sampled_results_df)
    plt.savefig(os.path.join(output_dir, "distance_histograms.pdf"), dpi=300)
    plt.close()

    # Torsion Angle Decorrelation Times
    torsion_decorrelation_times, total_count = collect_torsion_angle_decorrelation_times(results_df)
    py_logger.info(
        f"Number of backbone torsions with valid decorrelation times: {len(torsion_decorrelation_times['ref_traj']['backbone'])} out of {total_count['backbone']}"
    )
    py_logger.info(
        f"Number of sidechain torsions with valid decorrelation times: {len(torsion_decorrelation_times['ref_traj']['sidechain'])} out of {total_count['sidechain']}"
    )

    # Backbone Torsion Angle Decorrelation
    plot_backbone_decorrelation_times(results_df, torsion_decorrelation_times)
    plt.savefig(os.path.join(output_dir, "backbone_torsion_decorrelation_times.pdf"), dpi=300)
    plt.close()

    # Backbone Torsion Angle Speedups
    plot_backbone_decorrelation_speedups(torsion_decorrelation_times)
    plt.savefig(os.path.join(output_dir, "backbone_torsion_speedups.pdf"), dpi=300)
    plt.close()

    # Sidechain Torsion Angle Decorrelation
    plot_sidechain_decorrelation_times(results_df, torsion_decorrelation_times)
    plt.savefig(os.path.join(output_dir, "sidechain_torsion_decorrelation_times.pdf"), dpi=300)
    plt.close()

    plot_sidechain_decorrelation_speedups(torsion_decorrelation_times)
    plt.savefig(os.path.join(output_dir, "sidechain_torsion_speedups.pdf"), dpi=300)
    plt.close()

    # Jenson-Shannon Divergences (JSD)
    JSD_final_results = get_all_JSD_results(results_df)

    JSD_table = make_JSD_table(JSD_final_results)
    JSD_table.to_csv(os.path.join(output_dir, "jsd_table.csv"))

    plot_JSD_table(results_df, JSD_table)
    plt.savefig(os.path.join(output_dir, "jsd_table.pdf"), dpi=300)
    plt.close()

    plot_JSD_distribution(JSD_final_results, key="JSD_metastable_probs")
    plt.savefig(os.path.join(output_dir, "jsd_metastable_probs.pdf"), dpi=300)
    plt.close()

    # JSD as a function of trajectory progress
    figs = plot_JSD_against_time(results_df)
    for quantity in figs:
        figs[quantity].savefig(os.path.join(output_dir, f"jsd_vs_time_{quantity}.pdf"), dpi=300)
        plt.close(figs[quantity])

    # TICA Projection Histograms
    plot_TICA_histograms(sampled_results_df)
    plt.savefig(os.path.join(output_dir, "tica_projections.pdf"), dpi=300)
    plt.close()

    # TICA Projection Decorrelation Speedups
    try:
        plot_TICA_0_speedups(results_df)
        plt.savefig(os.path.join(output_dir, "tica_0_speedups.pdf"), dpi=300)
        plt.close()
    except ValueError:
        py_logger.warning("No TICA-0 speedups found.")

    # MSM State Probabilities
    plot_metastable_probs(results_df)
    plt.savefig(os.path.join(output_dir, "metastable_probs.pdf"), dpi=300)
    plt.close()

    # Transition Matrices
    correlations = get_transition_matrix_correlations(results_df)
    py_logger.info(
        f"Number of systems with valid transition correlations: {len(correlations)} out of {len(results_df)}"
    )
    py_logger.info(
        f"Mean Spearman correlation for transition matrices: {np.mean(correlations):.2f} ± {np.std(correlations):.2f}"
    )

    plot_transition_matrices(sampled_results_df)
    plt.savefig(os.path.join(output_dir, "transition_matrices.pdf"), dpi=300)
    plt.close()

    # Flux Matrices
    flux_correlations = get_flux_matrix_correlations(results_df)
    py_logger.info(f"Number of systems with valid flux correlations: {len(flux_correlations)} out of {len(results_df)}")
    py_logger.info(
        f"Mean Spearman correlation for flux matrices: {np.mean(flux_correlations):.2f} ± {np.std(flux_correlations):.2f}"
    )

    plot_flux_matrices(sampled_results_df)
    plt.savefig(os.path.join(output_dir, "flux_matrices.pdf"), dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate analysis plots for the peptide simulations.")
    parser.add_argument("--experiment", type=str, help="The experiment to analyze.")
    parser.add_argument("--trajectory", type=str, help="The trajectory to analyze.")
    parser.add_argument("--reference", type=str, help="The reference trajectory to analyze.")
    parser.add_argument("--analysis-dir", type=str, help="Directory containing the results of the analysis.")
    parser.add_argument("--plot-dir", type=str, help="Directory to save the output plots.")
    args = parser.parse_args()

    analysis_dir = load_trajectory.get_analysis_path(args.analysis_dir)
    plot_dir = load_trajectory.get_plot_path(args.plot_dir)

    experiment = args.experiment
    traj_name = args.trajectory
    if traj_name is None:
        traj_name = os.listdir(os.path.join(analysis_dir, experiment))[0]

    ref_traj_name = args.reference
    if ref_traj_name is None:
        ref_traj_name = os.listdir(os.path.join(analysis_dir, experiment, traj_name))[0]
        ref_traj_name = ref_traj_name[len("ref=") :]

    py_logger.info(f"Experiment: {experiment}")
    py_logger.info(f"Trajectory: {traj_name}")
    py_logger.info(f"Reference: {ref_traj_name}")

    make_plots(experiment, traj_name, ref_traj_name, analysis_dir, plot_dir)
