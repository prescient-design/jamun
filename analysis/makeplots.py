#!/usr/bin/env python3

import pickle

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np

# Constants
BINS = np.linspace(-np.pi, np.pi, 50)
X_COORDS = (BINS[1:] + BINS[:-1]) / 2
TICKS = [-np.pi + np.pi/50, -np.pi/2, 0, np.pi/2, np.pi - np.pi/50]
TICK_LABELS = ["$-\\pi$", "$-\\pi/2$", "$0$", "$\\pi/2$", "$\\pi$"]
PLOT_COLORS = ["blue", "green", "darkorange", "purple", "darkred",
               "cyan", "crimson", "darkmagenta", "teal", "m"]

# Amino acid mappings
AMINO_ACID_DICT = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"
}


def make_phipsi_for_one_residue(phi, psi, protein, ax):
    """Create a Ramachandran plot for a single residue."""
    H, xedges, yedges = np.histogram2d(phi.T[0], psi.T[0], bins=BINS)
    pmf = -np.log(H.T) + np.max(np.log(H.T))

    ax.contourf(xedges[:-1], yedges[:-1], pmf, cmap='viridis', levels=50)
    ax.contour(xedges[:-1], yedges[:-1], pmf, colors='white',
              linestyles='solid', levels=30, linewidths=0.25)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("$\\phi$")
    ax.set_ylabel("$\\psi$")
    plt.show()


def make_phipsi(phis, psis, protein):
    """Create Ramachandran plots for multiple residues."""
    num_residues = phis.shape[1]
    fig, axs = plt.subplots(1, num_residues, figsize=(num_residues*5, 5), squeeze=False)

    for i in range(num_residues):
        phi, psi = phis[:, i], psis[:, i]
        make_phipsi_for_one_residue(phi, psi, protein, axs[0, i])
    plt.show()


def get_traj_from_tbg(protein: str):
    """Load and process trajectory from TBG data."""
    traj = md.load(f"timewarp/{protein}-traj-state0.pdb")
    z = np.load(f"tbg/tbg_full_{protein}_aligned_corrected.npz")
    traj.xyz = z["samples_np"]
    traj.time = np.arange(len(traj))

    phi, psi = md.compute_phi(traj)[1], md.compute_psi(traj)[1]

    with open("time_tbg_5000.pkl", "rb") as f:
        time = pickle.load(f)

    t = time[protein]/(len(phi[0])*5000)
    return (phi, psi), t


def plot_comparison_figure(proteins):
    """Create comparison figure with JS divergence plots."""
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    plt.rcParams.update({'font.size': 22})

    with open("time_md_10.pkl", "rb") as f:
        time_md_10_np = pickle.load(f)

    for i, protein in enumerate(proteins):
        protein_name = f"{AMINO_ACID_DICT[protein[0]]}_{AMINO_ACID_DICT[protein[1]]}"
        js_jamun = np.load(f"js_jamun_{protein}.npy")
        js_md = np.load(f"js_md_{protein}.npy")
        time = time_md_10_np[protein_name] * len(js_md) / 10

        # Plot JAMUN trajectory
        x_jamun = np.arange(1000) / 1000
        axs[0].plot(x_jamun, js_jamun[:1000], "-", color=PLOT_COLORS[i], linewidth=4)

        # Plot MD trajectory
        x_md = np.arange(len(js_md)) / len(js_md)
        axs[0].plot(x_md, js_md, "--", color=PLOT_COLORS[i], markersize=4, markevery=500)

        # Plot converged MD
        js_md_converged = np.load(f"js_md_converged_{protein}.npy")
        x_converged = np.arange(len(js_md_converged)) / len(js_md_converged)
        axs[1].plot(x_converged, js_md_converged, "-",
                   label=protein_name, color=PLOT_COLORS[i], linewidth=2)

    # Add legends and labels
    axs[0].plot(0, 0, "--", color="black",
                label="Molecular Dynamics trajectory", markersize=4, markevery=500)
    axs[0].plot(0, 0, "-", color="black", label="JAMUN", linewidth=4)

    for ax in axs:
        ax.set_xlabel("Fraction of trajectory progress")
        ax.set_ylabel("Shannon-Jensen divergence")
        ax.legend()

    plt.savefig("figs/js_comparison.pdf", bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    # Set global plot parameters
    plt.rcParams.update({'font.size': 27})

    # Example usage
    proteins = ["DW", "ET", "FA", "NE", "CW", "GN", "HP", "IG"]
    plot_comparison_figure(proteins)
