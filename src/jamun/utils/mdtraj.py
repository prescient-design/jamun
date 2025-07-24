import einops
import mdtraj as md
import numpy as np
import torch


def coordinates_to_trajectories(coords: torch.Tensor | np.ndarray, topology: md.Topology) -> list[md.Trajectory]:
    """Converts a tensor of coordinates to MDtraj trajectories."""
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().detach().numpy()

    if coords.ndim == 3:
        coords = coords[None, ...]

    coords = einops.rearrange(
        coords,
        "batch_size atoms num_sampling_steps coords -> batch_size num_sampling_steps atoms coords",
        atoms=topology.n_atoms,
    )

    return [md.Trajectory(traj_coords, topology) for traj_coords in coords]


def save_pdb(traj: md.Trajectory, path: str) -> None:
    """Saves a trajectory to a PDB file, fixing bugs in mdtraj.save_pdb."""
    topology = traj.topology
    unique_bonds = set()
    for bond in traj.topology.bonds:
        unique_bonds.add((bond.atom1.index, bond.atom2.index))

    with open(path, "w") as f:
        for frame_index, frame in enumerate(traj.xyz):
            f.write(f"MODEL        {frame_index}\n")

            for atom_index, positions in enumerate(frame):
                atom = topology.atom(atom_index)
                x, y, z = positions * 10
                f.write(
                    f"ATOM  {atom_index + 1:5d} {atom.name:<4s} {atom.residue.name:3s} {atom.residue.chain.index:1d}{atom.residue.index + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.element.symbol:>2s}\n"
                )

            num_atoms = topology.n_atoms
            final_atom = topology.atom(num_atoms - 1)
            f.write(
                f"TER   {num_atoms + 1:5d}      {final_atom.residue.name:3s} {final_atom.residue.chain.index:1d}{final_atom.residue.index + 1:4d}\n"
            )

            # Add bonds.
            bonds = [[i + 1] for i in range(topology.n_atoms)]
            for bond in unique_bonds:
                bonds[bond[0]].append(bond[1] + 1)
                bonds[bond[1]].append(bond[0] + 1)
            for bond in bonds:
                s = "".join([f"{atom:5d}" for atom in bond])
                f.write(f"CONECT{s}\n")

            f.write("ENDMDL\n")
        f.write("END\n")
