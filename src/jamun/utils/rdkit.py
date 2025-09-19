import logging
import tempfile
from typing import List, Union, TYPE_CHECKING, Sequence, Tuple, Optional

import mdtraj as md
from rdkit import Chem, rdBase, RDLogger
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

from jamun import utils

if TYPE_CHECKING:
    import numpy as np
    import torch


def to_rdkit_mols(traj: md.Trajectory) -> List[Chem.Mol]:
    """Converts an MDTraj trajectory to a list of RDKit molecules."""

    # Suppress RDKit warnings.
    blocker = rdBase.BlockLogs()

    # Write to a PDB.
    temp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb").name
    utils.save_pdb(traj, temp_pdb)
    traj_mol = Chem.MolFromPDBFile(temp_pdb, removeHs=False, sanitize=False)

    if traj_mol is None:
        py_logger = logging.getLogger("jamun")
        py_logger.warning("Could not convert the trajectory to RDKit mols.")
        return []

    # Check if the input molecule has multiple conformers.
    if traj_mol.GetNumConformers() <= 1:
        return [traj_mol]

    # Create separate molecules for each conformer.
    molecules = []
    for conf_id in range(traj_mol.GetNumConformers()):
        new_mol = Chem.Mol(traj_mol)
        new_mol.RemoveAllConformers()
        conf = traj_mol.GetConformer(conf_id)
        new_conf = Chem.Conformer(conf)
        new_mol.AddConformer(new_conf, assignId=True)
        molecules.append(new_mol)

    del blocker
    return molecules

def save_sdf_from_coords(
    coords: Union["np.ndarray", "torch.Tensor"],
    atomic_numbers: Sequence[int],
    bonds: Sequence[Tuple[int, int]],
    path: str,
    reference_mol: Optional[Chem.Mol] = None,
) -> None:
    """Write an SDF directly from coordinates, atomic numbers, and bonds.

    coords shape: (num_atoms, num_frames, 3)
    atomic_numbers length: num_atoms
    bonds: iterable of (i, j) 0-based indices, undirected
    reference_mol: Optional RDKit molecule to extract bond orders from
    """
    try:
        import numpy as np  # Local import to avoid hard dependency at module import time
        import torch  # type: ignore
    except Exception:
        np = None  # type: ignore
        torch = None  # type: ignore

    if torch is not None and isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()

    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must be (num_atoms, num_frames, 3), got {coords.shape}.")

    num_atoms, num_frames, _ = coords.shape

    if len(atomic_numbers) != num_atoms:
        raise ValueError(f"atomic_numbers length {len(atomic_numbers)} does not match num_atoms {num_atoms}.")

    # Build base RDKit molecule from atom numbers and bonds
    rw_mol = Chem.RWMol()
    for z in atomic_numbers:
        rd_atom = Chem.Atom(int(z))
        rw_mol.AddAtom(rd_atom)

    # Add bonds with proper bond orders if reference molecule is provided
    if reference_mol is not None:
        for i, j in bonds:
            if i == j:
                continue
            # Find corresponding bond in reference molecule
            bond = reference_mol.GetBondBetweenAtoms(i, j)
            if bond:
                bond_type = bond.GetBondType()
            else:
                bond_type = Chem.rdchem.BondType.SINGLE
            rw_mol.AddBond(int(i), int(j), bond_type)
    else:
        # Fallback to single bonds
        for i, j in bonds:
            if i == j:
                continue
            rw_mol.AddBond(int(i), int(j), Chem.rdchem.BondType.SINGLE)

    base_mol = rw_mol.GetMol()

    writer = Chem.SDWriter(path)
    try:
        # Convert from nm to Angstroms
        coords_A = coords * 10.0
        for frame_index in range(num_frames):
            conformer = Chem.Conformer(num_atoms)
            frame_coords = coords_A[:, frame_index, :]
            for atom_index in range(num_atoms):
                x, y, z = frame_coords[atom_index]
                conformer.SetAtomPosition(atom_index, Point3D(float(x), float(y), float(z)))

            mol = Chem.Mol(base_mol)
            mol.RemoveAllConformers()
            mol.AddConformer(conformer, assignId=True)
            writer.write(mol)
    finally:
        writer.close()
