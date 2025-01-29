import logging
import tempfile
from typing import List

import mdtraj as md
from rdkit import Chem, rdBase, RDLogger

RDLogger.DisableLog("rdApp.*")

from jamun import utils


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
