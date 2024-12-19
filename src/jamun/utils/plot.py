import tempfile
from typing import Dict, Optional

import mdtraj as md
import py3Dmol
from rdkit import Chem

from jamun import utils


def animate_trajectory_with_py3Dmol(
    traj: md.Trajectory, alignment_frame: Optional[md.Trajectory] = None
) -> py3Dmol.view:
    """Create an animation of this trajectory using py3Dmol."""
    if alignment_frame:
        traj = traj.superpose(alignment_frame)

    temp_pdb = tempfile.NamedTemporaryFile(suffix=".pdb").name
    utils.save_pdb(traj, temp_pdb)
    with open(temp_pdb) as f:
        pdb = f.read()

    view = py3Dmol.view(width=1500, height=500)
    view.addModelsAsFrames(pdb, "pdb")
    view.setStyle(
        {
            "stick": {"radius": 0.1, "colorscheme": "grayCarbon"},
            "sphere": {"scale": 0.2},
            "colorscheme": {"prop": "index", "gradient": "roygb", "min": 0, "max": 14},
        }
    )
    view.animate(
        {
            "reps": 0,
            "interval": 100,
        }
    )
    view.zoomTo()
    return view


def plot_molecules_with_py3Dmol(
    molecules: Dict[str, Chem.Mol],
    show_atom_types: bool = False,
    show_keys: bool = True,
) -> py3Dmol.view:
    """Visualize a dictionary of molecules with py3Dmol."""
    num_keys = len(molecules)
    max_molecules_per_row = max([len(molecules[key]) for key in molecules])
    view = py3Dmol.view(
        viewergrid=(num_keys, max_molecules_per_row),
        linked=True,
        width=max_molecules_per_row * 180,
        height=num_keys * 180,
    )
    for i, key in enumerate(molecules):
        for j, mol in enumerate(molecules[key]):
            if j >= max_molecules_per_row:
                break

            try:
                view.addModel(Chem.MolToMolBlock(mol), "mol", {"keepH": "true"}, viewer=(i, j))
            except ValueError:
                # Sometimes RDKit errors out on weird molecules.
                continue

            if show_atom_types:
                for atom in mol.GetAtoms():
                    position = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                    view.addLabel(
                        str(atom.GetSymbol()),
                        {
                            "fontSize": 6,
                            "fontColor": "white",
                            "backgroundOpacity": 0.2,
                            "backgroundColor": "black",
                            "position": {
                                "x": position.x,
                                "y": position.y,
                                "z": position.z,
                            },
                        },
                        viewer=(i, j),
                    )

    # Add text labels for each row.
    if show_keys:
        for i, key in enumerate(molecules):
            for j in range(len(molecules[key])):
                view.addLabel(
                    key,
                    {
                        "fontSize": 12,
                        "fontColor": "black",
                        "backgroundColor": "white",
                        "backgroundOpacity": 0.8,
                        "position": {"x": -10, "y": 0, "z": 0},
                    },
                    viewer=(i, j),
                )

    view.setStyle({"stick": {"color": "spectrum", "radius": 0.2}, "sphere": {"scale": 0.3}})
    view.zoomTo()
    return view
