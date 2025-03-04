"""Script to combine relaxed IDRome v4 all-atom data."""

import argparse
import logging
import os
import multiprocessing
import mdtraj as md
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("process_IDRome")



def combine_frames(name: str, input_dir: str, original_traj_dir: str, output_dir: str) -> None:
    """Combine relaxed IDRome v4 all-atom frames."""

    traj_AA = None
    frames = sorted(os.listdir(os.path.join(input_dir, name)),
                    key=lambda name: int(name.split('_')[0]))
    for frame in frames:
        if not frame.endswith(".pdb"):
            continue

        frame_path = os.path.join(input_dir, name, frame)
        frame_AA = md.load_pdb(frame_path)
        
        if traj_AA is None:
            traj_AA = frame_AA
        else:
            traj_AA += frame_AA

    top_AA = md.Topology()
    chain = top_AA.add_chain()
    for residue in traj_AA.top.residues:
        res = top_AA.add_residue(residue.name, chain, resSeq=residue.index+1)
        for atom in residue.atoms:
            top_AA.add_atom(atom.name, element=atom.element, residue=res)
    top_AA.create_standard_bonds()

    original_traj_path = os.path.join(original_traj_dir, name, 'traj.xtc')
    original_top_path = os.path.join(original_traj_dir, name, 'top.pdb')
    original_traj = md.load_xtc(original_traj_path, top=original_top_path)
    original_traj = original_traj[0:traj_AA.n_frames]

    os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    traj_AA = md.Trajectory(traj_AA.xyz, top_AA, original_traj.time, original_traj.unitcell_lengths, original_traj.unitcell_angles)
    traj_AA[0].save_pdb(os.path.join(output_dir, name, 'top.pdb'))
    traj_AA.save_xtc(os.path.join(output_dir, name, 'traj.xtc'))

    py_logger.info(f"Successfully processed {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert IDRome v4 data to all-atom.')
    parser.add_argument('--name', help='Name of the trajectory.', type=str, required=True)
    parser.add_argument('--input-dir', help='Directory of relaxed all-atom trajectories (stored in each folder).', type=str, required=True)
    parser.add_argument('--original-traj-dir', help='Directory of original coarse-grained trajectories (stored in each folder).', type=str, required=True)
    parser.add_argument('--output-dir', '-o', help='Output directory to save combined relaxed all-atom trajectories (stored in each folder).', type=str, required=True)
    args = parser.parse_args()

    combine_frames(args.name, args.input_dir, args.original_traj_dir, args.output_dir)
