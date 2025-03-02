"""Script to combine relaxed IDRome v4 all-atom data."""

import argparse
import logging
import os
import multiprocessing
import mdtraj as md
from concurrent.futures import ProcessPoolExecutor

logging.basicConfig(format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s", level=logging.INFO)
py_logger = logging.getLogger("process_IDRome")



def combine_frames(args, use_srun: bool = True) -> None:
    """Combine relaxed IDRome v4 all-atom frames."""
    name, input_dir, original_traj_dir, output_dir = args

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

    traj = md.load_xtc(os.path.join(original_traj_dir, f'{name}.xtc'), top=os.path.join(original_traj_dir, f'{name}.pdb'))
    traj_AA = md.Trajectory(traj_AA.xyz, top_AA, traj.time, traj.unitcell_lengths, traj.unitcell_angles)
    traj_AA[0].save_pdb(os.path.join(output_dir, f"{name}.pdb"))
    traj_AA.save_xtc(os.path.join(output_dir, f"{name}.xtc"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert IDRome v4 data to all-atom.')
    parser.add_argument('--input-dir', help='Directory of relaxed all-atom trajectories (stored in each folder).', type=str, required=True)
    parser.add_argument('--original-traj-dir', help='Directory of original coarse-grained trajectories (stored in each folder).', type=str, required=True)
    parser.add_argument('--output-dir', '-o', help='Output directory to save combined relaxed all-atom trajectories (stored in each folder).', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=multiprocessing.cpu_count(),
                      help='Number of parallel workers')
    args = parser.parse_args()

    # Run in parallel.
    names = sorted(os.listdir(args.input_dir))
    preprocess_args = list(
        zip(
            names,
            [args.input_dir] * len(names),
            [args.original_traj_dir] * len(names),
            [args.output_dir] * len(names),
        )
    )
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(executor.map(combine_frames, preprocess_args))


