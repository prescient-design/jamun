"""Breaks up a trajectory file into smaller chunks."""

import mdtraj as md
import os
import argparse
import shutil


def split_trajectory(input_traj_file: str, input_top_file: str, chunk_size: int, output_dir: str) -> None:
    """
    Split an MD trajectory into smaller chunks.
    
    Parameters:
    -----------
    input_traj_file : str
        Path to input trajectory file (e.g., .xtc, .dcd, etc.)
    input_top_file : str
        Path to topology file (e.g., .pdb, .gro)
    chunk_size : int
        Number of frames per chunk
    output_dir : str
        Directory to save output chunks
    """
    # Create output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    # Output topology path.
    base_name = os.path.splitext(os.path.basename(input_traj_file))[0]
    out_top_path = os.path.join(output_dir, f"{base_name}.pdb")
    if os.path.exists(out_top_path):
        return

    # Load trajectory.
    traj = md.load(input_traj_file, top=input_top_file)
    
    # Get total number of frames.
    n_frames = traj.n_frames
    
    # Calculate number of chunks.
    n_chunks = (n_frames + chunk_size - 1) // chunk_size
    
    # Split and save chunks.
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_frames)
        
        # Extract chunk.
        chunk = traj[start_idx:end_idx]
        
        # Generate output filename.
        out_path = os.path.join(output_dir, f"{base_name}_chunk_{i+1}.xtc")
        
        # Save chunk.
        chunk.save_xtc(out_path)
    
    # Save topology file.
    shutil.copy(input_top_file, out_top_path)


def main():
    # Set up argument parser.
    parser = argparse.ArgumentParser(description='Split MD trajectory into chunks')
    parser.add_argument('--trajectory', required=True, help='Input trajectory file')
    parser.add_argument('--topology', required=True, help='Topology file')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Output directory')
    parser.add_argument('-c', '--chunk-size', type=int, default=20000,
                        help='Number of frames per chunk (default: 20000)')
    
    # Parse arguments.
    args = parser.parse_args()
    
    # Split trajectory.
    split_trajectory(args.trajectory, args.topology, args.chunk_size, args.output_dir)

if __name__ == "__main__":
    main()
