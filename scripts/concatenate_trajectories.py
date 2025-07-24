#!/usr/bin/env python3

import os
import glob
import mdtraj as md
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def concatenate_trajectories(folder_path, pdb_file, output_name="ALA_ALA.xtc"):
    """
    Concatenate all .xtc trajectories in a folder into a single long trajectory.
    
    Args:
        folder_path (str): Path to the folder containing .xtc files
        pdb_file (str): Path to the PDB topology file
        output_name (str): Name of the output trajectory file
    """
    folder_path = Path(folder_path)
    
    # Find all .xtc files in the folder
    xtc_files = sorted(glob.glob(str(folder_path / "*.xtc")))
    
    # Filter out any existing ALA_ALA.xtc to avoid including it in concatenation
    xtc_files = [f for f in xtc_files if not f.endswith("ALA_ALA.xtc")]
    
    if not xtc_files:
        print(f"No .xtc files found in {folder_path}")
        return
    
    print(f"Found {len(xtc_files)} .xtc files in {folder_path}")
    print(f"First few files: {xtc_files[:5]}")
    
    # Load the first trajectory to get the topology
    print("Loading first trajectory to get topology...")
    first_traj = md.load(xtc_files[0], top=pdb_file)
    print(f"Topology: {first_traj.n_atoms} atoms, {first_traj.n_frames} frames")
    
    # Initialize the concatenated trajectory with the first one
    concat_traj = first_traj
    
    # Load and concatenate the rest of the trajectories
    for xtc_file in tqdm(xtc_files[1:], desc="Concatenating trajectories", unit="file"):
        try:
            traj = md.load(xtc_file, top=pdb_file)
            concat_traj = concat_traj.join(traj)
                
        except Exception as e:
            tqdm.write(f"Error loading {os.path.basename(xtc_file)}: {e}")
            continue
    
    # Save the concatenated trajectory
    output_path = folder_path / output_name
    print(f"Saving concatenated trajectory to {output_path}")
    print(f"Final trajectory: {concat_traj.n_frames} frames, {concat_traj.n_atoms} atoms")
    
    concat_traj.save_xtc(str(output_path))
    print(f"Successfully saved {output_path}")
    
    return concat_traj

def main():
    parser = argparse.ArgumentParser(description="Concatenate .xtc trajectories in folders")
    parser.add_argument("--base-dir", 
                       default="/data2/sules/fake_enhanced_data/ALA_ALA_organized",
                       help="Base directory containing train/val/test folders")
    parser.add_argument("--pdb-file", 
                       default="/data/bucket/kleinhej/capped_diamines/timewarp_splits/train/ALA_ALA.pdb",
                       help="PDB topology file")
    parser.add_argument("--folders", nargs='+', 
                       default=["train", "val", "test"],
                       help="Folders to process")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    pdb_file = args.pdb_file
    
    # Check if PDB file exists
    if not os.path.exists(pdb_file):
        print(f"Error: PDB file not found: {pdb_file}")
        return
    
    print(f"Using PDB file: {pdb_file}")
    print(f"Base directory: {base_dir}")
    
    # Process each folder
    for folder in args.folders:
        folder_path = base_dir / folder
        
        if not folder_path.exists():
            print(f"Folder {folder_path} does not exist, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing folder: {folder}")
        print(f"{'='*60}")
        
        try:
            concatenate_trajectories(folder_path, pdb_file)
        except Exception as e:
            print(f"Error processing {folder}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Concatenation completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 