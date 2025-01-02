#!/usr/bin/env python3

import argparse
import os
import subprocess
from typing import Dict, Tuple, List
import tempfile
import logging

from jamun.utils import convert_to_three_letter_code, convert_to_one_letter_code

logging.basicConfig(
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    level=logging.INFO
)
logging.basicConfig(
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    level=logging.INFO
)
py_logger = logging.getLogger("prepare_pdb")


def parse_sequence(sequence: str) -> List[str]:
    """Parse sequence string into list of three-letter codes.
    
    Accepts:
    - String of one-letter codes (e.g., "AGPF")
    - Hyphen-separated three-letter codes (e.g., "ALA-GLY-PRO-PHE")
    """
    sequence = sequence.upper()
    
    # Check if it's a hyphen-separated three-letter code sequence
    if '-' in sequence:
        sequence = sequence.split('-')

    return [convert_to_three_letter_code(aa) for aa in sequence]


def create_sequence(amino_acids: List[str], mode: str) -> str:
    """Create sequence string with optional capping."""
    if len(amino_acids) < 2:
        raise ValueError("Sequence must have at least two amino acids")

    amino_acids = amino_acids.copy()
    if mode == "capped":
        sequence = " ".join(amino_acids)
        return f"{{ ACE {sequence} NME }}"
    
    if mode == "uncapped":
        amino_acids[0] = f"N{amino_acids[0]}"
        amino_acids[-1] = f"C{amino_acids[-1]}"

        sequence = " ".join(amino_acids)
        return f"{{ {sequence} }}"

    raise ValueError(f"Invalid mode: {mode}")


def format_sequence(amino_acids: str, mode: str) -> str:
    """Format sequence string for display."""
    amino_acids = amino_acids.copy()
    if mode == "capped":
        amino_acids = ["(ACE)"] + amino_acids + ["(NME)"] 
    if mode == "uncapped":
        amino_acids[0] = f"(N){amino_acids[0]}"
        amino_acids[-1] = f"(C){amino_acids[-1]}"

    return " ".join(amino_acids)


def create_tleap_input(sequence: str, output_file: str) -> str:
    """Create tleap input string."""
    return f"""source leaprc.protein.ff14SB
x = sequence {sequence}
savepdb x {output_file}
quit
"""

def run_tleap(input_content: str) -> Tuple[bool, str]:
    """Run tleap with given input content."""
   
    # Write tleap input file.
    _, input_file = tempfile.mkstemp(suffix='.in', prefix='tleap_', text=True)
    with open(input_file, 'w') as f:
        f.write(input_content)

    py_logger.info("Running tleap with input: \n" + input_content)

    try:
        # Run tleap command
        result = subprocess.run(['tleap', '-f', input_file], 
                             capture_output=True, 
                             text=True,
                             check=True)
        success = True
        message = "Successfully generated structure"
    except subprocess.CalledProcessError as e:
        success = False
        message = f"tleap error: {e.stderr}"
    finally:
        # Clean up input file
        if os.path.exists(input_file):
            os.remove(input_file)

    return success, message

def create_output_filename(amino_acids: List[str], mode: str, outputdir: str) -> str:
    """Create output filename based on sequence and mode."""
    sequence_short = "".join([convert_to_one_letter_code(aa) for aa in amino_acids])
    return os.path.join(outputdir, f"{mode}_{sequence_short}.pdb")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Generate .pdb file for a peptide sequence using AmberTools tleap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sequence format examples:
  AGPF             (one-letter codes)
  ALA-GLY-PRO-PHE  (three-letter codes with hyphens)
        """)
    parser.add_argument('sequence',
                      help='Amino acid sequence (one-letter string or hyphen-separated three-letter codes)')
    parser.add_argument('--mode', choices=['capped', 'uncapped'], required=True,
                      help='Specify if peptide should be capped with ACE/NME')
    parser.add_argument('--outputdir', default='.',
                      help='Output directory (default: current directory)')

    # Parse arguments
    args = parser.parse_args()

    try:
        # Parse and validate sequence
        amino_acids = parse_sequence(args.sequence)

        # Validate output directory
        if not os.path.isdir(args.outputdir):
            py_logger.info(f"Output directory does not exist: {args.outputdir}. Creating...")
            os.makedirs(args.outputdir)

        py_logger.info(f"Output directory: {os.path.abspath(args.outputdir)}")

        if not os.access(args.outputdir, os.W_OK):
            raise ValueError(f"Output directory is not writable: {args.outputdir}")

        # Create sequence and output filename
        sequence = create_sequence(amino_acids, args.mode)
        output_file = create_output_filename(amino_acids, args.mode, args.outputdir)

        # Create tleap input
        tleap_input = create_tleap_input(sequence, output_file)

        # Run tleap
        success, message = run_tleap(tleap_input)

        if success:
            py_logger.info(f"Generated {args.mode} peptide sequence: {format_sequence(amino_acids, args.mode)}")
            py_logger.info(f"Output saved to: {os.path.abspath(output_file)}")
        else:
            py_logger.info(f"Error: {message}")
            exit(1)

    except Exception as e:
        py_logger.info(f"Unexpected error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()