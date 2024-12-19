#!/usr/bin/env python3

import argparse
import os
import subprocess
from typing import Dict, Tuple, List

import logging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
log = logging.getLogger("[prepare_pdb]")

# One to three letter code mapping
AA_CODES: Dict[str, str] = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}

def convert_aa_code(aa: str) -> str:
    """Convert one-letter amino acid code to three-letter code."""
    aa = aa.upper()
    if len(aa) == 1:
        if aa not in AA_CODES:
            raise ValueError(f"Invalid one-letter amino acid code: {aa}")
        return AA_CODES[aa]
    elif len(aa) == 3:
        if aa not in AA_CODES.values():
            raise ValueError(f"Invalid three-letter amino acid code: {aa}")
        return aa
    else:
        raise ValueError(f"Invalid amino acid code length: {aa}")

def parse_sequence(sequence: str) -> List[str]:
    """Parse sequence string into list of three-letter codes.
    
    Accepts:
    - String of one-letter codes (e.g., "AGPF")
    - Hyphen-separated three-letter codes (e.g., "ALA-GLY-PRO-PHE")
    """
    sequence = sequence.upper()
    
    # Check if it's a hyphen-separated three-letter code sequence
    if '-' in sequence:
        three_letter_codes = sequence.split('-')
        return [convert_aa_code(code) for code in three_letter_codes]

    # Otherwise, treat as string of one-letter codes
    return [convert_aa_code(aa) for aa in sequence]

def create_sequence(amino_acids: List[str], mode: str) -> str:
    """Create sequence string with optional capping."""
    sequence = " ".join(amino_acids)
    if mode == "capped":
        return f"{{ ACE {sequence} NME }}"
    return f"{{ {sequence} }}"


def format_sequence(amino_acids: str, mode: str) -> str:
    """Format sequence string for display."""
    if mode == "capped":
        amino_acids = ["(ACE)"] + amino_acids + ["(NME)"]
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
    # Write tleap input file
    with open('tleap.in', 'w') as f:
        f.write(input_content)

    try:
        # Run tleap command
        result = subprocess.run(['tleap', '-f', 'tleap.in'], 
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
        if os.path.exists('tleap.in'):
            os.remove('tleap.in')

    return success, message

def create_output_filename(amino_acids: List[str], mode: str, outputdir: str) -> str:
    """Create output filename based on sequence and mode."""
    sequence_name = "_".join(amino_acids)
    return os.path.join(outputdir, f"{mode}_{sequence_name}.pdb")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Generate .pdb file for a peptide sequence using AmberTools tleap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sequence format examples:
  AGPF              (one-letter codes)
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
            log.info(f"Output directory does not exist: {args.outputdir}. Creating...")
            os.makedirs(args.outputdir)

        log.info(f"Output directory: {os.path.abspath(args.outputdir)}")

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
            log.info(f"Generated {args.mode} peptide sequence: {format_sequence(amino_acids, args.mode)}")
            log.info(f"Output saved to: {os.path.abspath(output_file)}")
        else:
            log.info(f"Error: {message}")
            exit(1)

    except Exception as e:
        log.info(f"Unexpected error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()