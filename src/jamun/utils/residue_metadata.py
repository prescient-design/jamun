class ResidueMetadata:
    """Metadata for residues and atoms."""

    ATOM_TYPES = ["C", "O", "N", "F", "S"]
    ATOM_CODES = ["C", "O", "N", "S", "CA", "CB"]
    RESIDUE_CODES = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLU",
        "GLN",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "ACE",
        "NME",
    ]


def encode_atom_type(atom_type: str) -> int:
    """Encode atom symbol (eg. C) as an integer."""
    if atom_type in ResidueMetadata.ATOM_TYPES:
        return ResidueMetadata.ATOM_TYPES.index(atom_type)
    else:
        return len(ResidueMetadata.ATOM_TYPES)


def encode_atom_code(atom_code: str) -> int:
    """Encode atom code (eg. CA) as an integer."""
    if atom_code in ResidueMetadata.ATOM_CODES:
        return ResidueMetadata.ATOM_CODES.index(atom_code)
    else:
        return len(ResidueMetadata.ATOM_CODES)


def encode_residue(residue_name: str) -> int:
    """Encode residue name as an integer."""
    if residue_name in ResidueMetadata.RESIDUE_CODES:
        return ResidueMetadata.RESIDUE_CODES.index(residue_name)
    else:
        return len(ResidueMetadata.RESIDUE_CODES)
