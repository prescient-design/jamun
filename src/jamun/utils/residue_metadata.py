from typing import Dict, List


class ResidueMetadata:
    """Metadata for residues and atoms."""

    ATOM_TYPES: List[str] = ["C", "O", "N", "F", "S"]
    ATOM_CODES: List[str] = ["C", "O", "N", "S", "CA", "CB"]
    RESIDUE_CODES: List[str] = [
        #Canoncial amino acids
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
        # Add noncanonical amino acids and modifications
        "(N->O)Leu",
        "(N->O)Val",
        "(N->O)Val(3-OH)",
        "1-Nal",
        "2-pyridylmethyl_Gly",
        "3-pyridylethyl_Gly",
        "3Pal",
        "Abu",
        "Aib",
        "Ala(indol-2-yl)",
        "Ala(tBu)",
        "Bal",
        "Bn(4-Cl)_Gly",
        "Bn(4-OH)_Gly",
        "Bn_Gly",
        "Cha",
        "Cys(EtO2H)_NH2",
        "Et_Gly",
        "Hph",
        "MeOEt_Gly",
        "Me_Abu",
        "Me_Abu(morpholino)",
        "Me_Ala(indol-2-yl)",
        "Me_Cha",
        "Me_Nle",
        "Me_Tza",
        "Me_dAbu",
        "Me_dNva",
        "Mono1",
        "Mono100",
        "Mono14",
        "Mono16",
        "Mono2",
        "Mono29",
        "Mono30",
        "Mono31",
        "Mono32",
        "Mono33",
        "Mono34",
        "Mono38",
        "Mono39",
        "Mono4",
        "Mono40",
        "Mono41",
        "Mono42",
        "Mono44",
        "Mono45",
        "Mono48",
        "Mono5",
        "Mono6",
        "Mono67",
        "Mono7",
        "Nal",
        "Nle",
        "Nva(Ph)",
        "PhPr_Gly",
        "Phe(4-F)",
        "Phg",
        "Pr_Gly",
        "Pye",
        "Sar",
        "Ser(Bn)",
        "Ser(tBu)",
        "Sta",
        "Sta(3R,4R)",
        "Tle",
        "Tyr(Me)",
        "Tza",
        "bHph",
        "cHexCH2_Gly",
        "d(N->O)Gly(allyl)",
        "d(N->O)Leu",
        "d(N->O)Val",
        "d(N->O)aIle",
        "dAbu",
        "dAla(indol-2-yl)",
        "dAsn(Me2)",
        "dAsp(pyrrol-1-yl)",
        "dCha",
        "dGln(Me2)",
        "dGlu(OMe)",
        "dNva",
        "dPye",
        "iBu_Gly",
        "pentyl_Gly",
    ]

    Cremp_RESIDUE_Mapping: Dict[str, str] = {
        # Mapping for D-amino acids (lowercase)
        "dA": "a",
        "dC": "c",
        "dD": "d",
        "dE": "e",
        "dF": "f",
        "dG": "g",
        "dH": "h",
        "dI": "i",
        "dK": "k",
        "dL": "l",
        "dM": "m",
        "dN": "n",
        "dP": "p",
        "dQ": "q",
        "dR": "r",
        "dS": "s",
        "dT": "t",
        "dV": "v",
        "dW": "w",
        "dY": "y",

        # Mapping for N-methylated L-amino acids (uppercase)
        "meA": "MeA",
        "meC": "MeC",
        "meD": "MeD",
        "meE": "MeE",
        "meF": "MeF",
        "meG": "MeG",
        "meH": "MeH",
        "meI": "MeI",
        "meK": "MeK",
        "meL": "MeL",
        "meM": "MeM",
        "meN": "MeN",
        "meP": "MeP",
        "meQ": "MeQ",
        "meR": "MeR",
        "meS": "MeS",
        "meT": "MeT",
        "meV": "MeV",
        "meW": "MeW",
        "meY": "MeY",

        # Mapping for N-methylated D-amino acids
        "Me_dA": "Mea",
        "Me_dC": "Mec",
        "Me_dD": "Med",
        "Me_dE": "Mee",
        "Me_dF": "Mef",
        "Me_dG": "Meg",
        "Me_dH": "Meh",
        "Me_dI": "Mei",
        "Me_dK": "Mek",
        "Me_dL": "Mel",
        "Me_dM": "Mem",
        "Me_dN": "Men",
        "Me_dP": "Mep",
        "Me_dQ": "Meq",
        "Me_dR": "Mer",
        "Me_dS": "Mes",
        "Me_dT": "Met",
        "Me_dV": "Mev",
        "Me_dW": "Me",
        "Me_dY": "Mey",
    }

    # One to three letter code mapping
    AA_3CODES: Dict[str, str] = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "E": "GLU",
        "Q": "GLN",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
        # Add mappings for noncanonical amino acids
        "(N->O)Leu": "(N->O)Leu",
        "(N->O)Val": "(N->O)Val",
        "(N->O)Val(3-OH)": "(N->O)Val(3-OH)",
        "1-Nal": "1-Nal",
        "2-pyridylmethyl_Gly": "2-pyridylmethyl_Gly",
        "3-pyridylethyl_Gly": "3-pyridylethyl_Gly",
        "3Pal": "3Pal",
        "Abu": "Abu",
        "Aib": "Aib",
        "Ala(indol-2-yl)": "Ala(indol-2-yl)",
        "Ala(tBu)": "Ala(tBu)",
        "Bal": "Bal",
        "Bn(4-Cl)_Gly": "Bn(4-Cl)_Gly",
        "Bn(4-OH)_Gly": "Bn(4-OH)_Gly",
        "Bn_Gly": "Bn_Gly",
        "Cha": "Cha",
        "Cys(EtO2H)_NH2": "Cys(EtO2H)_NH2",
        "Et_Gly": "Et_Gly",
        "Hph": "Hph",
        "MeOEt_Gly": "MeOEt_Gly",
        "Me_Abu": "Me_Abu",
        "Me_Abu(morpholino)": "Me_Abu(morpholino)",
        "Me_Ala(indol-2-yl)": "Me_Ala(indol-2-yl)",
        "Me_Cha": "Me_Cha",
        "Me_Nle": "Me_Nle",
        "Me_Tza": "Me_Tza",
        "Me_dAbu": "Me_dAbu",
        "Me_dNva": "Me_dNva",
        "Mono1": "Mono1",
        "Mono100": "Mono100",
        "Mono14": "Mono14",
        "Mono16": "Mono16",
        "Mono2": "Mono2",
        "Mono29": "Mono29",
        "Mono30": "Mono30",
        "Mono31": "Mono31",
        "Mono32": "Mono32",
        "Mono33": "Mono33",
        "Mono34": "Mono34",
        "Mono38": "Mono38",
        "Mono39": "Mono39",
        "Mono4": "Mono4",
        "Mono40": "Mono40",
        "Mono41": "Mono41",
        "Mono42": "Mono42",
        "Mono44": "Mono44",
        "Mono45": "Mono45",
        "Mono48": "Mono48",
        "Mono5": "Mono5",
        "Mono6": "Mono6",
        "Mono67": "Mono67",
        "Mono7": "Mono7",
        "Nal": "Nal",
        "Nle": "Nle",
        "Nva(Ph)": "Nva(Ph)",
        "PhPr_Gly": "PhPr_Gly",
        "Phe(4-F)": "Phe(4-F)",
        "Phg": "Phg",
        "Pr_Gly": "Pr_Gly",
        "Pye": "Pye",
        "Sar": "Sar",
        "Ser(Bn)": "Ser(Bn)",
        "Ser(tBu)": "Ser(tBu)",
        "Sta": "Sta",
        "Sta(3R,4R)": "Sta(3R,4R)",
        "Tle": "Tle",
        "Tyr(Me)": "Tyr(Me)",
        "Tza": "Tza",
        "bHph": "bHph",
        "cHexCH2_Gly": "cHexCH2_Gly",
        "d(N->O)Gly(allyl)": "d(N->O)Gly(allyl)",
        "d(N->O)Leu": "d(N->O)Leu",
        "d(N->O)Val": "d(N->O)Val",
        "d(N->O)aIle": "d(N->O)aIle",
        "dAbu": "dAbu",
        "dAla(indol-2-yl)": "dAla(indol-2-yl)",
        "dAsn(Me2)": "dAsn(Me2)",
        "dAsp(pyrrol-1-yl)": "dAsp(pyrrol-1-yl)",
        "dCha": "dCha",
        "dGln(Me2)": "dGln(Me2)",
        "dGlu(OMe)": "dGlu(OMe)",
        "dNva": "dNva",
        "dPye": "dPye",
        "iBu_Gly": "iBu_Gly",
        "pentyl_Gly": "pentyl_Gly",
    }

    # Three to one letter code mapping
    AA_1CODES: Dict[str, str] = {v: k for k, v in AA_3CODES.items()}


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
    if residue_name.startswith("Me+"):
        return len(ResidueMetadata.RESIDUE_CODES) + encode_residue(residue_name[len("Me+"):])
    # Check if the residue name is already in canonical form
    if residue_name in ResidueMetadata.RESIDUE_CODES:
        return ResidueMetadata.RESIDUE_CODES.index(residue_name)
    else:
        raise ValueError(f"Invalid residue name: {residue_name}. Valid names are: {ResidueMetadata.RESIDUE_CODES}")
        return len(ResidueMetadata.RESIDUE_CODES)


def convert_to_three_letter_code(aa: str) -> str:
    """Convert one-letter amino acid code to three-letter code."""
    if aa.startswith("Me+"):
        return "Me+" + convert_to_three_letter_code(aa[len("Me+"):])  # Return the rest of the string after "Me"

    aa = aa.upper()
    if len(aa) == 1:
        if aa not in ResidueMetadata.AA_3CODES:
            raise ValueError(f"Invalid one-letter amino acid code: {aa}")
        return ResidueMetadata.AA_3CODES[aa]
    elif len(aa) == 3:
        if aa not in ResidueMetadata.AA_1CODES.values():
            raise ValueError(f"Invalid three-letter amino acid code: {aa}")
        return aa
    else:
        raise ValueError(f"Invalid amino acid code length: {aa}")


def convert_to_three_letter_codes(peptide: str) -> str:
    """Convert peptides with one-letter amino acid codes to peptides with three-letter codes."""
    if "_" in peptide:
        return peptide
    return "_".join([convert_to_three_letter_code(aa) for aa in peptide])


def convert_to_one_letter_code(aa: str) -> str:
    """Convert three-letter amino acid code to one-letter code."""
    aa = aa.upper()
    if len(aa) == 1:
        if aa not in ResidueMetadata.AA_3CODES:
            raise ValueError(f"Invalid one-letter amino acid code: {aa}")
        return aa
    elif len(aa) == 3:
        if aa not in ResidueMetadata.AA_1CODES:
            raise ValueError(f"Invalid three-letter amino acid code: {aa}")
        return ResidueMetadata.AA_1CODES[aa]
    else:
        raise ValueError(f"Invalid amino acid code length: {aa}")


def convert_to_one_letter_codes(peptide: str) -> str:
    """Convert peptides with three-letter amino acid codes to peptides with one-letter codes."""
    if "_" not in peptide:
        return peptide
    return "".join([convert_to_one_letter_code(aa) for aa in peptide.split("_")])

#Adding code to check for residue name in RESIDUE_CODES and convert it to the same naming convention as in ResidueMetadata.Cremp_RESIDUE_Mapping
def convert_to_canonical_residue_names(residue_name: str) -> str:
    """Convert a residue name to its canonical form."""
    if residue_name in ResidueMetadata.RESIDUE_CODES:
        return residue_name
    elif residue_name in ResidueMetadata.Cremp_RESIDUE_Mapping:
        return ResidueMetadata.Cremp_RESIDUE_Mapping[residue_name]
    else:
        raise ValueError(f"Invalid residue name: {residue_name}. Valid names are: {ResidueMetadata.RESIDUE_CODES}")