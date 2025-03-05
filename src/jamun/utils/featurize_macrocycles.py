import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Set, Iterator, List, Literal, Optional, Union, Tuple, FrozenSet


import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdchem import ChiralType, HybridizationType

# Constants for encoding
ATOMIC_NUMS = list(range(1, 100))
PEPTIDE_CHIRAL_TAGS = {"L": 1, "D": -1, None: 0}
CHIRAL_TAGS = {ChiralType.CHI_TETRAHEDRAL_CW: -1, ChiralType.CHI_TETRAHEDRAL_CCW: 1, ChiralType.CHI_UNSPECIFIED: 0, ChiralType.CHI_OTHER: 0}
HYBRIDIZATION_TYPES = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2]
DEGREES = [0, 1, 2, 3, 4, 5]
VALENCES = [0, 1, 2, 3, 4, 5, 6]
NUM_HYDROGENS = [0, 1, 2, 3, 4]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
RING_SIZES = [3, 4, 5, 6, 7, 8]
NUM_RINGS = [0, 1, 2, 3]

# Feature names
ATOMIC_NUM_FEATURE_NAMES = [f"anum{anum}" for anum in ATOMIC_NUMS] + ["anumUNK"]
CHIRAL_TAG_FEATURE_NAME = "chiraltag"
AROMATICITY_FEATURE_NAME = "aromatic"
HYBRIDIZATION_TYPE_FEATURE_NAMES = [f"hybrid{ht}" for ht in HYBRIDIZATION_TYPES] + ["hybridUNK"]
DEGREE_FEATURE_NAMES = [f"degree{d}" for d in DEGREES] + ["degreeUNK"]
VALENCE_FEATURE_NAMES = [f"valence{v}" for v in VALENCES] + ["valenceUNK"]
NUM_HYDROGEN_FEATURE_NAMES = [f"numh{nh}" for nh in NUM_HYDROGENS] + ["numhUNK"]
FORMAL_CHARGE_FEATURE_NAMES = [f"charge{c}" for c in FORMAL_CHARGES] + ["chargeUNK"]
RING_SIZE_FEATURE_NAMES = [f"ringsize{rs}" for rs in RING_SIZES]  # No "unknown" name needed
NUM_RING_FEATURE_NAMES = [f"numring{nr}" for nr in NUM_RINGS] + ["numringUNK"]

# Constants and data for amino acids
AMINO_ACID_DATA_PATH = "/homefs/home/davidsd5/jamun/jamun/src/jamun/data/amino_acids.csv"
AMINO_ACID_DATA = pd.read_csv(AMINO_ACID_DATA_PATH, index_col="aa")
AMINO_ACID_DATA["residue_mol"] = AMINO_ACID_DATA["residue_smiles"].map(Chem.MolFromSmiles)

RING_PEPTIDE_BOND_PATTERN = Chem.MolFromSmarts("[C;R:0](=[OX1:1])[C;R:2][N;R:3]")
GENERIC_AMINO_ACID_SMARTS = "[$([CX3](=[OX1]))][NX3,NX4+][$([CX4H]([CX3](=[OX1])[O,N]))][*]"

SIDE_CHAIN_TORSIONS_SMARTS_DICT = {
    "alanine": "[CH3X4]",
    "asparagine": "[CH2X4][$([CX3](=[OX1])[NX3H2])][NX3H2]",
    "aspartic acid": "[CH2X4][$([CX3](=[OX1])[OH0-,OH])][OH0-,OH]",
    "cysteine": "[CH2X4][SX2H,SX1H0-]",
    "glutamic acid": "[CH2X4][CH2X4][$([CX3](=[OX1])[OH0-,OH])][OH0-,OH]",
    "glutamine": "[CH2X4][CH2X4][$([CX3](=[OX1])[NX3H2])][NX3H2]",
    "histidine": "[CH2X4][$([#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1)]:[#6X3H]",
    "isoleucine": "[$([CHX4]([CH3X4])[CH2X4][CH3X4])][CH2X4][CH3X4]",
    "leucine": "[CH2X4][$([CHX4]([CH3X4])[CH3X4])][CH3X4]",
    "lysine": "[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]",
    "phenylalanine": "[CH2X4][$([cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1)][cX3H]",
    "serine": "[CH2X4][OX2H]",
    "threonine": "[$([CHX4]([OX2H])[CH3X4])][CH3X4]",
    "tryptophan": "[CH2X4][$([cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12)][cX3H0]",
    "tyrosine": "[CH2X4][$([cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1)][cX3H]",
    "valine": "[$([CHX4]([CH3X4])[CH3X4])][CH3X4]",
}

# Updating for proline which doesn't conform to the generic pattern
AMINO_ACID_TORSIONS_SMARTS_DICT = {
    name: GENERIC_AMINO_ACID_SMARTS.replace("[*]", smarts)
    for name, smarts in SIDE_CHAIN_TORSIONS_SMARTS_DICT.items()
}
AMINO_ACID_TORSIONS_SMARTS_DICT["proline"] = (
    "[$([CX3](=[OX1]))]"
    "[$([$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)"
    "[CX3](=[OX1])[OX2H,OX1-,N])]"
    "[$([CX4H]1[CH2][CH2][CH2][$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1)]"
    "[$([CX4H2]1[CH2][CH2][$([NX3H,NX4H2+]),$([NX3](C)(C)(C))][CX4H]1)]"
)
AMINO_ACID_TORSIONS_PATTERNS = {
    name: Chem.MolFromSmarts(smarts) for name, smarts in AMINO_ACID_TORSIONS_SMARTS_DICT.items()
}
AMINO_ACIDS_WITH_SYMMETRY = {"leucine", "phenylalanine", "tyrosine", "valine"}

# Carboxyl before nitrogen corresponds to N-to-C direction of peptide
PEPTIDE_PATTERN = Chem.MolFromSmarts("[OX1]=[C;R][N;R]")

def get_macrocycle_idxs(
    mol: Chem.Mol, min_size: int = 9, n_to_c: bool = True
) -> Optional[List[int]]:
    sssr = Chem.GetSymmSSSR(mol)
    if len(sssr) > 0:
        largest_ring = max(sssr, key=len)
        if len(largest_ring) >= min_size:
            idxs = list(largest_ring)
            if n_to_c:
                return macrocycle_idxs_in_n_to_c_direction(mol, idxs)
            return idxs
    return None

def get_wrapped_overlapping_sublists(list_: List[Any], size: int) -> Iterator[List[Any]]:
    for idx in range(len(list_)):
        idxs = [idx]
        for offset in range(1, size):
            # Wrap past end of list
            idxs.append((idx + offset) % len(list_))
        yield [list_[i] for i in idxs]


def get_overlapping_sublists(
    list_: List[Any], size: int, wrap: bool = True
) -> Iterator[List[Any]]:
    if wrap:
        for item in get_wrapped_overlapping_sublists(list_, size):
            yield item
    else:
        for i in range(len(list_) - size + 1):
            yield list_[i : i + size]

def macrocycle_idxs_in_n_to_c_direction(mol: Chem.Mol, macrocycle_idxs: List[int]) -> List[int]:
    # Obtain carbon and nitrogen idxs in peptide bonds in the molecule
    matches = mol.GetSubstructMatches(PEPTIDE_PATTERN)
    if not matches:
        raise ValueError("Did not match any peptide bonds")

    # We match 3 atoms each time (O, C, N), just need C and N in the ring
    carbon_and_nitrogen_idxs = {match[1:] for match in matches}

    for atom_idx_pair in get_overlapping_sublists(macrocycle_idxs, 2):
        # If the directionality of atom idxs is already in N to C direction, then pairs of these
        # atom indices should already be in the set of matched atoms, otherwise, we need to flip
        # the direction
        if tuple(atom_idx_pair) in carbon_and_nitrogen_idxs:
            break
    else:
        macrocycle_idxs = macrocycle_idxs[::-1]  # Flip direction

    # Always start at a nitrogen
    nitrogen_idx = next(iter(carbon_and_nitrogen_idxs))[1]  # Random nitrogen
    nitrogen_loc = macrocycle_idxs.index(nitrogen_idx)
    macrocycle_idxs = macrocycle_idxs[nitrogen_loc:] + macrocycle_idxs[:nitrogen_loc]

    return macrocycle_idxs

def extract_macrocycle(mol: Chem.Mol) -> Chem.Mol:
    macrocycle_idxs = get_macrocycle_idxs(mol)
    if macrocycle_idxs is None:
        raise ValueError(f"No macrocycle detected in '{Chem.MolToSmiles(Chem.RemoveHs(mol))}'")

    macrocycle_idxs = set(macrocycle_idxs)
    to_remove = sorted(
        (atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in macrocycle_idxs),
        reverse=True,
    )

    rwmol = Chem.RWMol(mol)
    for idx in to_remove:
        rwmol.RemoveAtom(idx)

    new_mol = rwmol.GetMol()
    return new_mol

def combine_mols(mols: List[Chem.Mol]) -> Chem.Mol:
    """Combine multiple molecules with one conformer each into one molecule with multiple
    conformers.

    Args:
        mols: List of molecules.

    Returns:
        Combined molecule.
    """
    new_mol = Chem.Mol(mols[0], quickCopy=True)
    for mol in mols:
        conf = Chem.Conformer(mol.GetConformer())
        new_mol.AddConformer(conf, assignId=True)
    return new_mol


def set_atom_positions(
    mol: Chem.Mol,
    xyzs: Union[np.ndarray, pd.DataFrame, List[np.ndarray], List[pd.DataFrame]],
    atom_idxs: Optional[List[int]] = None,
) -> Chem.Mol:
    """Set atom positions of a molecule.

    Args:
        mol: Molecule.
        xyzs: An array of coordinates; a dataframe with 'x', 'y', and 'z' columns (and optionally an index of atom indices); a list of arrays; or a list of dataframes.
        atom_idxs: Atom indices to set atom positions for. Not required if dataframe(s) contain(s) atom indices.

    Returns:
        A copy of the molecule with one conformer for each set of coordinates.
    """
    # If multiple xyxs are provided, make one conformer for each one
    if isinstance(xyzs, (np.ndarray, pd.DataFrame)):
        xyzs = [xyzs]

    if atom_idxs is None:
        assert all(isinstance(xyz, pd.DataFrame) for xyz in xyzs)

    # The positions that don't get set will be the same as in the first conformer of the given mol
    dummy_conf = mol.GetConformer()
    mol = Chem.Mol(mol, quickCopy=True)  # Don't copy conformers

    for xyz in xyzs:
        if isinstance(xyz, pd.DataFrame):
            atom_idxs = xyz.index.tolist()
            xyz = xyz[["x", "y", "z"]].to_numpy()

        xyz = xyz.reshape(-1, 3)

        # Set only the positions at the provided indices
        conf = Chem.Conformer(dummy_conf)
        for atom_idx, pos in zip(atom_idxs, xyz):
            conf.SetAtomPosition(atom_idx, [float(p) for p in pos])

        mol.AddConformer(conf, assignId=True)

    return mol


def dfs(
    root_atom_idx: int,
    mol: Chem.Mol,
    max_depth: int = float("inf"),
    blocked_idxs: Optional[List[int]] = None,
    include_hydrogens: bool = True,
) -> List[int]:
    """Traverse molecular graph with depth-first search from given root atom index.

    Args:
        root_atom_idx: Root atom index.
        mol: Molecule.
        max_depth: Only traverse to this maximum depth.
        blocked_idxs: Don't traverse across these indices. Defaults to None.
        include_hydrogens: Include hydrogen atom indices in returned list.

    Returns:
        List of traversed atom indices in DFS order.
    """
    root_atom = mol.GetAtomWithIdx(root_atom_idx)
    if blocked_idxs is not None:
        blocked_idxs = set(blocked_idxs)
    return _dfs(
        root_atom,
        max_depth=max_depth,
        blocked_idxs=blocked_idxs,
        include_hydrogens=include_hydrogens,
    )


def _dfs(
    atom: Chem.Atom,  # Start from atom so we don't have to get it from index each time
    depth: int = 0,
    max_depth: int = float("inf"),
    blocked_idxs: Optional[Set[int]] = None,
    include_hydrogens: bool = True,
    visited: Optional[Set[int]] = None,
    traversal: Optional[List[int]] = None,
) -> List[int]:
    if visited is None:
        visited = set()
    if traversal is None:
        traversal = []

    if include_hydrogens or atom.GetAtomicNum() != 1:
        atom_idx = atom.GetIdx()
        visited.add(atom_idx)
        traversal.append(atom_idx)

    if depth < max_depth:
        for atom_nei in atom.GetNeighbors():
            atom_nei_idx = atom_nei.GetIdx()
            if atom_nei_idx not in visited:
                if blocked_idxs is None or atom_nei_idx not in blocked_idxs:
                    _dfs(
                        atom_nei,
                        depth=depth + 1,
                        max_depth=max_depth,
                        blocked_idxs=blocked_idxs,
                        include_hydrogens=include_hydrogens,
                        visited=visited,
                        traversal=traversal,
                    )

    return traversal

def one_k_encoding(value: Any, choices: List[Any], include_unknown: bool = True) -> List[int]:
    """Create a one-hot encoding with an extra category for uncommon values."""
    encoding = [0] * (len(choices) + include_unknown)
    try:
        idx = choices.index(value)
    except ValueError:
        if include_unknown:
            idx = -1
        else:
            raise ValueError(f"Cannot encode '{value}' because it is not in {choices}")
    encoding[idx] = 1
    return encoding

def featurize_macrocycle_atoms(
    mol: Chem.Mol,
    macrocycle_idxs: Optional[List[int]] = None,
    use_peptide_stereo: bool = True,
    residues_in_mol: Optional[List[str]] = None,
    include_side_chain_fingerprint: bool = True,
    radius: int = 3,
    size: int = 2048,
) -> pd.DataFrame:
    """Create a sequence of features for each atom in `macrocycle_idxs`."""
    atom_features = {}
    ring_info = mol.GetRingInfo()
    morgan_fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size, includeChirality=True)
    fingerprint_feature_names = [f"fp{i}" for i in range(size)]

    if macrocycle_idxs is None:
        macrocycle_idxs = [atom.GetIdx() for atom in mol.GetAtoms()]

    if use_peptide_stereo:
        residues = get_residues(mol, residues_in_mol=residues_in_mol, macrocycle_idxs=macrocycle_idxs)
        atom_to_residue = {atom_idx: symbol for atom_idxs, symbol in residues.items() for atom_idx in atom_idxs}

    for atom_idx in macrocycle_idxs:
        atom_feature_dict = {}
        atom = mol.GetAtomWithIdx(atom_idx)

        atomic_num_onehot = one_k_encoding(atom.GetAtomicNum(), ATOMIC_NUMS)
        atom_feature_dict.update(dict(zip(ATOMIC_NUM_FEATURE_NAMES, atomic_num_onehot)))

        chiral_feature = CHIRAL_TAGS[atom.GetChiralTag()]
        if use_peptide_stereo:
            if chiral_feature != 0:
                chiral_feature = PEPTIDE_CHIRAL_TAGS[get_amino_acid_stereo(atom_to_residue[atom_idx])]
        atom_feature_dict[CHIRAL_TAG_FEATURE_NAME] = chiral_feature

        atom_feature_dict[AROMATICITY_FEATURE_NAME] = 1 if atom.GetIsAromatic() else 0

        hybridization_onehot = one_k_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPES)
        atom_feature_dict.update(dict(zip(HYBRIDIZATION_TYPE_FEATURE_NAMES, hybridization_onehot)))

        degree_onehot = one_k_encoding(atom.GetTotalDegree(), DEGREES)
        atom_feature_dict.update(dict(zip(DEGREE_FEATURE_NAMES, degree_onehot)))

        valence_onehot = one_k_encoding(atom.GetTotalValence(), VALENCES)
        atom_feature_dict.update(dict(zip(VALENCE_FEATURE_NAMES, valence_onehot)))

        num_hydrogen_onehot = one_k_encoding(atom.GetTotalNumHs(includeNeighbors=True), NUM_HYDROGENS)
        atom_feature_dict.update(dict(zip(NUM_HYDROGEN_FEATURE_NAMES, num_hydrogen_onehot)))

        charge_onehot = one_k_encoding(atom.GetFormalCharge(), FORMAL_CHARGES)
        atom_feature_dict.update(dict(zip(FORMAL_CHARGE_FEATURE_NAMES, charge_onehot)))

        in_ring_sizes = [int(ring_info.IsAtomInRingOfSize(atom_idx, size)) for size in RING_SIZES]
        atom_feature_dict.update(dict(zip(RING_SIZE_FEATURE_NAMES, in_ring_sizes)))

        num_rings_onehot = one_k_encoding(int(ring_info.NumAtomRings(atom_idx)), NUM_RINGS)
        atom_feature_dict.update(dict(zip(NUM_RING_FEATURE_NAMES, num_rings_onehot)))

        if include_side_chain_fingerprint:
            side_chain_idxs = dfs(atom_idx, mol, blocked_idxs=macrocycle_idxs)
            fingerprint = morgan_fingerprint_generator.GetCountFingerprintAsNumPy(mol, fromAtoms=side_chain_idxs)
            fingerprint = np.asarray(fingerprint.astype(np.int64), dtype=int)
            atom_feature_dict.update(dict(zip(fingerprint_feature_names, fingerprint)))

        atom_features[atom_idx] = atom_feature_dict

    atom_features = pd.DataFrame(atom_features).T
    atom_features.index.name = "atom_idx"

    return atom_features

def featurize_macrocycle_atoms_from_file(
    path: Union[str, Path],
    use_peptide_stereo: bool = True,
    residues_in_mol: Optional[List[str]] = None,
    include_side_chain_fingerprint: bool = True,
    radius: int = 3,
    size: int = 2048,
    return_mol: bool = False,
) -> Union[pd.DataFrame, Tuple[Chem.Mol, pd.DataFrame]]:
    """Featurize macrocycle atoms from a pickle file."""
    with open(path, "rb") as f:
        ensemble_data = pickle.load(f)
    mol = ensemble_data["rd_mol"]

    features = featurize_macrocycle_atoms(
        mol,
        use_peptide_stereo=use_peptide_stereo,
        residues_in_mol=residues_in_mol,
        include_side_chain_fingerprint=include_side_chain_fingerprint,
        radius=radius,
        size=size,
    )

    if return_mol:
        return mol, features
    return features

def get_amino_acid_stereo(symbol: str) -> Optional[str]:
    stereo = AMINO_ACID_DATA.loc[symbol]["alpha_carbon_stereo"]
    return stereo if isinstance(stereo, str) else None

def get_residues(
    mol: Chem.Mol,
    residues_in_mol: Optional[List[str]] = None,
    macrocycle_idxs: Optional[List[int]] = None,
) -> Dict[FrozenSet[int], str]:
    """Find the residues in a molecule by matching to a known dataset of amino acids."""
    if macrocycle_idxs is None:
        macrocycle_idxs = get_macrocycle_idxs(mol)
        if macrocycle_idxs is None:
            raise ValueError(f"Couldn't get macrocycle indices for '{Chem.MolToSmiles(Chem.RemoveHs(mol))}'")

    backbone_idxs = mol.GetSubstructMatches(RING_PEPTIDE_BOND_PATTERN)
    if residues_in_mol is None:
        potential_residues = AMINO_ACID_DATA.index
    else:
        potential_residues = residues_in_mol

    potential_residue_idxs = {}
    for residue in set(potential_residues):
        residue_data = AMINO_ACID_DATA.loc[residue]
        residue_matches = mol.GetSubstructMatches(residue_data["residue_mol"], useChirality=True)
        potential_residue_idxs.update({frozenset(match): residue for match in residue_matches})

    residue_idxs = [
        frozenset(
            side_chain_idx
            for atom_idx in atom_idxs
            for side_chain_idx in dfs(
                atom_idx, mol, blocked_idxs=macrocycle_idxs, include_hydrogens=False
            )
        )
        for atom_idxs in backbone_idxs
    ]

    residue_dict = {}
    for atom_idxs in residue_idxs:
        try:
            residue = potential_residue_idxs[atom_idxs]
        except KeyError:
            raise Exception(f"Cannot determine residue for backbone indices '{list(atom_idxs)}' of '{Chem.MolToSmiles(Chem.RemoveHs(mol))}'")
        else:
            residue_dict[atom_idxs] = residue

    return residue_dict

def get_side_chain_torsion_idxs(mol: Chem.Mol) -> Dict[int, List[int]]:
    """Get the indices of atoms in the side chains that we want to calculate internal coordinates for."""
    side_chain_torsion_idxs = {}

    for amino_acid_name, pattern in AMINO_ACID_TORSIONS_PATTERNS.items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            if amino_acid_name in AMINO_ACIDS_WITH_SYMMETRY:
                assert len(matches) % 2 == 0
                matches = matches[::2]

            for match in matches:
                alpha_carbon = match[2]
                assert alpha_carbon not in side_chain_torsion_idxs
                side_chain_torsion_idxs[alpha_carbon] = list(match)

    return side_chain_torsion_idxs

