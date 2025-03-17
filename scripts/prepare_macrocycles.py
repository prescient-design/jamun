from rdkit import Chem
import numpy as np
import argparse
import os

from jamun import utils

def preprocess_sdf(sdf_file, output_file): 
    suppl = Chem.SDMolSupplier(sdf_file)
    mols = [mol for mol in suppl if mol is not None]

    if not mols:
        raise ValueError(f"No valid molecules found in the SDF file: {sdf_file}")

    rdkit_mol_withH = mols[0]
    rdkit_mol = Chem.RemoveHs(rdkit_mol_withH)

    bonds = np.asarray([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in rdkit_mol.GetBonds()], dtype=np.int64).T
    residues = utils.featurize_macrocycles.get_residues(rdkit_mol, residues_in_mol=None, macrocycle_idxs=None)
    for atom_set, residue in residues.items():
        if residue.startswith("Me"):
            residues[atom_set] = residue.replace("Me", "Me+")
    
    residue_sequence = [v for k, v in residues.items()]
    residue_to_sequence_index = {residue: index for index, residue in enumerate(residue_sequence)}

    atom_to_residue = {atom_idx: symbol for atom_idxs, symbol in residues.items() for atom_idx in atom_idxs}
    atom_to_residue = dict(sorted(atom_to_residue.items(), key=lambda x: x[0]))
    atom_to_residue_sequence_index = {atom_idx: residue_to_sequence_index[symbol] for atom_idx, symbol in atom_to_residue.items()}
    atom_to_3_letter = {atom_idx: utils.convert_to_three_letter_code(symbol) for atom_idx, symbol in atom_to_residue.items()}
    atom_to_residue_index = {atom_idx: utils.encode_residue(residue) for atom_idx, residue in atom_to_3_letter.items()}
    atom_types = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]

    residue_sequence_index = np.asarray([atom_to_residue_sequence_index[atom_idx] for atom_idx in range(len(atom_to_residue))], dtype=np.int64)
    residue_code_index = np.asarray([v for v in atom_to_residue_index.values()], dtype=np.int64)
    atom_type_index = np.asarray([utils.encode_atom_type(atom_type) for atom_type in atom_types], dtype=np.int64)

    positions = np.stack([mol.GetConformer().GetPositions() for mol in mols], axis=0)
    positions = positions.astype(np.float32)  # Ensure positions are float32
    positions /= 10.0  # Convert to nanometers

    np.savez(output_file, positions=positions, edge_index=bonds, atom_type_index=atom_type_index, residue_code_index=residue_code_index, residue_sequence_index=residue_sequence_index)
    print(f"Preprocessed {sdf_file} and saved to {output_file}")

    return rdkit_mol_withH
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SDF files for macrocycles.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the SDF files.")
    parser.add_argument("--index", type=int, required=True, help="Index of the SDF file to preprocess.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the preprocessed output.")
    args = parser.parse_args()

    sdf_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.sdf')])
    
    os.makedirs(args.output_dir, exist_ok=True)

    start_index = args.index * 50
    end_index = start_index + 50

    for index in range(start_index, end_index):
        filename = sdf_files[index]
        code = os.path.splitext(filename)[0]
        sdf_file = os.path.join(args.input_dir, filename)

        output_file = os.path.join(args.output_dir, f"{code}.npz")
        mol = preprocess_sdf(sdf_file, output_file)

        # Save the RDKit molecule with hydrogens to a file
        output_mol_file = os.path.join(args.output_dir, f"{code}.sdf")
        writer = Chem.SDWriter(output_mol_file)
        writer.write(mol)
        writer.close()