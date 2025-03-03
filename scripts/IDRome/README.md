## Installation

```bash
conda create --name idrome python=3.10 --yes
conda activate idrome
conda install -c conda-forge matplotlib mdtraj openmm=7.7 ipykernel --yes
conda install pulchra -c bioconda --yes
```

```bash
python scripts/generate_data/run_simulation.py /homefs/home/daigavaa/jamun/145_181/all_atom/top_AA.pdb --energy-minimization-only --energy-minimization-steps=5000
```

```bash
sbatch to_all_atom_batched.sh /data/bucket/kleinhej/IDRome_v4_preprocessed/flat /data/bucket/kleinhej/IDRome_v4_preprocessed/flat_by_frame/ /data/bucket/kleinhej/IDRome_v4_preprocessed/all_atom/ 1000
```

```bash
sbatch relax_structures_batched.sh /data/bucket/kleinhej/IDRome_v4_preprocessed/all_atom /data/bucket/kleinhej/IDRome_v4_preprocessed/all_atom_relaxed 1000
```