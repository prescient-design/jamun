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