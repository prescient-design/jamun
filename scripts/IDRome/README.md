## Installation

```bash
conda create --name idrome python=3.10 --yes
conda activate idrome
conda install -c conda-forge matplotlib mdtraj openmm=7.7 ipykernel --yes
conda install pulchra -c bioconda --yes
```

```bash
source .env
sbatch scripts/IDRome/to_all_atom_batched.sh \
    ${JAMUN_DATA_PATH}/IDRome_v4_preprocessed/flat \
    ${JAMUN_DATA_PATH}/IDRome_v4_preprocessed/flat_by_frame/ \
    ${JAMUN_DATA_PATH}/IDRome_v4_preprocessed/all_atom/ \
    1000
```

```bash
source .env
sbatch scripts/IDRome/relax_structures_batched.sh \
    ${JAMUN_DATA_PATH}/IDRome_v4_preprocessed/all_atom \
    ${JAMUN_DATA_PATH}/IDRome_v4_preprocessed/all_atom_relaxed \
    1000
```

```bash
source .env
sbatch scripts/IDRome/combine_frames.sh \
    ${JAMUN_DATA_PATH}/IDRome_v4_preprocessed/all_atom_relaxed/ \
    ${JAMUN_DATA_PATH}/IDRome_v4_preprocessed/flat/ \
    ${JAMUN_DATA_PATH}/IDRome_v4_preprocessed/all_atom_relaxed_combined/
```
