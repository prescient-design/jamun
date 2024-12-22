# JAMUN: Transferable Molecular Conformational Ensemble Generation with Walk-Jump Sampling

This is the official implementation of the paper
[JAMUN: Transferable Molecular Conformational Ensemble Generation with Walk-Jump Sampling](https://arxiv.org/abs/2410.14621v1).

![JAMUN results on capped 2AA peptides](figures/jamun-results.png)

Conformational ensembles of protein structures are immensely important both to understanding protein function, and for drug discovery in novel modalities such as cryptic pockets. Current techniques for sampling ensembles are computationally inefficient, or do not transfer to systems outside their training data. We present walk-Jump Accelerated Molecular ensembles with Universal Noise (JAMUN), a step towards the goal of efficiently sampling the Boltzmann distribution of arbitrary proteins. By extending Walk-Jump Sampling to point clouds, JAMUN enables ensemble generation at orders of magnitude faster rates than traditional molecular dynamics or state-of-the-art ML methods. Further, JAMUN is able to predict the stable basins of small peptides that were not seen during training.

<p align="center">
  <img src="https://github.com/prescient-design/jamun/blob/main/figures/walk-jump-overview.png?raw=true" alt="Overview of walk-jump sampling in JAMUN" width="400"/>
</p>

## Setup

Clone the repository with HTTPS:
```bash
git clone https://github.com/prescient-design/jamun.git
```
or SSH:
```bash
git clone git@github.com:prescient-design/jamun.git
```

Navigate to the cloned repository:
```bash
cd jamun
```

We recommend using [`uv`](https://docs.astral.sh/uv/getting-started/installation/). Alternatively, you can install
[`mamba`](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) or [`conda`](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install).

### Setup with `uv`

Create a virtual enviroment, and install all dependencies:
```bash
uv sync
source .venv/bin/activate
```

### Setup with `mamba` or `conda`

Create an environment:
```bash
mamba create -n jamun-env python=3.11 -y
mamba activate jamun-env
```

Then, install all dependencies:
```bash
pip install -r env/requirements.txt
pip install -e .[dev]
```

## Data

The uncapped 2AA data from [Timewarp](https://arxiv.org/abs/2302.01170) can be obtained from [Hugging Face](https://huggingface.co/datasets/microsoft/timewarp).
```bash
cd /path/to/data/root/
git lfs install
git clone https://huggingface.co/datasets/microsoft/timewarp
```
where `/path/to/data/root/` is the path where you want to store the datasets.

This should be your directory structure:
```bash
/path/to/data/root/
└── timewarp/
    ├── 2AA-1-big/
    │   └── ...
    ├── 2AA-1-large/
    │   └── ...
```
Now, set the environment variable `JAMUN_DATA_PATH`:
```bash
export JAMUN_DATA_PATH=/path/to/data/root/
```
or, create a `.env` file and set `JAMUN_DATA_PATH`:
```txt
JAMUN_DATA_PATH=/path/to/data/root/
```

## Training

Once you have downloaded the data and set the appropriate data variables correctly, 
you can start training on Timewarp.

We recommend first running our test config to check that installation was successful:
```bash
jamun_train --config-dir=configs experiment=train_test.yaml
```

Then, you can train on the uncapped 2AA peptides dataset:
```bash
jamun_train --config-dir=configs experiment=train_uncapped_2AA.yaml
```

or the uncapped 4AA peptides dataset:
```bash
jamun_train --config-dir=configs experiment=train_uncapped_4AA.yaml
```

By default, all run outputs will be saved to `outputs` in the current directory.

We also provide example [SLURM](https://slurm.schedmd.com/documentation.html) launcher scripts for [training](https://github.com/prescient-design/jamun/blob/main/scripts/slurm/train.sh) and [sampling](https://github.com/prescient-design/jamun/blob/main/scripts/slurm/sample.sh) on SLURM clusters:
```bash
sbatch scripts/slurm/train.sh 
sbatch scripts/slurm/sample.sh 
```

## Inference

We provide trained weights for inference at ...

If you want to test out your own trained model, 
either specify the `wandb_train_run_path` (in the form `entity/project/run_id`, which can be obtained from the Overview tab in the Weights and Biases UI for your training run), or the `checkpoint_dir` of the trained model.

```bash
jamun_sample ... ++wandb_train_run_path=[WANDB_TRAIN_RUN_PATH]
jamun_sample ... ++checkpoint_dir=[CHECKPOINT_DIR]
```

### Sampling a Peptide Sequence

If you want to sample conformations for a particular peptide sequence, you need to first generate a `.pdb` file.

We provide a script that uses [AmberTools](https://ambermd.org/AmberTools.php), specifically `tleap`. If you have a `.pdb` file already, then you can skip this step.

#### Generate `.pdb` file
First, install AmberTools23 following [instructions here](https://ambermd.org/GetAmber.php#ambertools):
```bash
conda create --name AmberTools23
conda activate AmberTools23
conda install -c conda-forge ambertools=23 -y
```
Then, run:
```bash
python scripts/prepare_pdb.py [SEQUENCE] --mode [MODE] --outputdir [OUTPUTDIR]
```
where `SEQUENCE` is your peptide sequence entered as a string of one-letter codes (eg. AGPF) or a string of hyphenated three letter codes (eg. ALA-GLY-PRO-PHE), `MODE` is either `capped` or `uncapped` to add capping ACE and NME residues, and `OUTPUTDIR` is where your generated `.pdb` file will be saved (default is current directory).
The script will print out the path to the generated  `.pdb` file, `INIT_PDB`.

Deactivate the `AmberTools23` conda environment:
```bash
conda deactivate
```

#### Run sampling on `.pdb`
Run the sampling script, starting from the provided `.pdb` structure:
```bash
jamun_sample --config-dir=configs experiment=sample_custom ++init_pdb=[INIT_PDB]
```

### Sampling Test Peptides from Timewarp

We also provide some configs to sample from the uncapped 2AA and 4AA peptides from the test set in Timewarp.

```bash
jamun_sample --config-dir=configs experiment=sample_uncapped_2AA.yaml checkpoint_dir=...

jamun_sample --config-dir=configs experiment=sample_uncapped_4AA.yaml checkpoint_dir=...
```

## Analysis

Our sampling scripts produce visualizations and some simple analysis in the Weights and Biases UI.

For more in-depth exploration, we provide an analysis notebook, adapted from that of [MDGen](https://github.com/bjing2016/mdgen).
This notebook requires the run path from Weights and Biases of your sampling run.

The analysis notebook has certain dependencies (eg. `pyemma`) which are slightly tricky to install.
We create a new environment to avoid dependency clashes.
Again, we recommend `uv`, but we also provide instructions for `mamba` or `conda`.

### Setup with `uv`

```bash
uv venv .venv-analysis --python 3.11
source .venv-analysis/bin/activate

uv run --no-project -m pip install wheel pyemma
uv pip install -r env/requirements-analysis.txt
```

### Setup with `mamba` or `conda`

```bash
mamba create -n jamun-analysis python=3.11 -y
mamba activate jamun-analysis

mamba install pyemma ipykernel -y
pip install -r env/requirements-analysis.txt
```

## Citation

If you found this repository useful, please cite our preprint!

```bibtex
@misc{daigavane2024jamuntransferablemolecularconformational,
      title={JAMUN: Transferable Molecular Conformational Ensemble Generation with Walk-Jump Sampling}, 
      author={Ameya Daigavane and Bodhi P. Vani and Saeed Saremi and Joseph Kleinhenz and Joshua Rackers},
      year={2024},
      eprint={2410.14621},
      archivePrefix={arXiv},
      primaryClass={physics.bio-ph},
      url={https://arxiv.org/abs/2410.14621}, 
}
```
