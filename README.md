# JAMUN: Transferable Molecular Conformational Ensemble Generation with Walk-Jump Sampling

This is the official implementation of the paper
[JAMUN: Transferable Molecular Conformational Ensemble Generation with Walk-Jump Sampling](https://arxiv.org/abs/2410.14621v1).

![JAMUN results on capped 2AA peptides](figures/jamun-results.png)

Conformational ensembles of protein structures are immensely important both to understanding protein function, and for drug discovery in novel modalities such as cryptic pockets. Current techniques for sampling ensembles are computationally inefficient, or do not transfer to systems outside their training data. We present walk-Jump Accelerated Molecular ensembles with Universal Noise (JAMUN), a step towards the goal of efficiently sampling the Boltzmann distribution of arbitrary proteins. By extending Walk-Jump Sampling to point clouds, JAMUN enables ensemble generation at orders of magnitude faster rates than traditional molecular dynamics or state-of-the-art ML methods. Further, JAMUN is able to predict the stable basins of small peptides that were not seen during training.

<p align="center">
  <img src="https://github.com/prescient-design/jamun/blob/main/figures/walk-jump-overview.png?raw=true" alt="Overview of walk-jump sampling in JAMUN" width="400"/>
</p>

## Setup

Clone the repository (either HTTPS or SSH):
```bash
# HTTPS
git clone https://github.com/prescient-design/jamun.git
# or SSH:
git clone git@github.com:prescient-design/jamun.git
```

We recommend creating either a `mamba` or a `conda` environment:
```bash
mamba create -n jamun python=3.11 -y
mamba activate jamun
```

Then, install all dependencies:
```bash
pip install -r env/linux-cuda/requirements.txt
pip install -e .[dev]
```

## Data

The uncapped 2AA data from [Timewarp](https://arxiv.org/abs/2302.01170) can be obtained from [Hugging Face](https://huggingface.co/datasets/microsoft/timewarp).

Once you have downloaded the data, if this is your directory structure:
```bash
/path/to/data/root/
└── timewarp/
    ├── 2AA-1-big/
    │   └── ...
    ├── 2AA-1-large/
    │   └── ...
```
you have three options for JAMUN to find the data directory:
- Set the environment variable `JAMUN_DATA_PATH` to point to the directory containing `timewarp`:
```bash
export JAMUN_DATA_PATH=/path/to/data/root/
```

- Or, override `paths.data_path` in the command-line:
```bash
jamun_train paths.data_path=/path/to/data/root ...

jamun_sample paths.data_path=/path/to/data/root ...
```

- Or, change `paths.data_path` in the actual [hydra config](https://github.com/prescient-design/jamun/blob/main/src/jamun/hydra_config/paths/default.yaml).


## Training

Once you have imported the data and set the appropriate data variables correctly, 
you can start training on uncapped 2AA peptides from Timewarp:

```bash
jamun_train --config-dir=configs experiment=train_uncapped_2AA.yaml
```

or uncapped 4AA peptides from Timewarp:

```bash
jamun_train --config-dir=configs experiment=train_uncapped_4AA.yaml
```

## Inference

To sample conformations from the test set peptides once you have a trained model,
either specify the `wandb_train_run_path` (obtainable from the Weights and Biases UI for your training run):

<p align="center">
  <img src="https://github.com/prescient-design/jamun/blob/main/figures/wandb-run-path.png?raw=true" alt="Run path as indicated on the Weights and Biases 'Overview' page for your training run" width="200"/>
</p>

and start sampling with:

```bash
jamun_sample --config-dir=configs experiment=sample_uncapped_2AA.yaml wandb_train_run_path=...
```

Alternatively, you can specify the `checkpoint_dir` of the trained model:

```bash
jamun_sample --config-dir=configs experiment=sample_uncapped_2AA.yaml checkpoint_dir=...
```

If you want to sample conformations for a particular protein sequence:
```bash

```

We provide trained weights at ...

## Citation

If this repository was useful to you, please cite our preprint!

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