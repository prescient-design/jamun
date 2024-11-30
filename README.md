# JAMUN: Transferable Molecular Conformational Ensemble Generation with Walk-Jump Sampling

This is the official implementation of the paper
[JAMUN: Transferable Molecular Conformational Ensemble Generation with Walk-Jump Sampling](https://arxiv.org/abs/2410.14621v1).


## Setup

```bash
mamba create -n jamun python=3.11 -y
mamba activate jamun
pip install -r env/linux-cuda/requirements.txt
pip install -e .[dev]
```

## Data

## Training

```bash
jamun_train configs/train_2AA.yaml
```

## Inference

```bash
jamun_sample configs/sample_2AA.yaml
```