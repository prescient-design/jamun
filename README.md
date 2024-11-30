# JAMUN: Transferable Molecular Conformational Ensemble Generation with Walk-Jump Sampling

This is the official implementation of the paper
[JAMUN: Transferable Molecular Conformational Ensemble Generation with Walk-Jump Sampling](https://arxiv.org/abs/2410.14621v1).


## Setup

We recommend creating either a `mamba` or a `conda` environment:

```bash
mamba create -n jamun python=3.11 -y
mamba activate jamun
pip install -r env/linux-cuda/requirements.txt
pip install -e .[dev]
```

## Data

## Training

```bash
jamun_train --config-dir=configs experiment=train_2AA.yaml
```

## Inference

```bash
jamun_sample --config-dir=configs experiment=sample_2AA.yaml
```

## Citation

Please cite our preprint, if this repository was useful to you!
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