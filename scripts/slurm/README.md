# Example SLURM Scripts

These are some example [SLURM](https://slurm.schedmd.com/documentation.html) launcher scripts that have worked for us, but might need some modifications for your cluster and SLURM version.

For example, we would launch a training run with:
```bash
sbatch train_uncapped_2AA.sh
```
and then a corresponding sampling run with:
```bash
sbatch sample_uncapped_2AA.sh
```
