import einops
import lightning.pytorch as pl
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import torch


class PlotGradNormDistribution(pl.Callback):
    def __init__(self, sigma_continuous: bool = True):
        self.sigma_continuous = sigma_continuous

    def setup(self, trainer, pl_module, stage):
        if not self.sigma_continuous and pl_module.m_batch_size > 1:
            raise RuntimeError("PlotGradNormDistribution: {self.sigma_continuous=} but {pl_module.m_batch_size=}")

    def on_train_epoch_start(self, trainer, pl_module):
        self.grad_norms = []
        self.sigmas = []

    def on_train_epoch_end(self, trainer, pl_module):
        grad_norm = torch.stack(self.grad_norms, dim=0)
        sigma = torch.cat(self.sigmas, dim=0)

        if trainer.world_size > 0:
            grad_norm = einops.rearrange(pl_module.all_gather(grad_norm), "r b ... -> (r b) ...")
            sigma = einops.rearrange(pl_module.all_gather(sigma), "r b ... -> (r b) ...")

        if isinstance(trainer.logger, pl.loggers.WandbLogger):
            if self.sigma_continuous:
                fig = ff.create_2d_density(x=sigma.cpu().numpy(), y=grad_norm.cpu().numpy())
                fig.update_xaxes(title_text="sigma")
                fig.update_yaxes(title_text="grad_norm")
            else:
                df = pd.DataFrame(
                    {
                        "sigma": sigma.cpu().numpy(),
                        "grad_norm": grad_norm.cpu().numpy(),
                    }
                ).sort_values(by="sigma", ascending=False)
                fig = px.violin(df, y="grad_norm", color="sigma", points=False, box=True)

            trainer.logger.experiment.log({"grad_norms": fig})

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        sigma = outputs["sigma"]
        sigma_mean = sigma.mean(0, keepdim=True)
        self.sigmas.append(sigma_mean)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        norms = [p.grad.data.norm(p=2) for p in pl_module.parameters() if p.grad is not None]
        total_norm = torch.tensor(norms).norm(p=2)
        self.grad_norms.append(total_norm)
        pl_module.log("grad_2norm_norm_total", total_norm)
