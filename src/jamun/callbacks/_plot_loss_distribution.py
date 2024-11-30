import einops
import lightning.pytorch as pl
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import torch


class PlotLossDistribution(pl.Callback):
    def __init__(self, sigma_continuous: bool = True):
        self.sigma_continuous = sigma_continuous

    def on_train_epoch_start(self, trainer, pl_module):
        self.losses = []
        self.sigmas = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = torch.cat(self.losses, dim=0)
        sigma = torch.cat(self.sigmas, dim=0)

        if trainer.world_size > 0:
            loss = einops.rearrange(pl_module.all_gather(loss), "r b ... -> (r b) ...")
            sigma = einops.rearrange(pl_module.all_gather(sigma), "r b ... -> (r b) ...")

        if isinstance(trainer.logger, pl.loggers.WandbLogger):
            if self.sigma_continuous:
                fig = ff.create_2d_density(x=sigma.cpu().numpy(), y=loss.cpu().numpy())
                fig.update_xaxes(title_text="sigma")
                fig.update_yaxes(title_text="loss")
            else:
                df = pd.DataFrame({"sigma": sigma.cpu().numpy(), "loss": loss.cpu().numpy()}).sort_values(
                    by="sigma", ascending=False
                )
                fig = px.violin(df, y="loss", color="sigma", points=False, box=True)

            trainer.logger.experiment.log({"losses": fig})

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss_batch"]
        sigma = outputs["sigma_batch"]

        self.losses.append(loss.detach())
        self.sigmas.append(sigma)
