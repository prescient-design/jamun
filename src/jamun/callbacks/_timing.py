import time

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import rank_zero_only


class Timing(pl.Callback):
    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        if isinstance(trainer.logger, pl.loggers.WandbLogger):
            trainer.logger.experiment.config.update({"world_size": trainer.world_size})

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_start_steps = trainer.global_step
        self.train_epoch_batch_times = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.train_batch_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapsed = time.perf_counter() - self.train_batch_start_time
        self.train_epoch_batch_times.append(elapsed)

    def on_train_epoch_end(self, trainer, pl_module):
        mean_seconds_per_batch = torch.tensor(self.train_epoch_batch_times, device=pl_module.device).mean()
        effective_batch_size = (
            trainer.world_size * trainer.train_dataloader.batch_size * trainer.accumulate_grad_batches
        )
        samples_per_second = effective_batch_size / mean_seconds_per_batch
        batches_per_step = trainer.accumulate_grad_batches

        seconds_per_step = mean_seconds_per_batch * batches_per_step
        steps_per_second = 1 / seconds_per_step

        pl_module.log("samples_per_second", samples_per_second, on_epoch=True, sync_dist=True)
        pl_module.log("steps_per_second", steps_per_second, on_epoch=True, sync_dist=True)
