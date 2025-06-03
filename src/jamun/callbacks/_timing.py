import time

import lightning.pytorch as pl


class Timing(pl.Callback):
    """
    Callback to measure and log the time taken for each batch and epoch.
    """

    def __init__(self):
        super().__init__()
        self.train_epoch_start_steps = 0
        self.train_batch_start_time = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.train_batch_start_time = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapsed = time.perf_counter() - self.train_batch_start_time
        effective_batch_size = (
            trainer.world_size * trainer.train_dataloader.batch_size * trainer.accumulate_grad_batches
        )
        samples_per_second = effective_batch_size / elapsed

        pl_module.log("train_batch_elapsed", elapsed, on_step=True, on_epoch=True, sync_dist=False, batch_size=1)
        pl_module.log(
            "samples_per_second", samples_per_second, on_step=True, on_epoch=True, sync_dist=False, batch_size=1
        )
