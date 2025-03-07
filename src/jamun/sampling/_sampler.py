from typing import Any, Iterable, Optional, Union

import lightning
import torch
import torch_geometric
import torch_geometric.data
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from tqdm.auto import tqdm

from jamun import utils


class Sampler:
    """A sampler for molecular dynamics simulations."""

    def __init__(
        self,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[list[int], str, int] = "auto",
        num_nodes: int = 1,
        precision: Union[str, int] = "32-true",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[list[Any], Any]] = None,
        loggers: Optional[Union[Logger, list[Logger]]] = None,
    ):
        self.fabric = lightning.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            num_nodes=num_nodes,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )
        self.global_step = None

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def sample(
        self,
        model,
        batch_sampler,
        num_batches: int,
        init_graphs: torch_geometric.data.Data,
        continue_chain: bool = False,
    ):
        self.fabric.launch()
        self.fabric.setup(model)
        model.eval()

        init_graphs = init_graphs.to(self.fabric.device)
        model_wrapped = utils.ModelSamplingWrapper(
            model=model,
            init_graphs=init_graphs,
            sigma=batch_sampler.sigma,
        )

        y_init = model_wrapped.sample_initial_noisy_positions()
        v_init = "gaussian"

        self.fabric.call("on_sample_start", sampler=self)

        batches = torch.arange(num_batches)
        iterable = self.progbar_wrapper(batches, desc="Sampling", total=len(batches), leave=False)

        with torch.inference_mode():
            for batch_idx in iterable:
                self.global_step = batch_idx

                out = batch_sampler.sample(model=model_wrapped, y_init=y_init, v_init=v_init)
                samples = model_wrapped.unbatch_samples(out)

                # Start next chain from the end state of the previous chain?
                if continue_chain:
                    y_init = out["y"].to(model_wrapped.device)
                    v_init = out["v"].to(model_wrapped.device)
                else:
                    y_init = model_wrapped.sample_initial_noisy_positions()
                    v_init = "gaussian"

                self.fabric.call("on_after_sample_batch", sample=samples, sampler=self)
                self.fabric.log("sampler/global_step", batch_idx)

        self.fabric.call("on_sample_end", sampler=self)
