from typing import Optional

import hydra
from omegaconf import DictConfig


def format_resolver(x, pattern):
    return f"{x:{pattern}}"


def instantiate_dict_cfg(cfg: Optional[DictConfig], verbose=False):
    out = []

    if not cfg:
        return out

    if not isinstance(cfg, DictConfig):
        raise TypeError("cfg must be a DictConfig")

    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            if "_target_" in v:
                if verbose:
                    print(f"instantiating <{v._target_}>")
                out.append(hydra.utils.instantiate(v))
            else:
                out.extend(instantiate_dict_cfg(v, verbose=verbose))

    return out
