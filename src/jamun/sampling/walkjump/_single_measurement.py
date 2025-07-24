import torch
from torch import Tensor
from tqdm.auto import tqdm


class SingleMeasurementSampler:
    """Single Measurement Walk-Jump Sampler."""

    def __init__(
        self,
        mcmc,
        sigma: float,
        y_init_distribution: torch.distributions.Distribution | None = None,
    ):
        self.mcmc = mcmc
        self.sigma = float(sigma)
        self.y_init_distribution = y_init_distribution

    def walk(
        self,
        model,
        batch_size: int | None = None,
        y_init: torch.Tensor | None = None,
        v_init: str | Tensor = "gaussian",
    ):
        if y_init is None:
            if self.y_init_distribution is None:
                raise RuntimeError("either y_init and y_init_distribution must be supplied")
            y_init = self.y_init_distribution.sample(sample_shape=(batch_size,)).to(model.device)

        y, v, y_traj, score_traj = self.mcmc(y_init, lambda y: model.score(y, self.sigma), v_init=v_init)

        if y_traj is not None:
            t_traj = torch.ones(y_traj.size(0), device=y_traj.device, dtype=int)
        else:
            t_traj = None

        return {"y": y, "v": v, "y_traj": y_traj, "t_traj": t_traj, "score_traj": score_traj}

    def walk_jump(
        self,
        model,
        batch_size: int | None = None,
        y_init: torch.Tensor | None = None,
        v_init: str | Tensor = "gaussian",
    ):
        out = self.walk(
            model,
            batch_size=batch_size,
            y_init=y_init,
            v_init=v_init,
        )
        y, v, y_traj, t_traj, score_traj = out["y"], out["v"], out["y_traj"], out["t_traj"], out["score_traj"]

        xhat = model.xhat(y, sigma=self.sigma)

        if y_traj is not None:
            xhat_traj = torch.stack(
                [
                    model.xhat(y_traj[i, :].to(model.device), sigma=self.sigma)
                    for i in tqdm(range(y_traj.size(0)), leave=False, desc="Jump")
                ],
                dim=0,
            )
        else:
            xhat_traj = None

        return {
            "xhat": xhat,
            "y": y,
            "v": v,
            "xhat_traj": xhat_traj,
            "y_traj": y_traj,
            "t_traj": t_traj,
            "score_traj": score_traj,
        }

    def sample(
        self,
        model,
        batch_size: int | None = None,
        y_init: torch.Tensor | None = None,
        v_init: str | Tensor = "gaussian",
    ):
        out = self.walk_jump(model, batch_size=batch_size, y_init=y_init, v_init=v_init)
        out["sample"] = out["xhat"]
        return out


class SingleMeasurementSamplerMemory:
    """Single Measurement Walk-Jump Sampler."""

    def __init__(
        self,
        mcmc,
        sigma: float,
        y_init_distribution: Optional[torch.distributions.Distribution] = None
    ):
        self.mcmc = mcmc
        self.sigma = float(sigma)
        self.y_init_distribution = y_init_distribution

    def walk(
        self,
        model,
        batch_size: Optional[int] = None,
        y_init: Optional[torch.Tensor] = None,
        v_init: str | Tensor = "gaussian",
        y_hist_init: Optional[list] = None,
    ):
        if y_init is None:
            if self.y_init_distribution is None:
                raise RuntimeError("either y_init and y_init_distribution must be supplied")
            y_init = self.y_init_distribution.sample(sample_shape=(batch_size,)).to(model.device)
        if y_hist_init is None:
            raise RuntimeError("y_hist_init must be supplied")
        y, v, y_hist,y_traj, score_traj, y_hist_traj = self.mcmc(y_init, y_hist_init, lambda y, y_hist: model.score(y, y_hist, self.sigma), \
                                             v_init=v_init)

        if y_traj is not None:
            t_traj = torch.ones(y_traj.size(0), device=y_traj.device, dtype=int)
        else:
            t_traj = None

        return {"y": y, "v": v, "y_hist": y_hist, "y_traj": y_traj, "t_traj": t_traj, "score_traj": score_traj, "y_hist_traj": y_hist_traj}

    def walk_jump(
        self,
        model,
        batch_size: Optional[int] = None,
        y_init: Optional[torch.Tensor] = None,
        v_init: str | Tensor = "gaussian",
        y_hist_init: Optional[list] = None,
    ):
        out = self.walk(
            model,
            batch_size=batch_size,
            y_init=y_init,
            v_init=v_init,
            y_hist_init=y_hist_init,
        )
        y, v, y_hist, y_traj, t_traj, score_traj, y_hist_traj = out["y"], out["v"], out["y_hist"], out["y_traj"], out["t_traj"], out["score_traj"], out["y_hist_traj"]

        xhat = model.xhat(y, y_hist, sigma=self.sigma)

        if y_traj is not None:
            xhat_traj = torch.stack(
                [
                    model.xhat(y_traj[i, :].to(model.device),  y_hist_traj[i], sigma=self.sigma)
                    for i in tqdm(range(y_traj.size(0)), leave=False, desc="Jump")
                ],
                dim=0,
            )
        else:
            xhat_traj = None

        return {
            "xhat": xhat,
            "y": y,
            "v": v,
            "y_hist": y_hist,
            "xhat_traj": xhat_traj,
            "y_traj": y_traj,
            "y_hist_traj": y_hist_traj,
            "t_traj": t_traj,
            "score_traj": score_traj,
        }

    def sample(
        self,
        model,
        batch_size: Optional[int] = None,
        y_init: Optional[torch.Tensor] = None,
        v_init: str | Tensor = "gaussian",
        y_hist_init: Optional[list] = None,
    ):
        out = self.walk_jump(model, batch_size=batch_size, y_init=y_init, v_init=v_init, y_hist_init=y_hist_init)
        out["sample"] = out["xhat"]
        return out