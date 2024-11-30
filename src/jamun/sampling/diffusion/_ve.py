import math

import torch
from lightning.pytorch.utilities import rank_zero_only
from tqdm.auto import tqdm

# class VESDE(SDE):
#  def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
#    """Construct a Variance Exploding SDE.
#
#    Args:
#      sigma_min: smallest sigma.
#      sigma_max: largest sigma.
#      N: number of discretization steps
#    """
#    super().__init__(N)
#    self.sigma_min = sigma_min
#    self.sigma_max = sigma_max
#    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
#    self.N = N
#
#  @property
#  def T(self):
#    return 1
#
#  def sde(self, x, t):
#    sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
#    drift = torch.zeros_like(x)
#    diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
#                                                device=t.device))
#    return drift, diffusion
#
#  def marginal_prob(self, x, t):
#    std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
#    mean = x
#    return mean, std
#
#  def prior_sampling(self, shape):
#    return torch.randn(*shape) * self.sigma_max
#
#  def prior_logp(self, z):
#    shape = z.shape
#    N = np.prod(shape[1:])
#    return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)
#
#  def discretize(self, x, t):
#    """SMLD(NCSN) discretization."""
#    timestep = (t * (self.N - 1) / self.T).long()
#    sigma = self.discrete_sigmas.to(t.device)[timestep]
#    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
#                                 self.discrete_sigmas.to(t.device)[timestep - 1].to(t.device))
#    f = torch.zeros_like(x)
#    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
#    return f, G
#
# class EulerMaruyamaPredictor:
#    def __init__(self, sde, score_fn, probability_flow=False):
#        pass
#
#    def update_fn(self, x, t):
#        dt = -1.0 / self.rsde.N
#        z = torch.randn_like(x)
#        drift, diffusion = self.rsde.sde(x, t)
#        x_mean = x + drift * dt
#        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
#        return x, x_mean
#
# class ReverseDiffusionPredictor:
#    def __init__(self, sde, score_fn, probability_flow=False):
#        super().__init__(sde, score_fn, probability_flow)
#
#    def update_fn(self, x, t):
#        f, G = self.rsde.discretize(x, t)
#        z = torch.randn_like(x)
#        x_mean = x - f
#        x = x_mean + G[:, None, None, None] * z
#        return x, x_mean

#    def pc_sampler(model):
#        """The PC sampler funciton.
#
#        Args:
#          model: A score model.
#        Returns:
#          Samples, number of function evaluations.
#        """
#        with torch.no_grad():
#            # Initial sample
#            x = sde.prior_sampling(shape).to(device)
#            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
#
#            r = tqdm(range(sde.N), desc="sde", leave=False) if verbose else range(sde.N)
#            for i in r:
#                t = timesteps[i]
#                vec_t = torch.ones(shape[0], device=t.device) * t
#                x, x_mean = corrector_update_fn(x, vec_t, model=model)
#                x, x_mean = predictor_update_fn(x, vec_t, model=model)
#
#            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)
#
#    return pc_sampler


# def get_score_fn(sde, model, train=False, continuous=False):
#  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.
#
#  Args:
#    sde: An `sde_lib.SDE` object that represents the forward SDE.
#    model: A score model.
#    train: `True` for training and `False` for evaluation.
#    continuous: If `True`, the score-based model is expected to directly take continuous time steps.
#
#  Returns:
#    A score function.
#  """
#  model_fn = get_model_fn(model, train=train)
#
#  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
#    def score_fn(x, t):
#      # Scale neural network output by standard deviation and flip sign
#      if continuous or isinstance(sde, sde_lib.subVPSDE):
#        # For VP-trained models, t=0 corresponds to the lowest noise level
#        # The maximum value of time embedding is assumed to 999 for
#        # continuously-trained models.
#        labels = t * 999
#        score = model_fn(x, labels)
#        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
#      else:
#        # For VP-trained models, t=0 corresponds to the lowest noise level
#        labels = t * (sde.N - 1)
#        score = model_fn(x, labels)
#        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]
#
#      score = -score / std[:, None, None, None]
#      return score
#
#  elif isinstance(sde, sde_lib.VESDE):
#    def score_fn(x, t):
#      if continuous:
#        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
#      else:
#        # For VE-trained models, t=0 corresponds to the highest noise level
#        labels = sde.T - t
#        labels *= sde.N - 1
#        labels = torch.round(labels).long()
#
#      score = model_fn(x, labels)
#      return score
#
#  else:
#    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
#
#  return score_fn


class VESDEReverseDiffusionSampler:
    def __init__(self, sample_shape, sigma_min=0.01, sigma_max=50.0, N=1000, eps=1e-5, verbose=False):
        self.sample_shape = sample_shape
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
        self.eps = eps
        self.verbose = verbose
        self.sigmas = torch.linspace(math.log(self.sigma_min), math.log(self.sigma_max), self.N).exp()

    def sample(self, model, batch_size: int = 32):
        # initial sample
        y_i = self.sigma_max * torch.randn(batch_size, *self.sample_shape).to(model.device)  # step N
        y_i_mean = torch.zeros_like(y_i)

        t = torch.linspace(1.0, self.eps, self.N)
        r = range(self.N - 1, -1, -1)  # steps N-1,...,0
        if self.verbose and rank_zero_only.rank == 0:
            r = tqdm(r, desc="sde", leave=False)

        y_traj = []
        y_mean_traj = []
        xhat_traj = []
        for i, ti in zip(r, t):
            # EQ 20 Song and Ermon et. al 2021
            # x_i = x_{i-1} + \sqrt{\sigma^2_i - \sigma^2_{i-1}} z_{i-1}

            sigma_i = self.sigmas[i].item()
            sigma_i_minus_one = self.sigmas[i - 1].item() if i > 0 else 0.0

            #            s = model.score(y_i, sigma_i) # in original implementation this is score(y_i, t) where t goes from [T, eps]
            sigma_ti = self.sigma_min * (self.sigma_max / self.sigma_min) ** ti.item()
            s = model.score(y_i, sigma_ti)

            f_fwd = torch.zeros_like(s)
            G_fwd = math.sqrt(sigma_i**2 - sigma_i_minus_one**2) * torch.ones_like(s)

            # EQ. 46 from Song and Ermon et. al 2021
            f_rev = f_fwd - (G_fwd.pow(2)) * s
            G_rev = G_fwd

            z = torch.randn_like(s)

            xhat_i = y_i + sigma_i**2 * s
            y_i_mean = y_i - f_rev
            y_i = y_i_mean + G_rev * z

            xhat_traj.append(xhat_i.cpu())
            y_traj.append(y_i.cpu())
            y_mean_traj.append(y_i_mean.cpu())

        sample = y_i_mean
        xhat_traj = torch.stack(xhat_traj, dim=0)
        y_traj = torch.stack(y_traj, dim=0)
        y_mean_traj = torch.stack(y_mean_traj, dim=0)

        return {"sample": sample, "xhat_traj": xhat_traj, "y_traj": y_traj, "y_mean_traj": y_mean_traj}
