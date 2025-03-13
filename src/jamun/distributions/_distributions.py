import einops
import torch


class CategoricalValue(torch.distributions.Distribution):
    def __init__(self, values: torch.Tensor, categorical: torch.distributions.Categorical):
        if values.shape[0] != categorical.probs.shape[0]:
            raise RuntimeError(f"{values.shape[0]=} != {categorical.probs.shape[0]=}")
        self.values = values
        self.categorical = categorical

    def sample(self, sample_shape=torch.Size([])):
        return self.values[self.categorical.sample(sample_shape)]

    @property
    def mean(self):
        return einops.einsum(self.values, self.categorical.probs, "i ..., i ... -> ...")

    def __repr__(self):
        return "CategoricalValue"


class WeightedMeasurement(CategoricalValue):
    def __init__(self, sigma: float, probs: torch.Tensor):
        self.sigma = sigma
        self.m = probs.shape[0]
        values = sigma * torch.arange(1, self.m + 1).pow(-0.5)
        super().__init__(values=values, categorical=torch.distributions.Categorical(probs=probs))

    def __repr__(self):
        return f"WeightedMeasurement(sigma={self.sigma}, logits={self.logits})"


class UniformMeasurement(WeightedMeasurement):
    def __init__(self, sigma: float, m: int):
        probs = torch.ones(m)
        super().__init__(sigma=sigma, probs=probs)

    def __repr__(self):
        return f"UniformMeasurement(sigma={self.sigma}, m={self.m})"


class UniformSigma(torch.distributions.Uniform):
    def __init__(self, sigma_max, sigma_min=1e-4):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        super().__init__(low=sigma_min, high=sigma_max)

    def __repr__(self):
        return f"UniformSigma(sigma_max={self.sigma_max}, sigma_min={self.sigma_min})"


class ExponentialSigma(torch.distributions.Distribution):
    def __init__(self, sigma_max=50.0, sigma_min=1e-2, epsilon=1e-5):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.epsilon = epsilon
        self.t_dist = torch.distributions.Uniform(self.epsilon, 1.0)

    def sample(self, sample_shape=torch.Size([])):
        t = self.t_dist.sample(sample_shape)
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return sigma

    def __repr__(self):
        return f"ExponentialSigma(sigma_max={self.sigma_max}, sigma_min={self.sigma_min})"


class UniformPlusNormal(torch.distributions.Distribution):
    def __init__(self, sigma, sample_shape, dtype=None):
        self.sigma = sigma
        self.sample_shape = sample_shape
        self.dtype = dtype

    def sample(self, sample_shape=torch.Size([])):
        low = torch.tensor(0.0, dtype=self.dtype)
        high = torch.tensor(1.0, dtype=self.dtype)
        x = torch.distributions.Uniform(low, high).sample((*sample_shape, *self.sample_shape))
        y = x + torch.randn_like(x) * self.sigma
        return y

    def __repr__(self):
        return f"UniformPlusNormal(sigma={self.sigma}, sample_shape={self.sample_shape})"


class ConstantSigma(torch.distributions.Distribution):
    def __init__(self, sigma: float):
        self.sigma = torch.tensor(sigma)

    def sample(self, sample_shape: torch.Size = torch.Size([])):
        return self.sigma.expand(sample_shape)

    def __repr__(self):
        return f"Constant(sigma={self.sigma})"


class ClippedLogNormalSigma(torch.distributions.Distribution):
    def __init__(self, log_sigma_mean: float, log_sigma_std: float, sigma_max: float = 100.0):
        self.log_sigma_mean = log_sigma_mean
        self.log_sigma_std = log_sigma_std
        self.log_sigma_dist = torch.distributions.Normal(log_sigma_mean, log_sigma_std)
        self.sigma_max = sigma_max

    def sample(self, sample_shape=torch.Size([])):
        log_sigma = self.log_sigma_dist.sample(sample_shape)
        sigma = log_sigma.exp()
        sigma = torch.clamp(sigma, max=self.sigma_max)
        return sigma

    def __repr__(self):
        return f"LogNormalSigma(log_sigma_mean={self.log_sigma_mean}, log_sigma_std={self.log_sigma_std}, sigma_max={self.sigma_max})"
