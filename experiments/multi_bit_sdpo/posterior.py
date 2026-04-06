from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .instances import GaussianBanditInstance


@dataclass(frozen=True)
class ExplorationSummary:
    counts: tuple[int, ...]
    sample_means: tuple[float, ...]
    posterior_means: tuple[float, ...]
    posterior_stds: tuple[float, ...]

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        data = asdict(self)
        return {key: list(value) for key, value in data.items()}


def sample_exploration_summary(
    instance: GaussianBanditInstance,
    rng: np.random.Generator,
) -> ExplorationSummary:
    counts = np.full(instance.num_arms, instance.pulls_per_arm, dtype=int)
    sample_mean_std = np.sqrt(instance.reward_variance / counts.astype(float))
    sample_means = rng.normal(loc=np.asarray(instance.means, dtype=float), scale=sample_mean_std)
    posterior_means, posterior_stds = gaussian_posterior_from_summary(
        counts=counts,
        sample_means=sample_means,
        prior_mean=instance.prior_mean,
        prior_variance=instance.prior_variance,
        reward_variance=instance.reward_variance,
    )
    return ExplorationSummary(
        counts=tuple(int(x) for x in counts),
        sample_means=tuple(float(x) for x in sample_means),
        posterior_means=tuple(float(x) for x in posterior_means),
        posterior_stds=tuple(float(x) for x in posterior_stds),
    )


def gaussian_posterior_from_summary(
    counts: np.ndarray,
    sample_means: np.ndarray,
    prior_mean: float,
    prior_variance: float,
    reward_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(counts, dtype=float)
    sample_means = np.asarray(sample_means, dtype=float)
    prior_precision = 1.0 / prior_variance
    noise_precision = counts / reward_variance
    posterior_variances = 1.0 / (prior_precision + noise_precision)
    posterior_means = posterior_variances * (
        prior_precision * prior_mean + noise_precision * sample_means
    )
    posterior_stds = np.sqrt(posterior_variances)
    return posterior_means, posterior_stds


def estimate_oracle_posterior(
    summary: ExplorationSummary,
    rng: np.random.Generator,
    num_mc: int = 200_000,
    chunk_size: int = 50_000,
) -> np.ndarray:
    posterior_means = np.asarray(summary.posterior_means, dtype=float)
    posterior_stds = np.asarray(summary.posterior_stds, dtype=float)
    counts = np.zeros_like(posterior_means)
    total = 0
    while total < num_mc:
        batch_size = min(chunk_size, num_mc - total)
        samples = rng.normal(
            loc=posterior_means,
            scale=posterior_stds,
            size=(batch_size, posterior_means.shape[0]),
        )
        winners = np.argmax(samples, axis=1)
        counts += np.bincount(winners, minlength=posterior_means.shape[0])
        total += batch_size
    posterior = counts / total
    posterior /= posterior.sum()
    return posterior
