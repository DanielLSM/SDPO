from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianBanditInstance:
    """Gaussian bandit instance used by the theory experiments."""

    name: str
    means: tuple[float, ...]
    pulls_per_arm: int
    reward_variance: float = 1.0
    prior_mean: float = 0.0
    prior_variance: float = 25.0

    @property
    def num_arms(self) -> int:
        return len(self.means)

    @property
    def best_arm(self) -> int:
        return int(np.argmax(np.asarray(self.means, dtype=float)))


PRESET_INSTANCES: dict[str, GaussianBanditInstance] = {
    "Easy-2": GaussianBanditInstance(name="Easy-2", means=(0.8, 0.2), pulls_per_arm=20),
    "Hard-2": GaussianBanditInstance(name="Hard-2", means=(0.52, 0.48), pulls_per_arm=20),
    "Medium-5": GaussianBanditInstance(
        name="Medium-5",
        means=(0.70, 0.50, 0.30, 0.20, 0.10),
        pulls_per_arm=15,
    ),
    "Hard-5": GaussianBanditInstance(
        name="Hard-5",
        means=(0.52, 0.50, 0.48, 0.46, 0.44),
        pulls_per_arm=30,
    ),
}


def get_instance(name: str) -> GaussianBanditInstance:
    if name not in PRESET_INSTANCES:
        valid = ", ".join(sorted(PRESET_INSTANCES))
        raise KeyError(f"Unknown instance '{name}'. Valid options: {valid}")
    return PRESET_INSTANCES[name]
