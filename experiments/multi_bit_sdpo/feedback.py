from __future__ import annotations

import numpy as np


def feedback_confirmation_probability(
    pgen: np.ndarray,
    proposal: int,
    alpha: float,
    beta: float,
) -> float:
    pgen = np.asarray(pgen, dtype=float)
    return float(alpha * pgen[proposal] + beta * (1.0 - pgen[proposal]))


def sample_feedback(
    pgen: np.ndarray,
    proposal: int,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> int:
    confirmation_prob = feedback_confirmation_probability(pgen, proposal, alpha, beta)
    return int(rng.random() < confirmation_prob)


def build_misaligned_distribution(
    p_star: np.ndarray,
    lambda_: float,
    contaminating_arm: int,
) -> np.ndarray:
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")
    p_star = np.asarray(p_star, dtype=float)
    contamination = np.zeros_like(p_star)
    contamination[contaminating_arm] = 1.0
    blended = (1.0 - lambda_) * p_star + lambda_ * contamination
    blended /= blended.sum()
    return blended
