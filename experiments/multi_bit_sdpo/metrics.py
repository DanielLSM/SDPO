from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


_EPS = 1e-12


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, _EPS, None)
    q = np.clip(q, _EPS, None)
    return float(np.sum(p * np.log(p / q)))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(0.5 * np.abs(p - q).sum())


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.abs(p - q).sum())


def simplex_boundary_distance(policy: np.ndarray) -> float:
    return float(np.min(np.asarray(policy, dtype=float)))


def top_arm_switch_count(argmax_history: Iterable[int]) -> int:
    switches = 0
    previous = None
    for arm in argmax_history:
        if previous is not None and arm != previous:
            switches += 1
        previous = arm
    return switches


def oscillation_amplitude(probability_history: np.ndarray) -> float:
    history = np.asarray(probability_history, dtype=float)
    if history.ndim != 2:
        raise ValueError(f"probability_history must be 2D, got shape {history.shape}")
    return float(np.max(history.max(axis=0) - history.min(axis=0)))


def best_arm_accuracy(policy: np.ndarray, best_arm: int) -> float:
    return float(int(np.argmax(np.asarray(policy, dtype=float)) == best_arm))


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return math.nan
    return float(sum(values) / len(values))
