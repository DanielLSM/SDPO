from __future__ import annotations

import math

import numpy as np


_EPS = 1e-12


def _normalize(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, _EPS, None)
    probs = probs / probs.sum()
    return probs


def bernoulli_kl(p: float, q: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    q = min(max(q, _EPS), 1.0 - _EPS)
    return float(p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q)))


def m_pi_y(pi_y: float, alpha: float, beta: float) -> float:
    return float(alpha * pi_y + beta * (1.0 - pi_y))


def exact_teacher_distribution(
    policy: np.ndarray,
    proposal: int,
    feedback: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    policy = _normalize(policy)
    denom = m_pi_y(policy[proposal], alpha, beta)
    if feedback == 1:
        teacher = policy.copy() * beta
        teacher[proposal] = policy[proposal] * alpha
        teacher /= max(denom, _EPS)
        return _normalize(teacher)

    denom = 1.0 - denom
    teacher = policy.copy() * (1.0 - beta)
    teacher[proposal] = policy[proposal] * (1.0 - alpha)
    teacher /= max(denom, _EPS)
    return _normalize(teacher)


def forward_branch_update(
    policy: np.ndarray,
    p: np.ndarray,
    proposal: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    policy = _normalize(policy)
    p = _normalize(p)
    pi_y = float(policy[proposal])
    p_y = float(p[proposal])
    m_y = m_pi_y(pi_y, alpha, beta)
    denom = max(m_y * (1.0 - m_y), _EPS)
    eta = (pi_y * alpha * (1.0 - alpha) + (1.0 - pi_y) * beta * (1.0 - beta)) / denom
    mu_y = p_y + eta * (pi_y - p_y)
    mu_y = float(np.clip(mu_y, _EPS, 1.0 - _EPS))

    updated = policy.copy()
    updated[proposal] = mu_y
    remaining_old_mass = max(1.0 - pi_y, _EPS)
    remaining_new_mass = max(1.0 - mu_y, _EPS)
    if updated.shape[0] > 1:
        mask = np.ones_like(updated, dtype=bool)
        mask[proposal] = False
        updated[mask] = policy[mask] * (remaining_new_mass / remaining_old_mass)
    return _normalize(updated)


def reverse_branch_update(
    policy: np.ndarray,
    p: np.ndarray,
    proposal: int,
    alpha: float,
    beta: float,
) -> np.ndarray:
    policy = _normalize(policy)
    p = _normalize(p)
    coeff = p[proposal] * bernoulli_kl(alpha, beta) - (1.0 - p[proposal]) * bernoulli_kl(beta, alpha)
    updated = policy.copy()
    updated[proposal] *= math.exp(coeff)
    return _normalize(updated)


def forward_expected_update(
    policy: np.ndarray,
    p: np.ndarray,
    proposal_probs: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    policy = _normalize(policy)
    proposal_probs = _normalize(proposal_probs)
    acc = np.zeros_like(policy)
    for arm in range(policy.shape[0]):
        acc += proposal_probs[arm] * forward_branch_update(policy, p, arm, alpha, beta)
    return _normalize(acc)


def reverse_expected_update(
    policy: np.ndarray,
    p: np.ndarray,
    proposal_probs: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    policy = _normalize(policy)
    proposal_probs = _normalize(proposal_probs)
    coeffs = np.array(
        [p[arm] * bernoulli_kl(alpha, beta) - (1.0 - p[arm]) * bernoulli_kl(beta, alpha) for arm in range(policy.shape[0])],
        dtype=float,
    )
    updated = policy * np.exp(proposal_probs * coeffs)
    return _normalize(updated)
