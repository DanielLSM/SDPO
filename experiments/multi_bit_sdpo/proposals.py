from __future__ import annotations

import numpy as np


def proposal_distribution(
    policy: np.ndarray,
    kind: str,
    epsilon: float = 0.1,
    temperature: float = 1.0,
) -> np.ndarray:
    policy = np.asarray(policy, dtype=float)
    num_arms = policy.shape[0]
    uniform = np.full(num_arms, 1.0 / num_arms, dtype=float)

    if kind == "epsilon_greedy":
        proposal = uniform * epsilon
        max_prob = np.max(policy)
        argmax = np.flatnonzero(np.isclose(policy, max_prob))
        proposal[argmax] += (1.0 - epsilon) / len(argmax)
        return proposal

    if kind == "temperature":
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        powered = np.power(np.clip(policy, 1e-12, None), 1.0 / temperature)
        base = powered / powered.sum()
        return (1.0 - epsilon) * base + epsilon * uniform

    raise ValueError(f"Unknown proposal kind '{kind}'")


def sample_proposal(proposal_probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(proposal_probs), p=np.asarray(proposal_probs, dtype=float)))
