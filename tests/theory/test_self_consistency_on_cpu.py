from __future__ import annotations

import numpy as np

from experiments.multi_bit_sdpo.oracle import forward_branch_update


def test_forward_projection_is_identity_under_self_consistent_feedback() -> None:
    policy = np.array([0.55, 0.30, 0.15], dtype=float)

    for proposal in range(policy.shape[0]):
        updated = forward_branch_update(policy, policy, proposal=proposal, alpha=0.85, beta=0.15)
        np.testing.assert_allclose(updated, policy, rtol=1e-8, atol=1e-8)
