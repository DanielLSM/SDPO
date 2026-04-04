from __future__ import annotations

import numpy as np

from experiments.multi_bit_sdpo.oracle import reverse_branch_update


def test_reverse_projection_sharpens_selected_arm_and_preserves_other_ratios() -> None:
    policy = np.array([0.50, 0.30, 0.20], dtype=float)
    target = np.array([0.80, 0.10, 0.10], dtype=float)
    proposal = 0

    updated = reverse_branch_update(policy, target, proposal=proposal, alpha=0.9, beta=0.1)

    assert updated[proposal] > policy[proposal]

    old_rest_ratio = policy[1] / policy[2]
    new_rest_ratio = updated[1] / updated[2]
    np.testing.assert_allclose(new_rest_ratio, old_rest_ratio, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(updated.sum(), 1.0, rtol=1e-10, atol=1e-10)
