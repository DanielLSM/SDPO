from __future__ import annotations

import numpy as np

from experiments.multi_bit_sdpo.oracle import exact_teacher_distribution


def test_exact_teacher_moves_toward_and_away_from_proposal_under_feedback() -> None:
    policy = np.array([0.40, 0.35, 0.25], dtype=float)

    positive = exact_teacher_distribution(policy, proposal=0, feedback=1, alpha=0.9, beta=0.1)
    negative = exact_teacher_distribution(policy, proposal=0, feedback=0, alpha=0.9, beta=0.1)

    assert positive[0] > policy[0]
    assert negative[0] < policy[0]
    np.testing.assert_allclose(positive.sum(), 1.0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(negative.sum(), 1.0, rtol=1e-10, atol=1e-10)
