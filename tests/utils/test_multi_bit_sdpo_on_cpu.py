from __future__ import annotations

import numpy as np

from experiments.multi_bit_sdpo.feedback import build_misaligned_distribution
from experiments.multi_bit_sdpo.instances import PRESET_INSTANCES
from experiments.multi_bit_sdpo.oracle import (
    exact_teacher_distribution,
    forward_branch_update,
    reverse_branch_update,
)
from experiments.multi_bit_sdpo.runner import OracleRunConfig, run_single_seed


def test_forward_projection_moves_selected_arm_toward_target_and_preserves_rest_ratio() -> None:
    policy = np.array([0.60, 0.25, 0.15], dtype=float)
    target = np.array([0.20, 0.50, 0.30], dtype=float)
    proposal = 0

    updated = forward_branch_update(policy, target, proposal=proposal, alpha=0.9, beta=0.1)

    assert target[proposal] < updated[proposal] < policy[proposal]

    old_rest_ratio = policy[1] / policy[2]
    new_rest_ratio = updated[1] / updated[2]
    np.testing.assert_allclose(new_rest_ratio, old_rest_ratio, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(updated.sum(), 1.0, rtol=1e-10, atol=1e-10)


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


def test_forward_projection_is_identity_under_self_consistent_feedback() -> None:
    policy = np.array([0.55, 0.30, 0.15], dtype=float)

    for proposal in range(policy.shape[0]):
        updated = forward_branch_update(policy, policy, proposal=proposal, alpha=0.85, beta=0.15)
        np.testing.assert_allclose(updated, policy, rtol=1e-8, atol=1e-8)


def test_exact_teacher_prefers_proposal_after_positive_feedback_and_away_after_negative() -> None:
    policy = np.array([0.40, 0.35, 0.25], dtype=float)

    teacher_pos = exact_teacher_distribution(policy, proposal=0, feedback=1, alpha=0.9, beta=0.1)
    teacher_neg = exact_teacher_distribution(policy, proposal=0, feedback=0, alpha=0.9, beta=0.1)

    assert teacher_pos[0] > policy[0]
    assert teacher_neg[0] < policy[0]
    np.testing.assert_allclose(teacher_pos.sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(teacher_neg.sum(), 1.0, atol=1e-10)


def test_misaligned_distribution_can_target_non_best_arm() -> None:
    p_star = np.array([0.7, 0.2, 0.1], dtype=float)
    blended = build_misaligned_distribution(p_star, lambda_=0.5, contaminating_arm=1)

    assert blended[1] > p_star[1]
    np.testing.assert_allclose(blended.sum(), 1.0, atol=1e-10)


def test_instance_presets_match_theory_plan_counts() -> None:
    assert PRESET_INSTANCES["Easy-2"].pulls_per_arm == 20
    assert PRESET_INSTANCES["Hard-2"].pulls_per_arm == 20
    assert PRESET_INSTANCES["Medium-5"].pulls_per_arm == 15
    assert PRESET_INSTANCES["Hard-5"].pulls_per_arm == 30


def test_forward_self_consistency_rollout_has_zero_drift() -> None:
    config = OracleRunConfig(
        experiment="forward_self_consistency",
        instance_name="Easy-2",
        divergence="forward",
        pgen_mode="self",
        proposal_kind="epsilon_greedy",
        epsilon=0.1,
        alpha=0.9,
        beta=0.1,
        iterations=5,
        num_mc=5_000,
    )

    result = run_single_seed(config, seed=0)

    assert result.trajectory
    assert all(abs(float(record["step_size_l1"])) < 1e-12 for record in result.trajectory)


def test_run_single_seed_logs_expected_diagnostics() -> None:
    config = OracleRunConfig(
        experiment="misalignment_sweep",
        instance_name="Easy-2",
        divergence="forward",
        pgen_mode="misaligned",
        proposal_kind="epsilon_greedy",
        epsilon=0.1,
        alpha=0.9,
        beta=0.1,
        iterations=2,
        num_mc=5_000,
        misalignment_lambda=0.25,
    )

    result = run_single_seed(config, seed=0)

    assert len(result.trajectory) == 2
    record = result.trajectory[0]
    assert "q_exact" in record
    assert "pgen" in record
    assert "proposal_label" in record
    assert "best_arm_label" in record
    assert "contaminating_arm" in record
    assert "observed_kl_decrement" in record
    assert "predicted_forward_decrement" in record
    assert record["predicted_forward_decrement"] is not None
