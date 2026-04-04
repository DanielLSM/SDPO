from __future__ import annotations

import numpy as np

from experiments.multi_bit_sdpo.instances import PRESET_INSTANCES
from experiments.multi_bit_sdpo.oracle import (
    exact_teacher_distribution,
    forward_branch_update,
    reverse_branch_update,
)
from experiments.multi_bit_sdpo.run_oracle_experiments import _write_csv
from experiments.multi_bit_sdpo.runner import OracleRunConfig, run_single_seed


def test_multi_bit_sdpo_instance_presets_match_appendix_b_counts() -> None:
    assert PRESET_INSTANCES["Easy-2"].pulls_per_arm == 20
    assert PRESET_INSTANCES["Hard-2"].pulls_per_arm == 20
    assert PRESET_INSTANCES["Medium-5"].pulls_per_arm == 15
    assert PRESET_INSTANCES["Hard-5"].pulls_per_arm == 30


def test_multi_bit_sdpo_forward_projection_preserves_rest_ratio() -> None:
    policy = np.array([0.60, 0.25, 0.15], dtype=float)
    target = np.array([0.20, 0.50, 0.30], dtype=float)
    proposal = 0

    updated = forward_branch_update(policy, target, proposal=proposal, alpha=0.9, beta=0.1)

    assert target[proposal] < updated[proposal] < policy[proposal]
    np.testing.assert_allclose(updated.sum(), 1.0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(updated[1] / updated[2], policy[1] / policy[2], rtol=1e-8, atol=1e-8)


def test_multi_bit_sdpo_reverse_projection_preserves_non_selected_ratio() -> None:
    policy = np.array([0.50, 0.30, 0.20], dtype=float)
    target = np.array([0.80, 0.10, 0.10], dtype=float)
    proposal = 0

    updated = reverse_branch_update(policy, target, proposal=proposal, alpha=0.9, beta=0.1)

    assert updated[proposal] > policy[proposal]
    np.testing.assert_allclose(updated.sum(), 1.0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(updated[1] / updated[2], policy[1] / policy[2], rtol=1e-8, atol=1e-8)


def test_multi_bit_sdpo_forward_projection_is_identity_under_self_consistency() -> None:
    policy = np.array([0.55, 0.30, 0.15], dtype=float)

    for proposal in range(policy.shape[0]):
        updated = forward_branch_update(policy, policy, proposal=proposal, alpha=0.85, beta=0.15)
        np.testing.assert_allclose(updated, policy, rtol=1e-8, atol=1e-8)


def test_multi_bit_sdpo_teacher_conditioning_moves_toward_and_away_from_proposal() -> None:
    policy = np.array([0.40, 0.35, 0.25], dtype=float)

    positive = exact_teacher_distribution(policy, proposal=0, feedback=1, alpha=0.9, beta=0.1)
    negative = exact_teacher_distribution(policy, proposal=0, feedback=0, alpha=0.9, beta=0.1)

    assert positive[0] > policy[0]
    assert negative[0] < policy[0]
    np.testing.assert_allclose(positive.sum(), 1.0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(negative.sum(), 1.0, rtol=1e-10, atol=1e-10)


def test_multi_bit_sdpo_forward_self_consistency_rollout_has_zero_drift() -> None:
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


def test_multi_bit_sdpo_csv_export_handles_mixed_arm_counts(tmp_path) -> None:
    output_path = tmp_path / "forward_convergence.csv"
    rows = [
        {
            "experiment": "forward_convergence",
            "instance_name": "Easy-2",
            "pi_A": 0.6,
            "pi_B": 0.4,
        },
        {
            "experiment": "forward_convergence",
            "instance_name": "Medium-5",
            "pi_A": 0.4,
            "pi_B": 0.25,
            "pi_C": 0.15,
            "pi_D": 0.10,
            "pi_E": 0.10,
        },
    ]

    _write_csv(rows, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "pi_E" in text
    assert "Medium-5" in text
