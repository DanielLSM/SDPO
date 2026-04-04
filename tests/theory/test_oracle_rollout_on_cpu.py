from __future__ import annotations

from experiments.multi_bit_sdpo.runner import OracleRunConfig, run_single_seed


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
