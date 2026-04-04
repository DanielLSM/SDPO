from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.multi_bit_sdpo.instances import get_instance
from experiments.multi_bit_sdpo.prompts import arm_labels
from experiments.multi_bit_sdpo.runner import OracleRunConfig, run_single_seed


def _experiment_grid(
    experiment: str,
    num_seeds: int,
    num_mc: int,
) -> list[tuple[OracleRunConfig, int]]:
    configs: list[tuple[OracleRunConfig, int]] = []

    if experiment == "forward_convergence":
        proposal_settings = [
            {"proposal_kind": "epsilon_greedy", "epsilon": 0.1, "temperature": 1.0},
            {"proposal_kind": "temperature", "epsilon": 0.05, "temperature": 1.0},
        ]
        for seed in range(num_seeds):
            for instance_name in ("Easy-2", "Hard-2", "Medium-5"):
                for proposal in proposal_settings:
                    configs.append(
                        (
                            OracleRunConfig(
                                experiment=experiment,
                                instance_name=instance_name,
                                divergence="forward",
                                pgen_mode="truth",
                                alpha=0.9,
                                beta=0.1,
                                iterations=500,
                                num_mc=num_mc,
                                **proposal,
                            ),
                            seed,
                        )
                    )
        return configs

    if experiment == "forward_self_consistency":
        for seed in range(num_seeds):
            for instance_name in ("Easy-2", "Medium-5"):
                configs.append(
                    (
                        OracleRunConfig(
                            experiment=experiment,
                            instance_name=instance_name,
                            divergence="forward",
                            pgen_mode="self",
                            proposal_kind="epsilon_greedy",
                            epsilon=0.1,
                            alpha=0.9,
                            beta=0.1,
                            iterations=300,
                            num_mc=num_mc,
                        ),
                        seed,
                    )
                )
        return configs

    if experiment == "reverse_dynamics":
        proposal_settings = [
            {"proposal_kind": "epsilon_greedy", "epsilon": 0.05, "temperature": 1.0},
            {"proposal_kind": "temperature", "epsilon": 0.05, "temperature": 0.5},
            {"proposal_kind": "temperature", "epsilon": 0.05, "temperature": 1.0},
            {"proposal_kind": "temperature", "epsilon": 0.05, "temperature": 2.0},
        ]
        for seed in range(num_seeds):
            for instance_name in ("Hard-2", "Hard-5"):
                for proposal in proposal_settings:
                    configs.append(
                        (
                            OracleRunConfig(
                                experiment=experiment,
                                instance_name=instance_name,
                                divergence="reverse",
                                pgen_mode="truth",
                                alpha=0.9,
                                beta=0.1,
                                iterations=500,
                                num_mc=num_mc,
                                **proposal,
                            ),
                            seed,
                        )
                    )
        return configs

    if experiment == "informativeness_sweep":
        for seed in range(num_seeds):
            for alpha, beta in ((1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3)):
                configs.append(
                    (
                        OracleRunConfig(
                            experiment=experiment,
                            instance_name="Easy-2",
                            divergence="forward",
                            pgen_mode="truth",
                            proposal_kind="epsilon_greedy",
                            epsilon=0.1,
                            alpha=alpha,
                            beta=beta,
                            iterations=300,
                            num_mc=num_mc,
                        ),
                        seed,
                    )
                )
        return configs

    if experiment == "misalignment_sweep":
        for seed in range(num_seeds):
            for divergence in ("forward", "reverse"):
                for instance_name in ("Easy-2", "Hard-2"):
                    for misalignment_lambda in (0.0, 0.1, 0.25, 0.5):
                        configs.append(
                            (
                                OracleRunConfig(
                                    experiment=experiment,
                                    instance_name=instance_name,
                                    divergence=divergence,
                                    pgen_mode="misaligned",
                                    proposal_kind="epsilon_greedy",
                                    epsilon=0.1,
                                    alpha=0.9,
                                    beta=0.1,
                                    iterations=300,
                                    num_mc=num_mc,
                                    misalignment_lambda=misalignment_lambda,
                                ),
                                seed,
                            )
                        )
        return configs

    raise ValueError(
        "Unsupported experiment. Expected one of: forward_convergence, "
        "forward_self_consistency, reverse_dynamics, informativeness_sweep, misalignment_sweep."
    )


def _flatten_record(config: OracleRunConfig, seed: int, record: dict[str, Any]) -> dict[str, Any]:
    instance = get_instance(config.instance_name)
    labels = arm_labels(instance.num_arms)
    row: dict[str, Any] = {
        "experiment": config.experiment,
        "seed": seed,
        "instance_name": config.instance_name,
        "num_arms": instance.num_arms,
        "divergence": config.divergence,
        "pgen_mode": config.pgen_mode,
        "proposal_kind": config.proposal_kind,
        "epsilon": config.epsilon,
        "temperature": config.temperature if config.proposal_kind == "temperature" else "",
        "alpha": config.alpha,
        "beta": config.beta,
        "misalignment_lambda": config.misalignment_lambda if config.pgen_mode == "misaligned" else "",
        "contaminating_arm": record.get("contaminating_arm", ""),
        "step": record["step"],
        "proposal": record["proposal"],
        "proposal_label": record.get("proposal_label", labels[int(record["proposal"])]),
        "feedback": record["feedback"],
        "feedback_confirmation_probability": record["feedback_confirmation_probability"],
        "kl_to_p_star_before": record["kl_to_p_star_before"],
        "kl_to_p_star_after": record["kl_to_p_star_after"],
        "observed_kl_decrement": record["observed_kl_decrement"],
        "predicted_forward_decrement": record["predicted_forward_decrement"] if record["predicted_forward_decrement"] is not None else "",
        "l1_to_p_star_after": record["l1_to_p_star_after"],
        "best_arm_probability_after": record["best_arm_probability_after"],
        "best_arm_accuracy_after": record["best_arm_accuracy_after"],
        "step_size_l1": record["step_size_l1"],
        "teacher_student_gap_tv": record["teacher_student_gap_tv"],
        "simplex_boundary_distance_after": record["simplex_boundary_distance_after"],
    }

    policy_after = record["policy_after"]
    q_exact = record["q_exact"]
    p_star = record["p_star"]
    pgen = record["pgen"]
    for idx, label in enumerate(labels):
        row[f"pi_{label}"] = policy_after[idx]
        row[f"q_exact_{label}"] = q_exact[idx]
        row[f"p_star_{label}"] = p_star[idx]
        row[f"pgen_{label}"] = pgen[idx]
    return row


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paper-default oracle sweeps for multi_bit_sdpo.")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "forward_convergence",
            "forward_self_consistency",
            "reverse_dynamics",
            "informativeness_sweep",
            "misalignment_sweep",
        ],
    )
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--num-mc", type=int, default=200_000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multi_bit_sdpo"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    run_configs = _experiment_grid(args.experiment, args.num_seeds, args.num_mc)
    for config, seed in run_configs:
        result = run_single_seed(config, seed=seed)
        rows.extend(_flatten_record(config, seed, record) for record in result.trajectory)

    output_path = args.output_dir / f"{args.experiment}.csv"
    _write_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
