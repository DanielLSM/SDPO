from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .feedback import build_misaligned_distribution, feedback_confirmation_probability, sample_feedback
from .instances import PRESET_INSTANCES, get_instance
from .metrics import (
    best_arm_accuracy,
    kl_divergence,
    l1_distance,
    oscillation_amplitude,
    simplex_boundary_distance,
    top_arm_switch_count,
    total_variation,
)
from .oracle import bernoulli_kl, exact_teacher_distribution, forward_branch_update, reverse_branch_update
from .posterior import estimate_oracle_posterior, sample_exploration_summary
from .prompts import arm_labels
from .proposals import proposal_distribution, sample_proposal


EXPERIMENT_PRESETS: dict[str, dict[str, Any]] = {
    "forward_convergence": {"divergence": "forward", "pgen_mode": "truth"},
    "forward_self_consistency": {"divergence": "forward", "pgen_mode": "self"},
    "reverse_dynamics": {"divergence": "reverse", "pgen_mode": "truth"},
    "informativeness_sweep": {"divergence": "forward", "pgen_mode": "truth"},
    "misalignment_sweep": {"pgen_mode": "misaligned"},
}


@dataclass
class OracleRunConfig:
    instance_name: str
    divergence: str = "forward"
    proposal_kind: str = "epsilon_greedy"
    epsilon: float = 0.1
    temperature: float = 1.0
    alpha: float = 1.0
    beta: float = 0.0
    iterations: int = 300
    num_mc: int = 200_000
    pgen_mode: str = "truth"
    misalignment_lambda: float = 0.0
    contaminating_arm: int | None = None
    initial_policy: str = "uniform"
    experiment: str = "custom"


@dataclass
class SeedResult:
    seed: int
    summary: dict[str, Any]
    trajectory: list[dict[str, Any]]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _initial_policy(mode: str, p_star: np.ndarray, posterior_means: tuple[float, ...]) -> np.ndarray:
    if mode == "uniform":
        return np.full_like(p_star, 1.0 / len(p_star), dtype=float)
    if mode == "p_star":
        return p_star.copy()
    if mode == "posterior_mean_softmax":
        means = np.asarray(posterior_means, dtype=float)
        centered = means - means.max()
        weights = np.exp(centered)
        return weights / weights.sum()
    raise ValueError(f"Unknown initial policy mode '{mode}'")


def _resolve_pgen(
    config: OracleRunConfig,
    p_star: np.ndarray,
    policy: np.ndarray,
    instance_best_arm: int,
) -> tuple[np.ndarray, int | None]:
    if config.pgen_mode == "truth":
        return p_star.copy(), None
    if config.pgen_mode == "self":
        return policy.copy(), None
    if config.pgen_mode == "misaligned":
        contaminating_arm = config.contaminating_arm
        if contaminating_arm is None:
            descending = np.argsort(-p_star)
            contamination_candidates = [idx for idx in descending if idx != instance_best_arm]
            contaminating_arm = int(contamination_candidates[0]) if contamination_candidates else instance_best_arm
        return build_misaligned_distribution(p_star, config.misalignment_lambda, contaminating_arm), contaminating_arm
    raise ValueError(f"Unknown pgen_mode '{config.pgen_mode}'")


def run_single_seed(config: OracleRunConfig, seed: int) -> SeedResult:
    rng = np.random.default_rng(seed)
    instance = get_instance(config.instance_name)
    exploration = sample_exploration_summary(instance, rng)
    p_star = estimate_oracle_posterior(exploration, rng=rng, num_mc=config.num_mc)
    policy = _initial_policy(config.initial_policy, p_star, exploration.posterior_means)

    trajectory: list[dict[str, Any]] = []
    argmax_history = [int(np.argmax(policy))]
    policy_history = [policy.copy()]
    labels = arm_labels(instance.num_arms)

    for step in range(1, config.iterations + 1):
        proposal_probs = proposal_distribution(
            policy,
            kind=config.proposal_kind,
            epsilon=config.epsilon,
            temperature=config.temperature,
        )
        proposal = sample_proposal(proposal_probs, rng)
        pgen, contaminating_arm = _resolve_pgen(config, p_star, policy, instance.best_arm)
        feedback = sample_feedback(pgen, proposal, config.alpha, config.beta, rng)
        teacher = exact_teacher_distribution(policy, proposal, feedback, config.alpha, config.beta)

        kl_before = kl_divergence(p_star, policy)
        predicted_forward_decrement = None
        if config.divergence == "forward":
            next_policy = forward_branch_update(policy, pgen, proposal, config.alpha, config.beta)
            predicted_forward_decrement = bernoulli_kl(float(pgen[proposal]), float(policy[proposal])) - bernoulli_kl(
                float(pgen[proposal]), float(next_policy[proposal])
            )
        elif config.divergence == "reverse":
            next_policy = reverse_branch_update(policy, pgen, proposal, config.alpha, config.beta)
        else:
            raise ValueError(f"Unknown divergence '{config.divergence}'")
        kl_after = kl_divergence(p_star, next_policy)

        record = {
            "step": step,
            "instance_name": config.instance_name,
            "instance_best_arm": instance.best_arm,
            "instance_best_arm_label": labels[instance.best_arm],
            "proposal": proposal,
            "proposal_label": labels[proposal],
            "feedback": feedback,
            "proposal_probs": proposal_probs.tolist(),
            "policy_before": policy.tolist(),
            "policy_after": next_policy.tolist(),
            "p_star": p_star.tolist(),
            "pgen": pgen.tolist(),
            "teacher": teacher.tolist(),
            "q_exact": teacher.tolist(),
            "contaminating_arm": contaminating_arm,
            "feedback_confirmation_probability": feedback_confirmation_probability(
                pgen, proposal, config.alpha, config.beta
            ),
            "kl_to_p_star_before": kl_before,
            "kl_to_p_star_after": kl_after,
            "observed_kl_decrement": kl_before - kl_after,
            "predicted_forward_decrement": predicted_forward_decrement,
            "l1_to_p_star_after": l1_distance(next_policy, p_star),
            "best_arm_probability_after": float(next_policy[instance.best_arm]),
            "best_arm_accuracy_after": best_arm_accuracy(next_policy, instance.best_arm),
            "step_size_l1": l1_distance(next_policy, policy),
            "teacher_student_gap_tv": total_variation(teacher, policy),
            "simplex_boundary_distance_after": simplex_boundary_distance(next_policy),
        }
        trajectory.append(record)
        policy = next_policy
        argmax_history.append(int(np.argmax(policy)))
        policy_history.append(policy.copy())

    policy_history_array = np.asarray(policy_history)
    summary = {
        "seed": seed,
        "instance_name": config.instance_name,
        "experiment": config.experiment,
        "divergence": config.divergence,
        "proposal_kind": config.proposal_kind,
        "pgen_mode": config.pgen_mode,
        "alpha": config.alpha,
        "beta": config.beta,
        "iterations": config.iterations,
        "initial_policy": config.initial_policy,
        "exploration": exploration.to_dict(),
        "p_star": p_star.tolist(),
        "final_policy": policy.tolist(),
        "final_kl_to_p_star": kl_divergence(p_star, policy),
        "final_l1_to_p_star": l1_distance(policy, p_star),
        "final_best_arm_probability": float(policy[instance.best_arm]),
        "final_best_arm_accuracy": best_arm_accuracy(policy, instance.best_arm),
        "top_arm_switch_count": top_arm_switch_count(argmax_history),
        "oscillation_amplitude": oscillation_amplitude(policy_history_array),
        "min_simplex_boundary_distance": float(np.min(policy_history_array)),
    }
    return SeedResult(seed=seed, summary=summary, trajectory=trajectory)


def aggregate_results(results: list[SeedResult]) -> dict[str, Any]:
    if not results:
        return {}
    keys = [
        "final_kl_to_p_star",
        "final_l1_to_p_star",
        "final_best_arm_probability",
        "final_best_arm_accuracy",
        "top_arm_switch_count",
        "oscillation_amplitude",
        "min_simplex_boundary_distance",
    ]
    aggregated = {
        "num_seeds": len(results),
        "seed_summaries": [result.summary for result in results],
    }
    for key in keys:
        values = np.asarray([result.summary[key] for result in results], dtype=float)
        aggregated[f"{key}_mean"] = float(values.mean())
        aggregated[f"{key}_std"] = float(values.std(ddof=0))
    return aggregated


def write_results(output_dir: Path, config: OracleRunConfig, results: list[SeedResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, default=_json_default) + "\n")
    aggregated = aggregate_results(results)
    (output_dir / "summary.json").write_text(json.dumps(aggregated, indent=2, default=_json_default) + "\n")
    for result in results:
        trajectory_path = output_dir / f"seed_{result.seed:04d}.jsonl"
        with trajectory_path.open("w") as fh:
            for record in result.trajectory:
                fh.write(json.dumps(record, default=_json_default) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run oracle multi_bit_sdpo theory experiments.")
    parser.add_argument("--experiment", default="custom", choices=["custom", *EXPERIMENT_PRESETS.keys()])
    parser.add_argument("--instance", dest="instance_name", default="Easy-2", choices=sorted(PRESET_INSTANCES))
    parser.add_argument("--divergence", default="forward", choices=["forward", "reverse"])
    parser.add_argument("--proposal-kind", default="epsilon_greedy", choices=["epsilon_greedy", "temperature"])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--num-mc", type=int, default=200_000)
    parser.add_argument("--pgen-mode", default="truth", choices=["truth", "self", "misaligned"])
    parser.add_argument("--misalignment-lambda", type=float, default=0.0)
    parser.add_argument("--contaminating-arm", type=int, default=None)
    parser.add_argument(
        "--initial-policy",
        default="uniform",
        choices=["uniform", "p_star", "posterior_mean_softmax"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> OracleRunConfig:
    config = OracleRunConfig(
        instance_name=args.instance_name,
        divergence=args.divergence,
        proposal_kind=args.proposal_kind,
        epsilon=args.epsilon,
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        iterations=args.iterations,
        num_mc=args.num_mc,
        pgen_mode=args.pgen_mode,
        misalignment_lambda=args.misalignment_lambda,
        contaminating_arm=args.contaminating_arm,
        initial_policy=args.initial_policy,
        experiment=args.experiment,
    )
    if args.experiment != "custom":
        for key, value in EXPERIMENT_PRESETS[args.experiment].items():
            setattr(config, key, value)
    return config


def default_output_dir(config: OracleRunConfig) -> Path:
    instance_slug = config.instance_name.lower().replace("-", "_")
    return Path("outputs") / "multi_bit_sdpo" / f"{config.experiment}_{instance_slug}_{config.divergence}"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    results = [run_single_seed(config, seed=seed) for seed in args.seeds]
    aggregated = aggregate_results(results)
    print(json.dumps(aggregated, indent=2, default=_json_default))

    output_dir = args.output_dir or default_output_dir(config)
    write_results(output_dir, config, results)
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()
