from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from experiments.multi_bit_sdpo.feedback import (  # type: ignore
        build_misaligned_distribution,
        feedback_confirmation_probability,
        sample_feedback,
    )
    from experiments.multi_bit_sdpo.instances import PRESET_INSTANCES, get_instance  # type: ignore
    from experiments.multi_bit_sdpo.llm_backend import (  # type: ignore
        load_model_and_tokenizer,
        restricted_next_token_distribution,
    )
    from experiments.multi_bit_sdpo.metrics import (  # type: ignore
        best_arm_accuracy,
        kl_divergence,
        l1_distance,
        oscillation_amplitude,
        simplex_boundary_distance,
        top_arm_switch_count,
        total_variation,
    )
    from experiments.multi_bit_sdpo.oracle import exact_teacher_distribution  # type: ignore
    from experiments.multi_bit_sdpo.posterior import estimate_oracle_posterior, sample_exploration_summary  # type: ignore
    from experiments.multi_bit_sdpo.prompts import arm_labels, build_history_prompt, validate_single_token_labels  # type: ignore
    from experiments.multi_bit_sdpo.proposals import proposal_distribution, sample_proposal  # type: ignore
else:
    from .feedback import build_misaligned_distribution, feedback_confirmation_probability, sample_feedback
    from .instances import PRESET_INSTANCES, get_instance
    from .llm_backend import load_model_and_tokenizer, restricted_next_token_distribution
    from .metrics import (
        best_arm_accuracy,
        kl_divergence,
        l1_distance,
        oscillation_amplitude,
        simplex_boundary_distance,
        top_arm_switch_count,
        total_variation,
    )
    from .oracle import exact_teacher_distribution
    from .posterior import estimate_oracle_posterior, sample_exploration_summary
    from .prompts import arm_labels, build_history_prompt, validate_single_token_labels
    from .proposals import proposal_distribution, sample_proposal


@dataclass
class LLMRolloutConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    instance_name: str = "Easy-2"
    proposal_kind: str = "epsilon_greedy"
    epsilon: float = 0.1
    temperature: float = 1.0
    alpha: float = 0.9
    beta: float = 0.1
    iterations: int = 8
    num_mc: int = 200_000
    pgen_mode: str = "truth"
    misalignment_lambda: float = 0.0
    contaminating_arm: int | None = None
    device: str = "cuda"
    dtype: str = "auto"
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    experiment: str = "llm_conditioned_rollout"


@dataclass
class SeedResult:
    seed: int
    summary: dict[str, Any]
    trajectory: list[dict[str, Any]]


def _json_default(value: Any) -> Any:
    try:
        import numpy as np  # local import for environments without numpy during import time

        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _resolve_pgen(
    config: LLMRolloutConfig,
    p_star: Any,
    policy: Any,
    instance_best_arm: int,
    np: Any,
) -> tuple[Any, int | None]:
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


def _evaluate_policy(
    model: Any,
    tokenizer: Any,
    torch: Any,
    np: Any,
    config: LLMRolloutConfig,
    summary: Any,
    label_token_ids: list[int],
    history: list[tuple[int, int]],
) -> tuple[Any, dict[str, Any], list[dict[str, str]]]:
    messages = build_history_prompt(summary, history=history, alpha=config.alpha, beta=config.beta)
    evaluation = restricted_next_token_distribution(
        model,
        tokenizer,
        messages,
        label_token_ids,
        config.device,
        torch,
    )
    policy = np.asarray(evaluation["restricted_probs"], dtype=float)
    return policy, evaluation, messages


def run_single_seed(
    config: LLMRolloutConfig,
    seed: int,
    tokenizer: Any,
    model: Any,
    np: Any,
    torch: Any,
) -> SeedResult:
    rng = np.random.default_rng(seed)
    instance = get_instance(config.instance_name)
    exploration = sample_exploration_summary(instance, rng)
    p_star = estimate_oracle_posterior(exploration, rng=rng, num_mc=config.num_mc)

    label_pairs = validate_single_token_labels(tokenizer, instance.num_arms)
    labels = [label for label, _ in label_pairs]
    label_token_ids = [token_id for _, token_id in label_pairs]

    history: list[tuple[int, int]] = []
    policy, evaluation, messages = _evaluate_policy(model, tokenizer, torch, np, config, exploration, label_token_ids, history)

    trajectory: list[dict[str, Any]] = []
    argmax_history = [int(np.argmax(policy))]
    policy_history = [policy.copy()]

    for step in range(1, config.iterations + 1):
        proposal_probs = proposal_distribution(
            policy,
            kind=config.proposal_kind,
            epsilon=config.epsilon,
            temperature=config.temperature,
        )
        proposal = sample_proposal(proposal_probs, rng)
        pgen, contaminating_arm = _resolve_pgen(config, p_star, policy, instance.best_arm, np)
        feedback = sample_feedback(pgen, proposal, config.alpha, config.beta, rng)
        teacher = exact_teacher_distribution(policy, proposal, feedback, config.alpha, config.beta)

        next_history = [*history, (proposal, feedback)]
        next_policy, next_evaluation, next_messages = _evaluate_policy(
            model,
            tokenizer,
            torch,
            np,
            config,
            exploration,
            label_token_ids,
            next_history,
        )

        kl_before = kl_divergence(p_star, policy)
        kl_after = kl_divergence(p_star, next_policy)

        record = {
            "step": step,
            "seed": seed,
            "instance_name": config.instance_name,
            "instance_best_arm": instance.best_arm,
            "instance_best_arm_label": labels[instance.best_arm],
            "proposal": proposal,
            "proposal_label": labels[proposal],
            "proposal_probs": proposal_probs.tolist(),
            "feedback": feedback,
            "feedback_confirmation_probability": feedback_confirmation_probability(pgen, proposal, config.alpha, config.beta),
            "history_length_before": len(history),
            "history_length_after": len(next_history),
            "prompt_num_messages_before": len(messages),
            "prompt_num_messages_after": len(next_messages),
            "policy_before": policy.tolist(),
            "policy_after": next_policy.tolist(),
            "teacher": teacher.tolist(),
            "p_star": p_star.tolist(),
            "pgen": pgen.tolist(),
            "restricted_logits_before": evaluation["restricted_logits"],
            "restricted_logits_after": next_evaluation["restricted_logits"],
            "kl_to_p_star_before": kl_before,
            "kl_to_p_star_after": kl_after,
            "observed_kl_decrement": kl_before - kl_after,
            "l1_to_p_star_after": l1_distance(next_policy, p_star),
            "best_arm_probability_after": float(next_policy[instance.best_arm]),
            "best_arm_accuracy_after": best_arm_accuracy(next_policy, instance.best_arm),
            "step_size_l1": l1_distance(next_policy, policy),
            "teacher_student_gap_tv_before": total_variation(teacher, policy),
            "teacher_student_gap_tv_after": total_variation(teacher, next_policy),
            "simplex_boundary_distance_after": simplex_boundary_distance(next_policy),
            "proposal_shift": float(next_policy[proposal] - policy[proposal]),
            "argmax_before": labels[int(np.argmax(policy))],
            "argmax_after": labels[int(np.argmax(next_policy))],
        }
        trajectory.append(record)

        history = next_history
        policy = next_policy
        evaluation = next_evaluation
        messages = next_messages
        argmax_history.append(int(np.argmax(policy)))
        policy_history.append(policy.copy())

    policy_history_array = np.asarray(policy_history, dtype=float)
    positive_shifts = [float(r["proposal_shift"]) for r in trajectory if int(r["feedback"]) == 1]
    negative_shifts = [float(r["proposal_shift"]) for r in trajectory if int(r["feedback"]) == 0]
    gap_after_values = [float(r["teacher_student_gap_tv_after"]) for r in trajectory]
    summary = {
        "seed": seed,
        "experiment": config.experiment,
        "model": config.model,
        "instance_name": config.instance_name,
        "proposal_kind": config.proposal_kind,
        "pgen_mode": config.pgen_mode,
        "alpha": config.alpha,
        "beta": config.beta,
        "iterations": config.iterations,
        "exploration": exploration.to_dict(),
        "p_star": p_star.tolist(),
        "label_names": labels,
        "label_token_ids": label_token_ids,
        "final_policy": policy.tolist(),
        "final_kl_to_p_star": kl_divergence(p_star, policy),
        "final_l1_to_p_star": l1_distance(policy, p_star),
        "final_best_arm_probability": float(policy[instance.best_arm]),
        "final_best_arm_accuracy": best_arm_accuracy(policy, instance.best_arm),
        "top_arm_switch_count": top_arm_switch_count(argmax_history),
        "oscillation_amplitude": oscillation_amplitude(policy_history_array),
        "min_simplex_boundary_distance": float(np.min(policy_history_array)),
        "mean_teacher_student_gap_tv_after": float(np.mean(gap_after_values)) if gap_after_values else 0.0,
        "max_teacher_student_gap_tv_after": float(np.max(gap_after_values)) if gap_after_values else 0.0,
        "num_positive_feedback": int(sum(int(r["feedback"]) for r in trajectory)),
        "num_negative_feedback": int(sum(1 - int(r["feedback"]) for r in trajectory)),
        "mean_positive_proposal_shift": float(np.mean(positive_shifts)) if positive_shifts else 0.0,
        "mean_negative_proposal_shift": float(np.mean(negative_shifts)) if negative_shifts else 0.0,
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
        "mean_teacher_student_gap_tv_after",
        "max_teacher_student_gap_tv_after",
        "mean_positive_proposal_shift",
        "mean_negative_proposal_shift",
    ]
    aggregated = {
        "num_seeds": len(results),
        "seed_summaries": [result.summary for result in results],
    }
    for key in keys:
        values = [float(result.summary[key]) for result in results]
        mean_value = sum(values) / len(values)
        std_value = (sum((value - mean_value) ** 2 for value in values) / len(values)) ** 0.5
        aggregated[f"{key}_mean"] = float(mean_value)
        aggregated[f"{key}_std"] = float(std_value)
    return aggregated


def write_results(output_dir: Path, config: LLMRolloutConfig, results: list[SeedResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, default=_json_default) + "\n")
    aggregated = aggregate_results(results)
    (output_dir / "summary.json").write_text(json.dumps(aggregated, indent=2, default=_json_default) + "\n")
    for result in results:
        trajectory_path = output_dir / f"seed_{result.seed:04d}.jsonl"
        with trajectory_path.open("w", encoding="utf-8") as handle:
            for record in result.trajectory:
                handle.write(json.dumps(record, default=_json_default) + "\n")


def default_output_dir(config: LLMRolloutConfig) -> Path:
    model_slug = config.model.rstrip("/").split("/")[-1].replace(".", "_")
    instance_slug = config.instance_name.lower().replace("-", "_")
    return Path("outputs") / "multi_bit_sdpo" / "llm_rollout" / f"{instance_slug}_{model_slug}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal multi-step LLM-conditioned multi_bit_sdpo rollout.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--instance", dest="instance_name", default="Easy-2", choices=sorted(PRESET_INSTANCES))
    parser.add_argument("--proposal-kind", default="epsilon_greedy", choices=["epsilon_greedy", "temperature"])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--num-mc", type=int, default=200_000)
    parser.add_argument("--pgen-mode", default="truth", choices=["truth", "self", "misaligned"])
    parser.add_argument("--misalignment-lambda", type=float, default=0.0)
    parser.add_argument("--contaminating-arm", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> LLMRolloutConfig:
    return LLMRolloutConfig(
        model=args.model,
        instance_name=args.instance_name,
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
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    np, torch, tokenizer, model = load_model_and_tokenizer(
        config.model,
        device=config.device,
        dtype_name=config.dtype,
        trust_remote_code=config.trust_remote_code,
        attn_implementation=config.attn_implementation,
    )

    results = [run_single_seed(config, seed=seed, tokenizer=tokenizer, model=model, np=np, torch=torch) for seed in args.seeds]
    aggregated = aggregate_results(results)
    print(json.dumps(aggregated, indent=2, default=_json_default))

    output_dir = args.output_dir or default_output_dir(config)
    write_results(output_dir, config, results)
    print(f"\nWrote LLM-conditioned rollout results to {output_dir}")


if __name__ == "__main__":
    main()
