from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
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
        restricted_next_token_logits,
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
    from experiments.multi_bit_sdpo.prompts import (  # type: ignore
        arm_labels,
        build_base_prompt,
        build_single_message_conditioned_prompt,
        validate_single_token_labels,
    )
    from experiments.multi_bit_sdpo.proposals import proposal_distribution, sample_proposal  # type: ignore
else:
    from .feedback import build_misaligned_distribution, feedback_confirmation_probability, sample_feedback
    from .instances import PRESET_INSTANCES, get_instance
    from .llm_backend import load_model_and_tokenizer, restricted_next_token_distribution, restricted_next_token_logits
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
    from .prompts import arm_labels, build_base_prompt, build_single_message_conditioned_prompt, validate_single_token_labels
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
    interleaved_grad_steps: int = 0
    interleaved_lr: float = 5e-6
    interleaved_optimizer: str = "adamw"
    train_prompt_variant: str = "single_message"
    train_objective: str = "forward_kl"
    freeze_embeddings: bool = True
    freeze_norms: bool = True
    experiment: str = "llm_conditioned_rollout"


@dataclass
class SeedResult:
    seed: int
    summary: dict[str, Any]
    trajectory: list[dict[str, Any]]


def _trainable_parameter_names(model: Any, freeze_embeddings: bool, freeze_norms: bool) -> list[str]:
    names: list[str] = []
    for name, parameter in model.named_parameters():
        lower_name = name.lower()
        trainable = True
        if freeze_embeddings and ("embed" in lower_name or "lm_head" in lower_name):
            trainable = False
        if freeze_norms and "norm" in lower_name:
            trainable = False
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    return names


def _build_optimizer(model: Any, config: LLMRolloutConfig, torch: Any) -> Any:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("No trainable parameters were left after applying freeze settings")
    optimizer_name = config.interleaved_optimizer.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=config.interleaved_lr)
    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=config.interleaved_lr)
    raise ValueError(f"Unsupported optimizer '{config.interleaved_optimizer}'")


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


def _build_messages(
    config: LLMRolloutConfig,
    summary: Any,
    proposal: int | None = None,
    feedback: int | None = None,
) -> list[dict[str, str]]:
    if proposal is None or feedback is None:
        return build_base_prompt(summary)
    if config.train_prompt_variant == "single_message":
        return build_single_message_conditioned_prompt(
            summary,
            proposal=proposal,
            feedback=feedback,
            alpha=config.alpha,
            beta=config.beta,
        )
    raise ValueError(f"Unsupported prompt variant '{config.train_prompt_variant}'")


def _evaluate_policy(
    model: Any,
    tokenizer: Any,
    torch: Any,
    np: Any,
    config: LLMRolloutConfig,
    summary: Any,
    label_token_ids: list[int],
    proposal: int | None = None,
    feedback: int | None = None,
) -> tuple[Any, dict[str, Any], list[dict[str, str]]]:
    messages = _build_messages(config, summary, proposal=proposal, feedback=feedback)
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


def _train_on_teacher_distribution(
    model: Any,
    tokenizer: Any,
    torch: Any,
    np: Any,
    optimizer: Any,
    config: LLMRolloutConfig,
    summary: Any,
    label_token_ids: list[int],
    proposal: int,
    feedback: int,
    teacher: Any,
) -> dict[str, Any]:
    messages = _build_messages(config, summary, proposal=proposal, feedback=feedback)
    teacher_tensor = torch.tensor(teacher, device=config.device, dtype=torch.float32).unsqueeze(0)
    teacher_log_probs = teacher_tensor.clamp_min(1e-12).log()
    losses: list[float] = []
    model.train()
    for _ in range(config.interleaved_grad_steps):
        optimizer.zero_grad(set_to_none=True)
        _, restricted_logits = restricted_next_token_logits(
            model,
            tokenizer,
            messages,
            label_token_ids,
            config.device,
        )
        student_log_probs = torch.log_softmax(restricted_logits.float(), dim=-1)
        student_probs = torch.softmax(restricted_logits.float(), dim=-1)
        if config.train_objective == "forward_kl":
            loss = torch.sum(teacher_tensor * (teacher_log_probs - student_log_probs), dim=-1).mean()
        elif config.train_objective == "reverse_kl":
            loss = torch.sum(student_probs * (student_log_probs - teacher_log_probs), dim=-1).mean()
        else:
            raise ValueError(f"Unsupported train objective '{config.train_objective}'")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    post_policy, post_evaluation, _ = _evaluate_policy(
        model,
        tokenizer,
        torch,
        np,
        config,
        summary,
        label_token_ids,
        proposal=proposal,
        feedback=feedback,
    )
    return {
        "losses": losses,
        "post_policy": post_policy,
        "post_evaluation": post_evaluation,
    }


def run_single_seed(
    config: LLMRolloutConfig,
    seed: int,
    tokenizer: Any,
    model: Any,
    np: Any,
    torch: Any,
    optimizer: Any = None,
    trainable_parameter_names: list[str] | None = None,
) -> SeedResult:
    rng = np.random.default_rng(seed)
    instance = get_instance(config.instance_name)
    exploration = sample_exploration_summary(instance, rng)
    p_star = estimate_oracle_posterior(exploration, rng=rng, num_mc=config.num_mc)

    label_pairs = validate_single_token_labels(tokenizer, instance.num_arms)
    labels = [label for label, _ in label_pairs]
    label_token_ids = [token_id for _, token_id in label_pairs]

    policy, evaluation, messages = _evaluate_policy(model, tokenizer, torch, np, config, exploration, label_token_ids)

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

        next_policy, next_evaluation, next_messages = _evaluate_policy(
            model,
            tokenizer,
            torch,
            np,
            config,
            exploration,
            label_token_ids,
            proposal=proposal,
            feedback=feedback,
        )

        train_metrics: dict[str, Any] = {
            "interleaved_loss_trace": [],
            "interleaved_policy_after_train": next_policy.tolist(),
            "interleaved_restricted_logits_after_train": next_evaluation["restricted_logits"],
            "interleaved_teacher_student_gap_tv_after_train": total_variation(teacher, next_policy),
            "interleaved_kl_to_teacher_after_train": kl_divergence(teacher, next_policy),
        }
        effective_next_policy = next_policy
        effective_next_evaluation = next_evaluation
        if config.interleaved_grad_steps > 0:
            if optimizer is None:
                raise ValueError("interleaved_grad_steps > 0 requires an optimizer")
            train_metrics = _train_on_teacher_distribution(
                model,
                tokenizer,
                torch,
                np,
                optimizer,
                config,
                exploration,
                label_token_ids,
                proposal,
                feedback,
                teacher,
            )
            effective_next_policy = np.asarray(train_metrics["post_policy"], dtype=float)
            effective_next_evaluation = train_metrics["post_evaluation"]
            train_metrics = {
                "interleaved_loss_trace": train_metrics["losses"],
                "interleaved_policy_after_train": effective_next_policy.tolist(),
                "interleaved_restricted_logits_after_train": effective_next_evaluation["restricted_logits"],
                "interleaved_teacher_student_gap_tv_after_train": total_variation(teacher, effective_next_policy),
                "interleaved_kl_to_teacher_after_train": kl_divergence(teacher, effective_next_policy),
            }

        kl_before = kl_divergence(p_star, policy)
        kl_after = kl_divergence(p_star, effective_next_policy)

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
            "prompt_num_messages_before": len(messages),
            "prompt_num_messages_after": len(next_messages),
            "policy_before": policy.tolist(),
            "policy_after_prompt_only": next_policy.tolist(),
            "policy_after": effective_next_policy.tolist(),
            "teacher": teacher.tolist(),
            "p_star": p_star.tolist(),
            "pgen": pgen.tolist(),
            "restricted_logits_before": evaluation["restricted_logits"],
            "restricted_logits_after_prompt_only": next_evaluation["restricted_logits"],
            "restricted_logits_after": effective_next_evaluation["restricted_logits"],
            "kl_to_p_star_before": kl_before,
            "kl_to_p_star_after": kl_after,
            "observed_kl_decrement": kl_before - kl_after,
            "l1_to_p_star_after": l1_distance(effective_next_policy, p_star),
            "best_arm_probability_after": float(effective_next_policy[instance.best_arm]),
            "best_arm_accuracy_after": best_arm_accuracy(effective_next_policy, instance.best_arm),
            "step_size_l1": l1_distance(effective_next_policy, policy),
            "teacher_student_gap_tv_before": total_variation(teacher, policy),
            "teacher_student_gap_tv_after_prompt_only": total_variation(teacher, next_policy),
            "teacher_student_gap_tv_after": total_variation(teacher, effective_next_policy),
            "simplex_boundary_distance_after": simplex_boundary_distance(effective_next_policy),
            "proposal_shift": float(effective_next_policy[proposal] - policy[proposal]),
            "argmax_before": labels[int(np.argmax(policy))],
            "argmax_after": labels[int(np.argmax(effective_next_policy))],
            "contaminating_arm": contaminating_arm,
            **train_metrics,
        }
        trajectory.append(record)

        policy = effective_next_policy
        evaluation = effective_next_evaluation
        messages = next_messages
        argmax_history.append(int(np.argmax(policy)))
        policy_history.append(policy.copy())

    policy_history_array = np.asarray(policy_history, dtype=float)
    positive_shifts = [float(r["proposal_shift"]) for r in trajectory if int(r["feedback"]) == 1]
    negative_shifts = [float(r["proposal_shift"]) for r in trajectory if int(r["feedback"]) == 0]
    gap_after_values = [float(r["teacher_student_gap_tv_after"]) for r in trajectory]
    teacher_kl_after_values = [float(r["interleaved_kl_to_teacher_after_train"]) for r in trajectory]
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
        "interleaved_grad_steps": config.interleaved_grad_steps,
        "interleaved_lr": config.interleaved_lr,
        "train_prompt_variant": config.train_prompt_variant,
        "train_objective": config.train_objective,
        "trainable_parameter_count": len(trainable_parameter_names or []),
        "trainable_parameter_names": trainable_parameter_names or [],
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
        "mean_teacher_kl_after": float(np.mean(teacher_kl_after_values)) if teacher_kl_after_values else 0.0,
        "max_teacher_kl_after": float(np.max(teacher_kl_after_values)) if teacher_kl_after_values else 0.0,
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
        "mean_teacher_kl_after",
        "max_teacher_kl_after",
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
    mode_slug = f"g{config.interleaved_grad_steps}_{config.train_objective}"
    return Path("outputs") / "multi_bit_sdpo" / "llm_rollout" / f"{instance_slug}_{model_slug}_{mode_slug}"


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
    parser.add_argument("--interleaved-grad-steps", type=int, default=0)
    parser.add_argument("--interleaved-lr", type=float, default=5e-6)
    parser.add_argument("--interleaved-optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--train-prompt-variant", default="single_message", choices=["single_message"])
    parser.add_argument("--train-objective", default="forward_kl", choices=["forward_kl", "reverse_kl"])
    parser.add_argument("--no-freeze-embeddings", action="store_true")
    parser.add_argument("--no-freeze-norms", action="store_true")
    parser.add_argument("--compare-modes", action="store_true")
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
        interleaved_grad_steps=args.interleaved_grad_steps,
        interleaved_lr=args.interleaved_lr,
        interleaved_optimizer=args.interleaved_optimizer,
        train_prompt_variant=args.train_prompt_variant,
        train_objective=args.train_objective,
        freeze_embeddings=not args.no_freeze_embeddings,
        freeze_norms=not args.no_freeze_norms,
    )


def main() -> None:
    args = build_parser().parse_args()
    base_config = config_from_args(args)

    configs = [base_config]
    if args.compare_modes:
        configs = [
            replace(base_config, interleaved_grad_steps=0, experiment=f"{base_config.experiment}_frozen"),
            replace(base_config, experiment=f"{base_config.experiment}_interleaved"),
        ]

    compare_summary: dict[str, Any] = {}
    for run_config in configs:
        np, torch, tokenizer, model = load_model_and_tokenizer(
            run_config.model,
            device=run_config.device,
            dtype_name=run_config.dtype,
            trust_remote_code=run_config.trust_remote_code,
            attn_implementation=run_config.attn_implementation,
        )
        trainable_parameter_names: list[str] = []
        optimizer = None
        if run_config.interleaved_grad_steps > 0:
            trainable_parameter_names = _trainable_parameter_names(
                model,
                freeze_embeddings=run_config.freeze_embeddings,
                freeze_norms=run_config.freeze_norms,
            )
            optimizer = _build_optimizer(model, run_config, torch)
        else:
            model.eval()

        results = [
            run_single_seed(
                run_config,
                seed=seed,
                tokenizer=tokenizer,
                model=model,
                np=np,
                torch=torch,
                optimizer=optimizer,
                trainable_parameter_names=trainable_parameter_names,
            )
            for seed in args.seeds
        ]
        aggregated = aggregate_results(results)
        mode_key = f"grad_steps_{run_config.interleaved_grad_steps}"
        compare_summary[mode_key] = aggregated
        print(json.dumps({mode_key: aggregated}, indent=2, default=_json_default))

        output_dir = args.output_dir or default_output_dir(run_config)
        if args.compare_modes and args.output_dir is not None:
            output_dir = args.output_dir / mode_key
        write_results(output_dir, run_config, results)
        print(f"\nWrote LLM-conditioned rollout results to {output_dir}")

    if args.compare_modes:
        print("\nCombined compare summary:")
        print(json.dumps(compare_summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
