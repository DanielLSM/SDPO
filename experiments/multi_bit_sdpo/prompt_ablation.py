from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from experiments.multi_bit_sdpo.instances import get_instance  # type: ignore
    from experiments.multi_bit_sdpo.llm_backend import (  # type: ignore
        load_model_and_tokenizer,
        restricted_next_token_distribution,
    )
    from experiments.multi_bit_sdpo.posterior import sample_exploration_summary  # type: ignore
    from experiments.multi_bit_sdpo.prompts import (  # type: ignore
        arm_labels,
        build_base_prompt,
        build_conditioned_prompt,
        validate_single_token_labels,
    )
else:
    from .instances import get_instance
    from .llm_backend import load_model_and_tokenizer, restricted_next_token_distribution
    from .posterior import sample_exploration_summary
    from .prompts import arm_labels, build_base_prompt, build_conditioned_prompt, validate_single_token_labels


def default_output_path(model_name: str, instance_name: str) -> Path:
    model_slug = model_name.rstrip("/").split("/")[-1].replace(".", "_")
    instance_slug = instance_name.lower().replace("-", "_")
    return Path("outputs") / "multi_bit_sdpo" / "prompt_ablation" / f"{instance_slug}_{model_slug}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare prompt variants for the multi_bit_sdpo LLM-conditioned update.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--instance", dest="instance_name", default="Easy-2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proposal-arm", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser


def build_single_message_conditioned_prompt(
    summary: Any,
    proposal: int,
    feedback: int,
    alpha: float,
    beta: float,
) -> list[dict[str, str]]:
    labels = arm_labels(len(summary.counts))
    rows = []
    for label, count, sample_mean, posterior_std in zip(
        labels,
        summary.counts,
        summary.sample_means,
        summary.posterior_stds,
        strict=True,
    ):
        rows.append(
            f"Arm {label}: pulls = {count}, sample mean = {sample_mean:.4f}, posterior std = {posterior_std:.4f}"
        )

    verdict = "CORRECT" if feedback == 1 else "INCORRECT"
    proposal_label = labels[proposal]
    user = (
        "There are arms labeled "
        + ", ".join(labels)
        + ". For each arm, you are given the number of pulls, the sample mean reward, and the posterior standard deviation of the arm mean.\n"
        + "\n".join(rows)
        + f"\nA previous proposal selected arm {proposal_label}. A noisy evaluator then returned the feedback {verdict}. "
        + f"If the proposed arm is truly optimal, the evaluator says CORRECT with probability {alpha:.3f}; "
        + f"if it is not optimal, it still says CORRECT with probability {beta:.3f}. "
        + "Treat this as Bayesian evidence about which arm is most likely to have the highest true mean. "
        + "Which arm is now most likely to be optimal? Answer with a single letter."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an expert statistician. Update your belief over which arm is optimal using the summary statistics "
                "and the noisy feedback observation. Respond with only a single letter."
            ),
        },
        {"role": "user", "content": user},
    ]


def main() -> None:
    args = build_parser().parse_args()
    np, torch, tokenizer, model = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype_name=args.dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )

    instance = get_instance(args.instance_name)
    rng = np.random.default_rng(args.seed)
    summary = sample_exploration_summary(instance, rng)

    label_pairs = validate_single_token_labels(tokenizer, instance.num_arms)
    label_names = [label for label, _ in label_pairs]
    label_token_ids = [token_id for _, token_id in label_pairs]

    base_messages = build_base_prompt(summary)
    base = restricted_next_token_distribution(model, tokenizer, base_messages, label_token_ids, args.device, torch)

    proposal_arm = args.proposal_arm
    if proposal_arm is None:
        proposal_arm = int(max(range(len(base["restricted_probs"])), key=lambda idx: base["restricted_probs"][idx]))
    if proposal_arm < 0 or proposal_arm >= instance.num_arms:
        raise ValueError(f"proposal_arm must be between 0 and {instance.num_arms - 1}, got {proposal_arm}")

    variants: dict[str, Any] = {}
    for variant_name, prompt_builder in {
        "chat_history_correct": lambda: build_conditioned_prompt(summary, proposal=proposal_arm, feedback=1, alpha=args.alpha, beta=args.beta),
        "chat_history_incorrect": lambda: build_conditioned_prompt(summary, proposal=proposal_arm, feedback=0, alpha=args.alpha, beta=args.beta),
        "single_message_correct": lambda: build_single_message_conditioned_prompt(summary, proposal=proposal_arm, feedback=1, alpha=args.alpha, beta=args.beta),
        "single_message_incorrect": lambda: build_single_message_conditioned_prompt(summary, proposal=proposal_arm, feedback=0, alpha=args.alpha, beta=args.beta),
    }.items():
        out = restricted_next_token_distribution(model, tokenizer, prompt_builder(), label_token_ids, args.device, torch)
        probs = out["restricted_probs"]
        variants[variant_name] = {
            **out,
            "proposal_shift": probs[proposal_arm] - base["restricted_probs"][proposal_arm],
            "argmax": label_names[int(max(range(len(probs)), key=lambda idx: probs[idx]))],
        }

    result = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "instance_name": instance.name,
        "seed": args.seed,
        "summary": summary.to_dict(),
        "label_names": label_names,
        "label_token_ids": label_token_ids,
        "proposal_arm": proposal_arm,
        "proposal_label": label_names[proposal_arm],
        "base": {
            **base,
            "argmax": label_names[int(max(range(len(base["restricted_probs"])), key=lambda idx: base["restricted_probs"][idx]))],
        },
        "variants": variants,
    }

    output_path = args.output_path or default_output_path(args.model, args.instance_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nWrote prompt ablation results to {output_path}")


if __name__ == "__main__":
    main()
