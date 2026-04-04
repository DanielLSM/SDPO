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
    from experiments.multi_bit_sdpo.posterior import sample_exploration_summary  # type: ignore
    from experiments.multi_bit_sdpo.prompts import (  # type: ignore
        build_base_prompt,
        build_conditioned_prompt,
        validate_single_token_labels,
    )
else:
    from .instances import get_instance
    from .posterior import sample_exploration_summary
    from .prompts import build_base_prompt, build_conditioned_prompt, validate_single_token_labels


def _runtime_imports() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - runtime-only dependency
        raise RuntimeError(
            "gpu_smoke requires runtime dependencies that are not installed in this Python environment. "
            "Please run it in the repo GPU environment with numpy + torch + transformers available."
        ) from exc
    return np, torch, AutoModelForCausalLM, AutoTokenizer


def _resolve_torch_dtype(torch: Any, dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'")
    return mapping[dtype_name]


def _render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _restricted_next_token_distribution(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    label_token_ids: list[int],
    device: str,
    torch: Any,
) -> dict[str, Any]:
    prompt_text = _render_messages(tokenizer, messages)
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        restricted_logits = outputs.logits[:, -1, label_token_ids]
        restricted_probs = torch.softmax(restricted_logits, dim=-1)

    return {
        "prompt_text": prompt_text,
        "restricted_token_ids": list(label_token_ids),
        "restricted_logits": restricted_logits[0].float().cpu().tolist(),
        "restricted_probs": restricted_probs[0].float().cpu().tolist(),
    }


def default_output_path(model_name: str, instance_name: str) -> Path:
    model_slug = model_name.rstrip("/").split("/")[-1].replace(".", "_")
    instance_slug = instance_name.lower().replace("-", "_")
    return Path("outputs") / "multi_bit_sdpo" / "gpu_smoke" / f"{instance_slug}_{model_slug}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a first GPU smoke pass for multi_bit_sdpo label-logit extraction.")
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


def main() -> None:
    args = build_parser().parse_args()
    np, torch, AutoModelForCausalLM, AutoTokenizer = _runtime_imports()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current runtime, so the GPU smoke layer cannot run here.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": _resolve_torch_dtype(torch, args.dtype),
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.to(args.device)
    model.eval()

    instance = get_instance(args.instance_name)
    rng = np.random.default_rng(args.seed)
    summary = sample_exploration_summary(instance, rng)

    label_pairs = validate_single_token_labels(tokenizer, instance.num_arms)
    label_names = [label for label, _ in label_pairs]
    label_token_ids = [token_id for _, token_id in label_pairs]

    base_messages = build_base_prompt(summary)
    base = _restricted_next_token_distribution(model, tokenizer, base_messages, label_token_ids, args.device, torch)

    proposal_arm = args.proposal_arm
    if proposal_arm is None:
        proposal_arm = int(max(range(len(base["restricted_probs"])), key=lambda idx: base["restricted_probs"][idx]))
    if proposal_arm < 0 or proposal_arm >= instance.num_arms:
        raise ValueError(f"proposal_arm must be between 0 and {instance.num_arms - 1}, got {proposal_arm}")

    conditioned_correct = _restricted_next_token_distribution(
        model,
        tokenizer,
        build_conditioned_prompt(summary, proposal=proposal_arm, feedback=1, alpha=args.alpha, beta=args.beta),
        label_token_ids,
        args.device,
        torch,
    )
    conditioned_incorrect = _restricted_next_token_distribution(
        model,
        tokenizer,
        build_conditioned_prompt(summary, proposal=proposal_arm, feedback=0, alpha=args.alpha, beta=args.beta),
        label_token_ids,
        args.device,
        torch,
    )

    result = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "cuda_device_name": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None,
        "instance_name": instance.name,
        "seed": args.seed,
        "summary": summary.to_dict(),
        "label_names": label_names,
        "label_token_ids": label_token_ids,
        "proposal_arm": proposal_arm,
        "proposal_label": label_names[proposal_arm],
        "base": base,
        "conditioned_correct": conditioned_correct,
        "conditioned_incorrect": conditioned_incorrect,
        "proposal_shift_correct": conditioned_correct["restricted_probs"][proposal_arm] - base["restricted_probs"][proposal_arm],
        "proposal_shift_incorrect": conditioned_incorrect["restricted_probs"][proposal_arm] - base["restricted_probs"][proposal_arm],
        "argmax_base": label_names[int(max(range(len(base["restricted_probs"])), key=lambda idx: base["restricted_probs"][idx]))],
        "argmax_conditioned_correct": label_names[
            int(max(range(len(conditioned_correct["restricted_probs"])), key=lambda idx: conditioned_correct["restricted_probs"][idx]))
        ],
        "argmax_conditioned_incorrect": label_names[
            int(max(range(len(conditioned_incorrect["restricted_probs"])), key=lambda idx: conditioned_incorrect["restricted_probs"][idx]))
        ],
    }

    output_path = args.output_path or default_output_path(args.model, args.instance_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nWrote GPU smoke results to {output_path}")


if __name__ == "__main__":
    main()
