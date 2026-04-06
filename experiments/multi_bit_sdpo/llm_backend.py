from __future__ import annotations

from typing import Any


def runtime_imports() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - runtime-only dependency
        raise RuntimeError(
            "LLM-side multi_bit_sdpo utilities require runtime dependencies that are not installed in this Python environment. "
            "Please run them in the repo GPU environment with numpy + torch + transformers available."
        ) from exc
    return np, torch, AutoModelForCausalLM, AutoTokenizer


def resolve_torch_dtype(torch: Any, dtype_name: str) -> Any:
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


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype_name: str,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
) -> tuple[Any, Any, Any, Any]:
    np, torch, AutoModelForCausalLM, AutoTokenizer = runtime_imports()

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current runtime, so the LLM-conditioned path cannot run here.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": resolve_torch_dtype(torch, dtype_name),
        "low_cpu_mem_usage": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    return np, torch, tokenizer, model


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def encode_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    device: str,
) -> tuple[str, dict[str, Any]]:
    prompt_text = render_messages(tokenizer, messages)
    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return prompt_text, encoded


def restricted_next_token_logits(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    label_token_ids: list[int],
    device: str,
) -> tuple[str, Any]:
    prompt_text, encoded = encode_messages(tokenizer, messages, device)
    outputs = model(**encoded)
    restricted_logits = outputs.logits[:, -1, label_token_ids]
    return prompt_text, restricted_logits


def restricted_next_token_distribution(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    label_token_ids: list[int],
    device: str,
    torch: Any,
) -> dict[str, Any]:
    prompt_text, encoded = encode_messages(tokenizer, messages, device)

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
