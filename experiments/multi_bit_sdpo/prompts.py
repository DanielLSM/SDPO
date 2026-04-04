from __future__ import annotations

from typing import Any

from .posterior import ExplorationSummary


def arm_labels(num_arms: int) -> list[str]:
    if num_arms > 26:
        raise ValueError("Only up to 26 arms are supported by the current label helper")
    return [chr(ord("A") + idx) for idx in range(num_arms)]


def validate_single_token_labels(tokenizer: Any, num_arms: int) -> list[tuple[str, int]]:
    labels = arm_labels(num_arms)
    label_ids: list[tuple[str, int]] = []
    for label in labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Label '{label}' is not a single tokenizer token: {token_ids}")
        label_ids.append((label, int(token_ids[0])))
    return label_ids


def build_base_prompt(summary: ExplorationSummary) -> list[dict[str, str]]:
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
    user = (
        "There are arms labeled "
        + ", ".join(labels)
        + ". For each arm, you are given the number of pulls, the sample mean reward, and the posterior standard deviation of the arm mean.\n"
        + "\n".join(rows)
        + "\nThe reward variance is 1. Which arm is most likely to have the highest true mean? Answer with a single letter."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an expert statistician. You will be given summary statistics from a multi-armed bandit "
                "experiment. Identify which arm has the highest true mean reward. Respond with only a single letter."
            ),
        },
        {"role": "user", "content": user},
    ]


def build_conditioned_prompt(
    summary: ExplorationSummary,
    proposal: int,
    feedback: int,
    alpha: float,
    beta: float,
) -> list[dict[str, str]]:
    messages = build_base_prompt(summary)
    label = arm_labels(len(summary.counts))[proposal]
    verdict = "CORRECT" if feedback == 1 else "INCORRECT"
    extra = (
        f"Additional information: arm {label} was checked and the feedback was {verdict}. "
        f"The feedback channel is noisy: if the proposed arm is truly optimal it is marked CORRECT with probability {alpha:.3f}; "
        f"if it is not optimal it is still marked CORRECT with probability {beta:.3f}. "
        "Given this additional information, which arm is now most likely to have the highest true mean? "
        "Answer with a single letter."
    )
    return [messages[0], {"role": "user", "content": messages[1]["content"] + "\n\n" + extra}]
