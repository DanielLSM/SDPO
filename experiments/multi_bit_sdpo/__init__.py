"""Standalone theory harness for multi_bit_sdpo experiments.

The current implementation focuses on the exact oracle dynamics from the theory
note. It intentionally stays separate from the main PPO trainer so we can
validate the theory with cheap CPU tests before layering in LLM conditioning.
"""

from .feedback import build_misaligned_distribution, feedback_confirmation_probability, sample_feedback
from .instances import PRESET_INSTANCES, GaussianBanditInstance, get_instance
from .oracle import exact_teacher_distribution, forward_branch_update, reverse_branch_update
from .posterior import ExplorationSummary, estimate_oracle_posterior, sample_exploration_summary
from .proposals import proposal_distribution, sample_proposal

__all__ = [
    "PRESET_INSTANCES",
    "ExplorationSummary",
    "GaussianBanditInstance",
    "build_misaligned_distribution",
    "estimate_oracle_posterior",
    "exact_teacher_distribution",
    "feedback_confirmation_probability",
    "forward_branch_update",
    "get_instance",
    "proposal_distribution",
    "reverse_branch_update",
    "sample_exploration_summary",
    "sample_feedback",
    "sample_proposal",
]
