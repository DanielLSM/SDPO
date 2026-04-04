# Multi-Bit SDPO Theory Validation Plan

This document captures a practical implementation plan for the theory-driven experiments described in Alessio Russo's note, while keeping the work isolated from the main PPO training stack.

## Naming

The appendix frames the toy setting as **best-arm identification (BAI)**, which is why that shorthand comes up.

For this repo, the better implementation name is:

- `multi_bit_sdpo`

Why use `multi_bit_sdpo` even though Appendix B starts with one-bit feedback:

- the first target is still the Appendix B one-bit channel
- but we want the harness name to scale to richer discrete feedback variants later
- it avoids baking the first experiment's exact feedback granularity into the package name

In docs and code, we can describe the first milestone as:

- **multi-bit SDPO theory validation**
- starting with the **one-bit feedback bandit** case from Appendix B

## High-Level Recommendation

Do **not** try to force these experiments directly into the main sequence-level PPO trainer.

The current SDPO trainer is built for full-sequence RL with prompts, rollouts, reward extraction, reprompting, and token-level distillation. Appendix B is a much smaller online decision process:

1. build a discrete policy over a small arm set
2. sample one proposal
3. sample feedback from a discrete channel
4. form a conditioned teacher
5. apply a forward- or reverse-KL SDPO update

That is much closer to a standalone simulator / theory-validation harness than to the main training loop.

So the right move is:

- follow the repo's experiment organization and SDPO semantics
- reuse the repo's loss conventions where they help
- implement the theory work as a dedicated top-level experiment harness

## Goal

Implement the Appendix B experiments from the theory PDF in a way that fits this branch's existing SDPO logic while staying faithful to the paper's core setup:

- restricted distribution over arm-label tokens only
- discrete feedback channel, beginning with the one-bit case
- exact oracle update run in parallel with the LLM
- forward and reverse KL SDPO updates

## Scope

The near-term goal is to support all six Appendix B experiments.

### Experiments to target

1. **Forward-KL convergence**
2. **Forward self-consistency**
3. **Reverse-KL dynamics**
4. **Feedback informativeness sweep**
5. **Feedback misalignment sweep**
6. **Conditioning-error diagnostic**

## Repo Fit

This branch already has the core SDPO pieces we want to reuse:

- teacher/student SDPO loss machinery
- forward/reverse/JSD behavior through `alpha`
- feedback-conditioned teacher passes
- optimizer and config conventions
- existing experiment and sweep script patterns

The main mismatch is architectural:

- this repo's current SDPO path is built around free-form sequence rollouts
- the theory note uses a tiny discrete decision process with an exact oracle baseline

That mismatch is why the plan below starts with an exact oracle harness first, then adds the LLM diagnostic layer only after the theory path is trusted.

## What We Reuse vs Build Fresh

Reuse from the repo:

- SDPO loss semantics
- optimizer and logging conventions
- experiment script style
- model-loading conventions where useful

Build fresh for this plan:

- exact Gaussian-bandit oracle
- restricted label-only probability extraction
- discrete feedback-channel simulation
- paper-specific metrics, plots, and sweep runners

Possible optional shared helpers later:

- `verl/utils/multi_bit_sdpo.py`

I would only promote helpers into `verl/` if the experiment code starts duplicating logic that is genuinely useful outside this theory harness.

## Recommended Architecture

Create a dedicated experiment area under:

- `experiments/multi_bit_sdpo/`

Use that area for:

- README and usage notes
- launch scripts
- configs
- plotting utilities
- the LLM-facing runner

Add lightweight validation coverage under:

- `tests/theory/`

The important design choice is separation:

- `experiments/multi_bit_sdpo/` for launches, configs, and experiment-facing code
- `tests/theory/` for fast CPU checks of oracle dynamics and invariants

## Proposed File Layout

- `experiments/multi_bit_sdpo/README.md`
- `experiments/multi_bit_sdpo/instances.py`
- `experiments/multi_bit_sdpo/posterior.py`
- `experiments/multi_bit_sdpo/proposals.py`
- `experiments/multi_bit_sdpo/feedback.py`
- `experiments/multi_bit_sdpo/oracle_updates.py`
- `experiments/multi_bit_sdpo/prompts.py`
- `experiments/multi_bit_sdpo/metrics.py`
- `experiments/multi_bit_sdpo/run_oracle_experiments.py`
- `experiments/multi_bit_sdpo/run_llm_diagnostics.py`
- `experiments/multi_bit_sdpo/plot_results.py`
- `experiments/multi_bit_sdpo/config/`
- `experiments/multi_bit_sdpo/scripts/`
- `tests/theory/test_forward_projection_on_cpu.py`
- `tests/theory/test_self_consistency_on_cpu.py`
- `tests/theory/test_reverse_projection_on_cpu.py`

If the code later needs to support multiple feedback alphabets beyond the one-bit case, the module layout above should already accommodate that without renaming the package again.

## Implementation Phases

### Phase 1: Exact Oracle Harness First

First implement the exact non-LLM version of the process.

This should include:

- problem definitions for `Easy-2`, `Hard-2`, `Medium-5`, `Hard-5`
- frozen exploration dataset generation
- Gaussian posterior computation for each arm
- Monte Carlo estimate of `p*`
- proposal rules:
  - epsilon-greedy
  - temperature-plus-epsilon
- discrete feedback channel abstraction
- one-bit channel implementation with configurable `(alpha, beta)`
- exact oracle update rules for:
  - forward KL
  - reverse KL
- structured per-step logging

### Why start here

This gives us the cleanest validation target with:

- no model-conditioning error
- no optimizer noise
- no prompt-design confounds
- no heavyweight GPU dependency

It also gives a reference trajectory that later LLM experiments can be compared against.

## Phase 2: Cheap, Theory-Critical Oracle Experiments First

Run these first in the exact oracle harness:

1. **Experiment 1: Forward-KL convergence**
2. **Experiment 2: Forward self-consistency**
3. **Experiment 4: Feedback informativeness sweep**
4. **Experiment 5: Feedback misalignment sweep**

### Why this order

These validate the main theory claims fastest and most cleanly.

- Experiment 1 checks convergence under truth-driven feedback.
- Experiment 2 checks the identity / no-drift property under self-consistent feedback.
- Experiment 4 checks dependence on channel quality.
- Experiment 5 checks what happens when the feedback law is biased away from the true posterior.

All of these can run without any LLM plumbing.

## Phase 3: Reverse-KL Stress Tests

After the forward-KL path is stable, add:

5. **Experiment 3: Reverse-KL dynamics**

This is still cheap, but likely more brittle numerically and behaviorally, so it is better to add after the forward-KL path is already trusted.

Expected behaviors to watch for:

- sharper updates than forward KL
- mode-seeking behavior
- oscillations under near-greedy proposals
- boundary-seeking / overconfidence

## Phase 4: LLM-In-The-Loop Diagnostic Harness

Only after the oracle suite is working should we add the model-based version.

This phase implements the Appendix B style loop:

- create a base prompt from frozen bandit summary statistics
- verify arm labels like `A`, `B`, `C`, ... are single tokenizer tokens
- extract restricted probabilities over arm-label tokens
- sample a proposal arm
- sample discrete feedback
- build the conditioned teacher prompt
- compute teacher distribution over arm-label tokens
- apply a student-teacher KL step
- fine-tune lightweight parameters first, for example LoRA

This phase is needed for:

- **Experiment 6: conditioning-error diagnostic**
- optional direct comparison between oracle and learned trajectories

### Important note

This should still be implemented as a separate theory harness, not by heavily mutating the main SDPO PPO path.

The main repo trainer is solving a different systems problem.

## Metrics and Diagnostics

For the oracle harness, log at least:

- `KL(p* || pi_k)`
- `||pi_k - p*||_1`
- probability on the true best arm
- argmax accuracy / decision accuracy
- step size `||pi_(k+1) - pi_k||_1`
- proposal arm
- sampled feedback symbol

For the LLM-linked phase, also log:

- teacher-student KL
- conditioning error `TV(q_llm, q_exact)`
- oracle-vs-model trajectory gap `TV(pi_llm, pi_oracle)`

Additional experiment-specific metrics:

- top-arm switch count
- oscillation amplitude
- distance to simplex boundary
- forward-KL decrement prediction error

## Plotting and Sweep Scripts

Add utilities to produce the plots recommended in Appendix B:

- KL trajectories
- arm-probability trajectories
- oracle-gap trajectories
- conditioning-error trajectories
- informativeness sweep summaries
- misalignment sweep summaries
- reverse-KL oscillation plots

Also add sweep scripts in the same spirit as the existing top-level experiment scripts in this repo.

## Recommended Experiment Order

Best order of implementation:

1. **Experiment 1** — Forward-KL convergence
2. **Experiment 2** — Forward self-consistency
3. **Experiment 4** — Feedback informativeness sweep
4. **Experiment 5** — Feedback misalignment sweep
5. **Experiment 3** — Reverse-KL dynamics
6. **Experiment 6** — Conditioning-error diagnostic

This gets the strongest signal fastest while keeping implementation risk low.

## Practical Recommendation

The best first milestone is:

- build the standalone `multi_bit_sdpo` theory harness
- implement the exact oracle dynamics first
- run Experiments **1, 2, 4, and 5** in the oracle harness
- add reverse-KL after that
- only then add the LLM conditioning diagnostic

That gives us fast, interpretable progress without bending the main training stack into a shape it was not built for, while still preserving a clean path to the full Appendix B comparison later.
