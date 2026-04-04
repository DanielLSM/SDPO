# SDPO Theory Experiment Plan

This document captures a practical implementation plan for the theory-driven experiments described in Alessio Russo’s note, without overloading the main PPO training stack.

## Naming

The appendix frames these as **best-arm identification** experiments, which is why the shorthand **BAI** comes up.

For this repo, a clearer name is better. In code and docs, we should refer to these as:

- **SDPO theory experiments**, or
- **one-bit feedback bandit experiments**

That says what they are without forcing readers to decode the acronym.

## High-level recommendation

Do **not** try to force these experiments directly into the main sequence-level PPO trainer.

The current SDPO trainer is built for full-sequence RL with prompts, rollouts, reward extraction, reprompting, and token-level distillation. The theory note’s Appendix B is a much smaller online decision process:

1. build a discrete policy over a small arm set
2. sample one proposal
3. sample one-bit feedback
4. form a conditioned teacher
5. apply a forward- or reverse-KL SDPO update

That is much closer to a standalone simulator / theory-validation harness than to the main training loop.

So the right move is:

- follow the repo’s **experiment organization and SDPO semantics**
- but implement the theory work as a **small standalone top-level experiment module**

## Scope

The goal is to support the experiments in Appendix B, especially the cheap and high-value ones first.

### Experiments to target

1. **Forward-KL convergence**
2. **Forward self-consistency**
3. **Reverse-KL dynamics**
4. **Feedback informativeness sweep**
5. **Feedback misalignment sweep**
6. **Conditioning-error diagnostic**

## Proposed implementation strategy

## Phase 1 — exact oracle harness first

First implement the exact non-LLM version of the process.

This should include:

- Gaussian bandit instance generator
- frozen exploration dataset
- oracle posterior \(p^*\) via Monte Carlo
- proposal kernels:
  - epsilon-greedy
  - temperature-plus-epsilon
- one-bit feedback channel with parameters \((\alpha, \beta)\)
- exact forward-KL projection update
- exact reverse-KL projection update
- structured logging of per-step metrics

### Why start here

This gives us the cleanest validation target with:

- no model-conditioning error
- no optimizer noise
- no prompt-design confounds
- no heavyweight GPU dependency

It also gives a reference trajectory that later LLM experiments can be compared against.

## Phase 2 — implement the cheap, theory-critical experiments first

Run these first:

- **Experiment 1: forward-KL convergence**
- **Experiment 2: forward self-consistency**
- **Experiment 4: feedback informativeness sweep**
- **Experiment 5: feedback misalignment sweep**

### Why this order

These validate the main theory claims fastest and most cleanly.

- Experiment 1 checks that forward-KL converges under truth-driven feedback.
- Experiment 2 checks the identity / no-drift property under self-consistent feedback.
- Experiment 4 checks dependence on feedback quality.
- Experiment 5 checks what happens when the feedback law is biased away from the true posterior.

All of these can be run in the oracle harness without any LLM plumbing.

## Phase 3 — reverse-KL stress tests

After the forward-KL path is stable, add:

- **Experiment 3: reverse-KL dynamics**

This is still cheap, but likely more brittle numerically and behaviorally. It is better to add after the forward-KL path is already trusted.

Expected behaviors to look for:

- sharper updates than forward-KL
- mode-seeking behavior
- oscillations under near-greedy proposals
- boundary-seeking / overconfidence

## Phase 4 — LLM-in-the-loop diagnostic harness

Only after the oracle suite is working should we add the model-based version.

This phase implements the Appendix-B style loop:

- create a base prompt from frozen bandit summary statistics
- extract restricted probabilities over arm-label tokens
- sample a proposal arm
- sample one-bit feedback
- build the conditioned teacher prompt
- compute teacher distribution over arm-label tokens
- apply a student-teacher KL step
- fine-tune only lightweight parameters first (for example LoRA)

This phase is needed for:

- **Experiment 6: conditioning-error diagnostic**
- optional direct comparison between oracle and learned trajectories

### Important note

This should still be implemented as a separate theory harness, not by heavily mutating the main SDPO PPO path.

The main repo trainer is solving a different systems problem.

## Recommended experiment order

Best order of implementation:

1. **Exp 1** — Forward-KL convergence
2. **Exp 2** — Forward self-consistency
3. **Exp 4** — Feedback informativeness sweep
4. **Exp 5** — Feedback misalignment sweep
5. **Exp 3** — Reverse-KL dynamics
6. **Exp 6** — Conditioning-error diagnostic

This gets the strongest signal fastest while keeping implementation risk low.

## Proposed repo layout

A clean layout would look something like this:

```text
SDPO/
├── SDPO_THEORY_EXPERIMENT_PLAN.md
├── experiments/
│   └── theory/
│       ├── run_forward_convergence.sh
│       ├── run_forward_self_consistency.sh
│       ├── run_informativeness_sweep.sh
│       ├── run_misalignment_sweep.sh
│       ├── run_reverse_dynamics.sh
│       └── run_conditioning_diagnostic.sh
├── theory_bandit/
│   ├── __init__.py
│   ├── instances.py
│   ├── posterior.py
│   ├── proposals.py
│   ├── feedback.py
│   ├── oracle_updates.py
│   ├── metrics.py
│   ├── runner.py
│   └── plots.py
└── tests/
    └── theory/
        ├── test_forward_projection_on_cpu.py
        ├── test_self_consistency_on_cpu.py
        └── test_reverse_projection_on_cpu.py
```

The exact names can change, but the important design choice is separation:

- `experiments/theory/` for launch scripts
- `theory_bandit/` for the actual logic
- `tests/theory/` for CPU validation

## Metrics to log

For the oracle harness, log at least:

- `kl_to_p_star`
- `l1_to_p_star`
- probability on the true best arm
- argmax accuracy / decision accuracy
- step size
- proposal arm
- sampled feedback bit

For the LLM-linked phase, also log:

- teacher-student KL
- conditioning error against exact teacher
- oracle-vs-model trajectory gap

## Suggested initial instance set

Use the representative family from the note:

- **Easy-2**
- **Hard-2**
- **Medium-5**
- **Hard-5**

These are enough to expose:

- easy convergence
- slow convergence
- multi-arm behavior
- reverse-KL brittleness

## What is realistically implementable

### Definitely implementable now

- Experiments **1–5** in the exact oracle harness

### Implementable next, with more engineering

- Experiment **6**
- optional model-based Appendix-B loop

## Final recommendation

The right first milestone is:

- build the standalone theory harness
- implement the oracle dynamics
- run experiments **1, 2, 4, 5** first
- add reverse-KL after that
- only then add the LLM conditioning diagnostic

That gets us fast, interpretable progress without contaminating the main training stack.
