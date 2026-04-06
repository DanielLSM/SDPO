# SDPO Theory Experiment Plan

This document captures a practical implementation plan for the theory-driven experiments described in Alessio Russo’s note, without overloading the main PPO training stack.

## Naming

The appendix frames these as **best-arm identification** experiments, which is why the shorthand **BAI** comes up.

For this repo, we should separate the human-facing name from the implementation slug:

- **Human-facing name:** **multi_bit_sdpo**
- **Descriptive long-form label:** **SDPO theory experiments**
- **Code/package slug:** `multi_bit_sdpo`

Even though the first appendix protocol uses one-bit feedback, the project name should stay broader so it can cover richer feedback variants later.

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
- but implement the theory work as a **small standalone experiment harness**

## Repo fit

This branch already contains several pieces we want to preserve conceptually:

- teacher/student SDPO loss semantics
- forward / reverse / interpolated KL behavior through `alpha`
- feedback-conditioned teacher logic
- existing experiment-script and sweep patterns

The mismatch is structural: the current trainer is designed for normal free-form sequence rollouts, while Appendix B is a much tighter one-token online protocol. So the implementation should be adjacent to the repo’s SDPO logic, not jammed directly into the main PPO path.

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

- load a base instruct model plus optional lightweight adapters
- verify arm labels like `A`, `B`, `C`, ... are single tokenizer tokens
- create a base prompt from frozen bandit summary statistics
- extract restricted probabilities over arm-label tokens only
- sample a proposal arm
- sample one-bit feedback
- build the conditioned teacher prompt
- compute teacher distribution over the same restricted label set
- apply a student-teacher KL step
- fine-tune only lightweight parameters first (for example LoRA)
- run the exact oracle on the same `(Y_k, F_k)` stream for direct comparison

This phase is needed for:

- **Experiment 6: conditioning-error diagnostic**
- optional direct comparison between oracle and learned trajectories

### Important note

This should still be implemented as a separate theory harness, not by heavily mutating the main SDPO PPO path.

The main repo trainer is solving a different systems problem.

## Phase 5 — plotting and sweep utilities

Once the oracle and model-linked loops exist, add a thin analysis layer for the plots recommended in Appendix B:

- KL trajectories
- arm-probability trajectories
- oracle-gap trajectories
- conditioning-error trajectories
- informativeness sweep summaries
- misalignment sweep summaries
- reverse-KL oscillation plots

This should stay lightweight and sit next to the theory harness, not inside the generic trainer.

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

A cleaner merged layout is to keep everything under a dedicated experiment package:

```text
SDPO/
├── SDPO_THEORY_EXPERIMENT_PLAN.md
├── experiments/
│   └── multi_bit_sdpo/
│       ├── README.md
│       ├── instances.py
│       ├── posterior.py
│       ├── proposals.py
│       ├── feedback.py
│       ├── oracle.py
│       ├── prompts.py
│       ├── metrics.py
│       ├── runner.py
│       ├── plots.py
│       ├── run_multi_bit_sdpo.py
│       ├── config/
│       └── scripts/
└── tests/
    └── theory/
        ├── test_forward_projection_on_cpu.py
        ├── test_self_consistency_on_cpu.py
        └── test_reverse_projection_on_cpu.py
```

Why this is better:

- `experiments/multi_bit_sdpo/` gives the work a concrete home
- oracle logic, prompt logic, runners, and plots stay together
- it still remains clearly separate from the generic PPO trainer
- tests stay cheap and CPU-oriented

A later optional step would be to move truly reusable helpers into `verl/utils/`, but only if the theory harness actually produces shared logic worth keeping.

## Metrics to log

For the oracle harness, log at least:

- `kl_to_p_star`
- `l1_to_p_star`
- probability on the true best arm
- argmax accuracy / decision accuracy
- step size
- proposal arm
- sampled feedback bit
- top-arm switch count
- oscillation amplitude
- distance to the simplex boundary

For the forward-KL runs, also log the predicted decrement / observed decrement mismatch from the theory where applicable.

For the LLM-linked phase, also log:

- teacher-student KL
- conditioning error against exact teacher
- oracle-vs-model trajectory gap

## What to reuse vs. build fresh

### Reuse from the repo

- SDPO loss semantics
- optimizer / config conventions
- experiment and sweep script style
- teacher / student naming and logging conventions

### Build fresh for this plan

- exact Gaussian-bandit oracle
- restricted label-only probability extraction
- one-bit feedback simulation
- paper-specific metrics and plots
- the dedicated `multi_bit_sdpo` runner

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

- build a dedicated `experiments/multi_bit_sdpo/` theory harness
- implement the oracle dynamics first
- run experiments **1, 2, 4, 5** first
- add reverse-KL after that
- only then add the LLM conditioning diagnostic
- add plotting and sweep polish once the core dynamics are trustworthy

That gets us fast, interpretable progress without contaminating the main training stack.
