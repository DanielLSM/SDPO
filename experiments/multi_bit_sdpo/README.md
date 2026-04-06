# multi_bit_sdpo

Standalone theory harness for the SDPO Appendix-B style experiments.

## Current scope

This first implementation covers the **exact oracle dynamics** on CPU:

- Gaussian bandit instance presets (`Easy-2`, `Hard-2`, `Medium-5`, `Hard-5`)
- frozen exploration summaries
- Monte Carlo estimation of the oracle posterior `p*`
- epsilon-greedy and temperature-plus-epsilon proposal rules
- one-bit feedback simulation with `(alpha, beta)`
- exact forward-KL and reverse-KL oracle branch updates
- JSONL logging, summary output, and CSV sweep export
- CPU tests for the key theory identities

The oracle path is now trusted enough that the harness also includes a **minimal LLM-conditioned rollout probe** on GPU. This is still not the full training loop: it iteratively conditions the prompt on proposal/feedback history and measures the restricted label distribution after each round.

## Layout

- `instances.py` — problem presets
- `posterior.py` — frozen exploration data + oracle posterior estimation
- `proposals.py` — proposal distributions
- `feedback.py` — feedback channel and misalignment helpers
- `oracle.py` — exact teacher and oracle updates
- `prompts.py` — prompt helpers for the LLM-conditioned phase
- `runner.py` — configurable JSONL/summary oracle runner
- `run_oracle_experiments.py` — paper-default sweep driver with CSV export
- `plots.py` — lightweight plotting helper
- `llm_backend.py` — shared runtime/model helpers for GPU-side probes
- `gpu_smoke.py` — first GPU smoke layer for restricted label-logit extraction
- `llm_rollout.py` — minimal multi-step LLM-conditioned rollout on fixed exploration summaries
- `scripts/` — shell wrappers for common runs

## Example usage

From the repo root:

```bash
python -m experiments.multi_bit_sdpo.runner \
  --experiment forward_convergence \
  --instance Easy-2 \
  --iterations 300 \
  --alpha 1.0 \
  --beta 0.0 \
  --seeds 0 1 2
```

Paper-default forward-KL sweep with CSV output:

```bash
python experiments/multi_bit_sdpo/run_oracle_experiments.py \
  --experiment forward_convergence \
  --output-dir outputs/multi_bit_sdpo
```

Truth-driven forward KL with the lighter single-config runner:

```bash
bash experiments/multi_bit_sdpo/scripts/run_forward_convergence.sh
```

Self-consistency check:

```bash
bash experiments/multi_bit_sdpo/scripts/run_forward_self_consistency.sh
```

Informativeness sweep starter:

```bash
bash experiments/multi_bit_sdpo/scripts/run_informativeness_sweep.sh
```

First GPU smoke layer (tiny model load + restricted label logits):

```bash
bash experiments/multi_bit_sdpo/scripts/run_gpu_smoke.sh
```

This runs a tiny LLM-side check that:
- loads an instruct model onto GPU
- validates that arm labels are single tokenizer tokens
- computes restricted next-token probabilities over arm labels for the base prompt
- recomputes them for positive and negative conditioned prompts

Minimal multi-step LLM-conditioned rollout:

```bash
bash experiments/multi_bit_sdpo/scripts/run_llm_rollout.sh
```

This runs a first iterative prompt-conditioned probe that:
- starts from the base summary prompt
- samples proposals from the current restricted label distribution
- samples noisy feedback from the chosen `pgen` mode
- re-evaluates the next restricted label distribution after each round
- optionally takes a few interleaved gradient steps toward the exact restricted teacher (`--interleaved-grad-steps N`) using either forward KL (`--train-objective forward_kl`) or reverse KL (`--train-objective reverse_kl`)

Tiny frozen-vs-interleaved comparison on the recommended small setting:

```bash
python3 -m experiments.multi_bit_sdpo.llm_rollout \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --instance Easy-2 \
  --proposal-kind epsilon_greedy \
  --epsilon 0.1 \
  --alpha 0.9 \
  --beta 0.1 \
  --iterations 4 \
  --seeds 0 \
  --interleaved-grad-steps 2 \
  --interleaved-lr 5e-6 \
  --train-objective forward_kl \
  --compare-modes
```

## Notes

- The project name is `multi_bit_sdpo`, even though the first oracle protocol currently uses one-bit feedback. The name is intentionally broader so richer feedback variants can live in the same harness later.
- The current runner uses the **branchwise oracle update** driven by the sampled proposal. That matches the stochastic-iteration viewpoint used in the theory note.
- `gpu_smoke.py` is intentionally a smoke layer, not the full training loop: it only checks model loading, label-token extraction, and conditioned-vs-base next-token distributions on GPU.
- `llm_rollout.py` now supports two tiny modes: a frozen prompt-only rollout (`--interleaved-grad-steps 0`) and a minimal interleaved-training sanity check that takes a few optimizer steps after each feedback round to match the exact restricted teacher distribution (`--interleaved-grad-steps N`). This is still only a lightweight proxy, not the full PPO/trainer loop.
