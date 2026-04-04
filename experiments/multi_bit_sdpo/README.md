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

The LLM-conditioned phase is intentionally left for later, after the oracle path is trusted.

## Layout

- `instances.py` — problem presets
- `posterior.py` — frozen exploration data + oracle posterior estimation
- `proposals.py` — proposal distributions
- `feedback.py` — feedback channel and misalignment helpers
- `oracle.py` — exact teacher and oracle updates
- `prompts.py` — prompt helpers for the later LLM phase
- `runner.py` — configurable JSONL/summary oracle runner
- `run_oracle_experiments.py` — paper-default sweep driver with CSV export
- `plots.py` — lightweight plotting helper
- `gpu_smoke.py` — first GPU smoke layer for restricted label-logit extraction
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

## Notes

- The project name is `multi_bit_sdpo`, even though the first oracle protocol currently uses one-bit feedback. The name is intentionally broader so richer feedback variants can live in the same harness later.
- The current runner uses the **branchwise oracle update** driven by the sampled proposal. That matches the stochastic-iteration viewpoint used in the theory note.
- `gpu_smoke.py` is intentionally a smoke layer, not the full training loop: it only checks model loading, label-token extraction, and conditioned-vs-base next-token distributions on GPU.
