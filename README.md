# Recurrent-Depth Transformers: controlled experiments on looped inference

Recurrent-depth transformers reuse one weight-tied block N times instead of stacking N distinct blocks (the Universal Transformer / Huginn / Ouro line), trading parameter count for inference-time compute. This repository reports controlled experiments on when the extra inference loops add capability that width does not already provide.

![python](https://img.shields.io/badge/python-3.8%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)

Two primary results are below. The remaining studies — sharpness / loss-landscape, a Q/K/V mechanism probe, a matched-data cross-architecture comparison, and composition — are in [`writeup/`](writeup/), with negatives in [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md). All experiments are small-scale and seed-pinned: controlled studies, not claims about production language models.

## Architecture

The model follows the Prelude–Core–Coda (Huginn) shape: unique pre-layers, one weight-tied core block applied N times, and unique post-layers.

```
tokens --> [ prelude ] --> [ core ] --> [ coda ] --> logits
                           ^       |
                           +-------+  x N
```

Prelude and coda are ordinary distinct layers; the core is a single weight-tied block reused on each of the N loops (`n_prelude` + core×N + `n_coda`).

- N is fixed during training. At inference, depth is either a fixed count or, in Result 2, set per example by a halt rule combined with multi-pass decoding.
- The base model uses no per-loop conditioning: no loop-index embedding is provided to the core, so an unconditioned shared block receives no external signal indicating which loop it is executing. This is relevant when interpreting the loop dynamics.

A reference implementation of all variants (vanilla / looped / pcc / gated / …) is in [`src/model.py`](src/model.py).

## Tasks

The extrapolation experiments use small synthetic sequence tasks for which the result after `r` steps is well-defined, giving a ground-truth target for applying the loop `r` times. Whether a task extrapolates past the trained depth depends on a single property: whether its per-step rule depends on the loop index (Result 1).

| task | per-step rule | position-invariant? |
|---|---|:---:|
| **chain** (pointer walk) | follow a random symbol→successor table one step | yes |
| **ListOps** | reduce the innermost `min` / `max` / `sum` operator | yes |
| **modular** | `x ← (a·x + b) mod P` | yes |
| **graph BFS** | expand the reachable frontier by one hop | yes |
| **parity** | XOR in the bit at position `r` | **no** (depends on `r`) |

Per-task definitions and results: [`writeup/01-supervision-lever.md`](writeup/01-supervision-lever.md).

## Result 1 — supervision, not architecture, determines length extrapolation

Training the same looped model under two supervision schemes produces opposite behaviour beyond the trained depth.

- Supervising only the final loop teaches the model to terminate; accuracy walls at the trained loop count.
- Supervising every loop against the partial result after that many steps (iterative-target) teaches the model to iterate; accuracy is retained beyond the trained depth, to approximately 24× the trained loop count on the synthetic chain task.

The boundary is governed by a position-invariance condition, and it includes a control that fails as predicted. Extrapolation holds only when the per-step rule is a function of state rather than of the loop index. Parity, whose per-step rule depends on the loop index, walls exactly at the trained depth under either supervision scheme. This failing control distinguishes the result from an unfalsifiable observation.

![length extrapolation under iterative-target supervision](results/length_extrap.png)

## Result 2 — inference compute can be set per example after training

Because iterative-target supervision installs genuine iteration, the trained loop count does not bound the inference loop count. After a short LoRA fine-tune (approximately 31K trainable parameters), running inference as multiple passes with a hardcoded halt rule (stop when the cumulative loop count reaches the requested depth) allows the inference compute to be set per example.

- 100% accuracy at a requested depth up to 256× the trained depth on the synthetic chain task.
- No learned halt head is required; the hardcoded rule matches a trained one.

This reproduces the adaptive-compute pattern — allocating more inference compute to harder inputs — at the recurrent-depth level. It is a controlled-scale demonstration on a synthetic task and is not a language-modeling result.

Details: [`writeup/02-test-time-compute.md`](writeup/02-test-time-compute.md); the minimal recipe is in [`writeup/05-production-recipe.md`](writeup/05-production-recipe.md).

## Limitations

At sub-1B parameters on natural language (four architectures trained on a 50B-token matched-data mixture), no recurrent variant exceeded a matched dense baseline beyond the run-to-run variance of pretraining: the per-checkpoint GSM8K standard deviation is ±0.6pp, and the cross-architecture differences fall within it. The extrapolation results above are on synthetic algorithmic tasks, not natural language.

Full negative results, retractions, and scope limits are in [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md) and [`writeup/08-cross-architecture-phase2.md`](writeup/08-cross-architecture-phase2.md).

---

Independent research, 2026. MIT license. All experiments are small-scale and seed-pinned; each experiment isolates a single question, and negative results are documented alongside positive ones.
