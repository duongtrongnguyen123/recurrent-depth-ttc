# Recurrent-Depth Transformers: when does looping beat width?

A controlled study of looped transformers — reusing one weight-tied block N times instead of stacking N distinct blocks (the Universal Transformer / Huginn / Ouro line). It maps where extra inference loops buy real capability and where they only reproduce what width already does.

![python](https://img.shields.io/badge/python-3.8%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)

Two results below. The rest of the program — a loss-landscape / sharpness study, a Q/K/V mechanism probe, a matched-data cross-architecture comparison, composition-via-orchestration, and the negative results — is in [`writeup/`](writeup/) and [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md). Everything here is small-scale and seed-pinned; treat it as controlled experiments, not claims about production LLMs.

## Architecture

Prelude–Core–Coda (the Huginn shape): a few unique pre-layers, **one weight-tied "core" block looped N times**, a few unique post-layers. Concretely `n_prelude` + core×N + `n_coda`, with the core sharing one set of weights across all N loops.

- **N is fixed during training.** At inference, depth is either a fixed number, or — in Result 2 — dialed per example by a halt rule plus multi-pass.
- **The base model has no per-loop conditioning** (no loop-index embedding fed to the core). This matters when interpreting loop dynamics: an unconditioned shared block has no external signal telling it which loop it is on.

Reference implementation with all variants (vanilla / looped / pcc / gated / …) is in [`src/model.py`](src/model.py).

## Tasks

The extrapolation experiments use small synthetic sequence tasks where the result after `r` steps is well-defined, so "run the loop `r` times" has a ground truth to score against.

- **chain** (pointer walk): a random table maps each of `V` symbols to a successor. From a start symbol, the result after `r` steps is the symbol reached by following the pointer `r` times.
- **parity**: over a bit string, the result after `r` steps is the XOR of the first `r` bits. Its per-step rule depends on the position `r` — which is exactly why it is used as a control.

The full set is five such tasks (chain, ListOps, modular arithmetic, graph BFS, parity); per-task definitions and results are in [`writeup/01-supervision-lever.md`](writeup/01-supervision-lever.md).

## Result 1 — how you supervise the loops decides whether it extrapolates

Train the same looped model two ways and you get opposite behaviour past the trained depth:

- **Supervise only the final loop** → the model learns to *terminate*. It walls exactly at the trained loop count.
- **Supervise every loop** against the partial result after that many steps ("iterative-target") → the model learns to *iterate*. It keeps working past the trained depth — on our synthetic pointer-chain task, out to ~24× the trained loop count.

The boundary is sharp and has a control that *fails as predicted*: this only holds when the per-step rule is **position-invariant** (a function of state, not of the loop index). **Parity** — whose per-step rule depends on the loop index — walls exactly at the trained depth, no matter the supervision. That failing control is what makes this an empirical rule rather than a just-so story.

![length extrapolation under iterative-target supervision](results/length_extrap.png)

Details: [`writeup/01-supervision-lever.md`](writeup/01-supervision-lever.md).

## Result 2 — a cheap way to dial inference compute after training

If per-step supervision teaches real iteration, the trained loop count shouldn't bind the *inference* loop count. It doesn't. After a short LoRA fine-tune (~31K trainable params), running inference as multiple passes plus a hardcoded halt (`stop when loop ≥ requested depth`) lets you set the compute per example:

- **100% accuracy at a requested depth up to 256× the trained depth**, on the synthetic chain task.
- No learned halt head needed; the hardcoded rule matches a trained one.

This is the o1-style "spend more compute on harder inputs" idea at the recurrent-depth level. It is a controlled-scale demonstration on a synthetic task, not a language-model result.

Details: [`writeup/02-test-time-compute.md`](writeup/02-test-time-compute.md), production floor in [`writeup/05-production-recipe.md`](writeup/05-production-recipe.md).

## What does not work

At sub-1B parameters on real text (four architectures on a 50B-token matched-data pretrain), **no recurrent variant beats a matched dense baseline beyond the run-to-run noise of pretraining** — the per-wave GSM8K std dev is ±0.6pp, and the cross-architecture differences sit inside it. The extrapolation numbers above are on synthetic algorithmic tasks, not natural language.

Full accounting, retractions, and scope limits: [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md) and [`writeup/08-cross-architecture-phase2.md`](writeup/08-cross-architecture-phase2.md).

---

_Independent research, 2026. MIT. Small-scale, seed-pinned; one question per experiment, negatives written up alongside positives._
