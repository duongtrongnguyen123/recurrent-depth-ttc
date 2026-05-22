# Recurrent-Depth Transformers: when does looping beat width?

> Training a recurrent-depth transformer for a **single loop**, then iterating
> that one block at inference, reaches **256× the trained depth at 100%
> accuracy** on algorithmic reasoning tasks — for ~7 minutes of training.
> I also show this advantage **does not (yet) survive to 1 B-token real-text
> language models**, and give the mechanistic reason why.

This repository is a curated writeup of a multi-week independent research
program on **recurrent-depth transformers** — a single shared transformer
block looped *N* times instead of *N* distinct blocks (the Universal
Transformer / Huginn / Ouro lineage). It trades parameters for
inference-time FLOPs: a 1-block × 8-loop model has the per-step compute of an
8-block model with 1/8 the parameters.

The program is organized around one question that the literature does not
have clean numbers on:

> **When does recurrent depth give capability that width cannot substitute
> for — and when is it just an expensive way to do what width already does?**

The honest answer, in one line: **recurrence pays rent only when the task
has irreducible per-step sequential structure; for everything else, width +
the right supervision wins.** The interesting part is *how sharp* that
boundary is, and that the boundary is set by the **training recipe**, not
the architecture.

---

## The four results

### 1. Length extrapolation is a supervision property, not an architecture property
[`writeup/01-supervision-lever.md`](writeup/01-supervision-lever.md)

The same looped architecture either walls exactly at its trained loop count
or extrapolates to 3×+ it — depending only on *how you supervise the loops*:

- **Per-final-answer** supervision (loss on the last loop only) teaches the
  model to *terminate*. It walls at the trained depth.
- **Per-step (iterative-target)** supervision (loss at every loop, target =
  the partial result after that many steps) teaches the model to *iterate*.
  It extrapolates: 100% accuracy through 3× the trained loop count, decaying
  gracefully with calibrated confidence beyond that.

A control on a different per-step rule (parity) pins down *why*: iter-target
generalises only when the per-step update is **position-invariant** (a
function of state, not of the loop index). This is a falsifiable mechanistic
claim, and the parity experiment is the falsification test that passes.

### 2. The cheapest known route to o1-style adaptive test-time compute
[`writeup/02-test-time-compute.md`](writeup/02-test-time-compute.md)

If per-step supervision teaches genuine iteration, the trained loop count
should not bound the *inference* loop count. It doesn't:

- Train the block for **a single loop** (n_loops=1) with iterative-target +
  noise injection.
- At inference, run it as a multi-pass loop (each pass's argmax feeds the
  next).
- Result: **100% accuracy at 256 effective loops — 256× the trained depth —
  in ~7 minutes of total training.**

Add a tiny halt head (or a hardcoded `halt(r,k)=r≥k` — they perform
identically) and the user *dials* the inference compute per example: halt at
*exactly* the requested loop count, anywhere from 1 to 256. This is the
o1/r1 adaptive-compute pattern, reproduced at the recurrent-depth substrate
for negligible cost.

### 3. Controllable recurrent solver: composition via orchestration, not architecture
[`writeup/03-controllable-solver.md`](writeup/03-controllable-solver.md)

A single recurrent model trained end-to-end on a *composed* task
(chain ∘ arithmetic) gets **0%** — it cannot switch programs mid-forward.
The same architecture, with each primitive trained *separately* and an
external orchestrator chaining the calls, gets **96/96 composition cells at
100%** across six primitive pairs and all depth combinations.

The lesson: the recurrent block is a **callable atomic operator**; program
structure belongs in the scheduler (a planner / LLM / search), not baked
into the weights. This is the concrete, minimal version of the
"reasoning = recurrent operator + external scratchpad" pattern.

### 4. The mechanism audit: a clean observation with no surviving explanation
[`writeup/04-mechanism-audit.md`](writeup/04-mechanism-audit.md)

Pin down the single-pass extrapolation ratio: under iter-target on chain
V=12 at NL ∈ {4, 8, 16}, **collapse depth ≈ 2.06 · NL** (slope of a linear
fit through three points; ratio drifts down with larger NL — a clean
empirical observation, not a proven law).

Then audit the natural follow-up: *why ≈ 2×?* Five mechanism hypotheses
were pre-registered, tested on two independent compute streams, and
filed under "falsified":

1. **Trajectory geometry** (angular spread) — sign reversed from prediction.
2. **State-space coverage** (exposure bias) — overlap measured = 1.0, no
   gap exists yet collapse still occurs.
3. **LayerNorm-induced drift** — predicts task-invariant collapse, fails to
   discriminate chain (collapse ≈ 2·NL) from modular sum (collapse ≈ NL).
4. **Contraction is causal** — forcing contraction with a regularizer
   **halved** the extrapolation horizon. Contraction is a co-occurring
   symptom, not the mechanism.
5. **Decoder Jacobian shape** — non-monotonic, non-predictive.

What survives the audit: the empirical observation, and writeup 1's
dissociation between supervision regimes. The mechanism question is
genuinely open.

The point of the audit is not the negative results themselves — it is the
**methodology**: pre-register the prediction, state what would falsify
it, verify on two independent streams. By that bar, much of the recurrent-
depth mechanism literature does not have testable claims. The cleanest
*describable* observation in the area is in search of its first
*falsifiable* mechanism.

---

## What does NOT work (and why that matters)

[`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md) is deliberately prominent.

- **The parameter-efficiency claim does not survive to scale (yet).** On 1 B
  tokens of real FineWeb-Edu text, a dense vanilla transformer beats every
  recurrent variant by a clear margin (0.1–0.6 nats). Recurrence's win is
  currently confined to algorithmic tasks with explicit per-step structure.
- **A claimed architecture win was retracted** when a parameter-matched
  control showed the "gain" was 1.9× capacity, not the architecture.
- **A halting signal result was nearly published before a trailing-space
  tokenization artifact was caught** — it was retracted and re-derived.
- A throughput optimization (larger micro-batch) was benchmarked,
  found to be net-negative, and dropped.
- **WSD decay can degrade reasoning** — a 900M model after CoT-SFT alone
  outperforms the same model after decay+CoT-SFT on algebra workflow,
  recursive code, and history facts. Order-of-operations matters.
- **Vanilla matches looped under the same recipe** — `vanilla + iter-target
  + multipass` reaches 100% at 16× train depth, breaking the multi-token
  state ceiling. Architecture is not the lever; supervision is.

These are here because they are the actual content of doing research
honestly. The boundary of a claim is part of the claim.

---

## Repo layout

```
README.md                     this file — the whole argument in 5 minutes
writeup/
  01-supervision-lever.md     iter-target vs per-answer; the parity control
  02-test-time-compute.md     train n=1 → 256× inference depth; halt control
  03-controllable-solver.md   composition via orchestration (0% → 100%)
  04-mechanism-audit.md       2-2.5× single-pass observation; 5 falsified mechanisms
NEGATIVE_RESULTS.md            retractions, scale limits, null results
src/model.py                  the looped/pcc/hr transformer (reference impl)
results/                      key figures (seed-pinned, methodology in writeups)
```

## Method notes

All algorithmic results are small-scale and seed-pinned (d ≤ 1280, vocab ≤ 27,
single RTX 4090/5090/6000, minutes-to-hours of compute). Real-text results
are at d=1024–2048, ≤ several B tokens. Every quantitative table in the
writeups states its exact config. The unifying experimental discipline:
**one question per experiment, parameter-matched controls before any
architecture claim, negative results written up as carefully as positive
ones.**

## Status

The real-text scaling arm (Phase 1 → 2) is an ongoing program; the
synthetic-task conclusions above are stable and reproduced across seeds.
This repo is the curated subset that survived scrutiny.

_License: MIT. Independent research, 2026._
