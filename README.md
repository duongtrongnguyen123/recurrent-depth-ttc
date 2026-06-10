# Recurrent-Depth Transformers: when does looping beat width?

> Training a recurrent-depth transformer for a **single loop**, then iterating
> that one block at inference, reaches **256× the trained depth at 100%
> accuracy** on algorithmic reasoning tasks — for ~7 minutes of training,
> with a **31K-parameter adapter** as the production recipe floor. Pushing
> the same recipe family wider extends this to **4096× train depth**
> (multi-pass) and **64× single-pass** with an architectural variant that
> contradicts the program's own 2× single-pass observation. The
> parameter-efficiency claim does **not (yet) survive** to 1 B-token raw
> web text, but a recurrent variant does win **head-to-head against a
> dense baseline on reasoning-mix at the same scale**.

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

## The eight results

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

### 5. Production recipe — 31K trainable params, hardcoded halt, multi-task
[`writeup/05-production-recipe.md`](writeup/05-production-recipe.md)

Tightening writeup 2's "9-minute LoRA r=8 + halt head" pipeline to its
floor:

- **Attn-only LoRA r=4 → 31K trainable params** (5.6× cheaper than the
  175K-param recipe, identical accuracy at trained depth and at
  user_k=256).
- **Hardcoded `halt(r,k) = r≥k` → 0 halt parameters** (identical
  calibration to a 367K-param trained halt head — the head was learning
  to approximate the hardcoded rule).
- **One 31K adapter holds chain + listops** without measurable accuracy
  loss, and halves catastrophic forgetting compared to sequential
  single-task FT.
- **Counterintuitive**: smaller `n_train` extrapolates *better* under
  multi-pass. `n_train=1` is the floor and the optimum.
- The whole recipe transfers to a **~600M-param base** (K-AZ), reaching
  user_k=256 at trained-depth accuracy with no recipe changes.

Min-cost adaptive-compute substrate: **61 s of base training, 31K trainable
adapter params, 0 halt params**. The result connects back to writeup 4:
the architecture is doing most of the work for free; supervision +
inference recipe is what makes it controllable.

### 6. The extrapolation frontier — and a recipe that breaks single-pass
[`writeup/06-extrapolation-frontier.md`](writeup/06-extrapolation-frontier.md)

Pushing the multi-pass envelope wider, and finding one architectural
variant that contradicts writeup 4's single-pass observation:

- **K = 4096 (512× train depth)** on chain V=12 with `pcc_hr_hybrid`
  d=1280 + noise injection — 16× the 256× number in writeup 2.
- **K = 2048 (1024× train depth)** on arithmetic reduction with `n=2 +
  noise` — the biggest extrapolation ratio in the program.
- **Modular sum solved at trained depth** (100% at K=8, vs 8% baseline)
  with noise + high-batch + extended `n_loops` — partial rescue of the
  boundary case writeup 1 flags.
- **Single-pass 64× extrapolation** with an `xloop + linear stabiliser`
  architecture — directly contradicts writeup 4's 2-2.5× single-pass
  observation, demonstrating that the 2× ratio is recipe-bounded, not
  architecture-bounded.
- **First real-text head-to-head won by a recurrent variant**:
  `pcc_hr_hybrid d=1280 × 6 loops` (118M) beats `vanilla d=2048` (153M)
  by 0.07 nats val loss on reasoning-mix at 1 B tokens — scoping
  NEGATIVE_RESULTS §1 from "vanilla wins everywhere" to "vanilla wins on
  raw web, hybrid wins on reasoning-mix."

Together with writeups 4 and 5, this sharpens the open mechanism question:
the 2× ratio is real *for the specific recipe family*, but it can be
shifted by an order of magnitude with the right architectural lever.

### 7. No universal recipe — task-family × training-recipe interaction matrix
[`writeup/07-task-family-recipe-interaction.md`](writeup/07-task-family-recipe-interaction.md)

4-variant × 4-seed × 3-task matrix testing whether Huginn paper's
"production" recipe (random-r training + truncated BPTT + LTI) generalizes
across task families. **The same engineering choice (random-r training)
HELPS one task by +0.51 and HURTS another by −0.51, with symmetric
absolute effect on the same lever.**

| variant | chain (V=12/24) | modular P=13 | parity (uk=4, 2× trained) |
|---|:---:|:---:|:---:|
| bp_baseline | 1.000 | **0.962 ± 0.045** | 0.486 ± 0.029 |
| bp_plus_lti | 1.000 | 0.961 ± 0.065 | 0.484 ± 0.030 |
| bp_plus_random_r | 1.000 | **0.450 ± 0.142** (HURTS) | **1.000 ± 0.000** (RESCUES) |
| bp_full_huginn | 1.000 | 0.237 ± 0.032 (catastrophe) | 1.000 ± 0.000 then walls |

**Mechanism**: parity needs a *counter* the iter-target supervision lacks at
small `n_train`; random-r exposure provides it. Modular needs *consistent
carry across loops*; random-r breaks per-depth gradient signal density.
The mechanism that helps one structure hurts the other.

This falsifies both opposing claims simultaneously:
- *"Huginn engineering is universally good"* (paper's framing) — refuted
  by modular.
- *"Huginn engineering is universally bad at small scale"* — refuted by
  parity.

Production implication: **identify per-step rule structure first; recipe
follows from it**, not the other way around. A single "default" recipe
applied to all task families will produce 50+ pp regressions on the wrong
family.

This writeup also retracts an earlier single-seed CC2 claim that LTI
architecturally helped modular (phantom Δ from baseline outlier seed —
detected on 4-seed verify of both variants). The standing methodological
rule: **when claiming variant A beats variant B, multi-seed BOTH A and B**
— multi-seeding only the new variant lets baseline outliers create
phantom effects.

### 8. Cross-architecture at real-text scale: the synthetic wins do not (yet) transfer
[`writeup/08-cross-architecture-phase2.md`](writeup/08-cross-architecture-phase2.md)

Four architectures (PCC 356M, xloop 356M, vanilla 500M/912M) pretrained on the
**same 50B-token canonical mix**, compared head-to-head. The headline is a
quantified negative: **no recurrent variant beats the matched dense baseline
beyond the noise of pretraining itself** (per-wave GSM8K std dev = ±0.60pp;
cross-architecture differences sit inside that band). The one reasoning signal
that clears the noise — HARD50 sampling-TTC — favours the *dense* model
(vanilla 500M best-of-K=20 = 90%, matching vanilla 912M at K=100).

What *is* recurrent-specific and measurable is the geometry: a sharper local
minimum (κ ≈ 33 PCC vs 12 vanilla 912M), a U-shaped loss over inference depth,
and a **state fixed-point** mechanism — applying the shared W_Q/W_K/W_V across
loops, all three projections cohere at once (cos ≈ 0.97/0.98/0.86 for PCC),
which only the hypothesis "`h_r` itself reaches a fixed point of `Block(·)`"
explains (a W_Q-only power iteration would not collapse K and V in lockstep).
These observations are correlational — the pretrains are not token-matched.

Bottom line, consistent with writeups 1–7: **recurrence is a post-training
mechanism for position-invariant per-step tasks, not a competitive pretraining
architecture at this scale.** We make no claim to refute the published
Huginn-3.5B numbers, which were obtained at 20–30× our token budget.

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
  05-production-recipe.md     31K-param adapter, hardcoded halt, multi-task, scaling
  06-extrapolation-frontier.md K=4096 multi-pass; 64× single-pass; reasoning-mix win
  07-task-family-recipe-interaction.md  no universal recipe — same lever ±0.51 on diff tasks
  08-cross-architecture-phase2.md  50B-token 4-arch comparison; noise band; sharpness; Q/K/V fixed-point
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

The real-text scaling arm (Phase 1 → 2) is summarised in writeup 8; its
cross-architecture comparisons are correlational, pending a token-matched
retrain. The synthetic-task conclusions (writeups 1–7) are stable and
reproduced across seeds. This repo is the curated subset that survived scrutiny.

_License: MIT. Independent research, 2026._
