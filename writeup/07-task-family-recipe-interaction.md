# 7. No universal recipe — task-family × training-recipe interaction matrix

## The question

The Huginn paper proposes a specific training recipe (randomized recurrence
depth + truncated BPTT + LTI re-injection) as the *production* setting for
recurrent-depth transformers. Earlier writeups (1, 5) showed that
**supervision** is the lever, but did not rigorously test whether the
remaining engineering choices generalize across task families. The
question this writeup answers: **does any single "production" recipe
work across task families, or do different per-step rule structures need
different recipes?**

## Setup

Same PCC backbone, 4 variants stacking Huginn-paper engineering on top of
the BP recipe (iter-target + noise + multipass):

| variant | LTI re-inject | random-r training | truncated BPTT |
|---|:---:|:---:|:---:|
| `bp_baseline` | — | — | — |
| `bp_plus_lti` | ✓ | — | — |
| `bp_plus_random_r` | — | ✓ | — |
| `bp_full_huginn` | ✓ | ✓ | ✓ |

3 algorithmic task families spanning the per-step rule space:

- **chain V=12, V=24** (table lookup; position-invariant rule; from writeup 1)
- **modular P=13** (accumulator: `x_{r+1} = (a·x_r + b) mod P`; accumulator
  needs consistent carry across loops)
- **parity** (XOR over first r bits; position-DEPENDENT rule — the
  falsification test from writeup 1 where iter-target alone walls)

Same `d=384`, `n_train=2`, 4000 steps. 4 seeds per variant per task →
40 kernels total (16 modular + 16 parity + 8 chain), parallel on 8
Kaggle accounts. Total wall-clock ~1 hour.

## Result — the task × recipe matrix

| variant | chain V=12/V=24 | modular P=13 (uk=256) | parity (uk=4, 2× trained) | parity (uk=8, 4× trained) |
|---|:---:|:---:|:---:|:---:|
| **bp_baseline** | 1.000 | **0.962 ± 0.045** | 0.486 ± 0.029 | 0.488 ± 0.022 |
| **bp_plus_lti** | 1.000 | 0.961 ± 0.065 | 0.484 ± 0.030 | 0.498 ± 0.019 |
| **bp_plus_random_r** | 1.000 | 0.450 ± 0.142 | **1.000 ± 0.000** | **1.000 ± 0.001** |
| **bp_full_huginn** | 1.000 | 0.237 ± 0.032 | 1.000 ± 0.000 | 0.508 ± 0.017 |

(Chain saturates at 1.000 for all variants — too easy at this scale to
discriminate. Parity at user_k=16 ≈ chance 0.50 for all variants — the
counter mechanism exhausts past 8× train depth.)

## The headline pattern

**The same engineering choice (random-r training) HELPS one task family by
+0.51 and HURTS another by −0.51.**

- On **modular** (accumulator structure): random_r training drops accuracy
  from 0.962 to 0.450 — a 51 pp regression.
- On **parity** (position-dependent rule): random_r training raises
  accuracy from 0.486 to 1.000 — a 51 pp improvement, exactly the
  extrapolation rescue that iter-target alone cannot achieve.

Symmetric absolute effect size (~0.51 in both directions), opposite
sign, on the same recipe lever. This is the cleanest task-family-specific
recipe interaction in the program.

## Mechanism

The two task families load the recurrent block differently:

**Parity needs a counter** — the per-step rule depends on the loop index
itself ("at loop r, XOR with bit at position r"). iter-target at
`n_train=2` only ever shows the model r=1 and r=2; the model has no
training signal to learn a counter for r > 2. **Random-r training
exposes the model to varying loop counts during training**, which
provides exactly the counter signal iter-target lacks. The
extrapolation rescue is mechanistically clean: the model now sees
r ∈ {1..8} at training, generalises to r=4 and r=8 at test.

**Modular needs consistent carry** — the per-step rule
`x_{r+1} = (a·x_r + b) mod P` requires the model to learn a stable
multiplicative-additive update applied identically at every loop. The
gradient signal that lets it learn this is the **stable per-depth
backprop** that fixed-depth training provides. Random-r training
disrupts this — at loop 7, sometimes the model is "finishing loop 7 of
7" (full gradient path) and sometimes "running loop 7 of 3 with the
rest dropped" (no downstream gradient). The block sees inconsistent
gradient pressure at each depth → fails to learn consistent carry.
The mechanism that *provides* the position counter for parity
*disrupts* the carry-stability the accumulator needs.

The interaction is not a curiosity — it is the predictable consequence
of one inductive bias (depth-variance) being orthogonal-to-positive for
one task structure and orthogonal-to-negative for another.

## Falsifies two opposing claims

This writeup retracts two simpler claims that previous batches in this
program made on single-task evidence:

1. **"Huginn engineering is universally good"** (the paper's framing /
   default reading) — falsified by the modular result. random_r training
   hurts an accumulator task by 51 pp.

2. **"Huginn engineering is universally bad at small scale"** (the
   narrow reading of the CC4 modular finding alone) — falsified by the
   parity result. random_r training rescues a position-dependent task
   by 51 pp.

Neither generalization holds. The honest statement is the matrix.

## Honest scope (single-seed retraction)

CC2 initially reported (single-seed) that LTI architecturally helped
modular (0.979 vs 0.910). That comparison turned out to be a phantom
of seed-selection bias on bp_baseline (seed=42 was −1.2σ below baseline
mean, by chance — pulled the apparent Δ to +0.069). Four-seed verify on
both variants showed mean Δ = −0.001, well within ±0.05 std. **The LTI
"win" was retracted before publication; the surviving signals are the
random-r effects on modular (−0.51) and parity (+0.51).** The standing
rule that came out of this retraction: when claiming variant A beats
variant B, multi-seed *both* A and B — multi-seeding only the new
variant lets baseline outliers create phantom effects.

## Production implications

For a practitioner choosing a recurrent-depth training recipe, the rule
the matrix establishes:

- **Identify the per-step rule structure of your task first.** Don't
  apply a "default" recipe.
- **Position-invariant + bounded per-step rule** (chain table-lookup,
  ListOps-style reductions): BP recipe (PCC + iter-target + noise +
  multipass) saturates the task; Huginn engineering is unnecessary
  overhead and contributes nothing measurable.
- **Position-dependent rule** (parity, anything requiring a counter):
  add random-r training. iter-target alone walls; random-r provides
  the counter signal.
- **Accumulator structure** (modular sums, running aggregates): use
  BP recipe alone. Adding random-r or truncated BPTT is actively
  harmful — gradient signal must be stable per depth.

The "production" Huginn recipe assumes the position-dependent or
counter-required regime. On accumulator regimes it produces a 50+ pp
regression. Practitioners deploying recurrent-depth on tasks they
haven't audited will get unpredictable results.

## Why this matters for the broader research

Writeup 1 established that supervision dissociates dynamical regime.
Writeup 4 established that the 2× single-pass observation has no
falsified mechanism. Writeup 6 showed an architectural variant
(xloop + linear stabiliser) shifts the ratio by 30×. This writeup adds
the recipe-level interaction: **architecture, supervision, and
training-schedule are three distinct levers, and their effects depend
on per-step rule structure of the task**. A single "best" configuration
across task families does not exist on the evidence in this program.

This is the empirical foundation for the cleaner production-recipe
recommendations in writeup 5 — the recipe ablation there
(`bp_baseline` + iter-target + noise + multipass) is the *minimum
common viable recipe*, and additions on top should be made
task-conditionally rather than as defaults.

## Method notes

- All 40 kernels seed-pinned (seeds 0, 1, 2, 42).
- Single-pass eval for parity (continuous hidden state through r loops,
  no argmax feedback because input is fixed bits, no x-position to
  re-inject into).
- Multipass eval for chain & modular per the standard writeup-2 protocol.
- 4-batches × 128 examples per (variant, seed, user_k) cell — 512
  examples per data point.
- Code: `archlab/scripts/CC2_huginn_combo_hardertask.py` parameterized by
  `--variant {bp_baseline, bp_plus_lti, bp_plus_random_r, bp_full_huginn}`
  `--task {chain_v24, modular_p13, parity_iter}` `--seed N`.
- All std reported in tables; per-seed values in
  `archlab/FINDINGS_KAGGLE.md` (private repo).
