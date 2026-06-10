# 6. The extrapolation frontier — and one architectural recipe that breaks single-pass

## The headline numbers

Writeup 2 demonstrates `train n=1 → 256× train depth at 100%` via
multi-pass. Pushing the same recipe across width, task family, and
inference protocol, the actual frontier sits much higher — and one
architectural variant produces a single-pass result that **contradicts**
the 2-2.5× single-pass observation in writeup 4.

| Recipe | Task | Train depth | Effective depth | Accuracy |
|---|---|---:|---:|---:|
| pcc_hr_hybrid d=1024 (76M) | chain V=12 | 8 | **K = 2048 (256×)** | **1.00** |
| pcc_hr_hybrid d=1280 (118M) + noise | chain V=12 | 8 | **K = 4096 (512×)** | **1.00** |
| n=2 + noise | arithmetic reduction (N=12 operands) | 2 | **K = 2048 (1024×)** | **1.00** |
| pcc_hr_hybrid + iter + noise | modular sum P=13 | 12 | K = 12 (1×, fully solved) | **1.00** (vs 8% baseline) |
| **xloop + linear stabiliser** | **chain V=12** | **n_train** | **64 × n_train, SINGLE-PASS** | **1.00** |

Each of these results is a separate experiment with its own checkpoint and
its own evaluation harness. None of them is just a fresh seed of writeup 2.

## Result 1 — 4096 effective loops at d=1280 (512× train depth)

A pcc_hr_hybrid model (118M parameters, d=1280, 6 trained loops) with
noise injection during training, evaluated multi-pass at varying K:

| K (effective loops) | acc | n_passes |
|---:|---:|---:|
| 8 (trained) | 1.00 | 1 |
| 64 | 1.00 | 8 |
| 256 | 1.00 | 32 |
| 1024 | 1.00 | 128 |
| **4096** | **1.00** | **512** |

Single-pass accuracy at K=4096 with the same checkpoint: ≈ 0.27. The
multi-pass loop is doing the work; without it the latent trajectory drifts
off the decoder's manifold per writeup 4. With noise injection during
training the noisy state distribution at each loop covers a wider basin
around the deterministic trajectory, so the "snap-to-token" reset between
passes does not land in a region the decoder cannot read.

**Caveat (writeup 2's 256× was robust; this 4096× number is more
fragile):** the K=4096 d=1280 result was initially seed-dependent
(reversal observed at d=1280 hi-batch on a different seed). Adding noise
injection during training rescued it back to 1.00 at K=4096 on the
problem seed and held across the seeds we tested. Without noise it is not
robust. The recipe boundary is *batch-size + noise injection* — not just
"more depth."

## Result 2 — 1024× train depth on arithmetic reduction

Arithmetic reduction (N=12 operands, 5 operator types, modular result) is
the closest synthetic analogue we have to a recursive numerical
program. With n_train=2 + iter-target + noise, multi-pass reaches
**K=2048 at 100% accuracy** — 1024× the trained inference depth.

This is the biggest extrapolation ratio in the program. It uses arithmetic
reduction's specific property that each per-step rule (reduce one operator
node) is local and the state-update Jacobian is well-conditioned. The
same recipe at the *same* `n_train` does not reach K=2048 on chain V=12 (it
saturates around K=1024). The task structure matters; the recipe is not
universally 1024×.

## Result 3 — Accumulator task closure (modular sum) at trained depth

Writeup 4 flags modular sum `(a·x + b) mod P` as the boundary case where
the 2× single-pass observation breaks. The follow-up experiment was to
ask: does *any* recipe in this program solve modular sum at trained
depth?

| recipe | acc at trained K=8 |
|---|---:|
| pcc_hr_hybrid plain iter-target | 0.08 (chance ≈ 1/13) |
| + noise injection (R-recipe) | 0.51 |
| + noise + high-batch (192) | 1.00 |
| + noise + high-batch + n_loops=12 | **1.00 across K=4..12** |

Modular sum is *solvable* at the trained depth with the right recipe —
noise injection + a sufficiently large batch + extending `n_loops` from 8
to 12. Beyond the trained depth, the 2× ceiling **still holds** —
multi-pass on modular does not extrapolate cleanly, because the per-step
state-update lacks the fixed-point recovery property writeup 1 identifies
as necessary for iter-target extrapolation. The task is solved *at* its
trained depth but does not extrapolate, which is exactly the boundary
predicted in writeup 1.

## Result 4 — Single-pass 64× extrapolation with `xloop + linear stabiliser`

**This is the result that contradicts writeup 4's 2-2.5× single-pass
observation, and the contradiction is real.**

Writeup 4 establishes: under iter-target on chain V=12 at NL ∈ {4, 8, 16},
single-pass collapse depth is ≈ 2 · NL. This is a three-point linear-fit trend
for the pcc / pcc_hr_hybrid recipe family used in writeups 1-5, and the ratio
drifts down with larger NL — so it is a recipe-bounded empirical observation,
**not a law**. The result below shows just how recipe-bounded it is.

A different architectural variant — `xloop` (per-loop QKV bias + cross-loop
attention) augmented with a **linear stabiliser** that bounds per-step
displacement — produces **single-pass extrapolation to 64 × n_train** with
no multi-pass and no halt head:

| variant | trained depth | single-pass collapse depth | ratio |
|---|---:|---:|---:|
| pcc / pcc_hr_hybrid + iter-target (writeup 4) | 4–16 | ≈ 2 · NL | **2.0–2.5×** |
| **xloop + linear stabiliser + iter-target** | NL | **64 · NL** | **64×** |

The linear stabiliser is a per-channel decay on the residual:
`h_{r+1} = α · h_r + Block(h_r)`, with α learned per-channel. It bounds
the per-step state growth that writeup 4 attributed to drift; bounded
growth keeps the latent trajectory inside the decoder's training manifold
for ~30× more loops than the bare iter-target recipe.

**Implication for writeup 4:** the 2-2.5× single-pass ratio is **not a
universal property of recurrent depth + iter-target supervision**. It is a
property of *that specific recipe family*. A different architectural
choice (xloop + linear stabiliser) can move the single-pass collapse
depth by an order of magnitude. The mechanism question writeup 4 leaves
open — "what determines the 2× ratio?" — is sharpened, not closed: the
linear stabiliser provides one *constructive* answer (bounded per-step
displacement extends the ratio), but it is one architectural intervention
among many, and the audit's five hypotheses still do not predict which
recipes shift the ratio.

This single positive result motivates the path forward: **future
recurrent-depth work should look for additional architectural levers that
extend the single-pass envelope.** Multi-pass works at 4096× on chain
V=12, but single-pass at 64× with the right architecture is a much
cleaner deployment story (no per-pass plumbing, no token-level reset).

## Result 5 — One real-text head-to-head where the hybrid beats the dense baseline

NEGATIVE_RESULTS §1 reports that on 1 B tokens of FineWeb-Edu, dense
vanilla beats every recurrent variant on next-token val loss. That
*scoping* claim still stands. But the same architectures swap leadership
on a **reasoning-mix** training set at 1 B tokens:

| arch | train data | trained tokens | val loss on held-out reasoning-mix |
|---|---|---:|---:|
| vanilla d=2048 (8 blocks × 1 loop, 153M) | reasoning-mix | 1 B | **1.66** |
| **pcc_hr_hybrid d=1280 × 6 loops (118M)** | **reasoning-mix** | **1 B** | **1.59** |

A 23%-smaller hybrid recurrent model beats the dense baseline by 0.07
nats on a reasoning-flavored distribution at matched tokens. This is the
**first non-synthetic regime in the program where a recurrent architecture
outright wins.** It is on a curated reasoning-mix, not on raw web text,
so it does not retract NEGATIVE_RESULTS §1 — it scopes it. The honest
phrasing of the negative result becomes: *"vanilla wins on raw web text;
hybrid wins on reasoning-mix; the architecture-wins-where question is
distribution-dependent, not architecture-dependent in isolation."*

A separate run (HR(8,2) S2 chain, 3.15 B tokens, reasoning-mix) reached
**val_loss 1.239** — the best LM val loss observed in the program at this
scale. The trajectory across architectures + data mixes suggests the real
question for Phase 2 is not "vanilla vs. recurrent" but "**which data
mix actually rewards recurrent depth at scale**" — currently
reasoning-mix does, raw web does not.

## Why this matters

Three things change for the program after these results.

1. **The headline extrapolation number is 4096× (multi-pass) and 64×
   (single-pass), not 256×.** Writeup 2 was conservative; this writeup
   states the actual frontier.

2. **The modular-task boundary is more nuanced than writeup 1 implies.**
   The accumulator task is solvable *at* its trained depth with the right
   recipe; only extrapolation past trained depth respects the boundary.

3. **Writeup 4's 2-2.5× single-pass observation is recipe-bounded, not
   architecture-bounded.** A different architectural recipe shifts it
   by 30×. This is the constructive complement to the audit: the
   mechanism question is open in the *negative* direction (no
   explanatory mechanism survives) and *positive* (an architectural lever
   extends the envelope, suggesting a mechanism exists to be found).

The honest scope of the recurrent-depth thesis as of now: the substrate
works at small algorithmic scale (4096× multi-pass, 64× single-pass with
the right architecture, 31K-param production recipes from writeup 5), the
mechanism behind why is open (writeup 4), and the real-text question is
distribution-dependent (vanilla on raw text, hybrid on reasoning-mix).

## Method notes

- All multi-pass runs use `r_per = n_train` for the in-distribution
  envelope; `r_per > n_train` compounds per-pass error and is documented
  in NEGATIVE_RESULTS §6 scope limit.
- The xloop + linear stabiliser variant is a separately-trained
  architecture, not a recipe ablation on the pcc / pcc_hr_hybrid family.
  Reproducing the 64× single-pass result requires the specific stabiliser
  formulation, not just any per-loop normalisation.
- The reasoning-mix vs raw-text head-to-head uses the same optimizer,
  schedule, batch, and tokenizer. The only differences are the training
  data mix and the architecture.
- The K=4096 d=1280 result was seed-dependent without noise injection;
  with noise injection it held across the seeds we tested. The fragility
  is documented in NEGATIVE_RESULTS §6 as a scope limit, not papered over.
