# 3. Controllable recurrent solver: composition via orchestration, not architecture

## The failure that motivated this

A single `pcc_hr_hybrid` recurrent model — the strongest single
recurrent-depth model in this program — trained **end-to-end** on a
*composed* task `chain ∘ arithmetic` (do *k_chain* pointer hops, then
*k_arith* modular reductions on the result):

> **0% accuracy at the trained depth.** 7% at r=4.

Diagnosis: the architecture has one shared block and one goal register.
There is no mechanism to *switch programs* mid-forward. Iter-target
supervision asks it to predict arithmetic partial-sums at *every* loop —
including loops where the chain phase has not finished. The supervision is
internally inconsistent, so the model learns neither phase cleanly.

This looked like a fundamental limit of recurrent depth for compositional
reasoning. It is not. It is a training-recipe choice.

## The fix: separate the operator from the program

Do not ask one model to be both the operator *and* the program. Instead:

1. **Train each primitive separately**, each with full gradient budget and
   consistent iter-target supervision (chain-only model sees only chain
   targets; arith-only model sees only arith targets). Recipe: PCC+HR hybrid
   block + per-loop QKV bias + iter-target + noise, ~12K steps, ~76M params,
   ~15 min each on an RTX 5090.

2. **Orchestrate at inference in plain Python** (a stand-in for an LLM /
   planner in production): call `chain_model` for *k_chain* hops, decode its
   output token to an integer state, feed that as a fresh input to
   `arith_model` for *k_arith* reductions.

The hand-off is **in tokens, not in latent space** — explicit, lossless,
and program-structure-agnostic.

## Result

Each primitive alone: 1.000 across r=1..8.

Composition, evaluated on a 4×4 grid of (k_chain, k_arith) ∈ {2,4,6,8}²:

| | k_arith=2 | 4 | 6 | 8 |
|---|---|---|---|---|
| **k_chain=2** | 1.000 | 1.000 | 1.000 | 1.000 |
| **4** | 1.000 | 1.000 | 1.000 | 1.000 |
| **6** | 1.000 | 1.000 | 1.000 | 1.000 |
| **8** | 1.000 | 1.000 | 1.000 | 1.000 |

16/16 cells at 100% — effectively 16-step compositional reasoning at perfect
accuracy, vs **0%** for the monolithic end-to-end model.

A stress test generalizes this: **3 primitives** (chain = table lookup,
arith = modular add, modular = modular affine `a·x+b mod P`), **6 ordered
pairs**, 4×4 depth grid each = **96/96 composition cells at 100%**. No
primitive ordering, depth combination, or per-step-rule complexity disrupts
it. The argmax state hand-off is lossless across every primitive boundary.

## Why this works

| problem in the monolithic model | how separation fixes it |
|---|---|
| inconsistent cross-phase supervision | each model sees only its own primitive's targets |
| no mechanism to switch programs | the orchestrator switches; the model never has to |
| program structure baked into weights | structure lives in Python, recombined ad hoc |

Adding a new primitive = training one more small recurrent block. No
retraining of existing primitives, no retraining for new compositions.

## Why this matters

This is the minimal, concrete instance of the pattern frontier labs are
converging on: **a reasoning system is an atomic neural operator plus an
external program scheduler**, not one monolith asked to do everything in a
single forward pass. The recurrent-depth block is an unusually clean fit for
the "atomic operator" role because (writeups 1–2) its compute depth is
controllable and its iteration is genuine. The contribution here is the
controlled demonstration that **moving composition from the architecture to
the orchestrator turns 0% into 100%** — and that this is a recipe choice, not
a capability ceiling.

Reproduction: two/three `SpecializedPccHrHybrid` models (`src/model.py`,
HR + per-loop QKV bias), each trained alone with dense iter-target + p_noise
0.3; Python orchestrator decodes argmax → integer state → next primitive
input. d=1024, n_loops=8, vocab=27, batch 256–512, ~15 min/primitive on one
RTX 5090.
