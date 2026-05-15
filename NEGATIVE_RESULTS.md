# Negative results, retractions, and scope limits

A research program is only as trustworthy as the claims it *withdraws*. These
are here at full prominence, not in a footnote.

## 1. The parameter-efficiency claim does NOT survive to real-text scale

The headline appeal of recurrent depth is parameter efficiency: a 1-block ×
8-loop model has an 8-block model's per-step FLOPs at 1/8 the parameters. On
**synthetic algorithmic tasks** the looped model matches or beats width at a
fraction of the parameters.

On **1 B tokens of real FineWeb-Edu text** it does not. Final validation
loss (1 B tokens, d=1024):

| arch | val_loss | params | gap to vanilla |
|---|---|---|---|
| **vanilla** (8 blocks × 1 loop) | **3.856** | 153M | — |
| pcc (Huginn-shape hybrid) | 3.964 | 115M | +0.108 |
| pure looped | 4.388 | 64M | +0.532 |
| looped + aux loss | 4.493 | 64M | +0.637 |

A dense transformer wins by a clear margin, and — reversing the synthetic
finding — auxiliary (per-loop) loss is the *worst* variant on real text, not
the best. Working hypothesis: aux loss helps an under-trained model and
becomes a capacity tax once enough training compute arrives. The
parameter-efficiency win is, on current evidence, **confined to algorithmic
tasks with explicit per-step structure.** Any claim otherwise would be
unsupported. The ongoing scaling arm exists specifically to find where (if
anywhere) this reverses.

## 2. Retracted: a claimed architecture win that was a capacity artifact

An early result reported a hybrid recurrent architecture beating a baseline.
A **parameter-matched** control showed the comparison was not
parameter-matched: the "winning" architecture had ~1.9× the effective
capacity. With capacity equalized, the architectural gain vanished. The
result was retracted and the matched comparison became a standing
methodological rule: **no architecture claim without a parameter-matched
control, stated explicitly.**

## 3. Retracted then re-derived: a halting signal that was a tokenization artifact

A result claimed that hidden-state stability predicts correctness ("stable =
correct"). Before publication, the prompt format was found to contain a
trailing-space artifact (`"= "`) that made stability trivially correlate with
the answer position rather than with computation. The original claim was
**retracted**. Re-derived cleanly, the *true* finding is weaker and more
precise: top-1 stability alone is at chance (AUC≈0.5); **entropy and logit
margin** discriminate stable-correct from stable-wrong at AUC≈0.78 pooled,
and ≈0.94 per-r under the aux + heterogeneous-depth recipe. The honest
version is the one that shipped.

## 4. Null result: larger micro-batch is not free throughput

Intuition said a larger micro-batch (holding effective batch fixed via
gradient accumulation) would speed up the 50 B-token training chains by
amortizing accumulation overhead. Benchmarked properly on identical
hardware:

| arch | mb=8 tok/s | mb=12 tok/s | Δ |
|---|---|---|---|
| pcc (6 loops) | 38.6K | 37.5K | **−3%** (slower) + 18 GB memory |
| xloop (4 loops) | 30.6K | 31.7K | +3.5% + 16 GB memory |

The GPU was already compute-bound at mb=8 (100% util). The larger
micro-batch bought either a regression or a +3.5% that did not justify the
memory cost or the eff-batch confound. Dropped. Cost of finding out:
~24 GPU-minutes. Cheaper than committing a multi-day run to a wrong assumption.

## 5. Scope limits that bound the positive results

- **Algorithmic, small-scale.** The clean extrapolation / multi-pass /
  composition results are at d ≤ 1280, vocab ≤ 27, synthetic generators.
  They are seed-pinned and reproduced, but they are *not* claims about
  natural language.
- **The composition recipe has a per-step complexity ceiling.** Iter-target
  extrapolation transfers to chain (table lookup) and ListOps reduction but
  **fails for modular affine** `(a·x+b) mod P` — the per-step rule must fit
  in ≤1 transformer block of computation. The boundary is characterized, not
  papered over.
- **Multi-pass error compounds** when per-pass internal loops exceed the
  trained loop count (`r_per > n_loops_train`). The safe operating regime is
  stated alongside the headline number, not omitted.

## Why this page exists

Every positive result in this repository has a stated boundary, a
parameter-matched control, or a falsification test that passed. The negative
results are not damage control — they are the part of the work that makes the
positive part believable.
