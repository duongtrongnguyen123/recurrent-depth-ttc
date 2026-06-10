# 4. The mechanism audit: a clean observation with no surviving explanation

## The setup

Writeup 1 gave a clean empirical dissociation: under iter-target
supervision, the same looped block walls vs. extrapolates depending only on
the loss. This writeup pins down **how far** the extrapolation goes
single-pass, and audits **why** — testing five concrete mechanism
hypotheses for the observation.

## The single-pass observation: ≈ 2–2.5 × NL

Train iter-target chain (V=12) at three trained-depth points and run
single-pass inference at every `r`. Define `collapse_depth = first r where
accuracy < 95%`. Measured:

| NL (train depth) | collapse depth | ratio |
|---:|---:|---:|
| 4  | ≈ 10 | 2.50 |
| 8  | ≈ 20 | 2.50 |
| 16 | ≈ 33 | 2.06 |

Linear fit through origin: **slope ≈ 2.06**. Three points, so this is a
clean *empirical observation*, not a proven law — and the ratio drifts
slightly downward as NL grows, suggesting the true asymptote may be below 2.

This is a *single-pass* statement. Multi-pass inference (writeup 2) extends
effective depth by orders of magnitude — the 2.06 here is the *latent
trajectory's* useful range before drift carries the state off the decoder's
training manifold.

Counter-example that scopes the observation: **modular sum** (running
accumulator structure) breaks it entirely — collapse at NL, not at 2·NL.
The observation is bounded to tasks with **position-invariant per-step
rules**; accumulator structure with no fixed-point recovery falls outside.

## The natural mechanism question

*Why is the ratio ≈ 2–2.5, rather than 1.5 or 3? What in the recurrent
block's dynamics determines this number?*

Five pre-registered mechanism hypotheses were tested — each with a
specific prediction, each measured on the same checkpoints, each verified
on two independent compute streams. All five were falsified.

### H1 — Trajectory geometry (angular spread)

*Hypothesis:* tasks that collapse earlier should show larger angular spread
in the hidden-state trajectory near the collapse depth.

*Test:* measure trajectory dispersion (`ρ_ang`) across tasks; correlate with
collapse depth.

*Result:* the task with **earliest collapse** had the **smallest** angular
spread (most contractive). Sign reversed from prediction. **Falsified.**

### H2 — Training-time state-space coverage

*Hypothesis:* collapse at `r > NL` is exposure bias — the model never saw
hidden states from `r > NL` regions of latent space during training, so it
decodes garbage there.

*Test:* measure overlap between the train-time hidden-state distribution
and the eval-time distribution at various r.

*Result:* support overlap = **1.0** for every task at every r. No coverage
gap exists, yet collapse still occurs. **Falsified.**

### H3 — LayerNorm-induced drift

*Hypothesis:* the per-loop LayerNorm rescaling causes the per-step
displacement to compound, and this one mechanism alone predicts collapse.

*Test:* derive collapse-depth prediction from LayerNorm dynamics alone,
compare to per-task measured collapse.

*Result:* the prediction is task-invariant — it gives the same collapse for
chain (collapse ≈ 2·NL) and modular sum (collapse ≈ NL). It cannot
discriminate the two regimes the data clearly shows. **Falsified by lack
of discriminative power.**

### H4 — Contraction is causal (the most important test)

*Hypothesis:* iter-target produces contractive trajectories (writeup 1's
plot); contraction *causes* extrapolation. Forcing contraction by other
means should therefore *increase* the extrapolation horizon.

*Test:* add a regularizer that explicitly penalizes per-step displacement,
forcing the block toward a contractive operator. Retrain; measure
extrapolation horizon.

*Result:* the forced-contraction model's extrapolation horizon was
**halved**, not improved. Contraction is therefore a **symptom that
co-occurs with iter-target's effect**, not the causal link to
extrapolation. **Falsified — the causal arrow we expected goes the wrong
way.**

This is the audit's sharpest result. It collapses the most intuitive
"mechanism" story for iter-target's extrapolation property.

### H5 — Decoder Jacobian shape at the answer position

*Hypothesis:* the spectral properties of `∂(logit)/∂h` at the answer
position predict when the decoder reads the state correctly.

*Test:* measure Jacobian rank / conditioning / top singular values at every
loop r; correlate with per-r accuracy.

*Result:* the relationship is non-monotonic and non-predictive — Jacobian
"good" cases collapse, "bad" cases sometimes don't. **Falsified.**

## What survived the audit

After five pre-registered tests on two independent streams:

- **The empirical observation** (≈ 2–2.5 × NL on chain V=12) holds for this
  recipe family, but is recipe-bounded — writeup 6 breaks it to 64× with a
  different architectural variant. Treat 2× as a property of the recipe, not
  of recurrent depth.
- **The dissociation** (writeup 1: supervision sets dynamical regime) is
  robust.
- **The mechanism question** ("what dynamical property of the block sets
  the 2× ratio?") remains open.

Three claims that did *not* survive but were tempting:

1. "Contraction is the mechanism" — directly falsified by H4 (forcing it
   makes things worse).
2. "Coverage gap is the mechanism" — falsified by H2 (no gap exists).
3. "Trajectory geometry alone predicts capability" — falsified by H1
   (sign reversed).

## Why this audit matters

The recurrent-depth literature is **saturated with descriptive mechanism
stories** — trajectory shapes, rotational structure, fixed-point claims —
that are presented as explanations without falsification tests. When those
stories are made specific enough to be tested (concrete prediction +
isolatable measurement), five of the most-cited ones do not survive on
this clean control task.

The point is not that mechanism research is impossible. The point is that
**a descriptive observation is not a mechanism claim**, and the field's
common practice of inferring mechanism from trajectory plots is producing
unfalsifiable results.

The discipline applied here:

1. **Pre-register the prediction** before running the test.
2. **Test on two independent compute streams** to rule out artifact.
3. **State a what-would-falsify-this condition.** If you cannot, the claim
   is not yet science.

All five hypotheses were filed under "falsified" — not as a negative
result to bury, but as the actual product of doing mechanism research with
discipline. The 2-2.5× observation is a real empirical regularity in
search of an explanation, and that is the honest state of the question.

## Method notes

- Single-pass collapse measured at greedy argmax, 4 batches × 128 examples
  per `r`, threshold 95% accuracy.
- H4 (forced contraction): regularizer `λ · ‖h_{r+1} - h_{r}‖²` with
  λ ∈ {0.01, 0.1, 1.0}, all reduce extrapolation horizon.
- Each hypothesis test independently re-run on a separate compute stream;
  verdicts agreed in all five cases.
- Code and configs reproducible from the recipe in `src/model.py` and the
  noise/iter-target switches in the chain training scripts.
