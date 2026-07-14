"""Halt-eval helpers for the chain-lookup task.

Mirrors `halt_eval.py` (which is wired to the addition task) but uses the
chain-task data generator and the single-token answer at position V+2.

Reuses `halt_eval.fixed_r_accuracy` and `halt_eval.adaptive_threshold_accuracy`
from the addition module — those are agnostic to the data source given a bundle
of (per_loop_logits, targets, answer_start, answer_len).

Adds chain-specific helpers:
  - `collect_per_loop_chain`  — runs forward_all_loops on heterogeneous-k examples
  - `fixed_r_per_k`           — per-(r, k) accuracy table (Phase 0 Result G shape)
  - `auc_correct_vs_wrong_at_r` — within-r AUC of entropy/margin (the diagnostic
                                   Phase 0 left out)
  - `oracle_adaptive`         — halt at the smallest r where prediction is correct;
                                  upper bound on any trajectory-only halter
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data_chain import (ANSWER_POS_UNARY, V, make_batch_chain,
                          make_batch_chain_unary)

ANSWER_POS = V + 2  # position predicting the answer in the default chain layout


@torch.no_grad()
def collect_per_loop_chain(model, n_eval: int, k_min: int, k_max: int,
                           max_loops: int, batch_size: int = 512,
                           device: str = "cuda",
                           unary_depth: bool = False) -> dict:
    """Run forward_all_loops on n_eval heterogeneous-k chain examples.

    Returns a bundle dict:
      per_loop_logits : [max_loops+1, N, 1, V_vocab]   (only the answer position)
      targets         : [N, T_full]                     (targets[:, answer_start] is the answer)
      k_values        : [N]                             (per-example k, 1..k_max)
      answer_start    : ANSWER_POS or ANSWER_POS_UNARY
      answer_len      : 1
    """
    if unary_depth:
        batch_fn = make_batch_chain_unary
        answer_pos = ANSWER_POS_UNARY
    else:
        batch_fn = make_batch_chain
        answer_pos = ANSWER_POS
    model.eval()
    bundles_logits, bundles_targets, bundles_k = [], [], []
    done = 0
    while done < n_eval:
        bs = min(batch_size, n_eval - done)
        tokens, targets, _mask, k = batch_fn(bs, k_min, k_max, device)
        logits_all = model.forward_all_loops(tokens, n_loops=max_loops)
        ans_logits = logits_all[:, :, answer_pos:answer_pos + 1, :].cpu()
        bundles_logits.append(ans_logits)
        bundles_targets.append(targets.cpu())
        bundles_k.append(k.cpu())
        done += bs
    return {
        "per_loop_logits": torch.cat(bundles_logits, dim=1),
        "targets":         torch.cat(bundles_targets, dim=0),
        "k_values":        torch.cat(bundles_k, dim=0),
        "answer_start":    answer_pos,
        "answer_len":      1,
    }


def fixed_r_per_k(bundle: dict) -> dict:
    """Per-(r, k) accuracy: replicates the Phase 0 Result G chain table.
    Returns {"per_r_overall": {r: acc}, "per_kr": {k: {r: acc}}}.
    """
    logits = bundle["per_loop_logits"]                                     # [R, N, 1, V]
    ans = bundle["answer_start"]
    target = bundle["targets"][:, ans:ans + 1]                              # [N, 1]
    ks = bundle["k_values"]                                                 # [N]
    R = logits.shape[0]
    out_overall: dict[int, float] = {}
    out_per_kr: dict[int, dict[int, float]] = {}
    unique_k = sorted(int(v) for v in torch.unique(ks).tolist())
    for k in unique_k:
        out_per_kr[k] = {}
    for r in range(R):
        preds = logits[r].argmax(-1)                  # [N, 1]
        ok = (preds == target).all(dim=1)             # [N]
        out_overall[r] = ok.float().mean().item()
        for k in unique_k:
            sel = (ks == k)
            out_per_kr[k][r] = ok[sel].float().mean().item() if sel.any() else float("nan")
    return {"per_r_overall": out_overall, "per_kr": out_per_kr,
            "k_values": unique_k}


def _signal_per_loop(per_loop_logits: torch.Tensor, signal: str) -> torch.Tensor:
    """Compute per-(loop, example) confidence signal at the answer position.
    Returns [R, N]: smaller entropy / larger margin = more confident.
    For uniform downstream handling we return the *raw* signal (entropy or margin);
    callers know the comparison direction.
    """
    if signal == "entropy":
        p = F.softmax(per_loop_logits, dim=-1)
        ent = -(p * (p + 1e-12).log()).sum(dim=-1)            # [R, N, 1]
        return ent.mean(dim=-1)                               # [R, N]
    if signal == "margin":
        srt = per_loop_logits.sort(dim=-1, descending=True).values
        return (srt[..., 0] - srt[..., 1]).mean(dim=-1)       # [R, N]
    raise ValueError(f"unknown signal: {signal}")


def auc_correct_vs_wrong_at_r(bundle: dict, signal: str,
                               min_r: int = 1) -> dict[int, dict]:
    """For each loop r in [min_r, R-1], compute the AUC of `signal[r]` for
    discriminating examples that are correct at r vs examples that are wrong at r.

    This is the diagnostic Phase 0 left untested: under aux + heterogeneous-k
    training, does the trajectory signal at intermediate r separate
    'this example is now correct' from 'this example is still wrong'?

    AUC > 0.5 at intermediate r means a halter using this signal could plausibly
    save compute. AUC ≈ 0.5 means halting is dead even under aux+het training.

    For entropy: lower-is-better, so we compute AUC of (-entropy) vs correct.
    For margin: higher-is-better, AUC of margin vs correct.
    Both should yield AUC > 0.5 if the signal separates correct from wrong.

    Returns {r: {auc, n_pos, n_neg, acc_at_r}}.
    """
    logits = bundle["per_loop_logits"]                                     # [R, N, 1, V]
    ans = bundle["answer_start"]
    target = bundle["targets"][:, ans:ans + 1]                              # [N, 1]
    R = logits.shape[0]
    sig = _signal_per_loop(logits, signal)                     # [R, N]
    if signal == "entropy":
        score = -sig                                            # higher = more confident
    else:
        score = sig

    out: dict[int, dict] = {}
    for r in range(min_r, R):
        preds = logits[r].argmax(-1)                            # [N, 1]
        correct = (preds == target).all(dim=1)                  # [N]
        n_pos = int(correct.sum().item())
        n_neg = int((~correct).sum().item())
        if n_pos == 0 or n_neg == 0:
            out[r] = {"auc": float("nan"), "n_pos": n_pos, "n_neg": n_neg,
                      "acc_at_r": float(n_pos) / max(n_pos + n_neg, 1)}
            continue
        out[r] = {
            "auc":      _roc_auc(score[r], correct),
            "n_pos":    n_pos,
            "n_neg":    n_neg,
            "acc_at_r": n_pos / (n_pos + n_neg),
        }
    return out


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann-Whitney U-style AUC. scores: [N] float, labels: [N] bool/0-1."""
    s = scores.detach().double().flatten()
    y = labels.detach().to(torch.bool).flatten()
    pos = s[y]
    neg = s[~y]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    # Pairwise: count positives ranked above negatives.
    # Tie counts 0.5. O(P*N) memory — fine for N <= ~10k.
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    above = (diff > 0).sum().item()
    ties = (diff == 0).sum().item()
    return (above + 0.5 * ties) / (pos.numel() * neg.numel())


def oracle_adaptive(bundle: dict, min_r: int = 1) -> dict:
    """Per-example oracle: halt at smallest r >= min_r where preds[r] == answer.
    If no such r exists, use the final r (R-1).

    This is the trajectory-only upper bound on adaptive halting — it knows the
    answer. Any practical halter must use only the trajectory signal and so
    cannot exceed this.
    """
    logits = bundle["per_loop_logits"]                                     # [R, N, 1, V]
    ans = bundle["answer_start"]
    target = bundle["targets"][:, ans:ans + 1]                              # [N, 1]
    ks = bundle["k_values"]
    R, N = logits.shape[0], logits.shape[1]
    halt_at = torch.full((N,), R - 1, dtype=torch.long)
    for r in range(min_r, R):
        preds_r = logits[r].argmax(-1)                          # [N, 1]
        ok_r = (preds_r == target).all(dim=1)                   # [N]
        # For each example that hasn't halted yet AND is correct now, halt here.
        not_yet = (halt_at == R - 1)
        new_halt = ok_r & not_yet
        halt_at[new_halt] = r
    # Recompute predictions at halt time and accuracy.
    preds = torch.zeros((N, 1), dtype=torch.long)
    for r in range(R):
        sel = (halt_at == r)
        if sel.any():
            preds[sel] = logits[r, sel].argmax(-1)
    acc = (preds == target).all(dim=1).float().mean().item()
    avg_loops = halt_at.float().mean().item()
    # Per-k breakdown.
    per_k: dict[int, dict] = {}
    for k in sorted(int(v) for v in torch.unique(ks).tolist()):
        sel = (ks == k)
        per_k[k] = {
            "avg_loops": halt_at[sel].float().mean().item(),
            "acc":       (preds[sel] == target[sel]).all(dim=1).float().mean().item(),
            "n":         int(sel.sum().item()),
        }
    return {"avg_loops": avg_loops, "acc": acc, "per_k": per_k}
