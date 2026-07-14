"""Soft-target combined iter+halt training (Result P candidate).

Per-loop target is a SOFT mixture of f^r(start) (iter) and f^k(start) (halt):
    target_distribution[r] = w(r,k) * onehot(f^r) + (1 - w(r,k)) * onehot(f^k)
where w decays smoothly with (r - k):
    w(r, k) = sigmoid((k - r) / temperature)
- r << k:  w ≈ 1 → target is f^r (iterate)
- r ~  k:  w ≈ 0.5 → mixture (transition)
- r >> k:  w ≈ 0 → target is f^k (halt)

Hypothesis: smooth transition between iter and halt regimes lets the model
learn BOTH per-step rule (low r) AND per-example stabilization at user's k
(high r), without breaking calibration as the hard-switch Result M did.

Calibration test: at intermediate r, does the model's confidence reflect
'have we reached k yet?' (a per-example signal that varies with k)?
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_chain import V, VOCAB_SIZE, make_batch_chain_combined
from .model import LoopedTransformer

ANSWER_POS = V + 2  # same default chain layout


def train_chain_soft(*, n_loops: int, d: int, n_heads: int, ff_mult: int,
                     steps: int, batch_size: int, lr: float,
                     eval_every: int, eval_size: int,
                     device: str, seed: int,
                     k_min: int, k_max: int,
                     n_loops_train: int,
                     n_loops_eval: int = 16,
                     soft_temperature: float = 1.0,
                     verbose: bool = True) -> dict:
    assert n_loops >= max(n_loops_train, n_loops_eval)
    torch.manual_seed(seed)
    max_len = V + 4
    model = LoopedTransformer(vocab=VOCAB_SIZE, max_len=max_len, d=d,
                              n_heads=n_heads, ff_mult=ff_mult,
                              n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log: dict[str, list] = {"step": [], "train_loss": [], "eval_acc_per_kr": []}
    t0 = time.time()

    for step in range(steps):
        # make_batch_chain_combined gives per_loop_targets[r] = f^r if r<=k else f^k
        # We need both f^r AND f^k at each (r, k) to build the soft mixture.
        # Re-derive: f^r is per_loop_targets[r] when r <= k_per_example, else
        # we need to recompute. Simpler: also call make_batch_chain_iter with
        # n_steps = n_loops_train. But that's a different (table, start) draw.
        # Solution: derive both from a single call to a new helper:
        tokens, targets_iter, targets_halt, k_per_ex = _make_batch_with_both(
            batch_size, k_min, k_max, n_loops_train, device)
        # targets_iter : [n_loops_train+1, B] = f^r(start) for r=0..n_loops_train
        # targets_halt : [B]                  = f^k(start) per example
        # k_per_ex     : [B]

        all_logits = model.forward_all_loops_grad(tokens, n_loops=n_loops_train)
        # all_logits : [n_loops_train+1, B, T, V]

        losses = []
        log_p_full = F.log_softmax(all_logits[:, :, ANSWER_POS, :], dim=-1)
        # log_p_full : [n_loops_train+1, B, V]
        for r in range(1, n_loops_train + 1):
            # Mixture weight per example
            r_minus_k = float(r) - k_per_ex.float()                       # [B]
            w = torch.sigmoid(-r_minus_k / soft_temperature)              # [B], in [0,1]
            # iter target loss component
            iter_loss = F.nll_loss(log_p_full[r], targets_iter[r],
                                    reduction="none")                       # [B]
            halt_loss = F.nll_loss(log_p_full[r], targets_halt,
                                    reduction="none")                       # [B]
            losses.append((w * iter_loss + (1 - w) * halt_loss).mean())
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            acc = _eval_soft(model, eval_size, k_min, k_max, n_loops_eval, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_acc_per_kr"].append(acc)
            if verbose:
                diag = [acc[(k, k)] for k in range(k_min, k_max + 1)
                        if (k, k) in acc]
                msg = " ".join(f"k{k}@r{k}={acc[(k, k)]:.2f}"
                               for k in range(k_min, k_max + 1)
                               if (k, k) in acc)
                print(f"  step {step:>5}  loss {loss.item():.4f}  diag mean "
                      f"{sum(diag)/len(diag):.2f}  {msg}")

    return {
        "config": {"n_loops": n_loops, "d": d, "k_min": k_min, "k_max": k_max,
                   "n_loops_train": n_loops_train, "n_loops_eval": n_loops_eval,
                   "soft_temperature": soft_temperature,
                   "steps": steps, "batch_size": batch_size, "lr": lr},
        "params": model.num_params(),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


def _make_batch_with_both(batch_size, k_min, k_max, n_loops_train, device):
    """Same as make_batch_chain_combined but also returns f^r(start) for all r
    (the iter targets) AND f^k(start) per-example (the halt target)."""
    from .data_chain import DEPTH_OFFSET, EQ, PAD
    table = torch.randint(0, V, (batch_size, V), device=device)
    start = torch.randint(0, V, (batch_size,), device=device)
    k = torch.randint(k_min, k_max + 1, (batch_size,), device=device)

    cur = start.clone()
    iters = [start.clone()]
    for _ in range(n_loops_train):
        cur = table[torch.arange(batch_size, device=device), cur]
        iters.append(cur.clone())
    iter_states = torch.stack(iters, dim=0)                    # [n_loops_train+1, B]

    bs_arange = torch.arange(batch_size, device=device)
    answer_at_k = iter_states[k.clamp(max=n_loops_train), bs_arange]   # [B]

    seq_len = V + 4
    tokens = torch.full((batch_size, seq_len), PAD, dtype=torch.long, device=device)
    tokens[:, :V] = table
    tokens[:, V] = start
    tokens[:, V + 1] = (k + DEPTH_OFFSET).long()
    tokens[:, V + 2] = EQ
    tokens[:, V + 3] = answer_at_k

    return tokens, iter_states, answer_at_k, k


@torch.no_grad()
def _eval_soft(model, eval_size, k_min, k_max, n_loops_eval, device):
    """Per-(r, k) accuracy at predicting f^k(start) (the user's target answer)."""
    model.eval()
    per_k = max(eval_size // (k_max - k_min + 1), 256)
    out = {}
    for k in range(k_min, k_max + 1):
        tokens, _iter_targets, target_at_k, _ = _make_batch_with_both(
            per_k, k, k, n_loops_eval, device)
        logits = model.forward_all_loops(tokens, n_loops=n_loops_eval)
        for r in range(0, n_loops_eval + 1):
            preds = logits[r, :, ANSWER_POS, :].argmax(-1)
            out[(k, r)] = (preds == target_at_k).float().mean().item()
    model.train()
    return out
