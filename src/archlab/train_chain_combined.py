"""Combined iter + halt chain training (Result M candidate).

Per-loop target schedule:
  loop r <= k   : f^r(start)  — per-step rule supervision (Result L)
  loop r > k    : f^k(start)  — hold-the-answer supervision (Result I, halt)

Hypothesis: the model learns BOTH the per-step iteration rule (extrapolates
past training k_max via Result L mechanism) AND the per-example halt behavior
(per-example calibration emerges via Result I mechanism). Same shared block.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_chain import V, VOCAB_SIZE, make_batch_chain_combined
from .model import LoopedTransformer

ANSWER_POS = V + 2  # same as default chain layout


def train_chain_combined(*, n_loops: int, d: int, n_heads: int, ff_mult: int,
                         steps: int, batch_size: int, lr: float,
                         eval_every: int, eval_size: int,
                         device: str, seed: int,
                         k_min: int, k_max: int,
                         n_loops_train: int,
                         n_loops_eval: int = 16,
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
        tokens, per_loop_targets, _k = make_batch_chain_combined(
            batch_size, k_min, k_max, n_loops_train, device)
        all_logits = model.forward_all_loops_grad(tokens, n_loops=n_loops_train)
        # all_logits : [n_loops_train+1, B, T, V]; supervise loop 1..n_loops_train at ANSWER_POS
        losses = []
        for r in range(1, n_loops_train + 1):
            losses.append(F.cross_entropy(
                all_logits[r, :, ANSWER_POS, :], per_loop_targets[r]))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            acc = _evaluate_combined(model, eval_size, k_min, k_max,
                                     n_loops_eval, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_acc_per_kr"].append(acc)
            if verbose:
                # report mean acc at r=k (the diagonal): if model has learned the iter+halt
                # behavior, pred at r=k should equal f^k for every k.
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
                   "steps": steps, "batch_size": batch_size, "lr": lr},
        "params": model.num_params(),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


@torch.no_grad()
def _evaluate_combined(model, eval_size: int, k_min: int, k_max: int,
                       n_loops_eval: int, device: str
                       ) -> dict[tuple[int, int], float]:
    """For each k in [k_min, k_max], evaluate per-r accuracy at predicting f^k(start)."""
    model.eval()
    per_k = max(eval_size // (k_max - k_min + 1), 256)
    out: dict[tuple[int, int], float] = {}
    for k in range(k_min, k_max + 1):
        tokens, per_loop_targets, _ = make_batch_chain_combined(
            per_k, k, k, n_loops_eval, device)
        logits = model.forward_all_loops(tokens, n_loops=n_loops_eval)
        target_at_k = per_loop_targets[k]  # = f^k(start)
        for r in range(0, n_loops_eval + 1):
            preds = logits[r, :, ANSWER_POS, :].argmax(-1)
            out[(k, r)] = (preds == target_at_k).float().mean().item()
    model.train()
    return out
