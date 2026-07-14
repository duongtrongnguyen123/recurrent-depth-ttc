"""PCC + iter-target + noise injection on chain (Result Y).

Extends Result X (PCC + iter) with Result R (noise injection): tests if noise
extends PCC's extrap range like it did for pure-looped (88.6% → 78% at r=32).
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_chain import (ANSWER_POS_ITER, SEQ_LEN_ITER, V, VOCAB_SIZE,
                          make_batch_chain_iter)
from .model_pcc import PCCLoopedTransformer


def train_chain_pcc_robust_iter(*, n_loops: int, d: int, n_heads: int,
                                  ff_mult: int, n_prelude: int, n_coda: int,
                                  steps: int, batch_size: int, lr: float,
                                  eval_every: int, eval_size: int,
                                  device: str, seed: int,
                                  n_loops_train: int,
                                  n_loops_eval: int = 32,
                                  p_noise: float = 0.5,
                                  noise_alpha: float = 0.1,
                                  verbose: bool = True) -> dict:
    assert n_loops >= max(n_loops_train, n_loops_eval)
    torch.manual_seed(seed)
    model = PCCLoopedTransformer(vocab=VOCAB_SIZE, max_len=SEQ_LEN_ITER, d=d,
                                  n_heads=n_heads, ff_mult=ff_mult,
                                  n_loops=n_loops,
                                  n_prelude=n_prelude, n_coda=n_coda).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                             weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log: dict[str, list] = {"step": [], "train_loss": [], "eval_acc_per_r": []}
    t0 = time.time()

    for step in range(steps):
        tokens, iter_targets = make_batch_chain_iter(batch_size, n_loops_train, device)
        all_logits = model.forward_all_loops_robust(
            tokens, n_loops=n_loops_train,
            p_noise=p_noise, noise_alpha=noise_alpha)
        losses = []
        for r in range(1, n_loops_train + 1):
            losses.append(F.cross_entropy(
                all_logits[r, :, ANSWER_POS_ITER, :], iter_targets[r]))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            acc = _eval_pcc_robust(model, eval_size, n_loops_eval, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_acc_per_r"].append(acc)
            if verbose:
                key = sorted({1, 2, 4, n_loops_train, n_loops_train + 4,
                              n_loops_train + 8, n_loops_eval} & set(acc.keys()))
                msg = " ".join(f"r{r}={acc[r]:.2f}" for r in key)
                print(f"  step {step:>5}  loss {loss.item():.4f}  {msg}")

    return {
        "config": {"n_loops": n_loops, "d": d, "n_loops_train": n_loops_train,
                   "n_loops_eval": n_loops_eval, "steps": steps,
                   "batch_size": batch_size, "lr": lr,
                   "n_prelude": n_prelude, "n_coda": n_coda,
                   "p_noise": p_noise, "noise_alpha": noise_alpha},
        "params": model.num_params(),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


@torch.no_grad()
def _eval_pcc_robust(model, eval_size: int, n_loops_eval: int,
                      device: str) -> dict[int, float]:
    model.eval()
    tokens, iter_targets = make_batch_chain_iter(eval_size, n_loops_eval, device)
    all_logits = model.forward_all_loops(tokens, n_loops=n_loops_eval)
    out: dict[int, float] = {}
    for r in range(1, n_loops_eval + 1):
        preds = all_logits[r, :, ANSWER_POS_ITER, :].argmax(-1)
        out[r] = (preds == iter_targets[r]).float().mean().item()
    model.train()
    return out
