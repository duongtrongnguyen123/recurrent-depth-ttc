"""Iter-target training on arith reduction with HETEROGENEOUS operand count.

Mirrors train_arith_long.train_arith_long_iter but draws each batch from
make_batch_arith_long_random_n. Tests Result AP follow-up: heterogeneous
N exposes the model to varying input lengths so multi-pass pass-i+1's
shorter input is in-distribution.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_arith_long import (ARITH_LONG_INPUT_LEN, ARITH_LONG_OUTPUT_START,
                                ARITH_LONG_SEQ_LEN,
                                make_batch_arith_long_random_n)
from .data_chain import VOCAB_SIZE
from .train_chain_iter_robust import RobustLoopedTransformer


def train_arith_long_hetN_iter(*, n_loops: int, d: int, n_heads: int, ff_mult: int,
                                 steps: int, batch_size: int, lr: float,
                                 eval_every: int, eval_size: int,
                                 device: str, seed: int,
                                 n_loops_train: int,
                                 n_loops_eval: int = 16,
                                 p_noise: float = 0.5,
                                 noise_alpha: float = 0.1,
                                 n_min: int = 3,
                                 n_max: int = 12,
                                 verbose: bool = True) -> dict:
    assert n_loops >= max(n_loops_train, n_loops_eval)
    torch.manual_seed(seed)
    model = RobustLoopedTransformer(vocab=VOCAB_SIZE, max_len=ARITH_LONG_SEQ_LEN,
                                     d=d, n_heads=n_heads, ff_mult=ff_mult,
                                     n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                             weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log: dict[str, list] = {"step": [], "train_loss": [],
                             "eval_acc_per_r": [], "eval_value_acc_per_r": []}
    t0 = time.time()

    for step in range(steps):
        tokens, iter_targets = make_batch_arith_long_random_n(
            batch_size, n_loops_train, n_min=n_min, n_max=n_max, device=device)
        all_logits = model.forward_all_loops_robust(
            tokens, n_loops=n_loops_train, p_noise=p_noise, noise_alpha=noise_alpha)
        losses = []
        for r in range(1, n_loops_train + 1):
            logits_r = all_logits[r, :,
                                   ARITH_LONG_OUTPUT_START:
                                   ARITH_LONG_OUTPUT_START + ARITH_LONG_INPUT_LEN, :]
            losses.append(F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)),
                                            iter_targets[r].reshape(-1)))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            acc, val_acc = _eval_arith_hetN(model, eval_size, n_loops_eval,
                                              n_min, n_max, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_acc_per_r"].append(acc)
            log["eval_value_acc_per_r"].append(val_acc)
            if verbose:
                key = sorted({1, n_loops_train, n_loops_train + 2,
                                n_loops_train + 4, 11, n_loops_eval}
                              & set(acc.keys()))
                msg = " ".join(f"r{r}=full{acc[r]:.2f}/val{val_acc[r]:.2f}"
                                for r in key)
                print(f"  step {step:>5}  loss {loss.item():.4f}  {msg}")

    return {
        "config": {"n_loops": n_loops, "d": d, "n_loops_train": n_loops_train,
                   "n_loops_eval": n_loops_eval, "steps": steps,
                   "batch_size": batch_size, "lr": lr,
                   "p_noise": p_noise, "noise_alpha": noise_alpha,
                   "n_min": n_min, "n_max": n_max},
        "params": model.num_params(),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


@torch.no_grad()
def _eval_arith_hetN(model, eval_size: int, n_loops_eval: int,
                       n_min: int, n_max: int, device: str
                       ) -> tuple[dict[int, float], dict[int, float]]:
    model.eval()
    tokens, iter_targets = make_batch_arith_long_random_n(
        eval_size, n_loops_eval, n_min=n_min, n_max=n_max, device=device)
    all_logits = model.forward_all_loops(tokens, n_loops=n_loops_eval)
    full: dict[int, float] = {}
    val: dict[int, float] = {}
    for r in range(1, n_loops_eval + 1):
        preds = all_logits[r, :,
                           ARITH_LONG_OUTPUT_START:
                           ARITH_LONG_OUTPUT_START + ARITH_LONG_INPUT_LEN, :].argmax(-1)
        target = iter_targets[r]
        full[r] = (preds == target).all(dim=-1).float().mean().item()
        val[r] = (preds[:, 0] == target[:, 0]).float().mean().item()
    model.train()
    return full, val
