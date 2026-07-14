"""Iterative-target chain training (Result L candidate).

Trains a looped transformer to compute f^r(start) at loop r, for every r in
[1, n_loops_train]. No depth token in the input. The model has no per-example
'how many hops to do' signal — only the loop count itself drives the output.

If the model internalizes the per-step rule ('apply f once per loop'), it
extrapolates: at inference loop r > n_loops_train the model continues iterating
and produces f^r(start) for arbitrary r. This is the central claim under test.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_chain import (ANSWER_POS_ITER, SEQ_LEN_ITER, V, VOCAB_SIZE,
                          make_batch_chain_iter)
from .model import LoopedTransformer


def train_chain_iter(*, n_loops: int, d: int, n_heads: int, ff_mult: int,
                     steps: int, batch_size: int, lr: float,
                     eval_every: int, eval_size: int,
                     device: str, seed: int,
                     n_loops_train: int,
                     n_loops_eval: int = 16,
                     verbose: bool = True) -> dict:
    """n_loops is the architectural depth (max loops the block can run).
    n_loops_train is how many of those are supervised during training (= n_loops typically).
    n_loops_eval is the inference-time loop budget — set > n_loops_train to test extrap.
    """
    assert n_loops >= max(n_loops_train, n_loops_eval), \
        "model n_loops must be >= max(train, eval) so forward_all_loops can step that far"
    torch.manual_seed(seed)
    model = LoopedTransformer(vocab=VOCAB_SIZE, max_len=SEQ_LEN_ITER, d=d,
                              n_heads=n_heads, ff_mult=ff_mult,
                              n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log: dict[str, list] = {"step": [], "train_loss": [], "eval_acc_per_r": []}
    t0 = time.time()

    for step in range(steps):
        tokens, iter_targets = make_batch_chain_iter(batch_size, n_loops_train, device)
        all_logits = model.forward_all_loops_grad(tokens, n_loops=n_loops_train)
        # all_logits: [n_loops_train+1, B, T, V]; supervise loops 1..n_loops_train
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
            acc_per_r = _evaluate_chain_iter(model, eval_size, n_loops_eval, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_acc_per_r"].append(acc_per_r)
            if verbose:
                # Print a few representative r values: one in train range, train-edge, OOD.
                key_rs = sorted({1, 2, n_loops_train // 2, n_loops_train,
                                 n_loops_train + 2, n_loops_train + 4,
                                 n_loops_eval} & set(acc_per_r.keys()))
                msg = " ".join(f"r{r}={acc_per_r[r]:.2f}" for r in key_rs)
                print(f"  step {step:>5}  loss {loss.item():.4f}  {msg}")

    return {
        "config": {"n_loops": n_loops, "d": d, "n_loops_train": n_loops_train,
                   "n_loops_eval": n_loops_eval,
                   "steps": steps, "batch_size": batch_size, "lr": lr},
        "params": model.num_params(),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


@torch.no_grad()
def _evaluate_chain_iter(model, eval_size: int, n_loops_eval: int,
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
