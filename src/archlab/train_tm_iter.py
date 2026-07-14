"""Train iter-target on TM-step (universal computation test).

Per-loop loss: cross-entropy at each of the W tape positions, summed.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_chain import VOCAB_SIZE
from .data_tm import TM_SEQ_LEN, TM_W, make_batch_tm_iter
from .train_chain_iter_robust import RobustLoopedTransformer


def train_tm_iter(*, n_loops, d, n_heads, ff_mult, steps, batch_size, lr,
                   eval_every, eval_size, device, seed,
                   n_loops_train, n_loops_eval=24,
                   p_noise=0.5, noise_alpha=0.1, verbose=True):
    assert n_loops >= max(n_loops_train, n_loops_eval)
    torch.manual_seed(seed)
    model = RobustLoopedTransformer(vocab=VOCAB_SIZE, max_len=TM_SEQ_LEN,
                                     d=d, n_heads=n_heads, ff_mult=ff_mult,
                                     n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                             weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log = {"step": [], "train_loss": [], "eval_acc_per_r": []}
    t0 = time.time()
    for step in range(steps):
        tokens, iter_t = make_batch_tm_iter(batch_size, n_loops_train, device)
        all_logits = model.forward_all_loops_robust(
            tokens, n_loops=n_loops_train,
            p_noise=p_noise, noise_alpha=noise_alpha)
        losses = []
        for r in range(1, n_loops_train + 1):
            logits_at_tape = all_logits[r, :, :TM_W, :]  # [B, W, V]
            target = iter_t[r]                           # [B, W]
            losses.append(F.cross_entropy(
                logits_at_tape.reshape(-1, logits_at_tape.size(-1)),
                target.reshape(-1)))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            acc = _eval(model, eval_size, n_loops_eval, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_acc_per_r"].append(acc)
            if verbose:
                key = sorted({1, 2, 4, n_loops_train, n_loops_train + 4,
                              n_loops_train + 8, n_loops_eval} & set(acc.keys()))
                msg = " ".join(f"r{r}={acc[r]:.2f}" for r in key)
                print(f"  step {step:>5}  loss {loss.item():.4f}  {msg}",
                       flush=True)

    return {
        "config": {"n_loops": n_loops, "d": d,
                   "n_loops_train": n_loops_train, "n_loops_eval": n_loops_eval,
                   "p_noise": p_noise, "noise_alpha": noise_alpha,
                   "steps": steps, "batch_size": batch_size, "lr": lr},
        "params": sum(p.numel() for p in model.parameters()),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


@torch.no_grad()
def _eval(model, eval_size, n_loops_eval, device):
    """Per-loop accuracy = fraction of examples where ALL tape cells match."""
    model.eval()
    tokens, iter_t = make_batch_tm_iter(eval_size, n_loops_eval, device)
    all_logits = model.forward_all_loops(tokens, n_loops=n_loops_eval)
    out = {}
    for r in range(1, n_loops_eval + 1):
        preds = all_logits[r, :, :TM_W, :].argmax(-1)  # [B, W]
        all_match = (preds == iter_t[r]).all(dim=-1)
        out[r] = all_match.float().mean().item()
    model.train()
    return out
