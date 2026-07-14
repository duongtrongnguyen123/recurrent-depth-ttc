"""Universal Interpreter Loop (UIL) training.

A single recurrent block, trained as a universal iterator across multiple
tasks. The instruction token at position 0 selects which per-step rule
applies. All per-step rules share the same loop core.

v1: chain + parity (50/50 mix per batch).

If this works, the recurrent block has learned *iteration as a general
capability* — not just one task's per-step rule. Extending to novel
instructions or instruction sequences becomes the next test.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_uil import (EQ_POS, INSTR_CHAIN, INSTR_PARITY, SEQ_LEN_UIL,
                        VOCAB_UIL, make_batch_uil_eval, make_batch_uil_iter)
from .train_chain_iter_robust import RobustLoopedTransformer


def train_uil(*, n_loops: int, d: int, n_heads: int, ff_mult: int,
              steps: int, batch_size: int, lr: float,
              eval_every: int, eval_size: int,
              device: str, seed: int,
              n_loops_train: int,
              n_loops_eval: int = 24,
              p_noise: float = 0.5,
              noise_alpha: float = 0.1,
              instr_mix: tuple[float, float] = (0.5, 0.5),
              verbose: bool = True) -> dict:
    assert n_loops >= max(n_loops_train, n_loops_eval)
    torch.manual_seed(seed)
    model = RobustLoopedTransformer(vocab=VOCAB_UIL, max_len=SEQ_LEN_UIL,
                                     d=d, n_heads=n_heads, ff_mult=ff_mult,
                                     n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log: dict[str, list] = {
        "step": [], "train_loss": [],
        "eval_acc_chain_per_r": [], "eval_acc_parity_per_r": [],
    }
    t0 = time.time()

    for step in range(steps):
        tokens, iter_targets = make_batch_uil_iter(
            batch_size, n_loops_train, device, instr_mix=instr_mix)
        all_logits = model.forward_all_loops_robust(
            tokens, n_loops=n_loops_train,
            p_noise=p_noise, noise_alpha=noise_alpha)
        losses = []
        for r in range(1, n_loops_train + 1):
            losses.append(F.cross_entropy(
                all_logits[r, :, EQ_POS, :], iter_targets[r]))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            chain_acc = _eval_uil(model, eval_size, n_loops_eval, INSTR_CHAIN, device)
            par_acc = _eval_uil(model, eval_size, n_loops_eval, INSTR_PARITY, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_acc_chain_per_r"].append(chain_acc)
            log["eval_acc_parity_per_r"].append(par_acc)
            if verbose:
                key = sorted({1, 2, n_loops_train, n_loops_train + 4,
                              n_loops_eval} & set(chain_acc.keys()))
                msg_c = " ".join(f"r{r}={chain_acc[r]:.2f}" for r in key)
                msg_p = " ".join(f"r{r}={par_acc[r]:.2f}" for r in key)
                print(f"  step {step:>5}  loss {loss.item():.4f}  "
                       f"chain[{msg_c}]  par[{msg_p}]")

    return {
        "config": {"n_loops": n_loops, "d": d, "n_loops_train": n_loops_train,
                   "n_loops_eval": n_loops_eval, "p_noise": p_noise,
                   "noise_alpha": noise_alpha, "steps": steps,
                   "batch_size": batch_size, "lr": lr,
                   "instr_mix": list(instr_mix)},
        "params": sum(p.numel() for p in model.parameters()),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


@torch.no_grad()
def _eval_uil(model, eval_size: int, n_loops_eval: int, instr: int,
              device: str) -> dict[int, float]:
    model.eval()
    tokens, iter_targets = make_batch_uil_eval(eval_size, n_loops_eval, instr, device)
    all_logits = model.forward_all_loops(tokens, n_loops=n_loops_eval)
    out: dict[int, float] = {}
    for r in range(1, n_loops_eval + 1):
        preds = all_logits[r, :, EQ_POS, :].argmax(-1)
        out[r] = (preds == iter_targets[r]).float().mean().item()
    model.train()
    return out
