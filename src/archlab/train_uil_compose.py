"""UIL Compositional training: one shared loop core applies chain rule
to chain segment AND parity rule to parity segment in the same forward pass.

Two iter-target losses summed per loop (one at EQ_CHAIN_POS, one at
EQ_PAR_POS).
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_uil_compose import (EQ_CHAIN_POS, EQ_PAR_POS, SEQ_LEN_COMPOSE,
                                make_batch_compose_iter)
from .data_chain import VOCAB_SIZE
from .data_uil import VOCAB_UIL
from .train_chain_iter_robust import RobustLoopedTransformer


def train_uil_compose(*, n_loops: int, d: int, n_heads: int, ff_mult: int,
                       steps: int, batch_size: int, lr: float,
                       eval_every: int, eval_size: int,
                       device: str, seed: int,
                       n_loops_train: int,
                       n_loops_eval: int = 16,
                       p_noise: float = 0.5,
                       noise_alpha: float = 0.1,
                       verbose: bool = True) -> dict:
    assert n_loops >= max(n_loops_train, n_loops_eval)
    torch.manual_seed(seed)
    model = RobustLoopedTransformer(vocab=VOCAB_UIL, max_len=SEQ_LEN_COMPOSE,
                                     d=d, n_heads=n_heads, ff_mult=ff_mult,
                                     n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log = {"step": [], "train_loss": [],
            "eval_chain_per_r": [], "eval_par_per_r": []}
    t0 = time.time()

    for step in range(steps):
        tokens, chain_t, par_t = make_batch_compose_iter(
            batch_size, n_loops_train, device)
        all_logits = model.forward_all_loops_robust(
            tokens, n_loops=n_loops_train,
            p_noise=p_noise, noise_alpha=noise_alpha)
        losses = []
        for r in range(1, n_loops_train + 1):
            losses.append(F.cross_entropy(
                all_logits[r, :, EQ_CHAIN_POS, :], chain_t[r]))
            losses.append(F.cross_entropy(
                all_logits[r, :, EQ_PAR_POS, :], par_t[r]))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            chain_acc, par_acc = _eval(model, eval_size, n_loops_eval, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_chain_per_r"].append(chain_acc)
            log["eval_par_per_r"].append(par_acc)
            if verbose:
                key = sorted({1, 2, n_loops_train, n_loops_train + 4,
                              n_loops_eval} & set(chain_acc.keys()))
                msg_c = " ".join(f"r{r}={chain_acc[r]:.2f}" for r in key)
                msg_p = " ".join(f"r{r}={par_acc[r]:.2f}" for r in key)
                print(f"  step {step:>5}  loss {loss.item():.4f}  "
                       f"chain[{msg_c}]  par[{msg_p}]", flush=True)

    return {
        "config": {"n_loops": n_loops, "d": d, "n_loops_train": n_loops_train,
                   "n_loops_eval": n_loops_eval, "p_noise": p_noise,
                   "noise_alpha": noise_alpha, "steps": steps,
                   "batch_size": batch_size, "lr": lr},
        "params": sum(p.numel() for p in model.parameters()),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


@torch.no_grad()
def _eval(model, eval_size, n_loops_eval, device):
    model.eval()
    tokens, chain_t, par_t = make_batch_compose_iter(
        eval_size, n_loops_eval, device)
    all_logits = model.forward_all_loops(tokens, n_loops=n_loops_eval)
    chain_acc, par_acc = {}, {}
    for r in range(1, n_loops_eval + 1):
        cp = all_logits[r, :, EQ_CHAIN_POS, :].argmax(-1)
        pp = all_logits[r, :, EQ_PAR_POS, :].argmax(-1)
        chain_acc[r] = (cp == chain_t[r]).float().mean().item()
        par_acc[r] = (pp == par_t[r]).float().mean().item()
    model.train()
    return chain_acc, par_acc
