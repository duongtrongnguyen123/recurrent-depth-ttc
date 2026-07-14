"""Train chain_compose with R-recipe (iter-target + noise injection).

Mirrors train_chain_iter_robust but supervises iter-target on the compositional
two-rule chain. Tests whether Branch 1's R recipe (Result U) extends to tasks
that require **per-loop rule selection from context**.

Hypothesis: as long as the per-step rule is position-self-referential (per
Result N — depends on input symbol at this position, not on a counter), the
recipe should give clean ID accuracy and gentle OOD extrapolation.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_compose import (ANSWER_POS_COMPOSE, SEQ_LEN_COMPOSE, V, VOCAB_SIZE,
                            make_batch_compose_iter,
                            make_batch_compose_iter_random_k)
from .train_chain_iter_robust import RobustLoopedTransformer


def train_compose_iter(*, n_loops: int, d: int, n_heads: int, ff_mult: int,
                       steps: int, batch_size: int, lr: float,
                       eval_every: int, eval_size: int,
                       device: str, seed: int,
                       n_loops_train: int,
                       n_loops_eval: int = 24,
                       p_noise: float = 0.5,
                       noise_alpha: float = 0.1,
                       random_k: bool = False,
                       k_min: int = 1,
                       k_max: int = 8,
                       verbose: bool = True) -> dict:
    assert n_loops >= max(n_loops_train, n_loops_eval)
    torch.manual_seed(seed)
    model = RobustLoopedTransformer(vocab=VOCAB_SIZE, max_len=SEQ_LEN_COMPOSE,
                                     d=d, n_heads=n_heads, ff_mult=ff_mult,
                                     n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log: dict[str, list] = {"step": [], "train_loss": [], "eval_acc_per_r": []}
    t0 = time.time()

    for step in range(steps):
        if random_k:
            tokens, iter_targets, _ = make_batch_compose_iter_random_k(
                batch_size, k_min, k_max, device)
            ksteps = k_max
        else:
            tokens, iter_targets = make_batch_compose_iter(
                batch_size, n_loops_train, device)
            ksteps = n_loops_train
        all_logits = model.forward_all_loops_robust(
            tokens, n_loops=ksteps, p_noise=p_noise, noise_alpha=noise_alpha)

        # iter_targets: [ksteps+1, B]. Loop r ∈ [1, ksteps] supervised at the
        # answer position with target iter_targets[r].
        losses = []
        for r in range(1, ksteps + 1):
            logits_r = all_logits[r][:, ANSWER_POS_COMPOSE]
            losses.append(F.cross_entropy(logits_r, iter_targets[r]))
        loss = torch.stack(losses).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            acc_per_r = _eval(model, eval_size, n_loops_eval, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_acc_per_r"].append(acc_per_r)
            if verbose:
                acc_str = " ".join(f"r{r}={acc_per_r[r-1]:.2f}"
                                   for r in [1, n_loops_train, n_loops_eval])
                print(f"  step {step:>5}  loss {loss.item():.4f}  {acc_str}",
                      flush=True)

    return {
        "config": {"n_loops": n_loops, "d": d, "ff_mult": ff_mult,
                   "steps": steps, "batch_size": batch_size, "lr": lr,
                   "n_loops_train": n_loops_train, "n_loops_eval": n_loops_eval,
                   "p_noise": p_noise, "noise_alpha": noise_alpha,
                   "random_k": random_k, "k_min": k_min, "k_max": k_max,
                   "task": "chain_compose"},
        "params": sum(p.numel() for p in model.parameters()),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


@torch.no_grad()
def _eval(model, eval_size: int, n_loops_eval: int, device: str) -> list[float]:
    model.eval()
    accs = [0.0] * n_loops_eval
    n_done = 0
    bs = min(eval_size, 256)
    while n_done < eval_size:
        n = min(bs, eval_size - n_done)
        # Eval at FIXED k = n_loops_eval (probes per-r accuracy when the model
        # is asked to apply that many compositional rules).
        tokens, iter_targets = make_batch_compose_iter(n, n_loops_eval, device)
        all_logits = model.forward_all_loops_robust(
            tokens, n_loops=n_loops_eval, p_noise=0.0, noise_alpha=0.0)
        for r in range(1, n_loops_eval + 1):
            preds = all_logits[r][:, ANSWER_POS_COMPOSE].argmax(-1)
            ok = (preds == iter_targets[r]).float().mean().item()
            accs[r - 1] += ok * n
        n_done += n
    accs = [a / eval_size for a in accs]
    model.train()
    return accs
