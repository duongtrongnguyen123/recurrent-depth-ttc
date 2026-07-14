from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_chain import (MAX_K, SEQ_LEN_UNARY, V, VOCAB_SIZE, make_batch_chain,
                          make_batch_chain_unary)
from .model import LoopedTransformer


def train_chain(*, n_loops: int, d: int, n_heads: int, ff_mult: int,
                steps: int, batch_size: int, lr: float, eval_every: int,
                eval_size: int, device: str, seed: int,
                k_min: int = 1, k_max: int = MAX_K,
                aux_loss: bool = False, aux_min_loops: int = 2,
                unary_depth: bool = False,
                verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    if unary_depth:
        max_len = SEQ_LEN_UNARY
        batch_fn = make_batch_chain_unary
    else:
        max_len = V + 4
        batch_fn = make_batch_chain
    model = LoopedTransformer(vocab=VOCAB_SIZE, max_len=max_len, d=d, n_heads=n_heads,
                              ff_mult=ff_mult, n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log: dict[str, list] = {"step": [], "train_loss": [], "eval_acc_overall": []}
    for k in range(k_min, k_max + 1):
        log[f"eval_acc_k{k}"] = []
    t0 = time.time()

    for step in range(steps):
        tokens, targets, mask, _ = batch_fn(batch_size, k_min, k_max, device)
        if aux_loss:
            all_logits = model.forward_all_loops_grad(tokens, n_loops=n_loops)
            losses = [_masked_loss(all_logits[r], targets, mask)
                      for r in range(aux_min_loops, n_loops + 1)]
            loss = torch.stack(losses).mean()
        else:
            logits = model(tokens)
            loss = _masked_loss(logits, targets, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            results = _evaluate_chain(model, eval_size, k_min, k_max, device, batch_fn)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_acc_overall"].append(results["overall"])
            for k in range(k_min, k_max + 1):
                log[f"eval_acc_k{k}"].append(results[f"k{k}"])
            if verbose:
                ks = " ".join(f"k{k}={results[f'k{k}']:.2f}"
                              for k in range(k_min, k_max + 1))
                print(f"  step {step:>5}  loss {loss.item():.4f}  "
                      f"all={results['overall']:.3f}  {ks}")

    return {
        "config": {"n_loops": n_loops, "d": d, "k_min": k_min, "k_max": k_max,
                   "aux_loss": aux_loss, "aux_min_loops": aux_min_loops,
                   "unary_depth": unary_depth,
                   "steps": steps, "batch_size": batch_size},
        "params": model.num_params(),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


def _masked_loss(logits, targets, mask):
    flat = logits.reshape(-1, logits.size(-1))
    return F.cross_entropy(flat[mask.reshape(-1)],
                           targets.reshape(-1)[mask.reshape(-1)])


@torch.no_grad()
def _evaluate_chain(model, eval_size: int, k_min: int, k_max: int,
                    device: str, batch_fn=make_batch_chain) -> dict[str, float]:
    model.eval()
    per_k = max(eval_size // (k_max - k_min + 1), 256)
    out: dict[str, float] = {}
    all_correct = []
    for k in range(k_min, k_max + 1):
        tokens, targets, mask, _ = batch_fn(per_k, k, k, device)
        preds = model(tokens).argmax(-1)
        ok = ((preds == targets) | ~mask).all(dim=1)
        out[f"k{k}"] = ok.float().mean().item()
        all_correct.append(ok)
    out["overall"] = torch.cat(all_correct).float().mean().item()
    model.train()
    return out
