"""Iterative-target training on tree BFS task (Branch 1 Result AN)."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_bfs_tree import (BFS_TREE_OUTPUT_START, BFS_TREE_SEQ_LEN,
                              V_BFS_TREE, make_batch_bfs_tree_iter)
from .data_chain import VOCAB_SIZE
from .model import LoopedTransformer


def train_bfs_tree_iter(*, n_loops: int, d: int, n_heads: int, ff_mult: int,
                          steps: int, batch_size: int, lr: float,
                          eval_every: int, eval_size: int,
                          device: str, seed: int,
                          n_loops_train: int,
                          n_loops_eval: int = 24,
                          verbose: bool = True) -> dict:
    assert n_loops >= max(n_loops_train, n_loops_eval)
    torch.manual_seed(seed)
    model = LoopedTransformer(vocab=VOCAB_SIZE, max_len=BFS_TREE_SEQ_LEN,
                               d=d, n_heads=n_heads, ff_mult=ff_mult,
                               n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                             weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log: dict[str, list] = {"step": [], "train_loss": [], "eval_acc_per_r": []}
    t0 = time.time()

    for step in range(steps):
        tokens, iter_targets = make_batch_bfs_tree_iter(batch_size, n_loops_train, device)
        all_logits = model.forward_all_loops_grad(tokens, n_loops=n_loops_train)
        losses = []
        for r in range(1, n_loops_train + 1):
            logits_r = all_logits[r, :,
                                   BFS_TREE_OUTPUT_START:BFS_TREE_OUTPUT_START + V_BFS_TREE, :]
            losses.append(F.cross_entropy(logits_r.reshape(-1, logits_r.size(-1)),
                                            iter_targets[r].reshape(-1)))
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
                print(f"  step {step:>5}  loss {loss.item():.4f}  {msg}")

    return {
        "config": {"n_loops": n_loops, "d": d, "n_loops_train": n_loops_train,
                   "n_loops_eval": n_loops_eval, "steps": steps,
                   "batch_size": batch_size, "lr": lr, "n_nodes": V_BFS_TREE,
                   "task": "tree_bfs"},
        "params": model.num_params(),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


@torch.no_grad()
def _eval(model, eval_size: int, n_loops_eval: int,
          device: str, chunk: int = 256) -> dict[int, float]:
    model.eval()
    correct = {r: 0 for r in range(1, n_loops_eval + 1)}
    total = 0
    remaining = eval_size
    while remaining > 0:
        bs = min(chunk, remaining)
        tokens, iter_targets = make_batch_bfs_tree_iter(bs, n_loops_eval, device)
        all_logits = model.forward_all_loops(tokens, n_loops=n_loops_eval)
        for r in range(1, n_loops_eval + 1):
            preds = all_logits[r, :, BFS_TREE_OUTPUT_START:BFS_TREE_OUTPUT_START + V_BFS_TREE, :].argmax(-1)
            correct[r] += (preds == iter_targets[r]).all(dim=-1).sum().item()
        del all_logits, tokens, iter_targets
        total += bs
        remaining -= bs
    model.train()
    return {r: correct[r] / total for r in range(1, n_loops_eval + 1)}
