"""HR on UIL multi-task (chain + parity, no instr)."""
from __future__ import annotations
import time
import torch, torch.nn.functional as F
from .data_uil_no_instr import (EQ_POS, SEQ_LEN_UIL,
                                 make_batch_uil_noinstr_eval,
                                 make_batch_uil_noinstr_iter)
from .data_uil import VOCAB_UIL
from .model_hier import HierLoopedTransformer


def train_uil_hier(*, n_inner, n_outer, d, n_heads, ff_mult, steps, batch_size,
                     lr, eval_every, eval_size, device, seed, n_loops_eval,
                     task_mix=(0.5, 0.5), verbose=True):
    torch.manual_seed(seed)
    n_total = n_inner * n_outer
    model = HierLoopedTransformer(vocab=VOCAB_UIL, max_len=SEQ_LEN_UIL,
        d=d, n_heads=n_heads, ff_mult=ff_mult,
        n_inner=n_inner, n_outer=n_outer).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                              weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    log = {"step": [], "train_loss": [], "eval_chain": [], "eval_par": []}
    t0 = time.time()
    for step in range(steps):
        tokens, iter_t, _ = make_batch_uil_noinstr_iter(
            batch_size, n_total, device, task_mix=task_mix)
        all_logits = model.forward_all_loops_grad(tokens, n_loops=n_total)
        losses = []
        for r in range(1, n_total + 1):
            losses.append(F.cross_entropy(
                all_logits[r, :, EQ_POS, :], iter_t[r]))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                chain_acc = {}
                par_acc = {}
                for task_id, name, dst in [(0, 'chain', chain_acc),
                                              (1, 'parity', par_acc)]:
                    tk, it = make_batch_uil_noinstr_eval(eval_size, n_loops_eval,
                                                              task_id, device)
                    al = model.forward_all_loops(tk, n_loops=min(n_loops_eval, n_total))
                    for r in range(1, min(n_loops_eval, n_total) + 1):
                        p = al[r, :, EQ_POS, :].argmax(-1)
                        dst[r] = (p == it[r]).float().mean().item()
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_chain"].append(chain_acc)
            log["eval_par"].append(par_acc)
            if verbose:
                key = sorted({1, 2, n_total} & set(chain_acc.keys()))
                msg_c = " ".join(f"r{r}={chain_acc[r]:.2f}" for r in key)
                msg_p = " ".join(f"r{r}={par_acc[r]:.2f}" for r in key)
                print(f"  step {step:>5}  loss {loss.item():.4f}  "
                       f"chain[{msg_c}]  par[{msg_p}]", flush=True)
    return {"config": {"n_inner": n_inner, "n_outer": n_outer, "d": d,
                         "steps": steps, "lr": lr},
             "params": sum(p.numel() for p in model.parameters()),
             "wall_time_sec": time.time() - t0,
             "log": log}
