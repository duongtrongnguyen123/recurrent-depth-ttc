"""Two-stream training with stream-B auxiliary loss + zero-init gate (Result AS).

Fixes AR's gate-collapse failure mode: zero-initializing the gate makes
g(x) = 0.5 at start (no preferred stream), and the auxiliary head on stream B
provides direct gradient to the looped block independent of gate value.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_mixed import (ANSWER_POS_MIXED, SEQ_LEN_MIXED, VOCAB_SIZE_MIXED,
                          make_batch_mixed_iter)
from .model_twostream import TwoStreamTransformer


def train_mixed_twostream_aux(*, n_loops: int, d: int, n_heads: int,
                                ff_mult: int, n_vanilla_blocks: int,
                                steps: int, batch_size: int, lr: float,
                                eval_every: int, eval_size: int,
                                device: str, seed: int,
                                n_loops_train: int,
                                n_loops_eval: int = 24,
                                p_chain: float = 0.5,
                                aux_b_weight: float = 0.5,
                                use_skip_loop: bool = False,
                                skip_alpha_init: float = 0.1,
                                verbose: bool = True) -> dict:
    assert n_loops >= max(n_loops_train, n_loops_eval)
    torch.manual_seed(seed)
    model = TwoStreamTransformer(vocab=VOCAB_SIZE_MIXED, max_len=SEQ_LEN_MIXED,
                                   d=d, n_heads=n_heads, ff_mult=ff_mult,
                                   n_vanilla_blocks=n_vanilla_blocks,
                                   n_loops=n_loops,
                                   zero_init_gate=True,
                                   aux_head=True,
                                   use_skip_loop=use_skip_loop,
                                   skip_alpha_init=skip_alpha_init).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log: dict[str, list] = {"step": [], "train_loss": [], "eval_acc_chain": [],
                              "eval_acc_id": [], "gate_chain": [], "gate_id": []}
    t0 = time.time()

    for step in range(steps):
        tokens, iter_targets, _ = make_batch_mixed_iter(batch_size, n_loops_train,
                                                          p_chain=p_chain,
                                                          device=device)
        combined_logits, b_logits = model.forward_all_loops_dual(
            tokens, n_loops=n_loops_train)
        losses_main = []
        losses_b = []
        for r in range(1, n_loops_train + 1):
            losses_main.append(F.cross_entropy(
                combined_logits[r, :, ANSWER_POS_MIXED, :], iter_targets[r]))
            losses_b.append(F.cross_entropy(
                b_logits[r, :, ANSWER_POS_MIXED, :], iter_targets[r]))
        loss_main = torch.stack(losses_main).mean()
        loss_b = torch.stack(losses_b).mean()
        loss = loss_main + aux_b_weight * loss_b
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            eval_out = _eval(model, eval_size, n_loops_eval, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_acc_chain"].append(eval_out["acc_chain"])
            log["eval_acc_id"].append(eval_out["acc_id"])
            log["gate_chain"].append(eval_out["gate_chain"])
            log["gate_id"].append(eval_out["gate_id"])
            if verbose:
                ac = eval_out["acc_chain"]
                ai = eval_out["acc_id"]
                key = sorted({1, 2, 4, n_loops_train, n_loops_train + 4,
                              n_loops_eval} & set(ac.keys()))
                msg_c = " ".join(f"r{r}={ac[r]:.2f}" for r in key)
                msg_i = " ".join(f"r{r}={ai[r]:.2f}" for r in key)
                print(f"  step {step:>5}  L={loss.item():.4f} (m={loss_main.item():.4f}, b={loss_b.item():.4f})  "
                      f"CHAIN[{msg_c}] g={eval_out['gate_chain']:.3f}  "
                      f"ID[{msg_i}] g={eval_out['gate_id']:.3f}")

    return {
        "config": {"n_loops": n_loops, "d": d,
                   "n_vanilla_blocks": n_vanilla_blocks,
                   "n_loops_train": n_loops_train,
                   "n_loops_eval": n_loops_eval, "steps": steps,
                   "batch_size": batch_size, "lr": lr, "p_chain": p_chain,
                   "aux_b_weight": aux_b_weight,
                   "zero_init_gate": True, "aux_head": True},
        "params": model.num_params(),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


@torch.no_grad()
def _eval(model, eval_size: int, n_loops_eval: int, device: str) -> dict:
    model.eval()
    tokens, iter_targets, is_chain = make_batch_mixed_iter(
        eval_size, n_loops_eval, p_chain=0.5, device=device)
    all_logits = model.forward_all_loops(tokens, n_loops=n_loops_eval)
    g = model.gate_diagnostic(tokens, n_loops=n_loops_eval)[:, ANSWER_POS_MIXED]
    acc_chain: dict[int, float] = {}
    acc_id: dict[int, float] = {}
    for r in range(1, n_loops_eval + 1):
        preds = all_logits[r, :, ANSWER_POS_MIXED, :].argmax(-1)
        correct = (preds == iter_targets[r])
        n_chain = is_chain.sum().item()
        n_id = (~is_chain).sum().item()
        acc_chain[r] = (correct[is_chain].float().mean().item() if n_chain > 0 else 0.0)
        acc_id[r] = (correct[~is_chain].float().mean().item() if n_id > 0 else 0.0)
    gate_chain = g[is_chain].mean().item() if is_chain.any() else 0.0
    gate_id = g[~is_chain].mean().item() if (~is_chain).any() else 0.0
    model.train()
    return {"acc_chain": acc_chain, "acc_id": acc_id,
             "gate_chain": gate_chain, "gate_id": gate_id}
