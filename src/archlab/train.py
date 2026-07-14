from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data import PAD, VOCAB_SIZE, make_batch
from .model import LoopedTransformer


def train(*, n_digits: int = 4, n_loops: int = 8, d: int = 1024, n_heads: int = 8,
          ff_mult: int = 4, steps: int = 6000, batch_size: int = 256, lr: float = 3e-4,
          eval_every: int = 200, eval_size: int = 4096, device: str = "cuda",
          seed: int = 0, verbose: bool = True, digits_min: int | None = None,
          aux_loss: bool = False, aux_min_loops: int = 1) -> dict:
    torch.manual_seed(seed)
    max_len = 2 * n_digits + 2 + (n_digits + 1)
    model = LoopedTransformer(vocab=VOCAB_SIZE, max_len=max_len, d=d, n_heads=n_heads,
                              ff_mult=ff_mult, n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log: dict[str, list] = {"step": [], "train_loss": [], "eval_acc_full": [],
                            "eval_acc_per_token": []}
    t0 = time.time()

    for step in range(steps):
        tokens, targets, mask = make_batch(batch_size, n_digits, device,
                                            digits_min=digits_min)
        if aux_loss:
            # Mean cross-entropy across loops [aux_min_loops, n_loops]. Forces every
            # loop's representation to be a valid prediction, so intermediate entropy
            # becomes calibrated for adaptive halting.
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
            acc_full, acc_tok = _evaluate(model, n_digits, eval_size, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_acc_full"].append(acc_full)
            log["eval_acc_per_token"].append(acc_tok)
            if verbose:
                print(f"  step {step:>5}  loss {loss.item():.4f}  "
                      f"acc_full {acc_full:.3f}  acc_tok {acc_tok:.3f}")

    return {
        "config": {"n_digits": n_digits, "n_loops": n_loops, "d": d,
                   "n_heads": n_heads, "ff_mult": ff_mult,
                   "steps": steps, "batch_size": batch_size},
        "params": model.num_params(),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


def _masked_loss(logits: torch.Tensor, targets: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])


@torch.no_grad()
def _evaluate(model, n_digits: int, eval_size: int, device: str
              ) -> tuple[float, float]:
    model.eval()
    tokens, targets, mask = make_batch(eval_size, n_digits, device, digits_min=None)
    logits = model(tokens)
    preds = logits.argmax(-1)
    correct_tok = (preds == targets) & mask
    # Per-token accuracy.
    acc_tok = correct_tok.sum().item() / mask.sum().item()
    # Full-answer accuracy: every answer token correct.
    per_example_correct = ((preds == targets) | ~mask).all(dim=1)
    acc_full = per_example_correct.float().mean().item()
    model.train()
    return acc_full, acc_tok
