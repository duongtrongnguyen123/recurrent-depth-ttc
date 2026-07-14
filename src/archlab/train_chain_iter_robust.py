"""Anti-degradation iter-target training (Result R candidate).

Result L's chain_iter degrades from 100% at r=15 to 88.6% at r=24 — graceful
but bounded by per-step noise accumulation. This trains the same task with
**latent noise injection** at random loops during training, forcing the
per-step rule to be robust to perturbations. At inference, the per-step rule
should generalize further with less drift.

Design: at each forward pass, with probability p_noise, inject Gaussian noise
of magnitude alpha * ||h|| into the latent at a randomly chosen loop r.
The model's loss at later loops sees the noisy state, so it must be robust
to recover the right output despite noise.

Hypothesis: noise-trained chain_iter extends usable range past r=24 with
slower accuracy decay.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_chain import (ANSWER_POS_ITER, SEQ_LEN_ITER, V, VOCAB_SIZE,
                          make_batch_chain_iter)
from .model import LoopedTransformer


class RobustLoopedTransformer(LoopedTransformer):
    """LoopedTransformer with optional latent noise injection at random loops."""

    def forward_all_loops_robust(self, x, n_loops, p_noise=0.0,
                                  noise_alpha=0.1, noise_loop=None):
        """Standard forward_all_loops_grad with optional noise.
        - p_noise: probability of injecting noise this batch
        - noise_alpha: noise magnitude relative to latent norm
        - noise_loop: which loop to inject noise at (random in [1, n_loops] if None)
        """
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        outs = [self.head(self.ln_f(h))]
        inject_at = -1
        if torch.rand(1).item() < p_noise:
            inject_at = (noise_loop if noise_loop is not None
                         else torch.randint(1, n_loops + 1, (1,)).item())
        for r in range(1, n_loops + 1):
            h = self.block(h, mask)
            if r == inject_at:
                noise_std = noise_alpha * h.norm(dim=-1, keepdim=True) / (h.shape[-1] ** 0.5)
                h = h + torch.randn_like(h) * noise_std
            outs.append(self.head(self.ln_f(h)))
        return torch.stack(outs, dim=0)


def train_chain_iter_robust(*, n_loops: int, d: int, n_heads: int, ff_mult: int,
                             steps: int, batch_size: int, lr: float,
                             eval_every: int, eval_size: int,
                             device: str, seed: int,
                             n_loops_train: int,
                             n_loops_eval: int = 24,
                             p_noise: float = 0.5,
                             noise_alpha: float = 0.1,
                             verbose: bool = True) -> dict:
    assert n_loops >= max(n_loops_train, n_loops_eval)
    torch.manual_seed(seed)
    model = RobustLoopedTransformer(vocab=VOCAB_SIZE, max_len=SEQ_LEN_ITER,
                                     d=d, n_heads=n_heads, ff_mult=ff_mult,
                                     n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log: dict[str, list] = {"step": [], "train_loss": [], "eval_acc_per_r": []}
    t0 = time.time()

    for step in range(steps):
        tokens, iter_targets = make_batch_chain_iter(batch_size, n_loops_train, device)
        all_logits = model.forward_all_loops_robust(
            tokens, n_loops=n_loops_train, p_noise=p_noise, noise_alpha=noise_alpha)
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
            acc = _eval_robust(model, eval_size, n_loops_eval, device)
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
                   "n_loops_eval": n_loops_eval, "p_noise": p_noise,
                   "noise_alpha": noise_alpha, "steps": steps,
                   "batch_size": batch_size, "lr": lr},
        "params": model.num_params(),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


@torch.no_grad()
def _eval_robust(model, eval_size: int, n_loops_eval: int,
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
