"""Halt head for decoupled iter + halt experiments (Result M's path forward).

Trains a small MLP halt head on top of a *frozen* iter-target base
(chain_iter, Result L checkpoint). The base outputs f^r(start) at every loop
r — it iterates correctly without any halt mechanism. The halt head learns to
detect 'user wants depth k, halt at this loop' from the per-loop latent and
explicit (r, k) inputs.

Training: heterogeneous k ∈ {1, ..., k_max_halt}; target at loop r is
1 if r >= k else 0 (BCE loss). Inference: walk r=1..max_r, halt at first r
where sigmoid(halt_logit) > threshold; output base argmax at halt time.

The base's length extrapolation (Result L) carries through: if the halt head
correctly halts at r=k for k > base's training depth, the output is the base's
extrapolated f^k(start) — accurate up to ~3× training depth per Result L.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_chain import (ANSWER_POS_ITER, SEQ_LEN_ITER, V, VOCAB_SIZE,
                          make_batch_chain_iter)
from .model import LoopedTransformer


class HaltHead(nn.Module):
    """Small MLP halt head: (h_r, r, k) → halt logit.

    h_r is the latent at the answer position at loop r (size d).
    r and k are integer indices embedded to small vectors.
    """
    def __init__(self, d: int, max_r: int, max_k: int,
                 r_emb_dim: int = 32, k_emb_dim: int = 32,
                 hidden: int = 256):
        super().__init__()
        self.r_embed = nn.Embedding(max_r + 1, r_emb_dim)
        self.k_embed = nn.Embedding(max_k + 1, k_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d + r_emb_dim + k_emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_r: torch.Tensor, r: torch.Tensor,
                k: torch.Tensor) -> torch.Tensor:
        """h_r: [B, d]; r, k: [B] long. Returns [B] halt logits."""
        feats = torch.cat([h_r, self.r_embed(r), self.k_embed(k)], dim=-1)
        return self.mlp(feats).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_halt_head(*, base_ckpt_path: str, base_cfg: dict,
                    n_loops: int, k_max_halt: int,
                    steps: int, batch_size: int, lr: float,
                    eval_every: int, eval_size: int,
                    device: str, seed: int,
                    verbose: bool = True) -> dict:
    """Train a halt head on a frozen chain_iter base.

    n_loops      : architectural max loops the base will run; head supervised on r=1..n_loops.
    k_max_halt   : max k sampled during halt-head training (covers eval range).
    """
    torch.manual_seed(seed)

    base = LoopedTransformer(vocab=VOCAB_SIZE, max_len=SEQ_LEN_ITER,
                             d=base_cfg["d"], n_heads=base_cfg["n_heads"],
                             ff_mult=base_cfg["ff_mult"],
                             n_loops=max(base_cfg["n_loops"], n_loops)).to(device)
    ck = torch.load(base_ckpt_path, weights_only=False)
    base.load_state_dict(ck["model_state"])
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    halt = HaltHead(d=base_cfg["d"], max_r=n_loops, max_k=k_max_halt).to(device)
    opt = torch.optim.AdamW(halt.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log: dict[str, list] = {"step": [], "train_loss": [],
                             "eval_per_k_halt_acc": []}
    t0 = time.time()

    for step in range(steps):
        tokens, _iter_targets = make_batch_chain_iter(batch_size, n_loops, device)
        ks = torch.randint(1, k_max_halt + 1, (batch_size,), device=device)

        with torch.no_grad():
            hidden, _logits = base.forward_all_loops_with_hidden(tokens, n_loops=n_loops)
            # hidden: [n_loops+1, B, T, d]
            h_at_eq = hidden[:, :, ANSWER_POS_ITER, :]            # [n_loops+1, B, d]

        losses = []
        for r in range(1, n_loops + 1):
            r_tensor = torch.full_like(ks, r)
            halt_logit = halt(h_at_eq[r], r_tensor, ks)            # [B]
            target = (r >= ks).float()                              # [B]
            losses.append(F.binary_cross_entropy_with_logits(halt_logit, target))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(halt.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            acc = _eval_halt(base, halt, eval_size, k_max_halt, n_loops, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_per_k_halt_acc"].append(acc)
            if verbose:
                halt_acc_mean = sum(v["halt_acc"] for v in acc.values()) / len(acc)
                halt_r_mean = sum(v["mean_halt_r"] for v in acc.values()) / len(acc)
                msg = (f"  step {step:>5}  loss {loss.item():.4f}  "
                       f"halt_acc_mean={halt_acc_mean:.3f}  "
                       f"halt_r_mean={halt_r_mean:.2f}")
                print(msg)

    return {
        "config": {"k_max_halt": k_max_halt, "n_loops": n_loops,
                   "steps": steps, "batch_size": batch_size, "lr": lr},
        "halt_params": halt.num_params(),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "halt_state": {k: v.cpu() for k, v in halt.state_dict().items()},
    }


@torch.no_grad()
def _eval_halt(base, halt, eval_size: int, k_max_halt: int,
               n_loops: int, device: str, halt_threshold: float = 0.5
               ) -> dict[int, dict]:
    """For each k in [1, k_max_halt], simulate inference: run base, walk loops,
    halt at first r where sigmoid(halt_logit) > threshold. Compare halt-time
    base output to f^k(start)."""
    halt.eval()
    out: dict[int, dict] = {}
    per_k = max(eval_size // k_max_halt, 256)
    for k in range(1, k_max_halt + 1):
        tokens, iter_targets = make_batch_chain_iter(per_k, max(k, n_loops), device)
        truth = iter_targets[k]                                     # [B]
        hidden, logits = base.forward_all_loops_with_hidden(tokens, n_loops=n_loops)
        # hidden, logits: [n_loops+1, B, T, *]
        h_at_eq = hidden[:, :, ANSWER_POS_ITER, :]                  # [n_loops+1, B, d]
        preds_per_r = logits[:, :, ANSWER_POS_ITER, :].argmax(-1)   # [n_loops+1, B]

        # Walk loops; halt at first r where prob(halt) > threshold
        ks_b = torch.full((per_k,), k, device=device)
        halt_at = torch.full((per_k,), n_loops, device=device, dtype=torch.long)
        already = torch.zeros(per_k, device=device, dtype=torch.bool)
        for r in range(1, n_loops + 1):
            r_tensor = torch.full_like(ks_b, r)
            halt_prob = torch.sigmoid(halt(h_at_eq[r], r_tensor, ks_b))  # [B]
            should_halt = (halt_prob > halt_threshold) & ~already
            halt_at[should_halt] = r
            already |= should_halt
        # Pred at halt time
        halt_preds = preds_per_r.gather(0, halt_at.unsqueeze(0)).squeeze(0)  # [B]
        halt_acc = (halt_preds == truth).float().mean().item()
        out[k] = {
            "halt_acc":     halt_acc,
            "mean_halt_r":  halt_at.float().mean().item(),
            "frac_halted_before_max": (halt_at < n_loops).float().mean().item(),
            "n":            per_k,
        }
    halt.train()
    return out
