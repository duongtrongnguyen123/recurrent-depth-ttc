"""Halt classifier head for per-token adaptive depth (user's Q2/Q3 recipe).

Trains a small MLP that predicts argmin_r CE per token, given h_1 (latent
after loop 1). At inference, picks the predicted-best loop for each token.

Recipe (frozen base, train classifier only):
1. Run aux-trained base for n_loops loops, get per-loop logits.
2. For each position, compute CE per loop r against the next-token target.
3. argmin_r → target class label per position (n_loops classes).
4. Classifier sees h_1 at that position (or h_0 if you want to predict before any loop).
   Output K logits, train with CE against argmin labels.
5. At inference: greedy-pick predicted argmin r per token, run base only that
   many loops at that position (or use as a halt signal in batched compute).

Reference: archlab Result Q2/Q3 (text d=1024) — classifier halt closes 39%
of oracle gap at 41% of compute.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class HaltCfg:
    hidden: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    steps: int = 2000
    batch_size: int = 4
    block_size: int = 2048
    eval_every: int = 100


class HaltClassifier(nn.Module):
    """Per-position MLP: h -> n_loops-way logits over best-r."""
    def __init__(self, d_model: int, n_loops: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_loops),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, d] -> [B, T, n_loops]
        return self.mlp(h)


@torch.no_grad()
def _argmin_targets(model, x: torch.Tensor, n_loops: int, dtype):
    """For batch x [B, T], return (h_1: [B, T, d], argmin_r: [B, T] in {0..n_loops-1}).

    argmin_r[i,t] = the loop index (0-indexed) that minimizes CE(logits_r[i,t], x[i,t+1]).
    """
    device = x.device
    device_type = device.type if hasattr(device, "type") else str(device)
    with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(dtype != torch.float32)):
        # Need both per-loop logits AND per-loop hidden states. Use forward_with_aux
        # for logits; we'll separately probe hidden if needed. Use loop-1 hidden
        # as classifier input — read by adding a small hook.
        out = model.forward_with_aux(x, n_loops=n_loops, aux_min_loops=1)
        per_loop_logits = out["per_loop_logits"]  # list of [B, T, V]
        # Compute h_1: same as standard forward at n_loops=1, before final coda.
        # Re-run forward with collect_loops=True to capture h at each loop.
        out_with_h = model(x, n_loops=n_loops, collect_loops=True)
        loop_hiddens = out_with_h["loop_hiddens"]  # list of [B, T, d]
        h_1 = loop_hiddens[0]                       # latent after loop 1

    B, T = x.shape
    target = torch.cat([x[:, 1:], torch.zeros(B, 1, dtype=x.dtype, device=device)], dim=1)
    valid = torch.ones(B, T, dtype=torch.bool, device=device)
    valid[:, -1] = False                           # last position has no target

    ce_per_r = []
    for r in range(n_loops):
        lg = per_loop_logits[r]                    # [B, T, V]
        ce = F.cross_entropy(
            lg.reshape(-1, lg.size(-1)),
            target.reshape(-1),
            reduction="none",
        ).reshape(B, T)
        ce_per_r.append(ce)
    ce_stack = torch.stack(ce_per_r, dim=-1)       # [B, T, n_loops]
    argmin = ce_stack.argmin(dim=-1)               # [B, T]
    return h_1, argmin, valid


def train_halt_classifier(*, model, val_bin_path: str | Path,
                            n_loops: int, d_model: int, device: str,
                            cfg: HaltCfg | None = None) -> dict:
    """Train a halt classifier on a frozen base. val_bin is the source of
    sequences for argmin label generation."""
    cfg = cfg or HaltCfg()
    val_bin_path = Path(val_bin_path)
    arr = np.memmap(str(val_bin_path), dtype=np.uint32, mode="r")

    cap = torch.cuda.get_device_capability(0) if device == "cuda" else (0, 0)
    dtype = torch.bfloat16 if cap[0] >= 8 else (torch.float16 if cap[0] >= 7 else torch.float32)

    classifier = HaltClassifier(d_model=d_model, n_loops=n_loops, hidden=cfg.hidden).to(device)
    opt = torch.optim.AdamW(classifier.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    log: list[dict] = []
    t0 = time.time()

    rng = np.random.default_rng(0)
    for step in range(cfg.steps):
        # Sample a batch from val.bin
        x_buf = np.empty((cfg.batch_size, cfg.block_size), dtype=np.int64)
        for i in range(cfg.batch_size):
            start = int(rng.integers(0, len(arr) - cfg.block_size - 1))
            x_buf[i] = arr[start: start + cfg.block_size].astype(np.int64)
        x = torch.from_numpy(x_buf).to(device)

        h_1, argmin, valid = _argmin_targets(model, x, n_loops, dtype)
        logits = classifier(h_1.float())                        # [B, T, n_loops]
        loss = F.cross_entropy(logits[valid].reshape(-1, n_loops),
                                argmin[valid].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

        if (step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1:
            with torch.no_grad():
                pred_r = logits.argmax(-1)
                acc = ((pred_r == argmin) & valid).float().sum() / valid.float().sum().clamp(min=1)
                # Counts of predicted r vs argmin r, plus per-r mean CE.
                counts_pred = torch.bincount(pred_r[valid], minlength=n_loops).cpu().tolist()
                counts_argmin = torch.bincount(argmin[valid], minlength=n_loops).cpu().tolist()
            entry = {"step": step + 1, "loss": loss.item(),
                     "argmin_acc": acc.item(),
                     "pred_dist": counts_pred,
                     "argmin_dist": counts_argmin,
                     "wall": time.time() - t0}
            log.append(entry)
            print(f"  halt step {step+1:>4}  loss={loss.item():.3f}  "
                  f"argmin_acc={acc.item():.3f}", flush=True)

    return {"halt_state": {k: v.cpu() for k, v in classifier.state_dict().items()},
            "halt_cfg": dataclasses.asdict(cfg),
            "log": log,
            "wall_sec": time.time() - t0}


@torch.no_grad()
def eval_with_halt(model, halt: HaltClassifier, val_bin_path: str | Path,
                    n_loops: int, device: str, n_batches: int = 16,
                    batch_size: int = 4, block_size: int = 2048) -> dict:
    """Eval: per-token apply halt, compute loss at predicted-best loop.

    Compare to fixed r=1, fixed r=n_loops, oracle argmin.
    """
    val_bin_path = Path(val_bin_path)
    arr = np.memmap(str(val_bin_path), dtype=np.uint32, mode="r")

    cap = torch.cuda.get_device_capability(0) if device == "cuda" else (0, 0)
    dtype = torch.bfloat16 if cap[0] >= 8 else (torch.float16 if cap[0] >= 7 else torch.float32)

    model.eval(); halt.eval()

    rng = np.random.default_rng(1)
    fixed_r1, fixed_rN, oracle, classifier_loss = [], [], [], []
    mean_halt_r = []
    halt_dist = [0] * n_loops
    for _ in range(n_batches):
        x_buf = np.empty((batch_size, block_size), dtype=np.int64)
        for i in range(batch_size):
            start = int(rng.integers(0, len(arr) - block_size - 1))
            x_buf[i] = arr[start: start + block_size].astype(np.int64)
        x = torch.from_numpy(x_buf).to(device)

        h_1, argmin, valid = _argmin_targets(model, x, n_loops, dtype)
        with torch.amp.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
            out = model.forward_with_aux(x, n_loops=n_loops, aux_min_loops=1)
        per_loop_logits = out["per_loop_logits"]   # list of [B, T, V]

        target = torch.cat([x[:, 1:], torch.zeros(batch_size, 1, dtype=x.dtype, device=device)], dim=1)

        # Fixed r=1 and r=N
        for arr_list, r in [(fixed_r1, 0), (fixed_rN, n_loops - 1)]:
            ce = F.cross_entropy(per_loop_logits[r].reshape(-1, per_loop_logits[r].size(-1)),
                                  target.reshape(-1), reduction="none").reshape(target.shape)
            arr_list.append(ce[valid].mean().item())

        # Oracle (argmin)
        ce_stack = torch.stack([
            F.cross_entropy(per_loop_logits[r].reshape(-1, per_loop_logits[r].size(-1)),
                            target.reshape(-1), reduction="none").reshape(target.shape)
            for r in range(n_loops)
        ], dim=-1)                                  # [B, T, n_loops]
        oracle.append(ce_stack[valid].min(dim=-1).values.mean().item())

        # Classifier
        pred_r = halt(h_1.float()).argmax(-1)       # [B, T] in {0..n_loops-1}
        # Gather CE at predicted r per position
        cls_ce = ce_stack.gather(-1, pred_r.unsqueeze(-1)).squeeze(-1)
        classifier_loss.append(cls_ce[valid].mean().item())
        mean_halt_r.append(((pred_r[valid] + 1).float()).mean().item())  # 1-indexed
        for r in range(n_loops):
            halt_dist[r] += int((pred_r[valid] == r).sum().item())

    def _mean(xs): return float(sum(xs) / len(xs))
    return {
        "fixed_r=1": _mean(fixed_r1),
        f"fixed_r={n_loops}": _mean(fixed_rN),
        "oracle_argmin": _mean(oracle),
        "classifier_halt": _mean(classifier_loss),
        "mean_halt_r": _mean(mean_halt_r),
        "halt_dist": halt_dist,
    }
