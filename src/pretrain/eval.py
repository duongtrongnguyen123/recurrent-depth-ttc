"""Tier-A eval: chain-task accuracy at variable r and val perplexity at varying n_loops."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def val_loss_at_loops(model, eval_loader, n_loops_list: list[int],
                      n_batches: int = 16, micro_batch: int = 8) -> dict[int, float]:
    """Validation loss as a function of inference-time n_loops."""
    model.eval()
    out: dict[int, float] = {}
    for n in n_loops_list:
        losses = []
        for _ in range(n_batches):
            x, y = eval_loader.get_batch(micro_batch, device="cuda")
            logits = model(x, n_loops=n)["logits"]
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                          y.reshape(-1)).item())
        out[n] = float(np.mean(losses))
    model.train()
    return out
