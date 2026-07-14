"""Position-register augmented LoopedTransformer.

Adds a per-loop sinusoidal encoding to the hidden state at every loop:

    for r in range(1, n_loops+1):
        h = block(h + loop_pos_enc(r), mask)

Sinusoidal (NOT learned) so loop_pos_enc(r) at unseen r > n_loops_train
extrapolates smoothly from the training pattern. Hypothesis: gives the
model an explicit r-counter inside the loop core, breaking position-driven
walls (parity / Result N) without changing the task spec.

Compatible with iter-target training: same forward_all_loops_grad /
forward_all_loops APIs as RobustLoopedTransformer.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .train_chain_iter_robust import RobustLoopedTransformer


def _sinusoidal_loop_enc(r: int, d: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal positional encoding for loop step r."""
    pe = torch.zeros(d, device=device)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32)
                     * -(math.log(10000.0) / d))
    pe[0::2] = torch.sin(r * div)
    pe[1::2] = torch.cos(r * div)
    return pe


class PosRegLoopedTransformer(RobustLoopedTransformer):
    """RobustLoopedTransformer + sinusoidal loop-step encoding added every loop.

    The encoding is broadcast across batch and sequence dimensions:
      h_r = block(h_{r-1} + loop_pos_enc(r), mask)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cache shape: [n_loops+1, d]
        self._loop_enc_cache: torch.Tensor | None = None
        self._loop_enc_device: torch.device | None = None

    def _ensure_loop_enc(self, n_loops: int, d: int, device: torch.device) -> torch.Tensor:
        if (self._loop_enc_cache is None or
                self._loop_enc_cache.size(0) < n_loops + 1 or
                self._loop_enc_device != device):
            cache = torch.zeros(n_loops + 1, d, device=device)
            for r in range(1, n_loops + 1):
                cache[r] = _sinusoidal_loop_enc(r, d, device)
            self._loop_enc_cache = cache
            self._loop_enc_device = device
        return self._loop_enc_cache

    def forward(self, x, n_loops=None):
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        loop_enc = self._ensure_loop_enc(n_loops, h.size(-1), x.device)
        for r in range(1, n_loops + 1):
            h = self.block(h + loop_enc[r], mask)
        return self.head(self.ln_f(h))

    def forward_all_loops_grad(self, x, n_loops=None):
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        loop_enc = self._ensure_loop_enc(n_loops, h.size(-1), x.device)
        outs = [self.head(self.ln_f(h))]
        for r in range(1, n_loops + 1):
            h = self.block(h + loop_enc[r], mask)
            outs.append(self.head(self.ln_f(h)))
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_all_loops(self, x, n_loops=None):
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        loop_enc = self._ensure_loop_enc(n_loops, h.size(-1), x.device)
        outs = [self.head(self.ln_f(h))]
        for r in range(1, n_loops + 1):
            h = self.block(h + loop_enc[r], mask)
            outs.append(self.head(self.ln_f(h)))
        return torch.stack(outs, dim=0)

    def forward_all_loops_robust(self, x, n_loops, p_noise=0.0,
                                  noise_alpha=0.1, noise_loop=None):
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        loop_enc = self._ensure_loop_enc(n_loops, h.size(-1), x.device)
        outs = [self.head(self.ln_f(h))]
        inject_at = -1
        if torch.rand(1).item() < p_noise:
            inject_at = (noise_loop if noise_loop is not None
                         else torch.randint(1, n_loops + 1, (1,)).item())
        for r in range(1, n_loops + 1):
            h = self.block(h + loop_enc[r], mask)
            if r == inject_at:
                noise_std = noise_alpha * h.norm(dim=-1, keepdim=True) / (h.shape[-1] ** 0.5)
                h = h + torch.randn_like(h) * noise_std
            outs.append(self.head(self.ln_f(h)))
        return torch.stack(outs, dim=0)
