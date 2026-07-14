"""Skip-connected looped transformer (Branch 1 Result AJ candidate).

Standard looped block with an ADDITIVE residual from the initial embedding
h_0 to every loop's output:

    h_{r+1} = block(h_r) + alpha * h_0

where alpha is a learnable scalar (initialized small). Provides a stable
anchor at every loop *without* introducing an attention shortcut path
(contrast to AE's cross-attention to anchor, which created a shortcut and
broke iter-target).

The per-step rule is unchanged — block(h_r) still has to compute the
right per-step transformation. The skip just prevents the iteration from
drifting too far from the input semantics.

Hypothesis: stability via additive residual gives slightly better
extrapolation than pure-looped (Result L), without the param cost of MoE
(Result AG) or the shortcut failure of cross-attention (AE).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model import Block


class SkipLoopedTransformer(nn.Module):
    """Pure-looped block + per-loop additive residual from h_0."""

    def __init__(self, vocab: int, max_len: int, d: int = 1024,
                 n_heads: int = 8, ff_mult: int = 4, n_loops: int = 8,
                 init_alpha: float = 0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.block = Block(d, n_heads, ff_mult)
        self.n_loops = n_loops
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        # Learnable mixing scalar; init small so initial behavior ≈ pure-looped.
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self._causal: torch.Tensor | None = None

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        cached = self._causal
        if cached is None or cached.size(0) < T or cached.device != device:
            cached = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
            self._causal = cached
        return cached[:T, :T]

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        return self.tok(x) + self.pos(pos)[None]

    def forward(self, x: torch.Tensor, n_loops: int | None = None) -> torch.Tensor:
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h0 = self._embed(x)
        mask = self._causal_mask(T, x.device)
        h = h0
        for _ in range(n_loops):
            h = self.block(h, mask) + self.alpha * h0
        return self.head(self.ln_f(h))

    def forward_all_loops_grad(self, x: torch.Tensor,
                                n_loops: int | None = None) -> torch.Tensor:
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h0 = self._embed(x)
        mask = self._causal_mask(T, x.device)
        h = h0
        outs = [self.head(self.ln_f(h))]
        for _ in range(n_loops):
            h = self.block(h, mask) + self.alpha * h0
            outs.append(self.head(self.ln_f(h)))
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_all_loops(self, x: torch.Tensor,
                           n_loops: int | None = None) -> torch.Tensor:
        return self.forward_all_loops_grad(x, n_loops)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
