"""Skip-connected loop + noise injection (Branch 1 Result AK candidate).

Combines AJ's additive skip residual from h_0 with Branch 1 Result R's
latent noise injection. Hypothesis: skip provides anchored stability + noise
provides per-step rule contraction → push extrap past Result R's 4× (78% at r=32).

13M params (same as pure-looped) + 1 learnable alpha scalar.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model import Block


class SkipLoopedRobustTransformer(nn.Module):
    """Skip-connected loop with optional latent noise injection at random loop."""

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

    def forward_all_loops_robust(self, x: torch.Tensor, n_loops: int,
                                  p_noise: float = 0.0,
                                  noise_alpha: float = 0.1) -> torch.Tensor:
        T = x.shape[1]
        h0 = self._embed(x)
        mask = self._causal_mask(T, x.device)
        h = h0
        outs = [self.head(self.ln_f(h))]
        inject_at = -1
        if torch.rand(1).item() < p_noise:
            inject_at = torch.randint(1, n_loops + 1, (1,)).item()
        for r in range(1, n_loops + 1):
            h = self.block(h, mask) + self.alpha * h0
            if r == inject_at:
                noise_std = noise_alpha * h.norm(dim=-1, keepdim=True) / (h.shape[-1] ** 0.5)
                h = h + torch.randn_like(h) * noise_std
            outs.append(self.head(self.ln_f(h)))
        return torch.stack(outs, dim=0)

    def forward(self, x: torch.Tensor, n_loops: int | None = None) -> torch.Tensor:
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h0 = self._embed(x)
        mask = self._causal_mask(T, x.device)
        h = h0
        for _ in range(n_loops):
            h = self.block(h, mask) + self.alpha * h0
        return self.head(self.ln_f(h))

    @torch.no_grad()
    def forward_all_loops(self, x: torch.Tensor,
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

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
