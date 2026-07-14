"""Mixture of Looped Experts (Branch 1 Result AG candidate).

K shared "expert" blocks. At each loop, a learned router computes a soft
weight per expert based on the current state, and the per-step rule is the
weighted average of all K experts' outputs:

    h_{r+1} = sum_k w_k(h_r) · expert_k(h_r)

Compute per loop: K × block forward (4× pure-looped at K=4). Params: K × block.

Importantly, the router takes h_r (current state) — no shortcut path to a
pre-computed encoding. Each expert sees the same h_r and produces a candidate
update; the router blends them. This preserves iter-target's per-step rule
constraint (no attention paths outside the iteration) while adding capacity.

For chain task: experts may specialize on different "phases" of computation
(early hops vs late, simple table vs complex chain).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Block


class MoECoreBlock(nn.Module):
    """Mixture-of-experts core: K shared blocks + soft router on pooled state."""

    def __init__(self, n_experts: int, d: int, n_heads: int, ff_mult: int = 4,
                 router_pos: int = -1):
        """router_pos: which sequence position to pool for the router input.
        -1 = last position; for chain task with answer at position V+1, set
        explicitly to V+1 via the model wrapper."""
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList(
            [Block(d, n_heads, ff_mult) for _ in range(n_experts)])
        self.router = nn.Linear(d, n_experts, bias=False)
        self.router_pos = router_pos

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Router: pool at router_pos, get logits over experts.
        rp = self.router_pos if self.router_pos >= 0 else x.size(1) + self.router_pos
        pooled = x[:, rp, :]                                # [B, d]
        logits = self.router(pooled)                         # [B, K]
        weights = F.softmax(logits, dim=-1)                  # [B, K]
        # Apply each expert
        expert_outs = [e(x, attn_mask) for e in self.experts]  # K of [B, T, d]
        stack = torch.stack(expert_outs, dim=0)              # [K, B, T, d]
        # Weighted combine
        w = weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)  # [K, B, 1, 1]
        return (w * stack).sum(dim=0)                        # [B, T, d]

    def expert_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return router weights (for analysis). Shape [B, K]."""
        rp = self.router_pos if self.router_pos >= 0 else x.size(1) + self.router_pos
        return F.softmax(self.router(x[:, rp, :]), dim=-1)


class MoELoopedTransformer(nn.Module):
    """Looped MoE: at every loop, the K experts vote (weighted by router) on
    the per-step update."""

    def __init__(self, vocab: int, max_len: int, d: int = 1024,
                 n_heads: int = 8, ff_mult: int = 4, n_loops: int = 8,
                 n_experts: int = 4, router_pos: int = -1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.core = MoECoreBlock(n_experts=n_experts, d=d, n_heads=n_heads,
                                  ff_mult=ff_mult, router_pos=router_pos)
        self.n_loops = n_loops
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
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
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        for _ in range(n_loops):
            h = self.core(h, mask)
        return self.head(self.ln_f(h))

    def forward_all_loops_grad(self, x: torch.Tensor,
                                n_loops: int | None = None) -> torch.Tensor:
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        outs = [self.head(self.ln_f(h))]
        for _ in range(n_loops):
            h = self.core(h, mask)
            outs.append(self.head(self.ln_f(h)))
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_all_loops(self, x: torch.Tensor,
                           n_loops: int | None = None) -> torch.Tensor:
        return self.forward_all_loops_grad(x, n_loops)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
