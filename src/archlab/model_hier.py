"""Hierarchical Recurrent Transformer: outer × inner loops.

Architecture:
  - INNER loop: standard recurrent block, run K_inner times per outer step
  - OUTER loop: a SEPARATE shared block applied AFTER each inner cycle,
    receives both current state AND a "goal" register
  - GOAL: a learned vector (1 token slot) that the OUTER loop updates
    each outer step

Forward:
    h = embed(x)              # [B, T, d]
    goal = self.goal_init.expand(B, 1, d)
    for t_outer in 1..K_outer:
        for t_inner in 1..K_inner:
            h = inner_block(h + goal, mask)
        goal = outer_block(torch.cat([h.mean(1, keepdim=True), goal], dim=1), no_mask)[:, 1:]
    return head(ln_f(h))

Total computation = K_outer × (K_inner inner steps + 1 outer step).
The OUTER block updates goal based on summarized inner state — a kind of
working-memory loop.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from .model import Block


class HierLoopedTransformer(nn.Module):
    def __init__(self, vocab, max_len, d, n_heads, ff_mult,
                  n_inner=4, n_outer=2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.inner = Block(d, n_heads, ff_mult)
        self.outer = Block(d, n_heads, ff_mult)
        self.goal_init = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.n_inner = n_inner
        self.n_outer = n_outer
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self._causal: torch.Tensor | None = None

    def _causal_mask(self, T, device):
        c = self._causal
        if c is None or c.size(0) < T or c.device != device:
            c = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
            self._causal = c
        return c[:T, :T]

    def _embed(self, x):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        return self.tok(x) + self.pos(pos)[None]

    def forward(self, x, n_outer=None, n_inner=None):
        n_outer = n_outer if n_outer is not None else self.n_outer
        n_inner = n_inner if n_inner is not None else self.n_inner
        B, T = x.shape
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        goal = self.goal_init.expand(B, 1, h.size(-1)).contiguous()
        for t_outer in range(n_outer):
            for t_inner in range(n_inner):
                h = self.inner(h + goal, mask)
            # Outer step: summarize h, update goal
            h_summary = h.mean(dim=1, keepdim=True)   # [B, 1, d]
            outer_in = torch.cat([h_summary, goal], dim=1)  # [B, 2, d]
            outer_out = self.outer(outer_in, attn_mask=None)
            goal = outer_out[:, 1:2]
        return self.head(self.ln_f(h))

    def forward_all_loops_grad(self, x, n_loops=None):
        """Returns logits at every (t_outer, t_inner) step. n_loops = n_outer * n_inner."""
        if n_loops is None:
            n_loops = self.n_outer * self.n_inner
        n_outer = self.n_outer
        n_inner = self.n_inner
        B, T = x.shape
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        goal = self.goal_init.expand(B, 1, h.size(-1)).contiguous()
        outs = [self.head(self.ln_f(h))]
        for t_outer in range(n_outer):
            for t_inner in range(n_inner):
                h = self.inner(h + goal, mask)
                outs.append(self.head(self.ln_f(h)))
            h_summary = h.mean(dim=1, keepdim=True)
            outer_in = torch.cat([h_summary, goal], dim=1)
            outer_out = self.outer(outer_in, attn_mask=None)
            goal = outer_out[:, 1:2]
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_all_loops(self, x, n_loops=None):
        return self.forward_all_loops_grad(x, n_loops)
