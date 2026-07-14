"""Anchor cross-attention looped transformer (Branch 1 Result AE candidate).

Architectural mix of vanilla and looped:
- 2 *distinct* anchor blocks produce a fixed latent A from the embedding
- 1 shared "core" looped block that, at every loop, applies:
  * self-attention over its own running state
  * cross-attention to the fixed anchor latent A
  * SwiGLU MLP

The cross-attention provides a stable reference at every loop so the iterative
state cannot drift away from the original input semantics. Distinct anchor
blocks give vanilla-style parameter capacity for input encoding; shared core
gives looped-style depth via inference loops.

Compared to PCC (which applies coda *after* iteration), here the anchor is
*inside* the iteration loop — every loop sees the anchor.

Compared to pure-looped (chain_iter from Result L), the per-step rule now has
access to the anchor's representation, not just self-state.

Hypothesis: at chain task, anchor prevents drift past trained depth → better
length extrapolation than pure-looped.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorBlock(nn.Module):
    """Standard self-attention + MLP block (used for the anchor stack)."""
    def __init__(self, d: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, ff_mult * d, bias=False),
            nn.GELU(),
            nn.Linear(ff_mult * d, d, bias=False),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class CoreBlockWithCrossAttn(nn.Module):
    """Looped core: self-attention + cross-attention to anchor + MLP."""
    def __init__(self, d: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.ln_self = nn.LayerNorm(d)
        self.ln_cross = nn.LayerNorm(d)
        self.ln_mlp = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, bias=False)
        # Cross-attention: query from current state, key/value from anchor latent A.
        # Use NO causal mask for cross-attention (full attention to anchor positions).
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, ff_mult * d, bias=False),
            nn.GELU(),
            nn.Linear(ff_mult * d, d, bias=False),
        )

    def forward(self, x: torch.Tensor, anchor: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention with causal mask.
        h = self.ln_self(x)
        a_self, _ = self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a_self
        # Cross-attention to fixed anchor (no causal mask — anchor is "context").
        h = self.ln_cross(x)
        a_cross, _ = self.cross_attn(h, anchor, anchor, need_weights=False)
        x = x + a_cross
        # MLP.
        x = x + self.mlp(self.ln_mlp(x))
        return x


class AnchorLoopedTransformer(nn.Module):
    """Hybrid: distinct anchor blocks + shared looped core with cross-attention to anchor."""

    def __init__(self, vocab: int, max_len: int, d: int = 1024,
                 n_heads: int = 8, ff_mult: int = 4,
                 n_anchor: int = 2, n_loops: int = 8):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.anchor_blocks = nn.ModuleList(
            [AnchorBlock(d, n_heads, ff_mult) for _ in range(n_anchor)])
        self.core = CoreBlockWithCrossAttn(d, n_heads, ff_mult)
        self.n_loops = n_loops
        self.n_anchor = n_anchor
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

    def _anchor(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the anchor stack — distinct blocks, applied once."""
        h = self._embed(x)
        for blk in self.anchor_blocks:
            h = blk(h, mask)
        return h

    def forward(self, x: torch.Tensor, n_loops: int | None = None) -> torch.Tensor:
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        mask = self._causal_mask(T, x.device)
        anchor = self._anchor(x, mask)
        # Initialize core state from the anchor (or just embedding — chose embed for symmetry
        # with chain_iter).
        h = anchor
        for _ in range(n_loops):
            h = self.core(h, anchor, mask)
        return self.head(self.ln_f(h))

    def forward_all_loops_grad(self, x: torch.Tensor,
                                n_loops: int | None = None) -> torch.Tensor:
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        mask = self._causal_mask(T, x.device)
        anchor = self._anchor(x, mask)
        h = anchor
        outs = [self.head(self.ln_f(h))]
        for _ in range(n_loops):
            h = self.core(h, anchor, mask)
            outs.append(self.head(self.ln_f(h)))
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_all_loops(self, x: torch.Tensor,
                           n_loops: int | None = None) -> torch.Tensor:
        return self.forward_all_loops_grad(x, n_loops)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
