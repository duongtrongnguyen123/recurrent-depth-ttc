"""OpenMythos faithful LTI: block sees h+e, plus loop-index embedding.

  h_loop = h + loop_pos_embed(t)           # loop index signal
  combined = LayerNorm(h_loop + e)         # block input is h+e
  t_out = block(combined)
  h = A·h + B·e + t_out                    # LTI injection
"""
from __future__ import annotations
import math
import torch, torch.nn as nn
from .model import Block


class LTIInjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def forward(self, h, e, transformer_out):
        A = torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))
        return A * h + self.B * e + transformer_out


def loop_pos_embed(t, dim, device):
    """Sinusoidal loop-index embedding for time t (loop number)."""
    div = torch.exp(torch.arange(0, dim, 2, device=device).float()
                    * -(math.log(10000.0) / dim))
    pe = torch.zeros(dim, device=device)
    val = float(t) * div
    pe[0::2] = torch.sin(val)
    pe[1::2] = torch.cos(val)
    return pe  # [dim]


class FaithfulLTILoopedTransformer(nn.Module):
    """OpenMythos-style: block(h+e), LTI injection, loop-index embedding."""
    def __init__(self, vocab, max_len, d, n_heads, ff_mult, n_loops):
        super().__init__()
        self.d = d; self.n_loops = n_loops; self.vocab = vocab
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.block = Block(d, n_heads, ff_mult)
        self.injection = LTIInjection(d)
        self.norm_combined = nn.LayerNorm(d)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self._causal = None

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _causal_mask(self, T, dev):
        if self._causal is None or self._causal.size(0) < T or self._causal.device != dev:
            self._causal = torch.triu(torch.full((T, T), float('-inf'), device=dev), diagonal=1)
        return self._causal[:T, :T]

    def _embed(self, x):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        return self.tok(x) + self.pos(pos)[None]

    def forward_all_loops_grad(self, x, n_loops=None):
        if n_loops is None: n_loops = self.n_loops
        T = x.shape[1]
        e = self._embed(x)
        h = e.clone()
        mask = self._causal_mask(T, x.device)
        outs = [self.head(self.ln_f(h))]
        for r in range(1, n_loops + 1):
            # Loop-index embedding
            lpe = loop_pos_embed(r - 1, self.d, x.device)
            h_loop = h + lpe[None, None, :]
            # Block input is h+e (faithful to OpenMythos)
            combined = self.norm_combined(h_loop + e)
            t_out = self.block(combined, mask)
            # LTI injection
            h = self.injection(h, e, t_out)
            outs.append(self.head(self.ln_f(h)))
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_all_loops(self, x, n_loops=None):
        return self.forward_all_loops_grad(x, n_loops)

    def forward(self, x, n_loops=None):
        return self.forward_all_loops_grad(x, n_loops)[-1]
