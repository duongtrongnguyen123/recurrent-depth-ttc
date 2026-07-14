"""LTI-injection looped transformer (OpenMythos / Parcae-style).

Update rule: h_{t+1} = A * h_t + B * e + Block(h_t)
where:
  - A is per-channel ∈ (0, 1): A = exp(-exp(log_dt + log_A))
  - B is per-channel scalar (init 0.1)
  - e is the post-embedding state (frozen across loops)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .model import Block


class LTIInjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self):
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(self, h, e, transformer_out):
        A = self.get_A()
        return A * h + self.B * e + transformer_out


class LTILoopedTransformer(nn.Module):
    """Looped transformer with LTI input re-injection per loop."""
    def __init__(self, vocab, max_len, d, n_heads, ff_mult, n_loops):
        super().__init__()
        self.vocab = vocab; self.d = d; self.n_loops = n_loops
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.block = Block(d, n_heads, ff_mult)
        self.injection = LTIInjection(d)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self._causal = None

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _causal_mask(self, T, device):
        if self._causal is None or self._causal.size(0) < T or self._causal.device != device:
            self._causal = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        return self._causal[:T, :T]

    def _embed(self, x):
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        return self.tok(x) + self.pos(pos)[None]

    def forward_all_loops_grad(self, x, n_loops=None):
        if n_loops is None:
            n_loops = self.n_loops
        T = x.shape[1]
        e = self._embed(x)
        h = e
        mask = self._causal_mask(T, x.device)
        outs = [self.head(self.ln_f(h))]
        for r in range(1, n_loops + 1):
            t_out = self.block(h, mask)
            h = self.injection(h, e, t_out)
            outs.append(self.head(self.ln_f(h)))
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_all_loops(self, x, n_loops=None):
        return self.forward_all_loops_grad(x, n_loops)

    def forward(self, x, n_loops=None):
        return self.forward_all_loops_grad(x, n_loops)[-1]
