"""PCC (Prelude-Core-Coda) looped transformer for Branch 2 architecture parity.

Architecture from Huginn / Branch 2 Phase 1 winner:
  embed(x) → prelude_1 → prelude_2 → core × N (shared, looped) → coda_1 → coda_2 → ln_f → head

For iter-target supervision (Result L recipe), output at loop r is the result
of applying core r times (then coda + head). The model outputs at every loop;
the per-loop target is f^r(start).

Cost note: each per-loop output requires running the coda + ln_f + head, so a
forward pass that supervises r=1..N applies coda N times. Training cost is
N× higher per step than vanilla forward_all_loops (which only applies head N
times). For N=8 and small d, this is still cheap.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model import Block


class PCCLoopedTransformer(nn.Module):
    """Prelude (n_prelude distinct blocks) + Core (1 shared block, looped n_loops)
    + Coda (n_coda distinct blocks). Per-loop output = head ∘ ln_f ∘ coda(...) ∘
    core^r(prelude(embed(x))). Output at loop 0 has no core applications.
    """

    def __init__(self, vocab: int, max_len: int, d: int = 1024, n_heads: int = 8,
                 ff_mult: int = 4, n_loops: int = 8,
                 n_prelude: int = 2, n_coda: int = 2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.prelude = nn.ModuleList([Block(d, n_heads, ff_mult)
                                       for _ in range(n_prelude)])
        self.core = Block(d, n_heads, ff_mult)
        self.coda = nn.ModuleList([Block(d, n_heads, ff_mult)
                                    for _ in range(n_coda)])
        self.n_loops = n_loops
        self.n_prelude = n_prelude
        self.n_coda = n_coda
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

    def _apply_coda_head(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply coda blocks then ln_f + head. Used per loop for output."""
        for blk in self.coda:
            h = blk(h, mask)
        return self.head(self.ln_f(h))

    def forward(self, x: torch.Tensor, n_loops: int | None = None) -> torch.Tensor:
        """Standard forward — returns final logits after running n_loops core iterations."""
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        for blk in self.prelude:
            h = blk(h, mask)
        for _ in range(n_loops):
            h = self.core(h, mask)
        return self._apply_coda_head(h, mask)

    def forward_all_loops_grad(self, x: torch.Tensor, n_loops: int | None = None
                                ) -> torch.Tensor:
        """Per-loop logits with grads. Applies coda separately at every loop."""
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        for blk in self.prelude:
            h = blk(h, mask)
        outs = [self._apply_coda_head(h, mask)]
        for _ in range(n_loops):
            h = self.core(h, mask)
            outs.append(self._apply_coda_head(h, mask))
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_all_loops(self, x: torch.Tensor, n_loops: int | None = None
                          ) -> torch.Tensor:
        return self.forward_all_loops_grad(x, n_loops)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_all_loops_robust(self, x: torch.Tensor, n_loops: int,
                                  p_noise: float = 0.0,
                                  noise_alpha: float = 0.1) -> torch.Tensor:
        """Per-loop logits with optional Gaussian noise injection (Result R recipe)
        applied to the latent at a randomly chosen loop. Coda is applied per loop.
        """
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        for blk in self.prelude:
            h = blk(h, mask)
        outs = [self._apply_coda_head(h, mask)]
        inject_at = -1
        if torch.rand(1).item() < p_noise:
            inject_at = torch.randint(1, n_loops + 1, (1,)).item()
        for r in range(1, n_loops + 1):
            h = self.core(h, mask)
            if r == inject_at:
                noise_std = noise_alpha * h.norm(dim=-1, keepdim=True) / (h.shape[-1] ** 0.5)
                h = h + torch.randn_like(h) * noise_std
            outs.append(self._apply_coda_head(h, mask))
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_all_loops_with_hidden(self, x: torch.Tensor,
                                       n_loops: int | None = None
                                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """For halt-head training. Returns (hidden_states, logits) at every loop.
        hidden = pre-head latent (post coda + ln_f), shape [n_loops+1, B, T, d].
        """
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        for blk in self.prelude:
            h = blk(h, mask)

        def coda_to_hidden(state):
            s = state
            for blk in self.coda:
                s = blk(s, mask)
            return self.ln_f(s)

        h_outs = [coda_to_hidden(h)]
        l_outs = [self.head(h_outs[0])]
        for _ in range(n_loops):
            h = self.core(h, mask)
            hh = coda_to_hidden(h)
            h_outs.append(hh)
            l_outs.append(self.head(hh))
        return torch.stack(h_outs, dim=0), torch.stack(l_outs, dim=0)
