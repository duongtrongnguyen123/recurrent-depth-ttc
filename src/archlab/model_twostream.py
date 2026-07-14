"""Two-stream parallel architecture (Branch 1 Result AQ).

Stream A: small vanilla stack (n_vanilla_blocks deep, single pass).
Stream B: 1 block looped n_loops times.
Combiner: per-position learned scalar gate, h = g*a + (1-g)*b.

Matched-compute comparison against pure-looped: total block-evals per token
= n_vanilla_blocks + n_loops. Stream A captures patterns; stream B handles
iteration. Gate observation diagnoses whether the model actually uses both.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model import Block


class TwoStreamTransformer(nn.Module):
    def __init__(self, vocab: int, max_len: int, d: int = 512,
                  n_heads: int = 8, ff_mult: int = 4,
                  n_vanilla_blocks: int = 2, n_loops: int = 6,
                  zero_init_gate: bool = False, aux_head: bool = False,
                  use_skip_loop: bool = False, skip_alpha_init: float = 0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.vanilla_blocks = nn.ModuleList([
            Block(d, n_heads, ff_mult) for _ in range(n_vanilla_blocks)
        ])
        self.loop_block = Block(d, n_heads, ff_mult)
        self.gate = nn.Linear(2 * d, 1)
        if zero_init_gate:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        # Auxiliary head on stream B (looped) alone, used when aux_head=True.
        # Ensures gradient flows to loop_block even if gate saturates to 1.
        self.aux_head = aux_head
        if aux_head:
            self.ln_b = nn.LayerNorm(d)
            self.head_b = nn.Linear(d, vocab, bias=False)
        self.n_loops = n_loops
        self.n_vanilla_blocks = n_vanilla_blocks
        self.use_skip_loop = use_skip_loop
        if use_skip_loop:
            self.skip_alpha = nn.Parameter(torch.tensor(skip_alpha_init))
        self._causal: torch.Tensor | None = None

    def stream_b_logits(self, b: torch.Tensor) -> torch.Tensor:
        assert self.aux_head, "stream_b_logits requires aux_head=True"
        return self.head_b(self.ln_b(b))

    def forward_all_loops_dual(self, x: torch.Tensor,
                                 n_loops: int | None = None
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (combined_logits_per_loop, stream_b_logits_per_loop).
        Both shaped [n_loops+1, B, T, V]. Stream-B logits use ln_b/head_b
        (different head than combined output) so each stream has its own
        objective.
        """
        n_loops = n_loops if n_loops is not None else self.n_loops
        mask = self._causal_mask(x.shape[1], x.device)
        h0 = self._embed(x)
        a = self._vanilla_pass(h0, mask)
        b = h0
        combined = [self._combine_logits(a, b)]
        b_outs = [self.stream_b_logits(b)] if self.aux_head else None
        for _ in range(n_loops):
            b = self._loop_step(b, h0, mask)
            combined.append(self._combine_logits(a, b))
            if self.aux_head:
                b_outs.append(self.stream_b_logits(b))
        return torch.stack(combined, dim=0), (torch.stack(b_outs, dim=0)
                                                if self.aux_head else None)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        cached = self._causal
        if cached is None or cached.size(0) < T or cached.device != device:
            cached = torch.triu(torch.full((T, T), float("-inf"), device=device),
                                 diagonal=1)
            self._causal = cached
        return cached[:T, :T]

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        return self.tok(x) + self.pos(pos)[None]

    def _vanilla_pass(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for blk in self.vanilla_blocks:
            h = blk(h, mask)
        return h

    def _combine_logits(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))  # [B,T,1]
        h = g * a + (1 - g) * b
        return self.head(self.ln_f(h))

    def _combine_with_gate(self, a: torch.Tensor, b: torch.Tensor):
        g_raw = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))  # [B,T,1]
        h = g_raw * a + (1 - g_raw) * b
        logits = self.head(self.ln_f(h))
        return logits, g_raw.squeeze(-1)

    def _loop_step(self, b: torch.Tensor, h0: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        out = self.loop_block(b, mask)
        if self.use_skip_loop:
            out = out + self.skip_alpha * h0
        return out

    def forward(self, x: torch.Tensor, n_loops: int | None = None) -> torch.Tensor:
        n_loops = n_loops if n_loops is not None else self.n_loops
        mask = self._causal_mask(x.shape[1], x.device)
        h0 = self._embed(x)
        a = self._vanilla_pass(h0, mask)
        b = h0
        for _ in range(n_loops):
            b = self._loop_step(b, h0, mask)
        return self._combine_logits(a, b)

    def forward_all_loops_grad(self, x: torch.Tensor,
                                n_loops: int | None = None) -> torch.Tensor:
        n_loops = n_loops if n_loops is not None else self.n_loops
        mask = self._causal_mask(x.shape[1], x.device)
        h0 = self._embed(x)
        a = self._vanilla_pass(h0, mask)
        b = h0
        outs = [self._combine_logits(a, b)]
        for _ in range(n_loops):
            b = self._loop_step(b, h0, mask)
            outs.append(self._combine_logits(a, b))
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_all_loops(self, x: torch.Tensor,
                            n_loops: int | None = None) -> torch.Tensor:
        return self.forward_all_loops_grad(x, n_loops=n_loops)

    @torch.no_grad()
    def gate_diagnostic(self, x: torch.Tensor,
                          n_loops: int | None = None) -> torch.Tensor:
        n_loops = n_loops if n_loops is not None else self.n_loops
        mask = self._causal_mask(x.shape[1], x.device)
        h0 = self._embed(x)
        a = self._vanilla_pass(h0, mask)
        b = h0
        for _ in range(n_loops):
            b = self._loop_step(b, h0, mask)
        _, g = self._combine_with_gate(a, b)
        return g

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
