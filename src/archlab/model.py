from __future__ import annotations

import torch
import torch.nn as nn


class Block(nn.Module):
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


class LoopedTransformer(nn.Module):
    """One shared transformer block looped n_loops times.

    Forward returns the final logits. `forward_all_loops` returns logits at every loop,
    enabling per-loop trajectory analysis without rerunning the model.
    """

    def __init__(self, vocab: int, max_len: int, d: int = 1024, n_heads: int = 8,
                 ff_mult: int = 4, n_loops: int = 8):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.block = Block(d, n_heads, ff_mult)
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
            h = self.block(h, mask)
        return self.head(self.ln_f(h))

    def forward_all_loops_grad(self, x: torch.Tensor, n_loops: int | None = None
                                ) -> torch.Tensor:
        """Same as forward_all_loops but with gradients enabled. Used for aux-loss training."""
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        outs = [self.head(self.ln_f(h))]
        for _ in range(n_loops):
            h = self.block(h, mask)
            outs.append(self.head(self.ln_f(h)))
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_all_loops(self, x: torch.Tensor, n_loops: int | None = None
                          ) -> torch.Tensor:
        """Return logits at every loop. Shape [n_loops+1, B, T, V] -- the +1 is loop 0
        (no recurrence, just embedding through coda).
        """
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        outs = [self.head(self.ln_f(h))]
        for _ in range(n_loops):
            h = self.block(h, mask)
            outs.append(self.head(self.ln_f(h)))
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_all_loops_with_hidden(self, x: torch.Tensor,
                                       n_loops: int | None = None
                                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (hidden_states, logits) at every loop. Hidden states are post-ln_f,
        pre-head — the natural input for an external halt head.

        hidden_states : [n_loops+1, B, T, d]
        logits        : [n_loops+1, B, T, V]
        """
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        h_outs = [self.ln_f(h)]
        l_outs = [self.head(h_outs[0])]
        for _ in range(n_loops):
            h = self.block(h, mask)
            h_outs.append(self.ln_f(h))
            l_outs.append(self.head(h_outs[-1]))
        return torch.stack(h_outs, dim=0), torch.stack(l_outs, dim=0)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
