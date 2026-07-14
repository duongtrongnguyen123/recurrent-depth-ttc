"""Self-conditioning looped transformer (Branch 1 Result AF candidate).

The model's per-loop output is fed back as additional context at the next loop.
Concretely: at each loop r, the answer-position latent h_r is appended to a
"thought stream" T = [h_1, h_2, ..., h_r]. At loop r+1, the core block has a
cross-attention layer that attends to T — letting the iteration use its own
prior outputs as scratch.

Compared to anchor (AE): anchor is FIXED across loops; thought-stream is
EVOLVING. Compared to pure looped: pure-looped's only memory is the current
state h; self-conditioning carries an explicit history.

This is essentially "latent chain-of-thought" — each loop r writes a thought
vector h_r into the stream, and subsequent loops can attend to all prior
thoughts. If iter-target supervision works here, the model is doing genuine
multi-step reasoning in latent space.

Cost: cross-attention at loop r is O(r) over thought stream. Total over n_loops:
O(n²). For n=8, this adds 36 extra attention applications across all loops —
~50% overhead vs pure-looped. At n=24 inference, 300 extra attentions —
significant. Still cheap on chain task.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfCondCoreBlock(nn.Module):
    """Core block with self-attention + cross-attention to thought stream."""
    def __init__(self, d: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        self.ln_self = nn.LayerNorm(d)
        self.ln_thought = nn.LayerNorm(d)
        self.ln_mlp = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, bias=False)
        self.thought_attn = nn.MultiheadAttention(d, n_heads, batch_first=True, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, ff_mult * d, bias=False),
            nn.GELU(),
            nn.Linear(ff_mult * d, d, bias=False),
        )

    def forward(self, x: torch.Tensor, thought_stream: torch.Tensor | None,
                attn_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention with causal mask.
        h = self.ln_self(x)
        a_self, _ = self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a_self
        # Cross-attention to thought stream (if non-empty). Each query position
        # attends to all entries in the thought stream.
        if thought_stream is not None and thought_stream.size(1) > 0:
            h = self.ln_thought(x)
            # thought_stream shape: [B, num_thoughts, d]
            a_thought, _ = self.thought_attn(h, thought_stream, thought_stream,
                                              need_weights=False)
            x = x + a_thought
        # MLP.
        x = x + self.mlp(self.ln_mlp(x))
        return x


class SelfCondLoopedTransformer(nn.Module):
    """Looped transformer with self-conditioning thought stream."""

    def __init__(self, vocab: int, max_len: int, d: int = 1024,
                 n_heads: int = 8, ff_mult: int = 4, n_loops: int = 8,
                 thought_position: int = -1):
        """thought_position: which sequence position contributes to the thought
        stream. -1 = last position; we'll use the answer position when predicting
        chain answers, set externally via the data layout's ANSWER_POS.
        """
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.core = SelfCondCoreBlock(d, n_heads, ff_mult)
        self.n_loops = n_loops
        self.thought_position = thought_position
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
        thought_stream = None
        for _ in range(n_loops):
            h = self.core(h, thought_stream, mask)
            # Append answer-position h to thought stream
            tp = self.thought_position if self.thought_position >= 0 else T + self.thought_position
            new_thought = h[:, tp:tp+1, :]   # [B, 1, d]
            thought_stream = (new_thought if thought_stream is None
                              else torch.cat([thought_stream, new_thought], dim=1))
        return self.head(self.ln_f(h))

    def forward_all_loops_grad(self, x: torch.Tensor,
                                n_loops: int | None = None) -> torch.Tensor:
        n_loops = n_loops if n_loops is not None else self.n_loops
        T = x.shape[1]
        h = self._embed(x)
        mask = self._causal_mask(T, x.device)
        thought_stream = None
        outs = [self.head(self.ln_f(h))]
        tp = self.thought_position if self.thought_position >= 0 else T + self.thought_position
        for _ in range(n_loops):
            h = self.core(h, thought_stream, mask)
            new_thought = h[:, tp:tp+1, :]
            thought_stream = (new_thought if thought_stream is None
                              else torch.cat([thought_stream, new_thought], dim=1))
            outs.append(self.head(self.ln_f(h)))
        return torch.stack(outs, dim=0)

    @torch.no_grad()
    def forward_all_loops(self, x: torch.Tensor,
                           n_loops: int | None = None) -> torch.Tensor:
        return self.forward_all_loops_grad(x, n_loops)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
