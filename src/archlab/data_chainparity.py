"""Chain-walk parity: state-driven XOR accumulation.

Per-step rule (state-driven, NOT position-driven):
    cur     = T[cur]          (chain step — same as data_chain)
    acc_xor = acc_xor XOR (cur & 1)

Target at depth k: XOR over i=1..k of (T^i(start) & 1) — the cumulative
parity of the LSBs of the visited chain nodes.

Why this matters: ordinary parity (data_parity) walls at training depth
because its per-step rule reads `bit[r]` — the input pointer advances
with r. Chain-walk parity's per-step rule reads `cur & 1` where `cur`
is state. State ranges over [0, V) during training, so the model sees
all relevant per-step inputs at training depths. If iter-target
extrapolates here, parity's wall is a TASK property (Result N),
fixable by reformulation.

Layout: same as data_chain — table | start | EQ. Output at EQ position
is the binary accumulated parity.
"""
from __future__ import annotations

import torch

from .data_chain import (ANSWER_POS_ITER, EQ, PAD, SEQ_LEN_ITER, V,
                          VOCAB_SIZE)  # noqa: F401  (re-export)


def make_batch_chainparity_iter(batch_size: int, n_steps: int,
                                 device: str = "cuda"
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (tokens, iter_targets):
      tokens       : [B, V+2]                      = [table | start | EQ]
      iter_targets : [n_steps + 1, B]               iter_targets[r] = acc_xor after r steps
    """
    table = torch.randint(0, V, (batch_size, V), device=device)
    start = torch.randint(0, V, (batch_size,), device=device)

    cur = start.clone()
    acc = torch.zeros_like(start)
    iters = [acc.clone()]
    idx = torch.arange(batch_size, device=device)
    for _ in range(n_steps):
        cur = table[idx, cur]
        acc = acc ^ (cur & 1)
        iters.append(acc.clone())
    iter_targets = torch.stack(iters, dim=0)  # [n_steps+1, B]

    tokens = torch.full((batch_size, SEQ_LEN_ITER), PAD, dtype=torch.long, device=device)
    tokens[:, :V] = table
    tokens[:, V] = start
    tokens[:, V + 1] = EQ
    return tokens, iter_targets
