"""Iterated multi-pointer chain task for Result T candidate.

State: M pointers (independent), each in [0, V). Per-step rule: each pointer
advances independently via the per-example table f. Per-loop r at pointer i:
f^r(start_i).

Position-invariant (same table-lookup applied at every pointer position).
State-dependent (each output position's value depends on its current pointer).
Cleanest multi-position iter test: extends chain to multi-token state without
introducing positional rules.

Sequence layout (length V + M + 2 + M):
  positions 0..V-1                  pointer table
  positions V..V+M-1                M starting pointers
  position  V+M                     EQ
  positions V+M+1..V+2M             output positions (predict f^r at each pointer)
"""
from __future__ import annotations

import torch

from .data_chain import EQ, PAD, V, VOCAB_SIZE  # noqa: F401

MULTI_M = 4                           # number of independent pointers
MULTI_SEQ_LEN = V + MULTI_M + 2 + MULTI_M  # 12 + 4 + 2 + 4 = 22
MULTI_OUTPUT_START = V + MULTI_M + 2  # first output position (V + M + 1 + 1)


def make_batch_multichain_iter(batch_size: int, n_steps: int,
                                device: str = "cuda"
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (tokens, iter_targets):
      tokens       : [B, MULTI_SEQ_LEN]
      iter_targets : [n_steps+1, B, MULTI_M]   — f^r at each pointer
    """
    table = torch.randint(0, V, (batch_size, V), device=device)
    starts = torch.randint(0, V, (batch_size, MULTI_M), device=device)

    iters = [starts.clone()]
    cur = starts.clone()
    for _ in range(n_steps):
        # cur: [B, M]; advance each pointer through the table independently
        bs_arange = torch.arange(batch_size, device=device).unsqueeze(1)        # [B, 1]
        cur = table[bs_arange, cur]                                              # [B, M]
        iters.append(cur.clone())
    iter_targets = torch.stack(iters, dim=0)                                     # [n_steps+1, B, M]

    tokens = torch.full((batch_size, MULTI_SEQ_LEN), PAD,
                         dtype=torch.long, device=device)
    tokens[:, :V] = table
    tokens[:, V:V + MULTI_M] = starts
    tokens[:, V + MULTI_M] = EQ
    # tokens[V+M+1] left as PAD (separator)
    # tokens[MULTI_OUTPUT_START..MULTI_OUTPUT_START+M-1] = output positions, left as PAD
    return tokens, iter_targets
