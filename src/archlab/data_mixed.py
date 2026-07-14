"""Heterogeneous-task batch: chain (needs looped) + identity (vanilla suffices).

Goal: test whether TwoStreamTransformer's gate learns to route per-task.
- task=CHAIN: answer at every loop r is f^r(start) — pure iter-target
- task=IDENTITY: answer at every loop r is start (constant) — trivial pattern match

Both share vocab and sequence length. A leading task token tells the model
which task it is. Vanilla stream should ace identity (1-hop attention to
position V+1). Looped stream is essential for chain at high r.
"""
from __future__ import annotations

import torch

from .data_chain import DEPTH_OFFSET, EQ, PAD, V

# Add new tokens for task signaling. Stay clear of HOP=199, EQ=200, PAD=201.
TASK_CHAIN = 210
TASK_IDENTITY = 211
VOCAB_SIZE_MIXED = 212

# Layout: [task_tok, table 0..V-1, start, EQ]   length = V + 3
SEQ_LEN_MIXED = V + 3
ANSWER_POS_MIXED = V + 2   # EQ position; answer predicted at this position


def make_batch_mixed_iter(batch_size: int, n_steps: int, p_chain: float = 0.5,
                            device: str = "cuda"
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (tokens, iter_targets, is_chain).
      tokens       : [B, V+3]              = [task_tok | table | start | EQ]
      iter_targets : [n_steps+1, B]
      is_chain     : [B] bool              True=chain, False=identity
    """
    is_chain = (torch.rand(batch_size, device=device) < p_chain)
    table = torch.randint(0, V, (batch_size, V), device=device)
    start = torch.randint(0, V, (batch_size,), device=device)

    cur = start.clone()
    iters_chain = [start.clone()]
    for _ in range(n_steps):
        cur = table[torch.arange(batch_size, device=device), cur]
        iters_chain.append(cur.clone())
    iters_chain_t = torch.stack(iters_chain, dim=0)         # [n+1, B]
    iters_id_t = start.unsqueeze(0).expand(n_steps + 1, -1)  # [n+1, B]

    iter_targets = torch.where(is_chain.unsqueeze(0), iters_chain_t, iters_id_t)

    task_tok = torch.where(is_chain,
                            torch.full_like(start, TASK_CHAIN),
                            torch.full_like(start, TASK_IDENTITY))

    tokens = torch.full((batch_size, SEQ_LEN_MIXED), PAD, dtype=torch.long,
                          device=device)
    tokens[:, 0] = task_tok
    tokens[:, 1:1 + V] = table
    tokens[:, 1 + V] = start
    tokens[:, ANSWER_POS_MIXED] = EQ
    return tokens, iter_targets, is_chain
