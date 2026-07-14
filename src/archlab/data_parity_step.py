"""Parity with EXPLICIT step-counter token in the input.

Each example carries a 'current step r' token at a designated position.
At training, r is sampled uniformly per batch from [1, n_loops_max].
Model reads this token + bits, predicts cumulative parity at step r.

If model learns "use step token to index into bits, accumulate XOR up
to that position", then at test r > train_max but within sampled
range, predictions should still work — wall is broken by curriculum.
"""
from __future__ import annotations

import torch
from .data_chain import EQ, PAD, VOCAB_SIZE  # noqa: F401

PARITY_STEP_MAX_N = 16
# Layout: [b_1, ..., b_16, STEP_r, EQ]
PARITY_STEP_SEQ_LEN = PARITY_STEP_MAX_N + 1 + 1
PARITY_STEP_POS = PARITY_STEP_MAX_N
PARITY_STEP_ANSWER_POS = PARITY_STEP_MAX_N + 1
# Step tokens encoded in tail vocab range to avoid clash with bits {0,1}
STEP_TOKEN_BASE = 100   # step r → token (STEP_TOKEN_BASE + r)


def make_batch_parity_step(batch_size: int, r_min: int = 1,
                              r_max: int = PARITY_STEP_MAX_N,
                              device: str = "cuda"
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (tokens, targets):
      tokens : [B, PARITY_STEP_SEQ_LEN] = bits + STEP_r + EQ
      targets : [B] — cumulative parity over first r bits
    """
    bits = torch.randint(0, 2, (batch_size, PARITY_STEP_MAX_N),
                            device=device, dtype=torch.long)
    rs = torch.randint(r_min, r_max + 1, (batch_size,), device=device)
    cum = bits.cumsum(dim=1) % 2
    targets = cum[torch.arange(batch_size, device=device), rs - 1]

    tokens = torch.full((batch_size, PARITY_STEP_SEQ_LEN), PAD,
                          dtype=torch.long, device=device)
    tokens[:, :PARITY_STEP_MAX_N] = bits
    tokens[:, PARITY_STEP_POS] = STEP_TOKEN_BASE + rs
    tokens[:, PARITY_STEP_ANSWER_POS] = EQ
    return tokens, targets, rs
