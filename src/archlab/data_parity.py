"""Iterated parity task for Result L generality validation.

Per-step rule: at loop r, output parity(b_1, ..., b_r) — XOR of the first r bits.
The model has all bits in the input from the start; loop count r determines
how many bits are 'consumed' for the cumulative XOR. Tests whether iter-target
supervision induces a position-aware iterative attention pattern that
extrapolates: train at r ≤ 8, test at r ≤ 16.

Vocab is shared with the chain task (VOCAB_SIZE=202 from data_chain). Bit
tokens 0 and 1 are interpreted as bits; EQ=200 marks the answer position.
"""
from __future__ import annotations

import torch

from .data_chain import EQ, PAD, VOCAB_SIZE  # noqa: F401  (re-export)

PARITY_MAX_N = 16
PARITY_SEQ_LEN = PARITY_MAX_N + 1     # 16 bits + EQ
PARITY_ANSWER_POS = PARITY_MAX_N      # the EQ position outputs the per-loop parity


def make_batch_parity_iter(batch_size: int, n_steps: int,
                            device: str = "cuda"
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (tokens, iter_targets):
      tokens       : [B, PARITY_SEQ_LEN]   = [b_1, ..., b_MAX_N, EQ]
      iter_targets : [n_steps+1, B]         with iter_targets[r] = parity(b_1..b_r)

    iter_targets[0] is unused (no recurrence). For r > PARITY_MAX_N, target is
    parity of all 16 bits (cannot extend further).
    """
    bits = torch.randint(0, 2, (batch_size, PARITY_MAX_N), device=device, dtype=torch.long)
    cum = bits.cumsum(dim=1) % 2          # [B, MAX_N]
    iter_targets = torch.zeros((n_steps + 1, batch_size), dtype=torch.long, device=device)
    n_avail = min(n_steps, PARITY_MAX_N)
    iter_targets[1:n_avail + 1] = cum[:, :n_avail].T
    if n_steps > PARITY_MAX_N:
        iter_targets[PARITY_MAX_N + 1:n_steps + 1] = cum[:, -1:].T

    tokens = torch.full((batch_size, PARITY_SEQ_LEN), PAD,
                         dtype=torch.long, device=device)
    tokens[:, :PARITY_MAX_N] = bits
    tokens[:, PARITY_ANSWER_POS] = EQ
    return tokens, iter_targets
