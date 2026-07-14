"""Iterated 1D cellular automaton (Rule 90) for Result Q candidate.

Rule 90: next[i] = b[i-1] XOR b[i+1] (with wraparound boundaries).
Per-step rule is **state-dependent** (uses current cells), **position-invariant**
(same rule applied at every position), and **local** (uses left and right
neighbors).

If iter-target supervision generalizes here, it confirms the recipe extends
beyond single-pointer chain to multi-position state with local interaction —
a stricter test than chain. Rule 90 produces a Sierpinski-triangle pattern;
its r-th iterate from a single-cell start is well-known and bounded.

Sequence layout (length 2N+1):
  positions 0..N-1     initial cell state b_1..b_N
  position  N          EQ
  positions N+1..2N    output positions — predict state at loop r at each cell
"""
from __future__ import annotations

import torch

from .data_chain import EQ, PAD, VOCAB_SIZE  # noqa: F401

CA_N = 16                                    # number of cells
CA_SEQ_LEN = 2 * CA_N + 1                     # 33
CA_OUTPUT_START = CA_N + 1                    # first output position


def _ca_step(state: torch.Tensor) -> torch.Tensor:
    """Apply Rule 90 one step. state: [B, N] -> [B, N] (with wraparound)."""
    left = torch.roll(state, shifts=1, dims=-1)
    right = torch.roll(state, shifts=-1, dims=-1)
    return (left ^ right).long()


def make_batch_ca_iter(batch_size: int, n_steps: int,
                        device: str = "cuda"
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (tokens, iter_targets):
      tokens       : [B, CA_SEQ_LEN]            — input cells + EQ + PAD output positions
      iter_targets : [n_steps+1, B, CA_N]        — state at each loop r

    Loss is at output positions N+1..2N for each cell, supervised against
    iter_targets[r]. iter_targets[0] is the initial state (not supervised).
    """
    state = torch.randint(0, 2, (batch_size, CA_N), device=device, dtype=torch.long)
    iters = [state.clone()]
    cur = state.clone()
    for _ in range(n_steps):
        cur = _ca_step(cur)
        iters.append(cur.clone())
    iter_targets = torch.stack(iters, dim=0)   # [n_steps+1, B, CA_N]

    tokens = torch.full((batch_size, CA_SEQ_LEN), PAD,
                         dtype=torch.long, device=device)
    tokens[:, :CA_N] = state
    tokens[:, CA_N] = EQ
    # Output positions left as PAD; the model fills in via per-loop logits.
    return tokens, iter_targets
