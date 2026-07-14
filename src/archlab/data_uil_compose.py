"""UIL Compositional: a single sequence containing TWO task segments
(chain segment + parity segment), each with its own instruction token,
each with its own EQ output.

The shared loop core must apply different per-step rules to different
parts of the input *in the same forward pass*. The strongest test of
"the loop is a general iterator that routes per segment".

Layout (SEQ_LEN_COMPOSE = 33):
  pos 0       : INSTR_CHAIN (220)
  pos 1..12   : chain table T (12 tokens)
  pos 13      : chain start
  pos 14      : EQ_CHAIN — read chain prediction here
  pos 15      : INSTR_PARITY (221)
  pos 16..31  : parity bits (16 tokens)
  pos 32      : EQ_PAR — read parity prediction here

Per-loop targets at depth k:
  EQ_CHAIN: T^k(start)
  EQ_PAR  : XOR_{i=0..k-1} bits[i]  (cumulative parity)

Two losses summed per loop. Both segments evolve in parallel under the
same shared block.
"""
from __future__ import annotations

import torch

from .data_chain import V, EQ, PAD, VOCAB_SIZE  # noqa: F401


INSTR_CHAIN = 220
INSTR_PARITY = 221
PARITY_MAX_N = 16

CHAIN_INSTR_POS = 0
CHAIN_TABLE_START = 1
CHAIN_START_POS = 1 + V                  # 13
EQ_CHAIN_POS = 1 + V + 1                 # 14
PAR_INSTR_POS = 1 + V + 2                # 15
PAR_BITS_START = 1 + V + 3               # 16
EQ_PAR_POS = 1 + V + 3 + PARITY_MAX_N    # 32
SEQ_LEN_COMPOSE = EQ_PAR_POS + 1         # 33


def make_batch_compose_iter(batch_size: int, n_steps: int,
                              device: str = "cuda"
                              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (tokens, chain_targets, par_targets):
      tokens         : [B, SEQ_LEN_COMPOSE]
      chain_targets  : [n_steps + 1, B] — target at EQ_CHAIN_POS
      par_targets    : [n_steps + 1, B] — target at EQ_PAR_POS
    """
    table = torch.randint(0, V, (batch_size, V), device=device)
    start = torch.randint(0, V, (batch_size,), device=device)
    bits = torch.randint(0, 2, (batch_size, PARITY_MAX_N), device=device, dtype=torch.long)
    cum_par = bits.cumsum(dim=1) % 2  # [B, PARITY_MAX_N]

    # chain trajectory
    cur = start.clone()
    chain_iters = [start.clone()]
    idx = torch.arange(batch_size, device=device)
    for _ in range(n_steps):
        cur = table[idx, cur]
        chain_iters.append(cur.clone())
    chain_targets = torch.stack(chain_iters, dim=0)

    # parity trajectory
    par_targets = torch.zeros((n_steps + 1, batch_size), dtype=torch.long, device=device)
    n_avail = min(n_steps, PARITY_MAX_N)
    par_targets[1:n_avail + 1] = cum_par[:, :n_avail].T
    if n_steps > PARITY_MAX_N:
        par_targets[PARITY_MAX_N + 1: n_steps + 1] = cum_par[:, -1:].T

    tokens = torch.full((batch_size, SEQ_LEN_COMPOSE), PAD,
                         dtype=torch.long, device=device)
    tokens[:, CHAIN_INSTR_POS] = INSTR_CHAIN
    tokens[:, CHAIN_TABLE_START:CHAIN_TABLE_START + V] = table
    tokens[:, CHAIN_START_POS] = start
    tokens[:, EQ_CHAIN_POS] = EQ
    tokens[:, PAR_INSTR_POS] = INSTR_PARITY
    tokens[:, PAR_BITS_START:PAR_BITS_START + PARITY_MAX_N] = bits
    tokens[:, EQ_PAR_POS] = EQ

    return tokens, chain_targets, par_targets
