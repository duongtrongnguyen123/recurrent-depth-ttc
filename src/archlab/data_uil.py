"""Universal Interpreter Loop (UIL) data.

A single recurrent block is trained as a *universal iterator*. The instruction
token at position 0 selects which per-step rule applies. All tasks share the
same input layout; per-loop targets depend on the instruction.

Tasks (v1):
- INSTR_CHAIN: pointer-table chain step (data_chain semantics)
- INSTR_PARITY: cumulative parity over a bit sequence (data_parity semantics)

Layout (V=12, MAX_BITS=16, max payload = max(V+1, MAX_BITS) = 16):
  pos 0          : INSTR token
  pos 1..16      : payload (interpretation depends on INSTR)
                    chain : pos 1..V        = table (12 tokens)
                            pos V+1         = start
                            pos V+2..16     = PAD (positions 13..16)
                    parity: pos 1..16       = bits (16 tokens)
  pos 17         : EQ — read predictions here

EQ is the LAST position so causal attention can see all payload tokens.
"""

from __future__ import annotations

import torch

from .data_chain import V, EQ, PAD, VOCAB_SIZE  # noqa: F401  (re-export VOCAB_SIZE)


INSTR_CHAIN = 220
INSTR_PARITY = 221
N_INSTRS = 2

PARITY_MAX_N = 16
PAYLOAD_LEN = max(V + 1, PARITY_MAX_N)  # 16
SEQ_LEN_UIL = 1 + PAYLOAD_LEN + 1       # 1 instr + 16 payload + 1 EQ = 18
EQ_POS = SEQ_LEN_UIL - 1                # = 17

# Vocab needs to include INSTR tokens
VOCAB_UIL = max(VOCAB_SIZE, INSTR_PARITY + 1)


def make_batch_uil_iter(batch_size: int, n_steps: int,
                         device: str = "cuda",
                         instr_mix: tuple[float, float] = (0.5, 0.5)
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixed-task iter-target batch for UIL.

    Returns (tokens, iter_targets):
      tokens       : [B, SEQ_LEN_UIL]   — instr-conditioned input
      iter_targets : [n_steps + 1, B]   — per-loop target at EQ position

    iter_targets[r] is the target *after r loops*. iter_targets[0] is unused.
    """
    p_chain, p_par = instr_mix
    assert abs(p_chain + p_par - 1.0) < 1e-6, "instr_mix must sum to 1"
    n_chain = int(round(batch_size * p_chain))
    n_par = batch_size - n_chain

    tokens = torch.full((batch_size, SEQ_LEN_UIL), PAD, dtype=torch.long, device=device)
    tokens[:, EQ_POS] = EQ
    iter_targets = torch.zeros((n_steps + 1, batch_size), dtype=torch.long, device=device)

    # Chain block: indices [0, n_chain)
    if n_chain > 0:
        table = torch.randint(0, V, (n_chain, V), device=device)
        start = torch.randint(0, V, (n_chain,), device=device)
        tokens[:n_chain, 0] = INSTR_CHAIN
        tokens[:n_chain, 1:1 + V] = table       # pos 1..12
        tokens[:n_chain, 1 + V] = start          # pos 13
        # pos 14..16 left as PAD
        cur = start.clone()
        iters = [start.clone()]
        idx = torch.arange(n_chain, device=device)
        for _ in range(n_steps):
            cur = table[idx, cur]
            iters.append(cur.clone())
        iter_targets[:, :n_chain] = torch.stack(iters, dim=0)

    # Parity block: indices [n_chain, batch_size)
    if n_par > 0:
        sl = slice(n_chain, batch_size)
        bits = torch.randint(0, 2, (n_par, PARITY_MAX_N), device=device, dtype=torch.long)
        cum = bits.cumsum(dim=1) % 2  # [n_par, PARITY_MAX_N]
        tokens[sl, 0] = INSTR_PARITY
        tokens[sl, 1:1 + PARITY_MAX_N] = bits  # pos 1..16
        # iter_targets[0] = 0 (placeholder)
        n_avail = min(n_steps, PARITY_MAX_N)
        iter_targets[1:n_avail + 1, sl] = cum[:, :n_avail].T
        if n_steps > PARITY_MAX_N:
            iter_targets[PARITY_MAX_N + 1: n_steps + 1, sl] = cum[:, -1:].T

    return tokens, iter_targets


def make_batch_uil_eval(batch_size: int, n_steps: int, instr: int,
                         device: str = "cuda"
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-task eval batch (instr ∈ {INSTR_CHAIN, INSTR_PARITY})."""
    if instr == INSTR_CHAIN:
        return make_batch_uil_iter(batch_size, n_steps, device, instr_mix=(1.0, 0.0))
    if instr == INSTR_PARITY:
        return make_batch_uil_iter(batch_size, n_steps, device, instr_mix=(0.0, 1.0))
    raise ValueError(f"unknown instr {instr}")
