"""UIL without explicit instruction tokens.

Same input layout as data_uil but the position-0 INSTR token is replaced
with PAD. The model has to infer task type from the rest of the input:
  - Chain: pos 1..12 = table tokens (range [0, 12)), pos 13 = start, pos 14..16 = PAD
  - Parity: pos 1..16 = bits (range [0, 2)), pos 14..16 also bits (not PAD)

Distinguishable by:
  - Token statistics (chain has values in [0, 12), parity has only {0, 1})
  - Padding pattern (chain has PAD at 14..16, parity does not)

Tests whether the loop core can learn to route per-step rule based on
input structure alone — a step toward instruction-free general reasoning.
"""
from __future__ import annotations

import torch

from .data_chain import V, EQ, PAD, VOCAB_SIZE  # noqa: F401
from .data_uil import EQ_POS, PARITY_MAX_N, SEQ_LEN_UIL  # reuse layout


def make_batch_uil_noinstr_iter(batch_size: int, n_steps: int,
                                  device: str = "cuda",
                                  task_mix: tuple[float, float] = (0.5, 0.5)
                                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (tokens, iter_targets, task_mask):
      tokens       : [B, SEQ_LEN_UIL] — INSTR pos 0 zeroed (PAD)
      iter_targets : [n_steps + 1, B]
      task_mask    : [B] — 0 = chain, 1 = parity (for eval analysis)
    """
    p_chain, p_par = task_mix
    assert abs(p_chain + p_par - 1.0) < 1e-6
    n_chain = int(round(batch_size * p_chain))
    n_par = batch_size - n_chain

    tokens = torch.full((batch_size, SEQ_LEN_UIL), PAD, dtype=torch.long, device=device)
    tokens[:, EQ_POS] = EQ
    iter_targets = torch.zeros((n_steps + 1, batch_size), dtype=torch.long, device=device)
    task_mask = torch.zeros(batch_size, dtype=torch.long, device=device)

    if n_chain > 0:
        table = torch.randint(0, V, (n_chain, V), device=device)
        start = torch.randint(0, V, (n_chain,), device=device)
        # NO INSTR token at pos 0 — leave as PAD
        tokens[:n_chain, 1:1 + V] = table
        tokens[:n_chain, 1 + V] = start
        cur = start.clone()
        iters = [start.clone()]
        idx = torch.arange(n_chain, device=device)
        for _ in range(n_steps):
            cur = table[idx, cur]
            iters.append(cur.clone())
        iter_targets[:, :n_chain] = torch.stack(iters, dim=0)

    if n_par > 0:
        sl = slice(n_chain, batch_size)
        bits = torch.randint(0, 2, (n_par, PARITY_MAX_N), device=device, dtype=torch.long)
        cum_par = bits.cumsum(dim=1) % 2
        # NO INSTR token at pos 0 — leave as PAD
        tokens[sl, 1:1 + PARITY_MAX_N] = bits
        n_avail = min(n_steps, PARITY_MAX_N)
        iter_targets[1:n_avail + 1, sl] = cum_par[:, :n_avail].T
        if n_steps > PARITY_MAX_N:
            iter_targets[PARITY_MAX_N + 1: n_steps + 1, sl] = cum_par[:, -1:].T
        task_mask[sl] = 1

    return tokens, iter_targets, task_mask


def make_batch_uil_noinstr_eval(batch_size: int, n_steps: int, task: int,
                                   device: str = "cuda"
                                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-task eval. task=0 chain, task=1 parity."""
    if task == 0:
        toks, tgt, _ = make_batch_uil_noinstr_iter(batch_size, n_steps, device, (1.0, 0.0))
    elif task == 1:
        toks, tgt, _ = make_batch_uil_noinstr_iter(batch_size, n_steps, device, (0.0, 1.0))
    else:
        raise ValueError(task)
    return toks, tgt
