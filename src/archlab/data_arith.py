"""Iterated left-to-right arithmetic reduction for Result V2 candidate.

Each example is a sequence of N operands interleaved with N-1 binary operators:
  [a_0 op_0 a_1 op_1 a_2 ... op_{N-2} a_{N-1}]

Per-step rule: reduce the leftmost (a_i, op_i, a_{i+1}) → op_i(a_i, a_{i+1}).
After r reductions, the leftmost operand is the partial result of the first
r+1 originals; expression length shrinks by 2 per reduction.

Per-step rule is state-dependent (the "leftmost active triple" depends on
state), and position-invariant (always "reduce leftmost"). Satisfies Result N's
criterion.

Sequence layout (length 2*N_OPERANDS + 1 + 2*N_OPERANDS):
  positions 0..2N-2     interleaved operands/operators (length 2N-1)
  position  2N-1        EQ
  positions 2N..4N-2    output positions, predict reduced expr each loop
"""
from __future__ import annotations

import torch

from .data_chain import EQ, PAD, VOCAB_SIZE  # noqa: F401

ARITH_N_OPERANDS = 8                                 # number of operands; N-1 reductions to single value
ARITH_VOCAB_OPERAND = 10                              # 0..9 are operand values
ARITH_OP_MIN = 100
ARITH_OP_MAX = 101                                    # 2 operator types
ARITH_INPUT_LEN = 2 * ARITH_N_OPERANDS - 1            # 15
ARITH_SEQ_LEN = ARITH_INPUT_LEN + 1 + ARITH_INPUT_LEN # 31 (input + EQ + output positions)
ARITH_OUTPUT_START = ARITH_INPUT_LEN + 1              # 16


def _apply_op(op: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized op application: op tokens in {ARITH_OP_MIN, ARITH_OP_MAX}."""
    is_min = (op == ARITH_OP_MIN)
    return torch.where(is_min, torch.minimum(a, b), torch.maximum(a, b))


def make_batch_arith_iter(batch_size: int, n_steps: int,
                           device: str = "cuda"
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (tokens, iter_targets):
      tokens       : [B, ARITH_SEQ_LEN]
      iter_targets : [n_steps+1, B, ARITH_INPUT_LEN]  — output sequence at each loop r,
                                                        right-padded with PAD as expr shrinks
    """
    operands = torch.randint(0, ARITH_VOCAB_OPERAND, (batch_size, ARITH_N_OPERANDS),
                              device=device)
    ops = torch.where(
        torch.randint(0, 2, (batch_size, ARITH_N_OPERANDS - 1), device=device).bool(),
        torch.full((batch_size, ARITH_N_OPERANDS - 1), ARITH_OP_MAX, device=device),
        torch.full((batch_size, ARITH_N_OPERANDS - 1), ARITH_OP_MIN, device=device))

    # Build expression: [a_0, op_0, a_1, op_1, a_2, ..., op_{N-2}, a_{N-1}]
    # Shape: [B, 2N-1]
    expr = torch.empty((batch_size, ARITH_INPUT_LEN), dtype=torch.long, device=device)
    expr[:, ::2] = operands           # positions 0, 2, 4, ... = operands
    expr[:, 1::2] = ops                # positions 1, 3, 5, ... = operators

    # Per-loop targets: simulate reductions
    iters = [expr.clone()]
    cur = expr.clone()
    cur_len = torch.full((batch_size,), ARITH_INPUT_LEN, dtype=torch.long, device=device)
    for _ in range(n_steps):
        # For examples with cur_len > 1: leftmost triple is at positions 0, 1, 2.
        # Reduce: new_value = op_0(operand_0, operand_1).
        bs_arange = torch.arange(batch_size, device=device)
        a = cur[:, 0]                  # leftmost operand
        op = cur[:, 1]                 # leftmost operator
        b = cur[:, 2]                  # second operand
        result = _apply_op(op, a, b)
        # Build new state: result, cur[3:], padded
        new_state = torch.full_like(cur, PAD)
        new_state[:, 0] = result
        # Shift the rest: original positions 3..N → new positions 1..N-2
        rest_len = ARITH_INPUT_LEN - 3
        new_state[:, 1:1 + rest_len] = cur[:, 3:3 + rest_len]
        # For examples with len <= 1 (already reduced), keep prior state
        already_done = cur_len <= 1
        new_state[already_done] = cur[already_done]
        cur = new_state
        cur_len = (cur_len - 2).clamp(min=1)
        iters.append(cur.clone())
    iter_targets = torch.stack(iters, dim=0)   # [n_steps+1, B, ARITH_INPUT_LEN]

    tokens = torch.full((batch_size, ARITH_SEQ_LEN), PAD,
                         dtype=torch.long, device=device)
    tokens[:, :ARITH_INPUT_LEN] = expr
    tokens[:, ARITH_INPUT_LEN] = EQ
    # Output positions left as PAD initially; model fills via per-loop logits
    return tokens, iter_targets
