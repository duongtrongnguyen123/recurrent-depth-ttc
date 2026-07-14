"""Iterated arithmetic reduction with N=12 operands for proper extrapolation test.

Same per-step rule as data_arith.py but with longer input so we can train at
n_loops_train < max_reductions and meaningfully test r > train_max.
"""
from __future__ import annotations

import torch

from .data_chain import EQ, PAD, VOCAB_SIZE  # noqa: F401

ARITH_LONG_N_OPERANDS = 12
ARITH_LONG_VOCAB_OPERAND = 10
ARITH_LONG_OP_MIN = 100
ARITH_LONG_OP_MAX = 101
ARITH_LONG_INPUT_LEN = 2 * ARITH_LONG_N_OPERANDS - 1            # 23
ARITH_LONG_SEQ_LEN = ARITH_LONG_INPUT_LEN + 1 + ARITH_LONG_INPUT_LEN  # 47
ARITH_LONG_OUTPUT_START = ARITH_LONG_INPUT_LEN + 1               # 24


def _apply_op(op: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    is_min = (op == ARITH_LONG_OP_MIN)
    return torch.where(is_min, torch.minimum(a, b), torch.maximum(a, b))


def make_batch_arith_long_iter(batch_size: int, n_steps: int,
                                 device: str = "cuda"
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    operands = torch.randint(0, ARITH_LONG_VOCAB_OPERAND,
                              (batch_size, ARITH_LONG_N_OPERANDS), device=device)
    ops = torch.where(
        torch.randint(0, 2, (batch_size, ARITH_LONG_N_OPERANDS - 1),
                       device=device).bool(),
        torch.full((batch_size, ARITH_LONG_N_OPERANDS - 1), ARITH_LONG_OP_MAX,
                    device=device),
        torch.full((batch_size, ARITH_LONG_N_OPERANDS - 1), ARITH_LONG_OP_MIN,
                    device=device))

    expr = torch.empty((batch_size, ARITH_LONG_INPUT_LEN), dtype=torch.long, device=device)
    expr[:, ::2] = operands
    expr[:, 1::2] = ops

    iters = [expr.clone()]
    cur = expr.clone()
    cur_len = torch.full((batch_size,), ARITH_LONG_INPUT_LEN, dtype=torch.long, device=device)
    for _ in range(n_steps):
        a = cur[:, 0]
        op = cur[:, 1]
        b = cur[:, 2]
        result = _apply_op(op, a, b)
        new_state = torch.full_like(cur, PAD)
        new_state[:, 0] = result
        rest_len = ARITH_LONG_INPUT_LEN - 3
        new_state[:, 1:1 + rest_len] = cur[:, 3:3 + rest_len]
        already_done = cur_len <= 1
        new_state[already_done] = cur[already_done]
        cur = new_state
        cur_len = (cur_len - 2).clamp(min=1)
        iters.append(cur.clone())
    iter_targets = torch.stack(iters, dim=0)

    tokens = torch.full((batch_size, ARITH_LONG_SEQ_LEN), PAD,
                         dtype=torch.long, device=device)
    tokens[:, :ARITH_LONG_INPUT_LEN] = expr
    tokens[:, ARITH_LONG_INPUT_LEN] = EQ
    return tokens, iter_targets


def make_batch_arith_long_random_n(batch_size: int, n_steps: int,
                                     n_min: int = 3, n_max: int = ARITH_LONG_N_OPERANDS,
                                     device: str = "cuda"
                                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Heterogeneous-N variant: each example has random operand count N ∈ [n_min, n_max].

    This addresses Result AP's full_acc artifact: the standard fixed-N=12 training
    means pass i+1's input (a shorter expression) is OOD for the model. Heterogeneous
    N exposes the model to inputs of varying length so multi-pass intermediate
    states are in-distribution.

    Trailing positions past 2*N-1 are PAD in BOTH the input expression AND
    every iter_target.
    """
    assert 2 <= n_min <= n_max <= ARITH_LONG_N_OPERANDS
    Ns = torch.randint(n_min, n_max + 1, (batch_size,), device=device)

    operands_full = torch.randint(0, ARITH_LONG_VOCAB_OPERAND,
                                    (batch_size, ARITH_LONG_N_OPERANDS), device=device)
    op_choice_full = torch.randint(0, 2, (batch_size, ARITH_LONG_N_OPERANDS - 1),
                                     device=device).bool()
    op_tokens_full = torch.where(
        op_choice_full,
        torch.full((batch_size, ARITH_LONG_N_OPERANDS - 1), ARITH_LONG_OP_MAX,
                    device=device),
        torch.full((batch_size, ARITH_LONG_N_OPERANDS - 1), ARITH_LONG_OP_MIN,
                    device=device))

    expr = torch.full((batch_size, ARITH_LONG_INPUT_LEN), PAD,
                       dtype=torch.long, device=device)
    arange = torch.arange(ARITH_LONG_INPUT_LEN, device=device).unsqueeze(0)
    operand_pos = (arange % 2 == 0)
    operand_idx = arange // 2
    op_idx = (arange - 1) // 2
    operand_active = operand_pos & (operand_idx < Ns.unsqueeze(1))
    op_active = (~operand_pos) & (op_idx < (Ns.unsqueeze(1) - 1))
    operand_take = torch.gather(operands_full, 1,
                                  operand_idx.expand_as(operand_active).clamp(
                                      min=0, max=ARITH_LONG_N_OPERANDS - 1))
    op_take = torch.gather(op_tokens_full, 1,
                             op_idx.expand_as(op_active).clamp(
                                 min=0, max=ARITH_LONG_N_OPERANDS - 2))
    expr = torch.where(operand_active, operand_take, expr)
    expr = torch.where(op_active, op_take, expr)

    iters = [expr.clone()]
    cur = expr.clone()
    cur_len = (2 * Ns - 1).long()
    for _ in range(n_steps):
        a = cur[:, 0]
        op = cur[:, 1]
        b = cur[:, 2]
        result = _apply_op(op, a, b)
        new_state = torch.full_like(cur, PAD)
        new_state[:, 0] = result
        rest_len = ARITH_LONG_INPUT_LEN - 3
        new_state[:, 1:1 + rest_len] = cur[:, 3:3 + rest_len]
        already_done = cur_len <= 1
        new_state[already_done] = cur[already_done]
        cur = new_state
        cur_len = (cur_len - 2).clamp(min=1)
        iters.append(cur.clone())
    iter_targets = torch.stack(iters, dim=0)

    tokens = torch.full((batch_size, ARITH_LONG_SEQ_LEN), PAD,
                         dtype=torch.long, device=device)
    tokens[:, :ARITH_LONG_INPUT_LEN] = expr
    tokens[:, ARITH_LONG_INPUT_LEN] = EQ
    return tokens, iter_targets
