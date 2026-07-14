"""Iterated Turing Machine step.

A small TM: W tape cells, T symbols, S states.
Per-step rule (state-driven):
  symbol = tape[head]
  (new_state, new_symbol, move) = program[state, symbol]
  tape[head] = new_symbol
  head = head + move (clamped to [0, W))
  state = new_state

Layout (W=8, T=4, S=4):
  pos 0..W-1                     : tape (T-vocab tokens)
  pos W..W+S*T-1                 : program new_symbols (S*T entries, T-vocab)
  pos W+S*T..W+2*S*T-1           : program new_states (S*T entries, S-vocab)
  pos W+2*S*T..W+3*S*T-1         : program moves (S*T entries, 0=left,1=right)
  pos W+3*S*T                    : initial head position (W-vocab)
  pos W+3*S*T+1                  : initial state (S-vocab)
  pos W+3*S*T+2                  : EQ
  total length: W + 3*S*T + 3 = 8 + 48 + 3 = 59

Output target at depth r: full tape state after r TM steps (W tokens).
We supervise at the W tape positions during training so each tape cell
gets per-loop iter-target supervision.

Vocab: TM uses small token range. We embed in chain's vocab (V=12), with
T <= 4 symbols, S <= 4 states, W <= 12 head positions all fitting.
"""
from __future__ import annotations

import torch

from .data_chain import EQ, PAD, VOCAB_SIZE  # noqa: F401


TM_W = 8
TM_T = 4
TM_S = 4
TM_PROG_LEN = TM_S * TM_T
TM_SEQ_LEN = TM_W + 3 * TM_PROG_LEN + 3  # 8 + 48 + 3 = 59
TM_PROG_SYM_START = TM_W
TM_PROG_STATE_START = TM_W + TM_PROG_LEN
TM_PROG_MOVE_START = TM_W + 2 * TM_PROG_LEN
TM_HEAD_POS = TM_W + 3 * TM_PROG_LEN
TM_STATE_POS = TM_HEAD_POS + 1
TM_EQ_POS = TM_STATE_POS + 1


def make_batch_tm_iter(batch_size: int, n_steps: int,
                        device: str = "cuda"
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (tokens, iter_targets):
      tokens       : [B, TM_SEQ_LEN]
      iter_targets : [n_steps + 1, B, TM_W]  — full tape per loop
    """
    tape0 = torch.randint(0, TM_T, (batch_size, TM_W), device=device)
    prog_sym = torch.randint(0, TM_T, (batch_size, TM_PROG_LEN), device=device)
    prog_state = torch.randint(0, TM_S, (batch_size, TM_PROG_LEN), device=device)
    prog_move = torch.randint(0, 2, (batch_size, TM_PROG_LEN), device=device)
    head0 = torch.randint(0, TM_W, (batch_size,), device=device)
    state0 = torch.randint(0, TM_S, (batch_size,), device=device)

    iters = [tape0.clone()]
    cur_tape = tape0.clone()
    cur_head = head0.clone()
    cur_state = state0.clone()
    idx = torch.arange(batch_size, device=device)
    for _ in range(n_steps):
        sym = cur_tape[idx, cur_head]
        prog_idx = cur_state * TM_T + sym
        new_sym = prog_sym[idx, prog_idx]
        new_state = prog_state[idx, prog_idx]
        move = prog_move[idx, prog_idx]
        cur_tape = cur_tape.clone()
        cur_tape[idx, cur_head] = new_sym
        delta = torch.where(move == 0, -1, 1)
        cur_head = (cur_head + delta).clamp(0, TM_W - 1)
        cur_state = new_state
        iters.append(cur_tape.clone())
    iter_targets = torch.stack(iters, dim=0)  # [n_steps+1, B, TM_W]

    tokens = torch.full((batch_size, TM_SEQ_LEN), PAD, dtype=torch.long, device=device)
    tokens[:, :TM_W] = tape0
    tokens[:, TM_PROG_SYM_START:TM_PROG_SYM_START + TM_PROG_LEN] = prog_sym
    tokens[:, TM_PROG_STATE_START:TM_PROG_STATE_START + TM_PROG_LEN] = prog_state
    tokens[:, TM_PROG_MOVE_START:TM_PROG_MOVE_START + TM_PROG_LEN] = prog_move
    tokens[:, TM_HEAD_POS] = head0
    tokens[:, TM_STATE_POS] = state0
    tokens[:, TM_EQ_POS] = EQ
    return tokens, iter_targets
