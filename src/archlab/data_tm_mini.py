"""Smaller TM for sanity-check: W=4 tape, T=2 symbols, S=2 states."""
from __future__ import annotations
import torch
from .data_chain import EQ, PAD, VOCAB_SIZE  # noqa: F401

TM_W = 4
TM_T = 2
TM_S = 2
TM_PROG_LEN = TM_S * TM_T  # 4
TM_SEQ_LEN = TM_W + 3 * TM_PROG_LEN + 3  # 4 + 12 + 3 = 19
TM_PROG_SYM_START = TM_W
TM_PROG_STATE_START = TM_W + TM_PROG_LEN
TM_PROG_MOVE_START = TM_W + 2 * TM_PROG_LEN
TM_HEAD_POS = TM_W + 3 * TM_PROG_LEN
TM_STATE_POS = TM_HEAD_POS + 1
TM_EQ_POS = TM_STATE_POS + 1


def make_batch_tm_mini_iter(batch_size, n_steps, device="cuda"):
    tape0 = torch.randint(0, TM_T, (batch_size, TM_W), device=device)
    prog_sym = torch.randint(0, TM_T, (batch_size, TM_PROG_LEN), device=device)
    prog_state = torch.randint(0, TM_S, (batch_size, TM_PROG_LEN), device=device)
    prog_move = torch.randint(0, 2, (batch_size, TM_PROG_LEN), device=device)
    head0 = torch.randint(0, TM_W, (batch_size,), device=device)
    state0 = torch.randint(0, TM_S, (batch_size,), device=device)
    iters = [tape0.clone()]
    cur_tape = tape0.clone(); cur_head = head0.clone(); cur_state = state0.clone()
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
    iter_targets = torch.stack(iters, dim=0)
    tokens = torch.full((batch_size, TM_SEQ_LEN), PAD, dtype=torch.long, device=device)
    tokens[:, :TM_W] = tape0
    tokens[:, TM_PROG_SYM_START:TM_PROG_SYM_START + TM_PROG_LEN] = prog_sym
    tokens[:, TM_PROG_STATE_START:TM_PROG_STATE_START + TM_PROG_LEN] = prog_state
    tokens[:, TM_PROG_MOVE_START:TM_PROG_MOVE_START + TM_PROG_LEN] = prog_move
    tokens[:, TM_HEAD_POS] = head0
    tokens[:, TM_STATE_POS] = state0
    tokens[:, TM_EQ_POS] = EQ
    return tokens, iter_targets
