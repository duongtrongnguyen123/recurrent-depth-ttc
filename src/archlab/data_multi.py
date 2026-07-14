"""Multi-operation arithmetic data: +, -, mod. Uniform sequence length so a single
model can be trained on all three with the same fixed-length head.

Layout: a_digits (N) | OP | b_digits (N) | EQ | answer (N+1)  -- least-sig first

Vocab: 0..9, PLUS, MINUS, MOD, EQ, PAD = 15 tokens.

Operations:
  add  - a + b, a,b in [0, 10^N)               answer fits in N+1 digits
  sub  - a - b, a in [0, 10^N), b in [0, a]    answer fits in N digits → pad with 0
  mod  - a mod b, a in [0, 10^N), b in [2, 99] answer < 99, fits in 2 digits → pad with 0

A 'difficulty' label captures the effective digit count of the operands (digits_min..N).
"""
from __future__ import annotations

import torch

VOCAB_SIZE = 15
PLUS = 10
MINUS = 11
MOD = 12
EQ = 13
PAD = 14

OP_TOKEN = {"add": PLUS, "sub": MINUS, "mod": MOD}


def make_batch_multi(batch_size: int, n_digits: int, ops: list[str],
                     digits_min: int | None = None, device: str = "cuda"
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each example uniformly samples an operation and an effective digit count.

    Returns (tokens, targets, target_mask, op_ids). op_ids is per-example for analysis.
    """
    op_choice = torch.randint(0, len(ops), (batch_size,), device=device)
    op_ids = torch.tensor([OP_TOKEN[ops[i]] for i in op_choice.tolist()], device=device)

    if digits_min is None:
        eff = torch.full((batch_size,), n_digits, device=device, dtype=torch.long)
    else:
        eff = torch.randint(digits_min, n_digits + 1, (batch_size,), device=device)
    upper = (10 ** eff.float()).long()

    a = (torch.rand(batch_size, device=device) * upper.float()).long()
    b = (torch.rand(batch_size, device=device) * upper.float()).long()

    # Per-op constraints / answer.
    c = torch.zeros_like(a)
    for op_str, tok in OP_TOKEN.items():
        sel = op_ids == tok
        if not sel.any(): continue
        if op_str == "add":
            c[sel] = a[sel] + b[sel]
        elif op_str == "sub":
            # ensure a >= b; if not, swap
            swap = sel & (a < b)
            tmp = a[swap].clone(); a[swap] = b[swap]; b[swap] = tmp
            c[sel] = a[sel] - b[sel]
        elif op_str == "mod":
            # b clamped to [2, 99]
            bs_sel = b[sel].clamp(min=2).clamp(max=99)
            b[sel] = bs_sel
            c[sel] = a[sel] % bs_sel

    a_d = _digits(a, n_digits, device)
    b_d = _digits(b, n_digits, device)
    c_d = _digits(c, n_digits + 1, device).flip(-1)

    op_col = op_ids.view(-1, 1)
    eq_col = torch.full((batch_size, 1), EQ, device=device)
    tokens = torch.cat([a_d, op_col, b_d, eq_col, c_d], dim=1)

    answer_start = 2 * n_digits + 2
    target_mask = torch.zeros_like(tokens, dtype=torch.bool)
    target_mask[:, answer_start - 1: answer_start - 1 + (n_digits + 1)] = True
    targets = torch.full_like(tokens, PAD)
    targets[:, answer_start - 1: answer_start - 1 + (n_digits + 1)] = tokens[
        :, answer_start: answer_start + (n_digits + 1)
    ]
    return tokens, targets, target_mask, op_ids


def _digits(x: torch.Tensor, n: int, device: str) -> torch.Tensor:
    out = torch.zeros((x.numel(), n), dtype=torch.long, device=device)
    for i in range(n):
        out[:, n - 1 - i] = (x // (10 ** i)) % 10
    return out
