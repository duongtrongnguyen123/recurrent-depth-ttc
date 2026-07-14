from __future__ import annotations

import torch

# Vocab: digits 0-9 (ids 0-9), '+' = 10, '=' = 11, PAD = 12.
VOCAB_SIZE = 13
PLUS = 10
EQ = 11
PAD = 12


def make_batch(batch_size: int, n_digits: int, device: str = "cuda",
               digits_min: int | None = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """N-digit addition. Returns (tokens, targets, target_mask).

    Each example uses n_digits unless digits_min is set, in which case the *effective*
    number of digits is sampled in [digits_min, n_digits] and the high digits are
    zero-padded — giving a mixed-difficulty batch at fixed sequence length.
    """
    if digits_min is None:
        a = torch.randint(0, 10 ** n_digits, (batch_size,), device=device)
        b = torch.randint(0, 10 ** n_digits, (batch_size,), device=device)
    else:
        # Sample effective digit counts uniformly per example.
        ds = torch.randint(digits_min, n_digits + 1, (batch_size,), device=device)
        upper = (10 ** ds.float()).long()
        a = (torch.rand(batch_size, device=device) * upper.float()).long()
        b = (torch.rand(batch_size, device=device) * upper.float()).long()
    c = a + b  # up to 10^N + 10^N - 2, so up to N+1 digits

    a_digits = _to_digits(a, n_digits, device)               # [B, N], most-significant first
    b_digits = _to_digits(b, n_digits, device)
    c_digits = _to_digits(c, n_digits + 1, device).flip(-1)  # least-significant first

    plus = torch.full((batch_size, 1), PLUS, device=device)
    eq = torch.full((batch_size, 1), EQ, device=device)

    tokens = torch.cat([a_digits, plus, b_digits, eq, c_digits], dim=1)
    # Targets shifted by 1 for next-token prediction. We compute loss only at the
    # positions where the model predicts answer tokens. Answer occupies the last
    # (N+1) positions of `tokens`. Predictions for those come from the previous (N+1)
    # positions of the *input* logits.
    target_mask = torch.zeros_like(tokens, dtype=torch.bool)
    answer_start = 2 * n_digits + 2  # index of c_0
    target_mask[:, answer_start - 1: answer_start - 1 + (n_digits + 1)] = True

    targets = torch.full_like(tokens, PAD)
    targets[:, answer_start - 1: answer_start - 1 + (n_digits + 1)] = tokens[
        :, answer_start: answer_start + (n_digits + 1)
    ]
    return tokens, targets, target_mask


def _to_digits(x: torch.Tensor, n: int, device: str) -> torch.Tensor:
    """Decompose into n digits, most-significant-first. Shape [B, n]."""
    out = torch.zeros((x.numel(), n), dtype=torch.long, device=device)
    for i in range(n):
        out[:, n - 1 - i] = (x // (10 ** i)) % 10
    return out
