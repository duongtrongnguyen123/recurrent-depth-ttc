"""Real-text iter-target via multi-step horizon prediction (Result AD candidate).

Tests whether iter-target supervision generalizes to autoregressive language
modeling. Per-step rule: at loop r, the model's output at position t predicts
the token at position t + r*stride. So loop count = "how far ahead to look."

Per-step rule is **position-self-referential** in the sense of Result N: each
output position predicts a token offset from its own position by r*stride,
using attention over its own context. No absolute-position-dependent shift.

Compatible with Branch 2's pretraining: this is a generalization of standard
next-token loss (which is the r=1 stride=1 case) to multi-step horizon under
a recurrent-depth architecture.

Tokenizer: char-level, ~65 chars for tiny-shakespeare (a-z, A-Z, punctuation).
Data: tiny-shakespeare (1MB, available at karpathy/char-rnn).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


def load_tiny_shakespeare(cache_dir: str = "/tmp/tiny_shakespeare"
                            ) -> tuple[torch.Tensor, dict[str, int], list[str]]:
    """Download (if needed) and tokenize tiny-shakespeare at character level.
    Returns (tokens, char_to_id, id_to_char). 1.1M chars, ~65 unique.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    raw_path = cache / "input.txt"
    if not raw_path.exists():
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, raw_path)
    text = raw_path.read_text()
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = chars
    tokens = torch.tensor([char_to_id[c] for c in text], dtype=torch.long)
    return tokens, char_to_id, id_to_char


def make_batch_text_iter(corpus: torch.Tensor, batch_size: int, seq_len: int,
                          n_steps: int, stride: int = 1,
                          device: str = "cuda"
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Random crops of text. Per-loop r target at position t = corpus[start_b + t + r*stride].

    tokens : [B, seq_len]
    iter_targets : [n_steps+1, B, seq_len]
      iter_targets[r, b, t] = corpus[start_b + t + r*stride]
      Loop 0 target = self (identity), so train uses r >= 1 only.

    Note: for the *last* (r * stride) positions, target reaches past the random
    crop and is taken from the contiguous corpus. We require the random start to
    leave room for n_steps*stride + seq_len tokens.
    """
    L = corpus.shape[0]
    max_start = L - seq_len - n_steps * stride - 1
    if max_start < 0:
        raise ValueError("corpus too short for these dims")
    starts = torch.randint(0, max_start, (batch_size,), device=corpus.device)
    base = starts.unsqueeze(1) + torch.arange(seq_len, device=corpus.device).unsqueeze(0)
    tokens = corpus[base].to(device)  # [B, seq_len]

    # Per-loop targets: r=0 is identity (the input itself), r=1..n_steps are
    # corpus[base + r*stride]
    iters = []
    for r in range(n_steps + 1):
        idx = base + r * stride
        iters.append(corpus[idx].to(device))
    iter_targets = torch.stack(iters, dim=0)  # [n_steps+1, B, seq_len]
    return tokens, iter_targets
