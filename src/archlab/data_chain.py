"""Iterated pointer-chain lookup. Each example specifies its own random function on
[0, V) via the first V positions; the query asks for f^k(start) for given start and k.

Depth requirement: at least k self-attention layers — there is no algebraic shortcut
because the function f changes per example, so the model can't memorize a closure.

Sequence layout (V=12, max_k=8):
  positions  0..V-1     pointer table tokens (each in [0, V))    V=12
  position   V          start token      (in [0, V))             1
  position   V+1        depth token      (k in [1, max_k])       1   *encoded as 100+k*
  position   V+2        EQ                                       1
  position   V+3        answer token     (the value f^k(start))

Loss is computed at position V+2 (predicting answer at position V+3).

Vocab:
  0..V-1               value/start tokens
  100..100+max_k       depth indicators k=1..max_k
  EQ                   = 200
  PAD                  = 201
"""
from __future__ import annotations

import torch

V = 12
MAX_K = 12  # depth tokens occupy DEPTH_OFFSET+1 .. DEPTH_OFFSET+MAX_K (must stay < EQ).
            # Bumped from 8 to 12 to support length-extrapolation eval beyond k=8.
DEPTH_OFFSET = 100
HOP = 199           # unary-depth encoding token (k repeated HOPs replace single depth token)
EQ = 200
PAD = 201
VOCAB_SIZE = 202

# Unary-depth layout constants (Result J disambiguation):
#   positions 0..V-1                pointer table
#   position  V                     start
#   positions V+1..V+k              HOP tokens (count = k)
#   positions V+k+1..V+MAX_K        PAD tokens (filler so all positions are populated)
#   position  V+MAX_K+1             EQ
#   position  V+MAX_K+2             answer
# Fixed sequence length lets train and OOD-eval share position embeddings.
SEQ_LEN_UNARY = V + MAX_K + 3
ANSWER_POS_UNARY = V + MAX_K + 1


def make_batch_chain(batch_size: int, k_min: int = 1, k_max: int = MAX_K,
                     device: str = "cuda"
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a batch of chain-lookup examples.

    Returns (tokens, targets, target_mask, k_values).
    Sequence length is V + 4. Loss is at position V+2 (the EQ token), predicting
    the answer at position V+3 (i.e. the next-token target there is the chain output).
    """
    # Random pointer tables: each row is a function from [0, V) -> [0, V).
    table = torch.randint(0, V, (batch_size, V), device=device)
    start = torch.randint(0, V, (batch_size,), device=device)
    k = torch.randint(k_min, k_max + 1, (batch_size,), device=device)

    # Compute f^k(start) by iterating k_max steps.
    cur = start.clone()
    answer = torch.zeros_like(start)
    for step in range(1, k_max + 1):
        cur = table[torch.arange(batch_size, device=device), cur]
        # Save when this step matches the desired k.
        sel = k == step
        answer[sel] = cur[sel]

    depth_tok = (k + DEPTH_OFFSET).long()
    eq = torch.full((batch_size, 1), EQ, device=device)
    pad = torch.full((batch_size, 1), PAD, device=device)

    tokens = torch.cat([table, start.view(-1, 1), depth_tok.view(-1, 1), eq, pad], dim=1)
    # We want to predict at position V+2 (EQ). The next-token target there is `answer`.
    # Layout: tokens[i] for i in 0..V+3, where tokens[V+3] is initially PAD; we replace
    # it with the answer so that the next-token target after position V+2 = answer.
    tokens[:, V + 3] = answer

    targets = torch.full_like(tokens, PAD)
    targets[:, V + 2] = answer
    target_mask = torch.zeros_like(tokens, dtype=torch.bool)
    target_mask[:, V + 2] = True
    return tokens, targets, target_mask, k


def make_batch_chain_unary(batch_size: int, k_min: int = 1, k_max: int = MAX_K,
                            device: str = "cuda"
                            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chain-task batch with depth k encoded as k repeated HOP tokens (unary)
    instead of a single depth token. Sequence length is fixed at SEQ_LEN_UNARY
    so position embeddings stay in-distribution even when k differs.

    Used for the Result J disambiguation: at test time, k > train_k_max means
    *more HOP tokens at later positions*, but the HOP token itself is always
    in-distribution (single embedding, seen for every training k).
    """
    table = torch.randint(0, V, (batch_size, V), device=device)
    start = torch.randint(0, V, (batch_size,), device=device)
    k = torch.randint(k_min, k_max + 1, (batch_size,), device=device)

    cur = start.clone()
    answer = torch.zeros_like(start)
    for step in range(1, MAX_K + 1):
        cur = table[torch.arange(batch_size, device=device), cur]
        sel = k == step
        answer[sel] = cur[sel]

    tokens = torch.full((batch_size, SEQ_LEN_UNARY), PAD, dtype=torch.long, device=device)
    tokens[:, :V] = table
    tokens[:, V] = start
    pos_idx = torch.arange(SEQ_LEN_UNARY, device=device).unsqueeze(0)   # [1, S]
    is_hop = (pos_idx > V) & (pos_idx <= V + k.unsqueeze(1))             # [B, S]
    tokens[is_hop] = HOP
    tokens[:, ANSWER_POS_UNARY] = EQ
    tokens[:, ANSWER_POS_UNARY + 1] = answer

    targets = torch.full_like(tokens, PAD)
    targets[:, ANSWER_POS_UNARY] = answer
    target_mask = torch.zeros_like(tokens, dtype=torch.bool)
    target_mask[:, ANSWER_POS_UNARY] = True
    return tokens, targets, target_mask, k


# Iterative-target layout (Result L candidate):
#   positions 0..V-1     pointer table
#   position  V          start
#   position  V+1        EQ — output here at loop r predicts f^r(start)
# No depth token. The model is trained to compute f^r(start) at every loop r,
# so at inference loop r > train_max the model (if it learned the per-step rule)
# extrapolates by continuing to apply f.
SEQ_LEN_ITER = V + 2
ANSWER_POS_ITER = V + 1


def make_batch_chain_iter(batch_size: int, n_steps: int,
                           device: str = "cuda"
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Iterative-target chain batch.
    Returns (tokens, iter_targets) where:
      tokens       : [B, V+2] = [table | start | EQ]
      iter_targets : [n_steps+1, B] with iter_targets[r] = f^r(start)

    Use iter_targets[1..n_steps] as the per-loop supervision; loop 0 is
    skipped during training (it's just the embedding pre-recurrence).
    """
    table = torch.randint(0, V, (batch_size, V), device=device)
    start = torch.randint(0, V, (batch_size,), device=device)

    cur = start.clone()
    iters = [start.clone()]
    for _ in range(n_steps):
        cur = table[torch.arange(batch_size, device=device), cur]
        iters.append(cur.clone())
    iter_targets = torch.stack(iters, dim=0)  # [n_steps+1, B]

    tokens = torch.full((batch_size, SEQ_LEN_ITER), PAD, dtype=torch.long, device=device)
    tokens[:, :V] = table
    tokens[:, V] = start
    tokens[:, V + 1] = EQ
    return tokens, iter_targets


# Combined layout (Result M candidate: iterate then halt):
#   Same as default chain layout — table | start | depth_k | EQ | answer
# Per-loop target is:
#   f^r(start) for r <= k     (per-step rule supervision, learns iteration)
#   f^k(start) for r > k      (halt supervision, learns to hold the answer)


def make_batch_chain_combined(batch_size: int, k_min: int, k_max: int,
                               n_loops_train: int, device: str = "cuda"
                               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combined iter+halt batch.

    Returns (tokens, per_loop_targets, k):
      tokens            : [B, V+4]                — table | start | depth_k | EQ | PAD
      per_loop_targets  : [n_loops_train+1, B]    — per-loop scalar target (chain value)
      k                 : [B]                      — per-example chain depth
    """
    table = torch.randint(0, V, (batch_size, V), device=device)
    start = torch.randint(0, V, (batch_size,), device=device)
    k = torch.randint(k_min, k_max + 1, (batch_size,), device=device)

    cur = start.clone()
    iters = [start.clone()]                           # iters[0] = start
    for _ in range(n_loops_train):
        cur = table[torch.arange(batch_size, device=device), cur]
        iters.append(cur.clone())
    iter_states = torch.stack(iters, dim=0)            # [n_loops_train+1, B]

    # answer_at_k[i] = iter_states[k_i, i] = f^{k_i}(start_i)
    idx = k.clamp(max=n_loops_train)
    bs_arange = torch.arange(batch_size, device=device)
    answer_at_k = iter_states[idx, bs_arange]          # [B]

    # Per-loop target: f^r(start) for r <= k, else f^k(start)
    per_loop_targets = iter_states.clone()
    for r in range(1, n_loops_train + 1):
        sel = r > k
        per_loop_targets[r, sel] = answer_at_k[sel]

    seq_len = V + 4   # match the default chain layout for compatibility with halt_eval_chain
    tokens = torch.full((batch_size, seq_len), PAD, dtype=torch.long, device=device)
    tokens[:, :V] = table
    tokens[:, V] = start
    tokens[:, V + 1] = (k + DEPTH_OFFSET).long()       # depth token (in-dist if k <= MAX_K)
    tokens[:, V + 2] = EQ
    tokens[:, V + 3] = answer_at_k                      # placeholder; loss is at V+2

    return tokens, per_loop_targets, k
