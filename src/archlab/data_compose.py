"""Compositional rule-selection chain task — Result AE candidate.

Generalizes data_chain.make_batch_chain_iter to TWO functions f and g per example,
where each example specifies a *rule string* indicating which function to apply
at each step. This tests whether the model can learn to *select among rules
per loop* using context, rather than hard-coding a single rule.

Layout (V=8, max_k=8):

  positions 0..V-1                 f-table (V tokens, each in [0, V))
  positions V..2V-1                g-table
  position  2V                     start (in [0, V))
  positions 2V+1..2V+k             rule indicators (RULE_F=100 or RULE_G=101)
  position  2V+k+1                 EQ
  -- iter-target supervised at this position --
  position  2V+k+2                 (final answer; only for sanity, unused as input)

Per-step rule (Result N test): at loop r, the model must apply
  - f if rule[r-1] == RULE_F,
  - g if rule[r-1] == RULE_G.
The selection at step r depends on the *input symbol* at sequence position 2V+r
(in context), NOT on a position counter. This is **position-self-referential** —
should fit the iter-target recipe per Result N.

Iter-target: at loop r ∈ [1, n_steps], target = state after applying rules[0..r-1]
to start. Recipe matches make_batch_chain_iter's per-loop supervision pattern.
"""
from __future__ import annotations

import torch

V = 8                      # smaller alphabet than data_chain (12) to limit seq length
MAX_K = 24                 # supports OOD eval up to r=24 (3× train depth at n=8)
RULE_F = 100
RULE_G = 101
EQ = 200
PAD = 201
VOCAB_SIZE = 202

# Layout: 2 tables (2V) + start (1) + rule_string (k) + EQ (1) = 2V + k + 2 positions.
# Use MAX_K for sequence length budget; pad rule positions when k < MAX_K.
SEQ_LEN_COMPOSE = 2 * V + MAX_K + 2     # answer slot is the EQ position (last)
ANSWER_POS_COMPOSE = 2 * V + MAX_K + 1  # last position (where iter-target reads)


def make_batch_compose_iter(batch_size: int, n_steps: int,
                            device: str = "cuda"
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compositional iter-target batch.

    Returns:
      tokens       : [B, SEQ_LEN_COMPOSE]
      iter_targets : [n_steps+1, B] where iter_targets[r] = state after r rule-applications
    """
    assert 1 <= n_steps <= MAX_K
    table_f = torch.randint(0, V, (batch_size, V), device=device)
    table_g = torch.randint(0, V, (batch_size, V), device=device)
    start = torch.randint(0, V, (batch_size,), device=device)
    # Generate rule tokens for ALL MAX_K rule positions so position embeddings
    # see rule-distribution at every position regardless of n_steps. The
    # iter-target trajectory only consumes the first n_steps rules; positions
    # past n_steps still carry rule tokens but the model never reads them
    # at training time (no per-loop supervision past r = n_steps). At eval
    # with n_steps_eval > n_steps_train, the higher loops attend to the
    # already-populated rule tokens at positions 2V+n_steps_train+1..2V+MAX_K.
    rule_choice = torch.randint(0, 2, (batch_size, MAX_K), device=device)

    # Trajectory: state after each rule application (only first n_steps used).
    cur = start.clone()
    iters = [start.clone()]
    arange = torch.arange(batch_size, device=device)
    for r in range(n_steps):
        nxt_f = table_f[arange, cur]
        nxt_g = table_g[arange, cur]
        cur = torch.where(rule_choice[:, r] == 0, nxt_f, nxt_g)
        iters.append(cur.clone())
    iter_targets = torch.stack(iters, dim=0)  # [n_steps+1, B]

    # Build the input sequence.
    tokens = torch.full((batch_size, SEQ_LEN_COMPOSE), PAD, dtype=torch.long,
                        device=device)
    tokens[:, :V] = table_f
    tokens[:, V:2 * V] = table_g
    tokens[:, 2 * V] = start
    rule_tokens = torch.where(rule_choice == 0,
                              torch.tensor(RULE_F, device=device),
                              torch.tensor(RULE_G, device=device))
    # Fill all MAX_K rule positions (not just n_steps) — see comment above.
    tokens[:, 2 * V + 1:2 * V + 1 + MAX_K] = rule_tokens
    tokens[:, ANSWER_POS_COMPOSE] = EQ
    return tokens, iter_targets


def make_batch_compose_iter_random_k(batch_size: int, k_min: int, k_max: int,
                                     device: str = "cuda"
                                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-example random k ∈ [k_min, k_max]. Returns (tokens, iter_targets, k_per_example).

    iter_targets is padded to [k_max+1, B]; entries past per-example k are repeats
    of the trajectory's last value (so the recipe trains "hold the answer past
    your true depth").
    """
    assert 1 <= k_min <= k_max <= MAX_K
    ks = torch.randint(k_min, k_max + 1, (batch_size,), device=device)
    table_f = torch.randint(0, V, (batch_size, V), device=device)
    table_g = torch.randint(0, V, (batch_size, V), device=device)
    start = torch.randint(0, V, (batch_size,), device=device)
    # Pad per-example rules to k_max; we'll only use the first per-example k.
    rule_choice = torch.randint(0, 2, (batch_size, k_max), device=device)

    cur = start.clone()
    iters = [start.clone()]
    arange = torch.arange(batch_size, device=device)
    for r in range(k_max):
        nxt_f = table_f[arange, cur]
        nxt_g = table_g[arange, cur]
        cur_new = torch.where(rule_choice[:, r] == 0, nxt_f, nxt_g)
        # Only step if we haven't exceeded this example's k.
        active = (r < ks).float()
        cur = (active * cur_new + (1 - active) * cur).long()
        iters.append(cur.clone())
    iter_targets = torch.stack(iters, dim=0)

    tokens = torch.full((batch_size, SEQ_LEN_COMPOSE), PAD, dtype=torch.long,
                        device=device)
    tokens[:, :V] = table_f
    tokens[:, V:2 * V] = table_g
    tokens[:, 2 * V] = start
    rule_tokens = torch.where(rule_choice == 0,
                              torch.tensor(RULE_F, device=device),
                              torch.tensor(RULE_G, device=device))
    # Mask rule tokens past per-example k to PAD (so the model's context only
    # contains the rules it should apply).
    pos_idx = torch.arange(k_max, device=device).unsqueeze(0)
    rule_tokens = torch.where(pos_idx < ks.unsqueeze(1), rule_tokens,
                              torch.tensor(PAD, device=device))
    tokens[:, 2 * V + 1:2 * V + 1 + k_max] = rule_tokens
    tokens[:, ANSWER_POS_COMPOSE] = EQ
    return tokens, iter_targets, ks
