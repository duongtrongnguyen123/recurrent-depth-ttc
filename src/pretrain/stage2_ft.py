"""Stage-2 LoRA iter-target FT on chain task.

Canonical Branch-2 recipe (CLAUDE.md item 13):
  Stage 1: pretrain LM (already done — load tanluu PCC or hduong vanilla ckpt)
  Stage 2: LoRA r=8 iter-FT at n_loops_train=8 on chain V=12 (THIS MODULE)
  Stage 3: hardcoded halt(r,k) = (r >= k)         (eval-time, no params)
  Stage 4: multi-pass with INT_TOKS-masked argmax (eval-time)

For PCC: model.forward_with_aux(x, n_loops=8) returns per-loop logits;
loop r predicts chain state after r iterations.
For vanilla with vanilla_blockwise_iter=True: each block i acts as loop i.

Train: ~1500 steps, ~3 minutes on RTX 6000 Pro.
Output: chain accuracy at user_k ∈ {1, 2, 4, 8, 16, 32, 64, 128}.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# Chain task — uses the SAME format as our reasoning-mix-mvp synth.
# To stay in the model's pretrain distribution we use the tokenizer for chain
# numerals (rather than a separate "task vocab" as in archlab synth).
#
# Format: "f(0)=a f(1)=b ... f(V-1)=z. f^k(start) = "
# Target: integer after the final "= ".


def _chain_prompt(table: list[int], start: int, k: int) -> tuple[str, int]:
    """Build a chain prompt; return (prompt, gold_answer)."""
    decl = ' '.join(f'f({i})={table[i]}' for i in range(len(table)))
    cur = start
    for _ in range(k):
        cur = table[cur]
    return f'{decl}. f^{k}({start}) = ', cur


def _listops_prompt(depth: int, V: int = 10, rng=None) -> tuple[str, int]:
    """Build a listops prompt: nested MIN/MAX/MED expression.
    depth=1: e.g., MIN[3,5,7]=3. depth=2: MIN[MAX[1,5],MED[3,7,2]]=...
    Returns (prompt, gold_int_answer).
    """
    import random as _r
    rng = rng or _r
    OPS = ['MIN', 'MAX', 'MED']
    def build(d):
        if d == 0 or rng.random() < 0.3:
            x = rng.randint(0, V - 1)
            return str(x), x
        op = rng.choice(OPS)
        nargs = rng.randint(2, 4)
        args = [build(d - 1) for _ in range(nargs)]
        text = f'{op}[' + ','.join(s for s, _ in args) + ']'
        vals = [v for _, v in args]
        if op == 'MIN':
            v = min(vals)
        elif op == 'MAX':
            v = max(vals)
        else:
            v = sorted(vals)[len(vals) // 2]
        return text, v
    expr, gold = build(depth)
    return f'Compute: {expr} = ', gold


def _modular_prompt(V: int = 12, rng=None) -> tuple[str, int]:
    """Modular (a + b) mod p problem. Returns (prompt, gold).
    p drawn from {3,5,7,11,13,17,19,23,29,31} ∩ [0, V-1]. Note V here is
    the *answer* range — gold ∈ [0, V-1], so we pick p ≤ V."""
    import random as _r
    rng = rng or _r
    PRIMES = [p for p in [3, 5, 7, 11, 13, 17, 19, 23] if p <= V]
    if not PRIMES:
        PRIMES = [V - 1]
    p = rng.choice(PRIMES)
    a = rng.randint(0, p * 5)
    b = rng.randint(0, p * 5)
    gold = (a + b) % p
    return f'Compute: ({a} + {b}) mod {p} = ', gold


def _composition_prompt(table_f: list[int], table_g: list[int], start: int,
                         k: int, j: int) -> tuple[str, int]:
    """Build f^k(g^j(start)) composition prompt.
    Two chain tables f and g. Apply g j times, then f k times. Tests
    whether iter-target FT can compose two chain operations."""
    V = len(table_f)
    decl_f = ' '.join(f'f({i})={table_f[i]}' for i in range(V))
    decl_g = ' '.join(f'g({i})={table_g[i]}' for i in range(V))
    cur = start
    for _ in range(j):
        cur = table_g[cur]
    for _ in range(k):
        cur = table_f[cur]
    return (f'{decl_f}. {decl_g}. f^{k}(g^{j}({start})) = ', cur)


def make_chain_batch(B: int, V: int, k_max: int, tokenizer, device,
                      seq_len: int) -> tuple[torch.Tensor, list[int], int]:
    """Generate a chain batch. Returns (input_ids, gold_answers, answer_pos).

    answer_pos is the position in the sequence where the next-token logit
    predicts the chain answer. The answer can be 1 or 2 tokens depending on
    tokenizer encoding ("3" = 1 tok, "10" = 1 or 2 tok).
    """
    prompts = []
    golds = []
    for _ in range(B):
        table = torch.randint(0, V, (V,)).tolist()
        start = torch.randint(0, V, (1,)).item()
        k = torch.randint(1, k_max + 1, (1,)).item()
        p, g = _chain_prompt(table, start, k)
        prompts.append(p)
        golds.append(g)
    # Tokenize with left-padding so all answer positions are the same offset.
    # Easier: tokenize separately, find max prompt len, right-align prompts.
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    Ls = [len(p) for p in prompt_ids]
    max_L = max(Ls)
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    x = torch.full((B, max_L), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(prompt_ids):
        x[i, max_L - len(ids):] = torch.tensor(ids, device=device)
    # Answer position = last index of each row = max_L - 1 (universal)
    return x, golds, max_L - 1


def make_listops_batch(B: int, V: int, depth_max: int, tokenizer, device,
                        n_loops: int,
                        max_prompt_tokens: int = 256) -> tuple[torch.Tensor, dict[int, list[int]], int]:
    """Listops with iter-target. Caps depth_max to keep prompt within
    max_prompt_tokens (default 256) — deep listops expressions can otherwise
    exceed model's max_seq_len. Also rejects oversize samples and resamples."""
    import random as _r
    # Listops at depth d has up to 4^d leaves; encoded prompt grows fast.
    # Bound depth_max conservatively at 4 (gives ~50-150 token prompts).
    bounded_depth_max = min(depth_max, 4)
    prompts = []
    golds = []
    for _ in range(B):
        for _retry in range(20):
            d = _r.randint(1, bounded_depth_max)
            p, g = _listops_prompt(d, V=V)
            # Quick token-count check (encode + len)
            n_tok = len(tokenizer.encode(p, add_special_tokens=False))
            if n_tok <= max_prompt_tokens:
                break
        prompts.append(p)
        golds.append(g)
    # Iter targets: for each r, gold_r = same gold (no sub-expression evals)
    # This is the simplest setup: model has n_loops to evaluate the expression.
    per_r_golds = {r: golds[:] for r in range(1, n_loops + 1)}
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    max_L = max(len(p) for p in prompt_ids)
    pad_id = tokenizer.eos_token_id or 0
    x = torch.full((B, max_L), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(prompt_ids):
        x[i, max_L - len(ids):] = torch.tensor(ids, device=device)
    return x, per_r_golds, max_L - 1


def make_modular_batch(B: int, V: int, tokenizer, device,
                        n_loops: int) -> tuple[torch.Tensor, dict[int, list[int]], int]:
    """Modular (a+b) mod p iter-target. Single-step task; supervise all
    loops with the same gold."""
    import random as _r
    prompts = []
    golds = []
    for _ in range(B):
        p, g = _modular_prompt(V=V)
        prompts.append(p)
        golds.append(g)
    per_r_golds = {r: golds[:] for r in range(1, n_loops + 1)}
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    max_L = max(len(p) for p in prompt_ids)
    pad_id = tokenizer.eos_token_id or 0
    x = torch.full((B, max_L), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(prompt_ids):
        x[i, max_L - len(ids):] = torch.tensor(ids, device=device)
    return x, per_r_golds, max_L - 1


def _parity_prompt(L: int, rng=None) -> tuple[str, list[int]]:
    """Cumulative parity. Returns (prompt, [parity(b[:r]) for r=1..L])."""
    import random as _r
    rng = rng or _r
    bits = [rng.randint(0, 1) for _ in range(L)]
    cum = []
    acc = 0
    for b in bits:
        acc ^= b
        cum.append(acc)
    bs = ''.join(str(b) for b in bits)
    return f'parity[{bs}] = ', cum


def make_parity_batch(B: int, V: int, n_loops: int, tokenizer, device
                       ) -> tuple[torch.Tensor, dict[int, list[int]], int]:
    """Cumulative-XOR iter-target. V acts as input bit-string length.
    Per loop r: gold_r = parity of first r bits. Predicted by Result N to
    WALL at trained depth (positional XOR has no fixed-radius reduction)."""
    import random as _r
    L = max(int(V), n_loops)   # need at least n_loops bits to supervise per-r golds
    prompts, all_cums = [], []
    for _ in range(B):
        p, cum = _parity_prompt(L, rng=_r)
        prompts.append(p); all_cums.append(cum)
    per_r_golds = {r: [c[r - 1] for c in all_cums] for r in range(1, n_loops + 1)}
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    max_L = max(len(p) for p in prompt_ids)
    pad_id = tokenizer.eos_token_id or 0
    x = torch.full((B, max_L), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(prompt_ids):
        x[i, max_L - len(ids):] = torch.tensor(ids, device=device)
    return x, per_r_golds, max_L - 1


def _arith_prompt(N: int, V: int, rng=None) -> tuple[str, list[int]]:
    """Signed cumulative-sum (mod V). N operations from N+1 operands.
    Returns (prompt, [cumsum_mod_V at step r] for r=0..N)."""
    import random as _r
    rng = rng or _r
    nums = [rng.randint(0, V - 1) for _ in range(N + 1)]
    ops = [rng.choice(['+', '-']) for _ in range(N)]
    text = str(nums[0])
    for i in range(N):
        text += ops[i] + str(nums[i + 1])
    cum = [nums[0] % V]
    for i in range(N):
        cum.append(((cum[-1] + nums[i + 1]) if ops[i] == '+' else (cum[-1] - nums[i + 1])) % V)
    return f'compute: {text} mod {V} = ', cum


def make_arith_batch(B: int, V: int, n_loops: int, tokenizer, device
                      ) -> tuple[torch.Tensor, dict[int, list[int]], int]:
    """Signed-cumsum (mod V) iter-target. Each loop r evaluates the prefix
    of r operations. State-dependent per-step rule (state=cum, parameter=
    next op + operand) — predicted by L-stream rule-complexity bound to WORK
    (single addition per loop). Contrast with parity (positional)."""
    import random as _r
    N = max(1, n_loops)
    prompts, all_cums = [], []
    for _ in range(B):
        p, cum = _arith_prompt(N, V, rng=_r)
        prompts.append(p); all_cums.append(cum)
    # per_r_golds: loop r predicts the result AFTER r operations (cum[r])
    per_r_golds = {r: [c[r] for c in all_cums] for r in range(1, n_loops + 1)}
    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    max_L = max(len(p) for p in prompt_ids)
    pad_id = tokenizer.eos_token_id or 0
    x = torch.full((B, max_L), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(prompt_ids):
        x[i, max_L - len(ids):] = torch.tensor(ids, device=device)
    return x, per_r_golds, max_L - 1


def iter_chain_targets(B: int, V: int, k_max: int, tokenizer, device, seq_len: int,
                        n_loops: int) -> tuple[torch.Tensor, dict[int, list[int]], int]:
    """Same as make_chain_batch but also returns per-loop golds:
    {r: golds_at_iteration_r for r=1..n_loops}.
    """
    prompts = []
    tables = []
    starts = []
    for _ in range(B):
        table = torch.randint(0, V, (V,)).tolist()
        start = torch.randint(0, V, (1,)).item()
        tables.append(table)
        starts.append(start)
        # For iter-target, the QUERY shows k=k_max (max depth); per-loop logits
        # predict gold at r=1, r=2, ..., r=k_max.
        p, _ = _chain_prompt(table, start, k_max)
        prompts.append(p)
    # Compute per-r golds: gold_r[i] = f^r(starts[i]) per tables[i]
    per_r_golds: dict[int, list[int]] = {}
    for r in range(1, n_loops + 1):
        gold_r = []
        for table, start in zip(tables, starts):
            cur = start
            for _ in range(r):
                cur = table[cur]
            gold_r.append(cur)
        per_r_golds[r] = gold_r

    prompt_ids = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    max_L = max(len(p) for p in prompt_ids)
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    x = torch.full((B, max_L), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(prompt_ids):
        x[i, max_L - len(ids):] = torch.tensor(ids, device=device)
    return x, per_r_golds, max_L - 1


# --- LoRA wrapper ---

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * ((x @ self.A.T) @ self.B.T)


def attach_lora(model, r: int = 8, alpha: float = 16.0,
                  target_modules: tuple = ('mlp',)) -> list[nn.Parameter]:
    """Attach LoRA adapters to all FFN/MLP Linear layers in the model.
    Freeze base weights. Return list of trainable LoRA parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    lora_params: list[nn.Parameter] = []
    device = next(model.parameters()).device

    def _attach_to_linear(parent, attr_name):
        base = getattr(parent, attr_name)
        if not isinstance(base, nn.Linear):
            return
        lora = LoRALinear(base, r=r, alpha=alpha).to(device)
        setattr(parent, attr_name, lora)
        lora_params.append(lora.A)
        lora_params.append(lora.B)

    # Walk modules; attach LoRA to FFN Linears (which contain "mlp" or "ffn" in path)
    for full_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Skip lm_head + embeddings
        if 'lm_head' in full_name or 'tok_emb' in full_name:
            continue
        # Decide if this Linear is in a "target" module
        is_target = any(t in full_name for t in target_modules)
        if not is_target:
            continue
        parts = full_name.split('.')
        parent_name = '.'.join(parts[:-1])
        attr_name = parts[-1]
        parent = model.get_submodule(parent_name)
        _attach_to_linear(parent, attr_name)

    return lora_params


# --- Iter-target loss + training loop ---

def iter_target_loss(model, x: torch.Tensor, per_r_golds: dict[int, list[int]],
                      answer_pos: int, n_loops: int, tokenizer,
                      device) -> torch.Tensor:
    """For each loop r, model's per-loop logits at answer_pos should match
    P(gold_token | prompt). Loss = mean_r CE(logits_r[answer_pos], gold_token_r)."""
    out = model.forward_with_aux(x, n_loops=n_loops, aux_min_loops=1)
    per_loop_logits = out['per_loop_logits']  # list[T, B, T, V] technically [B, T, V]
    losses = []
    for r in range(1, n_loops + 1):
        if r > len(per_loop_logits):
            break
        logits_r = per_loop_logits[r - 1][:, answer_pos, :]  # [B, V]
        # Map each gold integer to the single token id that encodes the digit.
        gold_strs = [str(g) for g in per_r_golds[r]]
        # Encode as single tokens (assume V<=99 → digit/multi-digit; pick first token)
        gold_tokens = []
        for s in gold_strs:
            ids = tokenizer.encode(s, add_special_tokens=False)
            gold_tokens.append(ids[0])
        gold_tokens = torch.tensor(gold_tokens, device=device)
        losses.append(F.cross_entropy(logits_r, gold_tokens))
    return torch.stack(losses).mean()


def train_lora_iter_ft(model, tokenizer, *,
                        V: int = 12, k_train: int = 8,
                        n_loops: int = 8,
                        steps: int = 1500,
                        batch_size: int = 32,
                        lr: float = 1e-3,
                        lora_r: int = 8, lora_alpha: float = 16.0,
                        device: str = 'cuda',
                        log_every: int = 100,
                        task: str = 'chain',
                        variant: str = 'bp',
                        use_lora: bool = True,
                        use_skip: bool = False,
                        skip_alpha_init: float = 0.1) -> dict:
    # Listops at depth-4 produces longer prompts that blow up activation memory
    # with batch=32 × n_loops=8. Drop batch_size for listops; same effective
    # capacity, just slower per step but fits.
    if task == 'listops':
        batch_size = min(batch_size, 8)
    # Full FT: smaller batch + smaller LR (CC4-CC5 from-scratch replay at large scale).
    if not use_lora:
        batch_size = min(batch_size, 16)
        lr = 5e-5
    """Run the canonical LoRA iter-target FT recipe (or full FT if use_lora=False).
    task: one of 'chain', 'listops', 'modular'.
    Returns {final_loss, train_log, lora_params_count}."""
    print(f'[stage2-ft] task={task} V={V} k_train={k_train} n_loops={n_loops} '
          f'steps={steps} use_lora={use_lora} batch={batch_size} lr={lr}',
          flush=True)
    if use_lora:
        lora_params = attach_lora(model, r=lora_r, alpha=lora_alpha)
        n_lora = sum(p.numel() for p in lora_params)
        print(f'[stage2-ft] LoRA params: {n_lora/1e3:.1f}K trainable', flush=True)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        lora_params = [p for p in model.parameters() if p.requires_grad]
    # Optional: add gated_h0 skip residual (Branch 1 AJ / archlab bp_plus_h0).
    if use_skip and getattr(model, 'skip_alpha', None) is None:
        model.skip_alpha = nn.Parameter(torch.tensor(skip_alpha_init, device=device))
        model.cfg.skip_alpha_init = skip_alpha_init
        lora_params.append(model.skip_alpha)
        n_lora = sum(p.numel() for p in lora_params if p.requires_grad)
        print(f'[stage2-ft] +gated_h0 skip_alpha init={skip_alpha_init} trainable; total={n_lora}', flush=True)
        n_lora = sum(p.numel() for p in lora_params)
        print(f'[stage2-ft] FULL FT: {n_lora/1e6:.1f}M trainable', flush=True)
    model.train()

    optim = torch.optim.AdamW(lora_params, lr=lr, betas=(0.9, 0.95))
    log = []
    t0 = time.time()
    last_loss = float('inf')

    def _make_batch():
        if task == 'chain':
            return iter_chain_targets(batch_size, V, k_train, tokenizer,
                                        device, seq_len=0, n_loops=n_loops)
        elif task == 'listops':
            return make_listops_batch(batch_size, V, k_train, tokenizer,
                                        device, n_loops=n_loops)
        elif task == 'modular':
            return make_modular_batch(batch_size, V, tokenizer, device,
                                        n_loops=n_loops)
        elif task == 'parity':
            return make_parity_batch(batch_size, V, n_loops, tokenizer, device)
        elif task == 'arith':
            return make_arith_batch(batch_size, V, n_loops, tokenizer, device)
        else:
            raise ValueError(f'unknown task: {task}')

    randomize_r = (variant == 'random_r')
    if randomize_r:
        print(f'[stage2-ft] variant=random_r — sampling n_loops_step ~ Uniform{{1..{n_loops}}} per step', flush=True)

    for step in range(1, steps + 1):
        x, per_r_golds, answer_pos = _make_batch()
        n_loops_step = (int(torch.randint(1, n_loops + 1, (1,)).item())
                          if randomize_r else n_loops)
        loss = iter_target_loss(model, x, per_r_golds, answer_pos, n_loops_step,
                                  tokenizer, device)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optim.step()
        last_loss = loss.item()
        if step % log_every == 0:
            elapsed = time.time() - t0
            print(f'[stage2-ft] step {step:>4}/{steps}  loss {last_loss:.4f}  '
                  f'elapsed {elapsed:.0f}s', flush=True)
            log.append({'step': step, 'loss': last_loss, 'elapsed': elapsed})

    return {'final_loss': last_loss, 'train_log': log, 'lora_n_params': n_lora}


# --- Multi-pass eval (Stage 4) ---

@torch.no_grad()
def eval_listops(model, tokenizer, *, V: int = 10,
                  depths: tuple = (1, 2, 3, 4),
                  n_loops_train: int = 8,
                  eval_examples: int = 100,
                  max_prompt_tokens: int = 256,
                  device: str = 'cuda') -> dict:
    """Listops eval. Caps depth and rejects oversize prompts."""
    import random as _r
    model.eval()
    digit_tokens = [tokenizer.encode(str(d), add_special_tokens=False)[0]
                      for d in range(V)]
    digit_tokens_t = torch.tensor(digit_tokens, device=device)
    results = {}
    for d in depths:
        correct = 0
        n_used = 0
        retries = 0
        while n_used < eval_examples:
            p, g = _listops_prompt(d, V=V)
            ids = tokenizer.encode(p, add_special_tokens=False)
            if len(ids) > max_prompt_tokens:
                retries += 1
                if retries > 100:
                    break
                continue
            x = torch.tensor([ids], device=device)
            n_used += 1
            with torch.amp.autocast(device_type=device,
                                      dtype=torch.bfloat16, enabled=True):
                out = model.forward_with_aux(x, n_loops=n_loops_train,
                                                aux_min_loops=1)
            logits = out['per_loop_logits'][-1][0, -1, :]
            digit_logits = logits[digit_tokens_t]
            pred = int(torch.argmax(digit_logits).item())
            if pred == g:
                correct += 1
        results[d] = correct / eval_examples
        print(f'[listops] depth={d:>2}  acc={results[d]:.3f}', flush=True)
    return results


@torch.no_grad()
def eval_modular(model, tokenizer, *, V: int = 12,
                  n_loops_train: int = 8,
                  eval_examples: int = 300,
                  device: str = 'cuda') -> dict:
    """Modular (a+b) mod p eval across all training primes; report per-p accuracy."""
    import random as _r
    model.eval()
    digit_tokens = [tokenizer.encode(str(d), add_special_tokens=False)[0]
                      for d in range(V)]
    digit_tokens_t = torch.tensor(digit_tokens, device=device)
    # Bucket by p
    PRIMES = [p for p in [3, 5, 7, 11, 13, 17, 19, 23] if p <= V]
    per_p_correct = {p: 0 for p in PRIMES}
    per_p_total = {p: 0 for p in PRIMES}
    for _ in range(eval_examples):
        p_text, gold = _modular_prompt(V=V)
        # Parse p from prompt: "Compute: (X + Y) mod P = "
        import re
        m = re.search(r'mod (\d+)', p_text)
        if not m:
            continue
        pmod = int(m.group(1))
        if pmod not in per_p_correct:
            continue
        ids = tokenizer.encode(p_text, add_special_tokens=False)
        x = torch.tensor([ids], device=device)
        with torch.amp.autocast(device_type=device,
                                  dtype=torch.bfloat16, enabled=True):
            out = model.forward_with_aux(x, n_loops=n_loops_train,
                                            aux_min_loops=1)
        logits = out['per_loop_logits'][-1][0, -1, :]
        digit_logits = logits[digit_tokens_t]
        pred = int(torch.argmax(digit_logits).item())
        per_p_total[pmod] += 1
        if pred == gold:
            per_p_correct[pmod] += 1
    results = {p: (per_p_correct[p] / max(per_p_total[p], 1)) for p in PRIMES}
    overall = sum(per_p_correct.values()) / max(sum(per_p_total.values()), 1)
    results['overall'] = overall
    for p in PRIMES:
        print(f'[modular] p={p:>2}  n={per_p_total[p]:>3}  acc={results[p]:.3f}',
              flush=True)
    print(f'[modular] OVERALL acc={overall:.3f}', flush=True)
    return results


@torch.no_grad()
def eval_multipass(model, tokenizer, *, V: int = 12,
                    user_k_values: tuple = (1, 2, 4, 8, 16, 32, 64, 128, 256),
                    n_loops_train: int = 8,
                    eval_examples: int = 100,
                    batch_size: int = 50,
                    device: str = 'cuda') -> dict:
    """TRUE multi-pass eval: pass i predicts f^{n_loops_train}(cur_state); use
    that as cur_state for pass i+1. After ceil(user_k / n_loops_train) passes,
    cumulative depth ≥ user_k.

    Halt rule: at the final pass, read loop r = (user_k % n_loops_train) if
    nonzero, else n_loops_train. For intermediate passes, always read loop r =
    n_loops_train (full depth). Hardcoded halt(r, k_local) = (r >= k_local).
    """
    model.eval()
    results = {}
    # Precompute digit token ids once (same tokenizer)
    digit_tokens = [tokenizer.encode(str(d), add_special_tokens=False)[0]
                      for d in range(V)]
    digit_tokens_t = torch.tensor(digit_tokens, device=device)

    for user_k in user_k_values:
        n_passes = (user_k + n_loops_train - 1) // n_loops_train   # ceil div
        final_pass_loops = user_k - n_loops_train * (n_passes - 1)  # 1..n_loops_train

        correct_total = 0
        n_done = 0
        while n_done < eval_examples:
            B = min(batch_size, eval_examples - n_done)

            # Generate a batch of (table, start, gold) tuples
            tables = [torch.randint(0, V, (V,)).tolist() for _ in range(B)]
            starts = [int(torch.randint(0, V, (1,)).item()) for _ in range(B)]
            golds = []
            for tbl, s in zip(tables, starts):
                cur = s
                for _ in range(user_k):
                    cur = tbl[cur]
                golds.append(cur)

            # Multi-pass: cur_state tracks the running chain state across passes
            cur_states = list(starts)

            for pass_idx in range(n_passes):
                # Build prompts: prompt at pass uses the table + cur_state as new "start"
                # and asks f^{loops_this_pass}(cur_state)
                loops_this_pass = (n_loops_train if pass_idx < n_passes - 1
                                     else final_pass_loops)
                # We always probe at loop r=loops_this_pass of the model — the
                # supervision trained loop r → state after r iterations.
                prompts = [_chain_prompt(tables[i], cur_states[i], loops_this_pass)[0]
                            for i in range(B)]
                prompt_ids = [tokenizer.encode(p, add_special_tokens=False)
                               for p in prompts]
                max_L = max(len(p) for p in prompt_ids)
                pad_id = tokenizer.eos_token_id or 0
                x = torch.full((B, max_L), pad_id, dtype=torch.long, device=device)
                for i, ids in enumerate(prompt_ids):
                    x[i, max_L - len(ids):] = torch.tensor(ids, device=device)

                with torch.amp.autocast(device_type=device,
                                          dtype=torch.bfloat16, enabled=True):
                    out = model.forward_with_aux(x, n_loops=n_loops_train,
                                                    aux_min_loops=1)
                # Read loop r = loops_this_pass at the answer position (last col)
                logits = out['per_loop_logits'][loops_this_pass - 1][:, -1, :]  # [B, vocab]
                digit_logits = logits[:, digit_tokens_t]  # [B, V]
                preds = torch.argmax(digit_logits, dim=-1).tolist()
                cur_states = preds  # feed forward to next pass

            # After all passes, cur_states is the final predicted f^user_k(start)
            for p, g in zip(cur_states, golds):
                if p == g:
                    correct_total += 1
            n_done += B

        results[user_k] = correct_total / eval_examples
        print(f'[multipass] user_k={user_k:>3}  passes={n_passes}  '
              f'final_loops={final_pass_loops}  acc={results[user_k]:.3f}',
              flush=True)
    return results


@torch.no_grad()
def eval_per_r(model, tokenizer, *, task: str, V: int, n_loops_train: int,
                user_k_values: tuple, eval_examples: int = 200,
                device: str = 'cuda') -> dict:
    """Per-r single-pass eval for parity / arith. For each user_k, runs the
    model with that many loops and measures gold-token accuracy at the answer
    position. No multipass — these tasks are sensitive to depth wall."""
    model.eval()
    results = {}
    B = min(eval_examples, 64)
    for user_k in user_k_values:
        correct = 0; total = 0
        for _ in range(max(1, eval_examples // B)):
            if task == 'parity':
                x, per_r_golds, ans_pos = make_parity_batch(B, V, user_k, tokenizer, device)
            else:
                x, per_r_golds, ans_pos = make_arith_batch(B, V, user_k, tokenizer, device)
            gold_r = per_r_golds[user_k]
            # Run model for user_k loops and check last-loop prediction at ans_pos
            try:
                logits = model(x, n_loops=user_k)['logits']
            except TypeError:
                logits = model(x).logits if hasattr(model(x), 'logits') else model(x)[0]
            last = logits[:, ans_pos, :]
            for i in range(B):
                gold_ids = tokenizer.encode(str(gold_r[i]), add_special_tokens=False)
                if gold_ids and int(last[i].argmax().item()) == gold_ids[0]:
                    correct += 1
                total += 1
        results[user_k] = round(correct / max(1, total), 3)
        print(f'[per-r] task={task} user_k={user_k:>3}  acc={results[user_k]:.3f}',
              flush=True)
    return results


def run(ckpt_dir: str, data_dir: str, out_path: str = 'stage2_ft.json',
         steps: int = 1500, n_loops_train: int = 8,
         V: int = 12,
         user_k_values: tuple = (1, 2, 4, 8, 16, 32, 64, 128),
         task: str = 'chain', variant: str = 'bp',
         use_lora: bool = True, batch_size: int = 32,
         use_skip: bool = False) -> dict:
    """Top-level: load model+tokenizer, run LoRA iter-FT (or full FT), eval multipass."""
    from transformers import AutoTokenizer
    from pretrain.model import PretrainConfig, build_model

    ckpt_path = Path(ckpt_dir)
    data_path = Path(data_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}', flush=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(data_path / 'tokenizer'))
    print(f'tokenizer vocab_size={tokenizer.vocab_size}', flush=True)

    # Load model
    cfg_path = ckpt_path.parent.parent / 'config.json'
    cfg_data = json.loads(cfg_path.read_text())
    field_names = {f.name for f in PretrainConfig.__dataclass_fields__.values()}
    arch_cfg = PretrainConfig(**{k: v for k, v in cfg_data['arch'].items()
                                  if k in field_names})
    print(f'arch={arch_cfg.arch} d={arch_cfg.d_model} trained_n_loops={arch_cfg.n_loops}',
          flush=True)
    model = build_model(arch_cfg).to(device)
    state = torch.load(ckpt_path / 'state.pt', map_location='cpu',
                        weights_only=False)
    model.load_state_dict(state['model'])
    print(f'[load] loaded weights from {ckpt_path / "state.pt"}', flush=True)

    print(f'[task] {task} V={V}  k_train={n_loops_train}', flush=True)
    # Stage 2: LoRA iter-target FT
    ft_out = train_lora_iter_ft(model, tokenizer, V=V, k_train=n_loops_train,
                                  n_loops=n_loops_train, steps=steps,
                                  device=device, task=task, variant=variant,
                                  use_lora=use_lora, batch_size=batch_size,
                                  use_skip=use_skip)

    # Stage 4: task-specific eval
    print(f'[eval] running {task} eval...', flush=True)
    if task == 'chain':
        eval_out = eval_multipass(model, tokenizer, V=V,
                                    user_k_values=user_k_values,
                                    n_loops_train=n_loops_train,
                                    eval_examples=100, device=device)
    elif task == 'listops':
        eval_out = eval_listops(model, tokenizer, V=V,
                                  depths=(1, 2, 3, 4, 5, 6),
                                  n_loops_train=n_loops_train,
                                  eval_examples=100, device=device)
    elif task == 'modular':
        eval_out = eval_modular(model, tokenizer, V=V,
                                  n_loops_train=n_loops_train,
                                  eval_examples=300, device=device)
    elif task == 'parity':
        eval_out = eval_per_r(model, tokenizer, task='parity', V=V,
                              n_loops_train=n_loops_train,
                              user_k_values=user_k_values,
                              eval_examples=200, device=device)
    elif task == 'arith':
        eval_out = eval_per_r(model, tokenizer, task='arith', V=V,
                              n_loops_train=n_loops_train,
                              user_k_values=user_k_values,
                              eval_examples=200, device=device)
    else:
        raise ValueError(f'unknown task: {task}')

    payload = {
        'ckpt_dir': str(ckpt_path),
        'arch': arch_cfg.arch,
        'd_model': arch_cfg.d_model,
        'pretrain_n_loops': arch_cfg.n_loops,
        'ft_n_loops_train': n_loops_train,
        'ft_steps': steps,
        'ft_use_lora': use_lora,
        'ft_final_loss': ft_out['final_loss'],
        'ft_lora_params': ft_out['lora_n_params'],
        'eval_chain_acc': eval_out,
        'ft_log': ft_out['train_log'],
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f'saved -> {out_path}', flush=True)
    return payload


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', required=True)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out', default='stage2_ft.json')
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--n-loops-train', type=int, default=8)
    ap.add_argument('--V', type=int, default=12)
    ap.add_argument('--user-k-values', default='1,2,4,8,16,32,64,128',
                     help='comma-separated user_k list for eval')
    args = ap.parse_args()
    uk = tuple(int(x) for x in args.user_k_values.split(','))
    run(args.ckpt_dir, args.data_dir, out_path=args.out,
        steps=args.steps, n_loops_train=args.n_loops_train, V=args.V,
        user_k_values=uk)
