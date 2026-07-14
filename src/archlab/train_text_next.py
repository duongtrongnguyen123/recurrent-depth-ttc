"""Real-text NEXT-token prediction with iter-target supervision.

Each loop predicts the SAME target — the next token at each position.
Model implicitly refines its prediction over loops. Loss = mean across loops
of standard next-token CE. R recipe (noise injection) optional.

This is the autoregressive-LM analog of iter-target on chain: same target
every loop, model gets compute amortized over multiple passes.

Adaptive halting: a small head trained on per-loop loss change predicts
"more loops will help" or "stop here". At inference, halt fires when
prediction stabilizes — EASY tokens halt early, HARD tokens use full depth.

The TTC story: same model, varying compute per token, average-FLOPs
(weighted by halt_r distribution) is lower than fixed r at matched val loss.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_text import load_tiny_shakespeare
from .train_chain_iter_robust import RobustLoopedTransformer


def _make_next_batch(corpus, batch_size, seq_len, device):
    """Random crop. Returns (tokens, targets) where targets[t] = corpus[start+t+1]."""
    starts = torch.randint(0, len(corpus) - seq_len - 2, (batch_size,), device=device)
    idx = starts[:, None] + torch.arange(seq_len, device=device)[None]
    tokens = corpus[idx]
    targets = corpus[idx + 1]
    return tokens, targets


def train_text_next(*, n_loops: int, d: int, n_heads: int, ff_mult: int,
                      steps: int, batch_size: int, lr: float,
                      eval_every: int, eval_size: int,
                      device: str, seed: int,
                      n_loops_train: int, n_loops_eval: int,
                      seq_len: int,
                      p_noise: float = 0.0, noise_alpha: float = 0.1,
                      verbose: bool = True) -> dict:
    assert n_loops >= max(n_loops_train, n_loops_eval)
    torch.manual_seed(seed)

    print("Loading tiny-shakespeare...", flush=True)
    tokens, char_to_id, id_to_char = load_tiny_shakespeare()
    vocab = len(id_to_char)
    print(f"  loaded {len(tokens)} chars, vocab={vocab}", flush=True)
    val_start = int(0.9 * len(tokens))
    train_corpus = tokens[:val_start].to(device)
    val_corpus = tokens[val_start:].to(device)

    model = RobustLoopedTransformer(vocab=vocab, max_len=seq_len, d=d,
                                     n_heads=n_heads, ff_mult=ff_mult,
                                     n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                             weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log = {"step": [], "train_loss": [], "eval_per_r_loss": []}
    t0 = time.time()

    for step in range(steps):
        b_tokens, b_targets = _make_next_batch(train_corpus, batch_size, seq_len, device)
        all_logits = model.forward_all_loops_robust(
            b_tokens, n_loops=n_loops_train,
            p_noise=p_noise, noise_alpha=noise_alpha)
        # Per-loop CE: at each loop r, predict next token at every position
        losses = []
        for r in range(1, n_loops_train + 1):
            losses.append(F.cross_entropy(
                all_logits[r].reshape(-1, vocab),
                b_targets.reshape(-1)))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            per_r_loss = _eval_text_next(model, val_corpus, eval_size, seq_len,
                                            n_loops_eval, vocab, batch_size, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_per_r_loss"].append(per_r_loss)
            if verbose:
                key = sorted({1, 2, 4, n_loops_train, n_loops_train + 4,
                              n_loops_eval} & set(per_r_loss.keys()))
                msg = " ".join(f"r{r}={per_r_loss[r]:.3f}" for r in key)
                print(f"  step {step:>5}  loss {loss.item():.4f}  val: {msg}",
                       flush=True)

    return {
        "config": {"n_loops": n_loops, "d": d, "n_loops_train": n_loops_train,
                   "n_loops_eval": n_loops_eval, "steps": steps,
                   "batch_size": batch_size, "lr": lr,
                   "p_noise": p_noise, "noise_alpha": noise_alpha,
                   "seq_len": seq_len, "vocab": vocab},
        "params": sum(p.numel() for p in model.parameters()),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "vocab": vocab,
    }


@torch.no_grad()
def _eval_text_next(model, val_corpus, eval_size, seq_len, n_loops_eval,
                     vocab, batch_size, device):
    model.eval()
    n_batches = max(1, eval_size // batch_size)
    losses_per_r = {r: 0.0 for r in range(1, n_loops_eval + 1)}
    for _ in range(n_batches):
        b_tokens, b_targets = _make_next_batch(val_corpus, batch_size, seq_len, device)
        all_logits = model.forward_all_loops(b_tokens, n_loops=n_loops_eval)
        for r in range(1, n_loops_eval + 1):
            losses_per_r[r] += F.cross_entropy(
                all_logits[r].reshape(-1, vocab),
                b_targets.reshape(-1)).item()
    out = {r: losses_per_r[r] / n_batches for r in losses_per_r}
    model.train()
    return out
