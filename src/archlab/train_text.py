"""Real-text iter-target training (Result AD candidate).

Multi-step horizon: at loop r, the model's logit at position t predicts the
character at position t + r * stride. Training supervises every (r, t) pair
with the appropriate target. Test extrapolation past trained n_loops by
evaluating per-r val loss past n_loops_train.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .data_text import load_tiny_shakespeare, make_batch_text_iter
from .train_chain_iter_robust import RobustLoopedTransformer


def train_text_iter(*, n_loops: int, d: int, n_heads: int, ff_mult: int,
                     steps: int, batch_size: int, lr: float,
                     eval_every: int, eval_size: int,
                     device: str, seed: int,
                     n_loops_train: int, n_loops_eval: int,
                     seq_len: int, stride: int,
                     p_noise: float = 0.0, noise_alpha: float = 0.1,
                     verbose: bool = True) -> dict:
    assert n_loops >= max(n_loops_train, n_loops_eval)
    torch.manual_seed(seed)

    print("Loading tiny-shakespeare...")
    tokens, char_to_id, id_to_char = load_tiny_shakespeare()
    vocab = len(id_to_char)
    print(f"  loaded {len(tokens)} chars, vocab={vocab}")
    # Train/val split: last 10% as val
    val_start = int(0.9 * len(tokens))
    train_corpus = tokens[:val_start]
    val_corpus = tokens[val_start:]
    train_corpus_dev = train_corpus.to(device)
    val_corpus_dev = val_corpus.to(device)

    model = RobustLoopedTransformer(vocab=vocab, max_len=seq_len, d=d,
                                     n_heads=n_heads, ff_mult=ff_mult,
                                     n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                             weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    log: dict[str, list] = {"step": [], "train_loss": [], "eval_per_r_loss": []}
    t0 = time.time()

    for step in range(steps):
        batch_tokens, iter_targets = make_batch_text_iter(
            train_corpus_dev, batch_size, seq_len, n_loops_train, stride, device)
        # iter_targets: [n_steps+1, B, seq_len], iter_targets[r, b, t] = corpus[start_b + t + r*stride]
        all_logits = model.forward_all_loops_robust(
            batch_tokens, n_loops=n_loops_train,
            p_noise=p_noise, noise_alpha=noise_alpha)
        # all_logits: [n_loops_train+1, B, seq_len, V]

        losses = []
        for r in range(1, n_loops_train + 1):
            logits_r = all_logits[r]  # [B, seq_len, V]
            target_r = iter_targets[r]  # [B, seq_len]
            losses.append(F.cross_entropy(logits_r.reshape(-1, vocab),
                                            target_r.reshape(-1)))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps - 1:
            per_r_loss = _eval_text(model, val_corpus_dev, eval_size,
                                      seq_len, n_loops_eval, stride,
                                      vocab, batch_size, device)
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_per_r_loss"].append(per_r_loss)
            if verbose:
                key = sorted({1, 2, 4, n_loops_train, n_loops_train + 4,
                              n_loops_train + 8, n_loops_eval} & set(per_r_loss.keys()))
                msg = " ".join(f"r{r}={per_r_loss[r]:.3f}" for r in key)
                print(f"  step {step:>5}  loss {loss.item():.4f}  val: {msg}")

    return {
        "config": {"n_loops": n_loops, "d": d, "n_loops_train": n_loops_train,
                   "n_loops_eval": n_loops_eval, "steps": steps,
                   "batch_size": batch_size, "lr": lr,
                   "p_noise": p_noise, "noise_alpha": noise_alpha,
                   "seq_len": seq_len, "stride": stride, "vocab": vocab},
        "params": model.num_params(),
        "wall_time_sec": time.time() - t0,
        "log": log,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
    }


@torch.no_grad()
def _eval_text(model, val_corpus, eval_size: int, seq_len: int,
                n_loops_eval: int, stride: int, vocab: int,
                batch_size: int, device: str) -> dict[int, float]:
    model.eval()
    losses = {r: 0.0 for r in range(1, n_loops_eval + 1)}
    n_seen = 0
    while n_seen < eval_size:
        bs = min(batch_size, eval_size - n_seen)
        batch_tokens, iter_targets = make_batch_text_iter(
            val_corpus, bs, seq_len, n_loops_eval, stride, device)
        all_logits = model.forward_all_loops(batch_tokens, n_loops=n_loops_eval)
        for r in range(1, n_loops_eval + 1):
            logits_r = all_logits[r]                  # [bs, seq_len, V]
            target_r = iter_targets[r]                # [bs, seq_len]
            loss_r = F.cross_entropy(logits_r.reshape(-1, vocab),
                                       target_r.reshape(-1)).item()
            losses[r] += loss_r * bs
        n_seen += bs
    out = {r: losses[r] / n_seen for r in losses}
    model.train()
    return out
