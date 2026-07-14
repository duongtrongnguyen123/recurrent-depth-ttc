"""Multi-Token Prediction (MTP) + iter-target hybrid.

At loop r: predict the token at position t + r (look r tokens ahead).
This makes deeper loops solve PROGRESSIVELY HARDER predictions:
  r=1: t+1 (next token)
  r=2: t+2 (2 tokens ahead)
  ...
  r=8: t+8 (8 tokens ahead)

Forces depth utilization — the model can't just mimic r=1 at deeper r
because the targets are different. Each loop must produce a genuinely
different prediction.

At inference: use r=1 for next-token (standard LM). The deeper r values
are training scaffolding — they force the recurrent block to learn
extended-horizon prediction, which should improve r=1 quality too.

Hypothesis: MTP gives base a per-loop curve where each r is the BEST
predictor for its specific horizon. r=1 is best for t+1 (the only one
that matters at inference). Bimodal pattern should disappear because
loops aren't competing on the same target.
"""
from __future__ import annotations
import time, torch, torch.nn.functional as F
from .data_text import load_tiny_shakespeare
from .train_chain_iter_robust import RobustLoopedTransformer
from .train_text_next import _make_next_batch


def _make_mtp_batch(corpus, batch_size, seq_len, n_horizons, device):
    """Returns (tokens, targets[h, B, T]) where targets[h, b, t] = corpus[start+t+h+1]."""
    starts = torch.randint(0, len(corpus) - seq_len - n_horizons - 2,
                              (batch_size,), device=device)
    idx = starts[:, None] + torch.arange(seq_len, device=device)[None]
    tokens = corpus[idx]
    targets = torch.stack([
        corpus[idx + h + 1] for h in range(n_horizons)
    ], dim=0)
    return tokens, targets


def train_text_mtp(*, n_loops, d, n_heads, ff_mult, steps, batch_size, lr,
                     eval_every, eval_size, device, seed, n_loops_train,
                     n_loops_eval, seq_len, p_noise=0.5, noise_alpha=0.1,
                     verbose=True):
    torch.manual_seed(seed)
    print("Loading tiny-shakespeare...", flush=True)
    tokens, _, ids = load_tiny_shakespeare()
    vocab = len(ids)
    val_start = int(0.9 * len(tokens))
    train_corpus = tokens[:val_start].to(device)
    val_corpus = tokens[val_start:].to(device)
    model = RobustLoopedTransformer(vocab=vocab, max_len=seq_len, d=d,
        n_heads=n_heads, ff_mult=ff_mult, n_loops=n_loops).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                              weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    log = {"step": [], "train_loss": [], "eval_per_horizon_loss": []}
    t0 = time.time()
    for step in range(steps):
        b_tokens, b_targets = _make_mtp_batch(train_corpus, batch_size, seq_len,
                                                  n_loops_train, device)
        # b_targets: [n_loops_train, B, T] — target at horizon h
        all_logits = model.forward_all_loops_robust(b_tokens, n_loops=n_loops_train,
                                                      p_noise=p_noise, noise_alpha=noise_alpha)
        # all_logits: [n_loops_train+1, B, T, V]
        # Loop r predicts horizon r-1 (0-indexed: loop 1 → t+1, loop 8 → t+8)
        losses = []
        for r in range(1, n_loops_train + 1):
            losses.append(F.cross_entropy(all_logits[r].reshape(-1, vocab),
                                              b_targets[r-1].reshape(-1)))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                # Eval each horizon
                per_h = {h: 0.0 for h in range(1, n_loops_eval + 1)}
                n_b = max(1, eval_size // batch_size)
                for _ in range(n_b):
                    tv, tgts = _make_mtp_batch(val_corpus, batch_size, seq_len,
                                                  n_loops_eval, device)
                    al = model.forward_all_loops(tv, n_loops=n_loops_eval)
                    for r in range(1, n_loops_eval + 1):
                        per_h[r] += F.cross_entropy(al[r].reshape(-1, vocab),
                                                          tgts[r-1].reshape(-1)).item()
                per_h = {h: per_h[h] / n_b for h in per_h}
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_per_horizon_loss"].append(per_h)
            if verbose:
                key = sorted({1, 2, 4, 8, 16} & set(per_h.keys()))
                msg = " ".join(f"h{h}={per_h[h]:.3f}" for h in key)
                print(f"  step {step:>5}  loss {loss.item():.4f}  val: {msg}", flush=True)
    return {"config": {"n_loops": n_loops, "d": d, "n_loops_train": n_loops_train,
                         "n_loops_eval": n_loops_eval, "steps": steps, "lr": lr},
             "params": sum(p.numel() for p in model.parameters()),
             "wall_time_sec": time.time() - t0,
             "log": log,
             "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
             "vocab": vocab}
