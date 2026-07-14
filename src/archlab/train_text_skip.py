"""Train skip-loop on text. Skip residual h_0 → every loop r adds alpha*h_0."""
from __future__ import annotations
import time
import torch, torch.nn.functional as F
from .data_text import load_tiny_shakespeare
from .model_skiploop_robust import SkipLoopRobustTransformer
from .train_text_next import _make_next_batch


def train_text_skip(*, n_loops, d, n_heads, ff_mult, steps, batch_size, lr,
                      eval_every, eval_size, device, seed, n_loops_train,
                      n_loops_eval, seq_len, p_noise=0.5, noise_alpha=0.1,
                      skip_alpha=0.1, verbose=True):
    torch.manual_seed(seed)
    print("Loading tiny-shakespeare...", flush=True)
    tokens, _, ids = load_tiny_shakespeare()
    vocab = len(ids)
    val_start = int(0.9 * len(tokens))
    train_corpus = tokens[:val_start].to(device)
    val_corpus = tokens[val_start:].to(device)
    model = SkipLoopRobustTransformer(vocab=vocab, max_len=seq_len, d=d,
        n_heads=n_heads, ff_mult=ff_mult, n_loops=n_loops,
        skip_alpha=skip_alpha).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                              weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    log = {"step": [], "train_loss": [], "eval_per_r_loss": []}
    t0 = time.time()
    for step in range(steps):
        b_tokens, b_targets = _make_next_batch(train_corpus, batch_size, seq_len, device)
        all_logits = model.forward_all_loops_robust(b_tokens, n_loops=n_loops_train,
                                                      p_noise=p_noise, noise_alpha=noise_alpha)
        losses = []
        for r in range(1, n_loops_train + 1):
            losses.append(F.cross_entropy(all_logits[r].reshape(-1, vocab),
                                              b_targets.reshape(-1)))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                losses_per_r = {r: 0.0 for r in range(1, n_loops_eval + 1)}
                n_b = max(1, eval_size // batch_size)
                for _ in range(n_b):
                    tv, ty = _make_next_batch(val_corpus, batch_size, seq_len, device)
                    al = model.forward_all_loops(tv, n_loops=n_loops_eval)
                    for r in range(1, n_loops_eval + 1):
                        losses_per_r[r] += F.cross_entropy(al[r].reshape(-1, vocab),
                                                             ty.reshape(-1)).item()
                per_r = {r: losses_per_r[r] / n_b for r in losses_per_r}
            log["step"].append(step)
            log["train_loss"].append(loss.item())
            log["eval_per_r_loss"].append(per_r)
            if verbose:
                key = sorted({1, 4, 8, 16} & set(per_r.keys()))
                msg = " ".join(f"r{r}={per_r[r]:.3f}" for r in key)
                print(f"  step {step:>5}  loss {loss.item():.4f}  val: {msg}", flush=True)
    return {"config": {"n_loops": n_loops, "d": d, "n_loops_train": n_loops_train,
                         "n_loops_eval": n_loops_eval, "steps": steps, "lr": lr,
                         "skip_alpha": skip_alpha},
             "params": sum(p.numel() for p in model.parameters()),
             "wall_time_sec": time.time() - t0,
             "log": log,
             "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
             "vocab": vocab}
