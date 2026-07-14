from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import make_batch


@torch.no_grad()
def collect_per_loop_logits(model, n_digits: int, n_eval: int, max_loops: int,
                            batch_size: int = 512, device: str = "cuda"
                            ) -> dict[str, torch.Tensor]:
    """Run model with forward_all_loops over n_eval examples, capturing per-loop
    answer-position logits. Returns dict with:
      - tokens:      [N, T] full token sequences
      - targets:     [N, T] target token at each position (or PAD)
      - mask:        [N, T] which positions are answer positions
      - per_loop_logits: [max_loops+1, N, T_ans, V]  (only the answer positions)
      - answer_idxs: list of (start, end) inclusive indices in T for the answer
    """
    model.eval()
    answer_start = 2 * n_digits + 2 - 1   # last input position before the answer
    answer_len = n_digits + 1
    all_tokens, all_targets, all_masks, all_logits = [], [], [], []
    done = 0
    while done < n_eval:
        bs = min(batch_size, n_eval - done)
        tokens, targets, mask = make_batch(bs, n_digits, device)
        logits_all = model.forward_all_loops(tokens, n_loops=max_loops)
        # logits_all: [max_loops+1, B, T, V]; keep only the answer positions.
        slc = slice(answer_start, answer_start + answer_len)
        ans_logits = logits_all[:, :, slc, :].cpu()
        all_tokens.append(tokens.cpu())
        all_targets.append(targets.cpu())
        all_masks.append(mask.cpu())
        all_logits.append(ans_logits)
        done += bs
    return {
        "tokens":   torch.cat(all_tokens, dim=0),
        "targets":  torch.cat(all_targets, dim=0),
        "mask":     torch.cat(all_masks, dim=0),
        "per_loop_logits": torch.cat(all_logits, dim=1),
        "answer_start": answer_start, "answer_len": answer_len,
    }


def fixed_r_accuracy(per_loop_logits: torch.Tensor, targets: torch.Tensor,
                     mask: torch.Tensor, answer_start: int, answer_len: int,
                     ) -> dict[int, dict[str, float]]:
    """For each loop count r in [0, n_loops], evaluate per-token and full-answer accuracy.
    per_loop_logits: [n_loops+1, N, T_ans, V]
    targets, mask:   [N, T]
    """
    target_slice = targets[:, answer_start:answer_start + answer_len]
    n_loops_plus_one = per_loop_logits.shape[0]
    out: dict[int, dict[str, float]] = {}
    for r in range(n_loops_plus_one):
        preds = per_loop_logits[r].argmax(-1)
        per_tok = (preds == target_slice).float().mean().item()
        full = (preds == target_slice).all(dim=1).float().mean().item()
        out[r] = {"acc_per_token": per_tok, "acc_full": full}
    return out


def adaptive_threshold_accuracy(per_loop_logits: torch.Tensor, targets: torch.Tensor,
                                answer_start: int, answer_len: int,
                                signal: str, thresholds: list[float],
                                min_loops: int = 1
                                ) -> list[dict]:
    """At each loop, compute a per-example halt signal. Halt the example at the first
    loop where signal_value crosses threshold (>= for margin, <= for entropy).

    Returns one dict per threshold:
      {threshold, avg_loops, acc_full, acc_per_token, frac_halted_by_max}
    """
    target_slice = targets[:, answer_start:answer_start + answer_len]   # [N, T_ans]
    R, N, T, V = per_loop_logits.shape   # R = n_loops+1
    # Compute per-loop signal averaged over T_ans positions per example.
    if signal == "entropy":
        p = F.softmax(per_loop_logits, dim=-1)
        ent = -(p * (p + 1e-12).log()).sum(dim=-1)  # [R, N, T]
        sig = ent.mean(dim=-1)                       # [R, N], smaller is more confident
        compare = lambda s, thr: s <= thr
    elif signal == "margin":
        srt = per_loop_logits.sort(dim=-1, descending=True).values
        margin = (srt[..., 0] - srt[..., 1]).mean(dim=-1)  # [R, N], larger is more confident
        sig = margin
        compare = lambda s, thr: s >= thr
    else:
        raise ValueError(signal)

    out = []
    for thr in thresholds:
        # First loop r >= min_loops where compare(sig[r, n], thr) is True. If none, use R-1.
        halt_at = torch.full((N,), R - 1, dtype=torch.long)
        crossed_any = torch.zeros(N, dtype=torch.bool)
        for r in range(min_loops, R):
            cond = compare(sig[r], thr) & ~crossed_any
            halt_at[cond] = r
            crossed_any |= cond
        # Gather predictions at halted loops.
        preds = torch.zeros((N, T), dtype=torch.long)
        for r in range(R):
            sel = (halt_at == r)
            if sel.any():
                preds[sel] = per_loop_logits[r, sel].argmax(-1)
        per_tok = (preds == target_slice).float().mean().item()
        full = (preds == target_slice).all(dim=1).float().mean().item()
        out.append({"threshold": float(thr), "signal": signal,
                    "avg_loops": halt_at.float().mean().item(),
                    "acc_full": full, "acc_per_token": per_tok,
                    "frac_used_max": float(crossed_any.logical_not().float().mean().item())})
    return out
