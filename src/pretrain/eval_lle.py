"""Likelihood-Locked Eval (LLE) for reasoning tasks.

Replaces greedy-generation + regex-extract (eval_stage3a) with direct
log-probability scoring of the candidate answer set. Fixes:

  - Decoding-strategy noise (different seeds picking different argmax paths
    even when likelihoods are nearly identical)
  - Regex-normalize fragility (model emits LaTeX/punctuation → wrong extraction)
  - EOS/newline edge cases

Definitions per example with prompt p, gold a:

  candidates = sorted set of all gold answers seen in this task's eval set
  for each c in candidates:
    nll(c | p) = -sum_i log P(tok_i | p, tok_<i)   # over tokens of c
  top1_pred = argmin_c nll(c | p)
  margin    = nll(best_wrong | p) - nll(gold | p)  # >0 if correct preferred

Metrics per task (chain/listops/modular):

  top1_acc      = mean[top1_pred == gold]                  (continuous-ranked)
  mean_nll_gold = mean[nll(gold | p)]                      (lower is better)
  mean_margin   = mean[margin]                             (calibration)
  ci95_top1     = bootstrap 95% confidence interval on top1_acc (B=1000)

Output JSON has the same per-task structure as eval_stage3a but with these
new fields. The greedy-gen `summary.acc` field is preserved as `top1_acc`
to keep the comparison plumbing identical.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


def _load_tokenizer(data_dir: Path):
    from transformers import AutoTokenizer
    tk_dir = data_dir / "tokenizer"
    if not (tk_dir / "tokenizer.json").exists():
        raise FileNotFoundError(f"tokenizer.json not in {tk_dir}")
    return AutoTokenizer.from_pretrained(str(tk_dir))


def _load_model(ckpt_dir: Path, device: str):
    state_path = ckpt_dir / "state.pt"
    cfg_path = ckpt_dir.parent.parent / "config.json"
    if not state_path.exists() or state_path.stat().st_size == 0:
        raise FileNotFoundError(f"state.pt missing or empty in {ckpt_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json missing at {cfg_path}")
    cfg_data = json.loads(cfg_path.read_text())

    from pretrain.model import PretrainConfig, build_model
    arch_cfg_dict = cfg_data["arch"]
    field_names = {f.name for f in PretrainConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in arch_cfg_dict.items() if k in field_names}
    arch_cfg = PretrainConfig(**filtered)
    model = build_model(arch_cfg).to(device)

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, arch_cfg, cfg_data


@torch.no_grad()
def _score_candidates(model, tokenizer, prompt: str, candidates: list[str],
                      n_loops: int, device: str, dtype: torch.dtype,
                      max_seq_len: int) -> tuple[list[float], list[int]]:
    """Return (nll, K) for each candidate, in order. Uses a single forward
    pass per candidate (concat prompt + candidate, score the candidate
    tokens).

    Returns:
        nlls: list of summed NLL across the candidate's tokens (lower = more likely)
        K_list: list of token counts per candidate (for length-normalization)
    """
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)  # [1, P]
    P = prompt_ids.size(1)

    nlls: list[float] = []
    K_list: list[int] = []
    for c in candidates:
        # Encode candidate WITHOUT bos, as a continuation.
        cand_ids = tokenizer.encode(c, return_tensors="pt",
                                     add_special_tokens=False).to(device)  # [1, K]
        K = cand_ids.size(1)
        K_list.append(K)
        if K == 0:
            nlls.append(float("inf"))
            continue
        ids = torch.cat([prompt_ids, cand_ids], dim=1)  # [1, P+K]
        if ids.size(1) > max_seq_len:
            # Truncate from the LEFT (preserve the recent prompt + the candidate)
            ids = ids[:, -max_seq_len:]
            # Recompute P relative to truncated input
            P_eff = ids.size(1) - K
        else:
            P_eff = P
        with torch.amp.autocast(device_type=device, dtype=dtype,
                                  enabled=(dtype != torch.float32)):
            out = model(ids, n_loops=n_loops)
        logits = out["logits"]  # [1, P+K, V]
        # Position i in logits predicts token i+1 (causal). So tokens P+0 ... P+K-1
        # are predicted by logits at positions P-1 ... P+K-2.
        # cand_ids[0, j] is target at position P + j; predicted by logits[0, P-1+j].
        log_probs = F.log_softmax(logits[0, P_eff - 1 : P_eff - 1 + K, :], dim=-1)  # [K, V]
        token_lp = log_probs[torch.arange(K, device=device), cand_ids[0]]  # [K]
        nll = -float(token_lp.sum().item())
        nlls.append(nll)
    return nlls, K_list


def _bootstrap_ci(values: list[float], B: int = 1000, alpha: float = 0.05,
                   seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of a 0/1 (or continuous) array."""
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))
    means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        means[b] = arr[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def run(ckpt_dir: str, data_dir: str, eval_file: str | None = None,
        inference_n_loops: int | None = None,
        out_path: str = "lle_eval.json",
        bootstrap_B: int = 1000) -> dict:
    ckpt_path = Path(ckpt_dir)
    data_path = Path(data_dir)
    eval_path = Path(eval_file) if eval_file else (data_path / "eval_synth.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cap = torch.cuda.get_device_capability(0)
        dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"device={device} dtype={dtype}", flush=True)

    print(f"loading model from {ckpt_path}", flush=True)
    model, arch_cfg, cfg_data = _load_model(ckpt_path, device)
    n_loops = inference_n_loops if inference_n_loops is not None else arch_cfg.n_loops
    print(f"  arch={arch_cfg.arch}  d={arch_cfg.d_model}  "
          f"trained n_loops={arch_cfg.n_loops}  inference n_loops={n_loops}",
          flush=True)

    print(f"loading tokenizer from {data_path / 'tokenizer'}", flush=True)
    tokenizer = _load_tokenizer(data_path)

    print(f"loading eval set from {eval_path}", flush=True)
    examples = json.loads(eval_path.read_text())
    print(f"  {len(examples)} examples", flush=True)
    by_kind: dict[str, list[dict]] = {}
    for ex in examples:
        by_kind.setdefault(ex["kind"], []).append(ex)
    print(f"  by kind: {{ {', '.join(f'{k}={len(v)}' for k, v in by_kind.items())} }}",
          flush=True)

    # Per-task candidate sets (union of golds in that task — covers the
    # answer space the model is actually being asked to discriminate).
    candidates_by_kind: dict[str, list[str]] = {}
    for kind, exs in by_kind.items():
        candidates_by_kind[kind] = sorted(set(ex["target"].strip() for ex in exs),
                                            key=lambda s: (len(s), s))
        print(f"  {kind}: {len(candidates_by_kind[kind])} candidates: "
              f"{candidates_by_kind[kind]}", flush=True)

    max_seq_len = arch_cfg.max_seq_len
    results: list[dict] = []
    t0 = time.time()
    for kind, exs in by_kind.items():
        cands = candidates_by_kind[kind]
        print(f"\n=== scoring {kind} ({len(exs)} examples × {len(cands)} candidates) ===",
              flush=True)
        for i, ex in enumerate(exs):
            prompt = ex["prompt"]
            gold = ex["target"].strip()
            nlls, K_list = _score_candidates(model, tokenizer, prompt, cands, n_loops,
                                       device, dtype, max_seq_len)
            # Per-token NLLs (length-normalized): nll(c) / K_c.
            nlls_per_tok = [nll / max(K, 1) for nll, K in zip(nlls, K_list)]
            # Pick top-1 BOTH ways (sum-NLL and per-token-NLL), compute margins.
            min_idx_sum = int(min(range(len(nlls)), key=lambda j: nlls[j]))
            min_idx_norm = int(min(range(len(nlls_per_tok)),
                                    key=lambda j: nlls_per_tok[j]))
            pred_sum = cands[min_idx_sum]
            pred_norm = cands[min_idx_norm]
            try:
                gold_idx = cands.index(gold)
                nll_gold = nlls[gold_idx]
                K_gold = K_list[gold_idx]
                nll_gold_per_tok = nll_gold / max(K_gold, 1)
            except ValueError:
                nll_gold = float("inf")
                K_gold = 0
                nll_gold_per_tok = float("inf")
            # Best wrong = min over candidates excluding gold (both metrics)
            nll_wrong = min((nll for j, nll in enumerate(nlls)
                             if cands[j] != gold), default=float("inf"))
            nll_wrong_per_tok = min((nlls_per_tok[j] for j in range(len(cands))
                                       if cands[j] != gold), default=float("inf"))
            margin_sum = nll_wrong - nll_gold
            margin_norm = nll_wrong_per_tok - nll_gold_per_tok
            results.append({
                "kind": kind,
                "gold": gold,
                "K_gold": K_gold,
                # Sum-NLL metrics (length-biased; kept for backwards compat)
                "pred_sum": pred_sum,
                "correct_sum": (pred_sum == gold),
                "nll_gold": nll_gold,
                "nll_wrong": nll_wrong,
                "margin": margin_sum,
                # Length-normalized metrics (PRIMARY going forward)
                "pred": pred_norm,
                "correct": (pred_norm == gold),
                "nll_gold_per_tok": nll_gold_per_tok,
                "nll_wrong_per_tok": nll_wrong_per_tok,
                "margin_per_tok": margin_norm,
                # Full per-candidate NLLs (for any future re-analysis)
                "all_nlls": nlls,
                "all_K": K_list,
            })
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1:>3}/{len(exs)}] {elapsed:.0f}s", flush=True)

    # Summary by kind. PRIMARY metric is length-normalized top1 (correct/pred).
    # Backwards-compat sum-NLL top1 reported as top1_sum_acc.
    print("\n=== SUMMARY (primary = length-normalized) ===", flush=True)
    summary: dict = {}
    for kind in by_kind:
        rs = [r for r in results if r["kind"] == kind]
        correct = [int(r["correct"]) for r in rs]            # length-normalized
        correct_sum = [int(r["correct_sum"]) for r in rs]    # sum-NLL legacy
        nll_g_pt = [r["nll_gold_per_tok"] for r in rs
                      if r["nll_gold_per_tok"] != float("inf")]
        nll_g_sum = [r["nll_gold"] for r in rs if r["nll_gold"] != float("inf")]
        margin_pt = [r["margin_per_tok"] for r in rs
                       if r["margin_per_tok"] != float("inf")
                          and r["nll_gold_per_tok"] != float("inf")]
        margin_sum = [r["margin"] for r in rs if r["margin"] != float("inf")
                                                    and r["nll_gold"] != float("inf")]
        ci_lo, ci_hi = _bootstrap_ci(correct, B=bootstrap_B)
        ci_lo_s, ci_hi_s = _bootstrap_ci(correct_sum, B=bootstrap_B)
        summary[kind] = {
            "n": len(rs),
            # PRIMARY: length-normalized top1
            "top1_acc": sum(correct) / len(rs) if rs else 0.0,
            "top1_ci95": [ci_lo, ci_hi],
            "mean_nll_gold_per_tok": (sum(nll_g_pt) / len(nll_g_pt))
                                       if nll_g_pt else float("nan"),
            "mean_margin_per_tok": (sum(margin_pt) / len(margin_pt))
                                       if margin_pt else float("nan"),
            # Legacy: sum-NLL top1 (length-biased)
            "top1_sum_acc": sum(correct_sum) / len(rs) if rs else 0.0,
            "top1_sum_ci95": [ci_lo_s, ci_hi_s],
            "mean_nll_gold": (sum(nll_g_sum) / len(nll_g_sum))
                                       if nll_g_sum else float("nan"),
            "mean_margin": (sum(margin_sum) / len(margin_sum))
                                       if margin_sum else float("nan"),
        }
        s = summary[kind]
        print(f"  {kind:<10}  n={s['n']:>3}  "
              f"top1[norm]={s['top1_acc']:.3f} CI95=[{ci_lo:.3f},{ci_hi:.3f}]  "
              f"top1[sum]={s['top1_sum_acc']:.3f}  "
              f"nll/tok={s['mean_nll_gold_per_tok']:.3f}  "
              f"margin/tok={s['mean_margin_per_tok']:+.3f}",
              flush=True)
    overall_correct = [int(r["correct"]) for r in results]
    overall_correct_sum = [int(r["correct_sum"]) for r in results]
    ci_lo, ci_hi = _bootstrap_ci(overall_correct, B=bootstrap_B)
    ci_lo_s, ci_hi_s = _bootstrap_ci(overall_correct_sum, B=bootstrap_B)
    summary["overall"] = {
        "n": len(results),
        "top1_acc": sum(overall_correct) / len(results) if results else 0.0,
        "top1_ci95": [ci_lo, ci_hi],
        "top1_sum_acc": sum(overall_correct_sum) / len(results) if results else 0.0,
        "top1_sum_ci95": [ci_lo_s, ci_hi_s],
    }
    print(f"  {'overall':<10}  n={summary['overall']['n']:>3}  "
          f"top1[norm]={summary['overall']['top1_acc']:.3f} "
          f"CI95=[{ci_lo:.3f},{ci_hi:.3f}]  "
          f"top1[sum]={summary['overall']['top1_sum_acc']:.3f}", flush=True)

    payload = {
        "ckpt_dir": str(ckpt_path),
        "arch": arch_cfg.arch,
        "d_model": arch_cfg.d_model,
        "n_loops": n_loops,
        "trained_n_loops": arch_cfg.n_loops,
        "n_train_tokens": cfg_data.get("step", 0),
        "wall_sec": time.time() - t0,
        "candidates_by_kind": candidates_by_kind,
        "summary": summary,
        "results": results,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"\nsaved -> {out_path}", flush=True)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="lle_eval.json")
    ap.add_argument("--inference-n-loops", type=int, default=None)
    args = ap.parse_args()
    run(args.ckpt_dir, args.data_dir, out_path=args.out,
        inference_n_loops=args.inference_n_loops)
