"""Stage 3a evaluation: load a trained ckpt and compute per-task accuracy on
eval_synth.json (chain / listops / modular).

Usage in Kaggle kernel:
    import sys, os
    sys.path.insert(0, '/kaggle/working/pretrain/src')
    from pretrain.scripts.eval_stage3a import run
    run(ckpt_dir='/kaggle/input/<train-kernel>/<arch>/ckpt/latest',
        data_dir='/kaggle/input/<reasoning-data>')

This script can also be invoked as a Python module locally.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_tokenizer(data_dir: Path):
    """Load the HF tokenizer saved alongside the data."""
    from transformers import AutoTokenizer
    tk_dir = data_dir / "tokenizer"
    if not (tk_dir / "tokenizer.json").exists():
        raise FileNotFoundError(f"tokenizer.json not in {tk_dir}")
    return AutoTokenizer.from_pretrained(str(tk_dir))


def _load_model(ckpt_dir: Path, device: str):
    """Load model from saved ckpt by reading config.json beside it."""
    state_path = ckpt_dir / "state.pt"
    cfg_path = ckpt_dir.parent.parent / "config.json"
    if not state_path.exists():
        raise FileNotFoundError(f"state.pt missing in {ckpt_dir}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json missing at {cfg_path}")
    cfg_data = json.loads(cfg_path.read_text())

    # Reconstruct arch_cfg
    from pretrain.model import PretrainConfig, build_model
    arch_cfg_dict = cfg_data["arch"]
    # PretrainConfig may have new fields not in older configs; use defaults.
    field_names = {f.name for f in PretrainConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in arch_cfg_dict.items() if k in field_names}
    arch_cfg = PretrainConfig(**filtered)
    model = build_model(arch_cfg).to(device)

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, arch_cfg, cfg_data


@torch.no_grad()
def _per_loop_val_loss(model, val_bin_path, max_n_loops: int,
                       device: str, dtype: torch.dtype,
                       block_size: int = 2048, n_batches: int = 16,
                       batch_size: int = 4) -> dict[int, float]:
    """Compute per-loop val_loss at horizon-1 (next-token) for r in 1..max_n_loops.

    For iter-target trained models, loop r predicts token at +r offset. To
    measure pure next-token prediction at loop r, we just take the logits at
    that loop and score against y = x[1:].
    """
    import numpy as np
    arr = np.memmap(str(val_bin_path), dtype=np.uint32, mode="r")
    rng = np.random.default_rng(0)
    losses_per_r: dict[int, list[float]] = {r: [] for r in range(1, max_n_loops + 1)}
    for _ in range(n_batches):
        x_buf = np.empty((batch_size, block_size), dtype=np.int64)
        for i in range(batch_size):
            start = int(rng.integers(0, len(arr) - block_size - 1))
            x_buf[i] = arr[start: start + block_size].astype(np.int64)
        x = torch.from_numpy(x_buf).to(device)
        y = torch.cat([x[:, 1:], torch.zeros(batch_size, 1, dtype=torch.long,
                                                device=device)], dim=1)
        with torch.amp.autocast(device_type=device, dtype=dtype,
                                  enabled=(dtype != torch.float32)):
            out = model.forward_with_aux(x, n_loops=max_n_loops, aux_min_loops=1)
        for r in range(1, max_n_loops + 1):
            logits_r = out["per_loop_logits"][r - 1]
            loss = torch.nn.functional.cross_entropy(
                logits_r[:, :-1, :].reshape(-1, logits_r.size(-1)),
                x[:, 1:].reshape(-1))
            losses_per_r[r].append(loss.item())
    return {r: float(sum(v) / len(v)) for r, v in losses_per_r.items()}


@torch.no_grad()
def _greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int,
                     n_loops: int, device: str, dtype: torch.dtype):
    """Greedy autoregressive generation. Stops at EOS or max_new_tokens.

    Does NOT stop at newline — model often emits a leading newline before
    the actual answer. Caller normalizes the decoded string.
    """
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated: list[int] = []
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

    for _ in range(max_new_tokens):
        with torch.amp.autocast(device_type=device, dtype=dtype,
                                  enabled=(dtype != torch.float32)):
            out = model(ids, n_loops=n_loops)
            next_logits = out["logits"][0, -1, :]
        nxt = int(next_logits.argmax().item())
        generated.append(nxt)
        if nxt == eos_id:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        if ids.size(1) >= model.cfg.max_seq_len:
            break

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _normalize(gen: str) -> str:
    """Extract a plausible short answer from generation. Tolerant of LaTeX
    wraps (\\boxed{X}, $X$, \\[ ... \\]) and leading punctuation/newlines.

    For chain/listops/modular tasks the answer is an integer, so when the
    generation starts with LaTeX markup, prefer the LAST integer found in
    the LaTeX content (the answer typically appears near the end of an
    expression like '\\[\\n f(3) = 1 + 8\\]').
    """
    import re
    s = gen.strip()
    if not s:
        return ""
    m = re.search(r"\\boxed\{([^{}]+)\}", s)
    if m:
        return m.group(1).strip().rstrip(".,;:!?)")
    m = re.search(r"\$([^$]+)\$", s)
    if m:
        return m.group(1).strip().rstrip(".,;:!?)")
    # Raw LaTeX environment: extract last integer
    if s.startswith("\\[") or s.startswith("\\("):
        ints = re.findall(r"-?\d+", s)
        if ints:
            return ints[-1]
    # If gen starts with non-digit punctuation but contains an integer, take the last one
    head = s.split()[0].rstrip(".,;:!?)")
    if not re.match(r"^-?\d+$", head):
        ints = re.findall(r"-?\d+", s[:64])  # only first ~64 chars to avoid long unrelated junk
        if ints:
            return ints[-1]
    return head


def run(ckpt_dir: str, data_dir: str, eval_file: str | None = None,
        max_new_tokens: int = 16,
        inference_n_loops: int | None = None,
        out_path: str = "stage3a_eval.json") -> dict:
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
    # For iter-target trained models, inference uses loop 1 (next-token horizon).
    # For others, inference uses arch_cfg.n_loops (their trained loop count).
    n_loops = inference_n_loops if inference_n_loops is not None else arch_cfg.n_loops
    print(f"  arch={arch_cfg.arch}  d={arch_cfg.d_model}  "
          f"trained n_loops={arch_cfg.n_loops}  inference n_loops={n_loops}",
          flush=True)

    # Per-loop val_loss on val.bin if available (max horizon = trained n_loops).
    val_path = data_path / "val.bin"
    per_loop_val: dict[int, float] = {}
    if val_path.exists():
        print(f"computing per-loop val_loss on {val_path}...", flush=True)
        per_loop_val = _per_loop_val_loss(model, val_path, arch_cfg.n_loops,
                                            device, dtype)
        for r, v in per_loop_val.items():
            print(f"  r={r}: val_loss={v:.4f}", flush=True)

    print(f"loading tokenizer from {data_path / 'tokenizer'}", flush=True)
    tokenizer = _load_tokenizer(data_path)

    print(f"loading eval set from {eval_path}", flush=True)
    examples = json.loads(eval_path.read_text())
    print(f"  {len(examples)} examples", flush=True)
    from collections import Counter
    print(f"  by kind: {Counter(e['kind'] for e in examples)}", flush=True)

    results: list[dict] = []
    correct_by_kind: dict[str, list[bool]] = {}
    t0 = time.time()
    for i, ex in enumerate(examples):
        prompt = ex["prompt"]
        gold = ex["target"].strip()
        kind = ex["kind"]
        gen = _greedy_generate(model, tokenizer, prompt, max_new_tokens,
                                 n_loops, device, dtype)
        gen_norm = _normalize(gen)
        is_correct = (gen_norm == gold)
        correct_by_kind.setdefault(kind, []).append(is_correct)
        results.append({"idx": i, "kind": kind, "prompt": prompt[:80] + "...",
                         "gold": gold, "gen": gen, "gen_norm": gen_norm,
                         "correct": is_correct})
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1:>3}/{len(examples)}] {elapsed:.0f}s  "
                  f"{rate:.1f}/s", flush=True)

    summary = {kind: {"n": len(v), "correct": sum(v),
                       "acc": sum(v) / len(v)}
                for kind, v in correct_by_kind.items()}
    overall_correct = sum(r["correct"] for r in results)
    summary["overall"] = {"n": len(results), "correct": overall_correct,
                            "acc": overall_correct / len(results)}

    print("\n=== SUMMARY ===", flush=True)
    for k, v in summary.items():
        print(f"  {k:<10}  n={v['n']:>3}  acc={v['acc']:.3f}", flush=True)

    payload = {
        "ckpt_dir": str(ckpt_path),
        "arch": arch_cfg.arch,
        "d_model": arch_cfg.d_model,
        "n_loops": n_loops,
        "trained_n_loops": arch_cfg.n_loops,
        "n_train_tokens": cfg_data.get("step", 0),
        "wall_sec": time.time() - t0,
        "per_loop_val_loss": per_loop_val,
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
    ap.add_argument("--out", default="stage3a_eval.json")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()
    run(args.ckpt_dir, args.data_dir, max_new_tokens=args.max_new_tokens,
        out_path=args.out)
