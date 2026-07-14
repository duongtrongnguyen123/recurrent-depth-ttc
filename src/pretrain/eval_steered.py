"""Probe A: prompt-prefix steering eval.

Tests the hypothesis that greedy chain-accuracy is hurt by a data-recipe
policy conflict — the model 'knows' the digit answer (LLE confirms this) but
greedy follows the more-frequent NuminaMath/LaTeX policy because chain
prompts (f(0)=... f^k(n)=) are ambiguous between the two policies.

We re-run GREEDY generation with a small priming prefix prepended to each
chain prompt, e.g. "Answer with a single digit:\n" or "0 1 2 3 4 5 6 7 8 9 10 11\n".
If chain greedy jumps from ~22% to ~40-50% (matching LLE), the hypothesis is
confirmed and the path forward is to (a) train with synth-dominant mix
(cfg_pcc_phase3b_synth) or (b) deploy this prefix at inference for any
model whose greedy is below its LLE.

We report side-by-side: baseline greedy (no prefix) vs steered greedy
(with prefix), on the SAME eval set, so the only variable is the prefix.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_tokenizer(data_dir: Path):
    from transformers import AutoTokenizer
    tk = data_dir / "tokenizer"
    return AutoTokenizer.from_pretrained(str(tk))


def _load_model(ckpt_dir: Path, device: str):
    state_path = ckpt_dir / "state.pt"
    cfg_path = ckpt_dir.parent.parent / "config.json"
    cfg_data = json.loads(cfg_path.read_text())
    from pretrain.model import PretrainConfig, build_model
    arch_cfg_dict = cfg_data["arch"]
    field_names = {f.name for f in PretrainConfig.__dataclass_fields__.values()}
    arch_cfg = PretrainConfig(**{k: v for k, v in arch_cfg_dict.items()
                                  if k in field_names})
    model = build_model(arch_cfg).to(device)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, arch_cfg, cfg_data


@torch.no_grad()
def _greedy(model, tokenizer, prompt: str, n_loops: int, device, dtype,
             max_new_tokens: int = 16) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    generated: list[int] = []
    for _ in range(max_new_tokens):
        with torch.amp.autocast(device_type=device, dtype=dtype,
                                  enabled=(dtype != torch.float32)):
            out = model(ids, n_loops=n_loops)
        nxt = int(out["logits"][0, -1, :].argmax().item())
        generated.append(nxt)
        if nxt == eos_id:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        if ids.size(1) >= model.cfg.max_seq_len:
            break
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _normalize(s: str) -> str:
    """Extract the first integer-looking token from generation (tolerant of
    LaTeX wraps, leading whitespace, trailing punctuation)."""
    import re
    s = s.strip()
    if not s:
        return ""
    m = re.search(r"\\boxed\{([^{}]+)\}", s)
    if m:
        return m.group(1).strip().rstrip(".,;:!?)")
    m = re.search(r"\$([^$]+)\$", s)
    if m:
        return m.group(1).strip().rstrip(".,;:!?)")
    if s.startswith("\\[") or s.startswith("\\("):
        ints = re.findall(r"-?\d+", s)
        return ints[-1] if ints else ""
    head = s.split()[0].rstrip(".,;:!?)")
    if not re.match(r"^-?\d+$", head):
        ints = re.findall(r"-?\d+", s[:64])
        if ints:
            return ints[-1]
    return head


# Priming prefix variants. Each one is prepended to chain prompts ONLY.
# (listops / modular keep baseline prompts so we measure incidental damage.)
PREFIXES = {
    "none": "",
    "digit_list": "0 1 2 3 4 5 6 7 8 9 10 11\n",
    "instruct": "Answer with a single digit only:\n",
    "both": "Answer with a single digit only.\n0 1 2 3 4 5 6 7 8 9 10 11\n",
}


def run(ckpt_dir: str, data_dir: str, out_path: str = "steered_eval.json",
        inference_n_loops: int | None = None,
        only_kind: str = "chain") -> dict:
    ckpt_path = Path(ckpt_dir)
    data_path = Path(data_dir)
    eval_path = data_path / "eval_synth.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda"
                                and torch.cuda.get_device_capability(0)[0] >= 8) \
              else (torch.float16 if device == "cuda" else torch.float32)
    print(f"device={device} dtype={dtype}", flush=True)

    print(f"loading model from {ckpt_path}", flush=True)
    model, arch_cfg, cfg_data = _load_model(ckpt_path, device)
    n_loops = inference_n_loops if inference_n_loops is not None else arch_cfg.n_loops
    print(f"  arch={arch_cfg.arch}  d={arch_cfg.d_model}  trained_n_loops={arch_cfg.n_loops}  "
          f"inference_n_loops={n_loops}", flush=True)
    tokenizer = _load_tokenizer(data_path)

    examples = json.loads(eval_path.read_text())
    examples = [e for e in examples if e["kind"] == only_kind]
    print(f"  {len(examples)} {only_kind} examples", flush=True)

    summary: dict[str, dict] = {}
    all_results: dict[str, list[dict]] = {}

    for prefix_name, prefix in PREFIXES.items():
        print(f"\n=== prefix={prefix_name!r} ({prefix!r}) ===", flush=True)
        t0 = time.time()
        correct = 0
        records: list[dict] = []
        for i, ex in enumerate(examples):
            steered_prompt = prefix + ex["prompt"] if prefix else ex["prompt"]
            gen = _greedy(model, tokenizer, steered_prompt, n_loops, device, dtype)
            pred = _normalize(gen)
            ok = (pred == ex["target"].strip())
            correct += int(ok)
            records.append({"gold": ex["target"], "gen": gen, "pred": pred, "ok": ok})
            if (i + 1) % 50 == 0:
                print(f"  [{i+1:>3}/{len(examples)}] {time.time()-t0:.0f}s "
                      f"acc={correct/(i+1):.3f}", flush=True)
        acc = correct / len(examples) if examples else 0.0
        print(f"  prefix={prefix_name} acc={acc:.3f} ({correct}/{len(examples)})",
              flush=True)
        summary[prefix_name] = {
            "n": len(examples),
            "correct": correct,
            "acc": acc,
            "wall_sec": time.time() - t0,
        }
        all_results[prefix_name] = records

    print("\n=== STEERING SUMMARY (chain) ===", flush=True)
    for k, v in summary.items():
        print(f"  prefix={k:<12}  acc={v['acc']:.3f}", flush=True)

    payload = {
        "ckpt_dir": str(ckpt_path),
        "arch": arch_cfg.arch,
        "d_model": arch_cfg.d_model,
        "trained_n_loops": arch_cfg.n_loops,
        "n_loops": n_loops,
        "kind": only_kind,
        "prefixes": PREFIXES,
        "summary": summary,
        "results": all_results,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"saved -> {out_path}", flush=True)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="steered_eval.json")
    ap.add_argument("--only-kind", default="chain")
    args = ap.parse_args()
    run(args.ckpt_dir, args.data_dir, out_path=args.out, only_kind=args.only_kind)
