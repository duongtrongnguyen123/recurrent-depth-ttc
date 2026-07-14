"""Per-block gradient flow analysis on real-text reasoning tasks.

For each chain/listops/modular prompt:
  1. Forward through the model.
  2. Take the logits at the final position; compute CE loss vs the gold answer's
     first token.
  3. Backward; capture the L2 norm of the gradient at every named parameter.
  4. Aggregate per-block (prelude_blocks.k / core_blocks.k / coda_blocks.k) and
     per-component (attn.qkv / attn.proj / mlp.w1 / mlp.w2 / mlp.w3 / ln1 / ln2).

Output JSON gives a 3-D view: task_kind x block x component -> grad_norm.

The point: see WHICH layers/components are most active when the model is asked
to commit to a reasoning answer. For looped (PCC/xloop) architectures, gradients
through the shared core block sum across all loop applications — the per-loop
grad contribution is implicit in the reported core_blocks.0 norm.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _component_of(name: str) -> str:
    """Map a parameter name like 'core_blocks.0.attn.qkv.weight' -> 'attn.qkv'."""
    parts = name.split(".")
    # Skip block container + index, take next 1-2 components.
    # e.g. core_blocks.0.attn.qkv.weight -> ['attn', 'qkv']
    if len(parts) >= 3:
        suffix = parts[2:]
        # drop trailing .weight / .bias
        if suffix and suffix[-1] in ("weight", "bias"):
            suffix = suffix[:-1]
        return ".".join(suffix) if suffix else parts[-1]
    return name


def _block_of(name: str) -> str | None:
    """Map 'prelude_blocks.0.<rest>' -> 'prelude_blocks.0'. None if not a block param."""
    parts = name.split(".")
    if len(parts) >= 2 and parts[0].endswith("_blocks") and parts[1].isdigit():
        return f"{parts[0]}.{parts[1]}"
    return None


@torch.enable_grad()
def per_prompt_grad_norms(model, x: torch.Tensor, gold_token_id: int,
                            device: str, dtype) -> tuple[dict, float]:
    """Returns {param_name: l2_grad_norm} for one prompt + target token."""
    model.zero_grad(set_to_none=True)
    device_type = device if isinstance(device, str) else device.type
    with torch.amp.autocast(device_type=device_type, dtype=dtype,
                              enabled=(dtype != torch.float32)):
        out = model(x)
        if isinstance(out, dict):
            logits = out["logits"]
        elif isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        # Take final-position logits. Cast to float32 for clean CE.
        last = logits[:, -1, :].float()
        target = torch.tensor([gold_token_id], device=last.device)
        loss = F.cross_entropy(last, target)
    loss.backward()
    norms = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            norms[n] = float(p.grad.detach().float().norm().item())
    model.zero_grad(set_to_none=True)
    return norms, float(loss.item())


def aggregate(per_prompt_norms: list[dict]) -> dict:
    """Aggregate per-prompt norms by (block, component). Returns mean per group."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    overall = []
    for norms in per_prompt_norms:
        prompt_total = 0.0
        for name, val in norms.items():
            blk = _block_of(name)
            comp = _component_of(name)
            key = (blk if blk else "_top", comp)
            grouped[key].append(val)
            prompt_total += val
        overall.append(prompt_total)
    out = {}
    for (blk, comp), vals in grouped.items():
        key = f"{blk}::{comp}"
        out[key] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "count": len(vals),
        }
    out["_overall"] = {
        "mean": float(np.mean(overall)),
        "median": float(np.median(overall)),
        "n_prompts": len(per_prompt_norms),
    }
    return out


def run(ckpt_dir: str, data_dir: str, out_path: str,
        eval_file: str | None = None,
        max_prompts_per_kind: int = 25,
        max_seq_len: int = 1024) -> dict:
    """Main entry point.
    ckpt_dir: path containing state.pt.
    data_dir: path containing eval_synth.json + tokenizer/.
    """
    from .model import PretrainConfig, build_model

    ckpt_path = Path(ckpt_dir)
    cfg_path = ckpt_path.parent.parent / "config.json"
    cfg_data = json.loads(cfg_path.read_text())
    field_names = {f.name for f in PretrainConfig.__dataclass_fields__.values()}
    arch_cfg = PretrainConfig(**{k: v for k, v in cfg_data["arch"].items() if k in field_names})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32
    model = build_model(arch_cfg).to(device)
    state = torch.load(ckpt_path / "state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.train()  # need grad enabled

    # Tokenizer
    from transformers import AutoTokenizer
    tok_dir = Path(data_dir) / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))

    # Load eval prompts
    ef = Path(eval_file) if eval_file else (Path(data_dir) / "eval_synth.json")
    examples = json.loads(Path(ef).read_text())

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_kind[ex["kind"]].append(ex)

    results: dict = {
        "arch": arch_cfg.arch,
        "d_model": arch_cfg.d_model,
        "n_blocks": arch_cfg.n_blocks,
        "n_loops": arch_cfg.n_loops,
        "n_prelude": arch_cfg.n_prelude,
        "n_coda": arch_cfg.n_coda,
        "by_kind": {},
        "wall_sec": 0.0,
    }

    t0 = time.time()
    for kind, exs in by_kind.items():
        per_prompt = []
        losses = []
        for ex in exs[:max_prompts_per_kind]:
            prompt = ex["prompt"]
            gold = ex.get("target") or ex.get("gold") or ""
            if not gold.strip():
                continue
            # First token of gold answer (often an integer like '3').
            gold_ids = tokenizer.encode(gold, add_special_tokens=False)
            if not gold_ids:
                continue
            gold_first = gold_ids[0]
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            if ids.size(1) > max_seq_len:
                ids = ids[:, -max_seq_len:]
            norms, loss = per_prompt_grad_norms(model, ids, gold_first, device, dtype)
            per_prompt.append(norms)
            losses.append(loss)
        agg = aggregate(per_prompt)
        agg["mean_loss_to_gold"] = float(np.mean(losses)) if losses else None
        results["by_kind"][kind] = agg
        print(f"[{kind}] n={len(per_prompt)} mean_loss={agg['mean_loss_to_gold']:.4f}", flush=True)

    results["wall_sec"] = time.time() - t0
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="grad_flow.json")
    ap.add_argument("--max-prompts-per-kind", type=int, default=25)
    args = ap.parse_args()
    run(args.ckpt_dir, args.data_dir, args.out,
        max_prompts_per_kind=args.max_prompts_per_kind)
