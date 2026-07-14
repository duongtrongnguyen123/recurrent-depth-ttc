"""Within-block dimension analysis for reasoning probes.

For each block in the network, capture (via forward + backward hooks):
  1. h_in: input activation [B, T, d_model] -- per-channel grad/value
  2. mlp.gate (after silu(w1) * w3): [B, T, hidden] -- FFN hidden activations
  3. attn output per head: [B, n_heads, T, head_dim] -- per-head grad

For each chain/listops/modular prompt:
  - Forward (with hooks recording h_in / mlp.gate / attn per-head out)
  - Compute CE loss vs gold first token at the final position
  - Backward populates input.grad on the captured h_in (using retain_grad)
  - Read off:
      - per-channel grad norm of h_in: ||h_in.grad[..., c]|| for c in 0..d
      - participation: # channels above 1% of max grad
      - effective rank from grad-channel-variance
      - mlp activation density: % of hidden units > threshold
      - attn head importance: sum of grad norms per head

Reports per-block, per-task aggregates.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _block_module_name(model, blk_module) -> str:
    """Find the dotted name for a block module (prelude_blocks.k / core_blocks.k / coda_blocks.k)."""
    for name, m in model.named_modules():
        if m is blk_module:
            return name
    return "unknown"


def _per_channel_grad_norm(grad: torch.Tensor) -> torch.Tensor:
    """grad: [B, T, D] -> [D] L2 norm per channel."""
    return grad.float().pow(2).sum(dim=(0, 1)).sqrt()


def _channel_stats(per_channel_norm: torch.Tensor) -> dict:
    v = per_channel_norm.cpu().numpy()
    total = float(v.sum())
    if total == 0:
        return {"top1_share": 0.0, "top10_share": 0.0, "n_active_1pct": 0,
                "n_active_5pct": 0, "effective_rank_grad": 0.0,
                "max": 0.0, "median": 0.0}
    sorted_v = np.sort(v)[::-1]
    cum = np.cumsum(sorted_v)
    n = len(v)
    # Effective rank from gradient-channel "softmax" distribution: (sum)^2 / sum_of_squares
    eff_rank = (v.sum() ** 2) / (np.sum(v ** 2) + 1e-12)
    return {
        "top1_share": float(sorted_v[0] / total),
        "top10_share": float(sorted_v[: min(10, n)].sum() / total),
        "n_active_1pct": int((v > 0.01 * sorted_v[0]).sum()),
        "n_active_5pct": int((v > 0.05 * sorted_v[0]).sum()),
        "effective_rank_grad": float(eff_rank),
        "max": float(sorted_v[0]),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
    }


def per_prompt_dim_flow(model, x: torch.Tensor, gold_token_id: int,
                          device: str, dtype) -> tuple[dict, float]:
    """Returns {block_name: {h_in_stats, mlp_stats, head_stats}}."""
    from .model import Block
    model.zero_grad(set_to_none=True)
    blocks: list[tuple[str, "Block"]] = []  # name, module
    for name, mod in model.named_modules():
        if isinstance(mod, Block):
            blocks.append((name, mod))

    # Hooks: capture h_in and mlp_gate via forward; record gradients.
    h_in_store: dict[str, torch.Tensor] = {}
    mlp_gate_store: dict[str, torch.Tensor] = {}
    head_out_store: dict[str, torch.Tensor] = {}
    handles = []

    def make_block_hook(name):
        def hook(module, inputs, output):
            # inputs is a tuple; first element is h
            h = inputs[0]
            if h.requires_grad:
                h.retain_grad()
                h_in_store[name] = h
        return hook

    def make_mlp_hook(name):
        def hook(module, inputs, output):
            # SwiGLU: w2(silu(w1) * w3). output is [B, T, d_model]. Hidden is the
            # silu(w1)*w3 product, but we don't have direct access here. Approximate
            # FFN activity from output magnitude + the residual flag.
            mlp_gate_store[name] = output.detach()
        return hook

    def make_attn_hook(name):
        def hook(module, inputs, output):
            # Attention.forward returns the full mixed output [B,T,D] OR (mixed, kv).
            o = output[0] if isinstance(output, tuple) else output
            head_out_store[name] = o.detach()
        return hook

    for bname, blk in blocks:
        handles.append(blk.register_forward_hook(make_block_hook(bname)))
        # Hook for the mlp output (post-FFN, pre-residual)
        if hasattr(blk, "mlp"):
            handles.append(blk.mlp.register_forward_hook(make_mlp_hook(bname)))
        if hasattr(blk, "attn"):
            handles.append(blk.attn.register_forward_hook(make_attn_hook(bname)))

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
        last = logits[:, -1, :].float()
        target = torch.tensor([gold_token_id], device=last.device)
        loss = F.cross_entropy(last, target)
    loss.backward()

    result = {}
    for bname, blk in blocks:
        rec = {}
        if bname in h_in_store and h_in_store[bname].grad is not None:
            grad = h_in_store[bname].grad
            chn = _per_channel_grad_norm(grad)
            rec["h_in_grad"] = _channel_stats(chn)
            # Activation magnitude per channel
            act = h_in_store[bname].detach()
            act_chn = act.float().pow(2).sum(dim=(0, 1)).sqrt()
            rec["h_in_act"] = _channel_stats(act_chn)
        if bname in mlp_gate_store:
            mlp = mlp_gate_store[bname]
            mlp_chn = mlp.float().pow(2).sum(dim=(0, 1)).sqrt()
            rec["mlp_out_act"] = _channel_stats(mlp_chn)
        if bname in head_out_store:
            ao = head_out_store[bname]
            B, T, D = ao.shape
            cfg = blk.attn
            n_heads = cfg.n_heads if hasattr(cfg, "n_heads") else 1
            head_dim = D // n_heads
            # reshape to [B, T, H, hd]
            ah = ao.view(B, T, n_heads, head_dim)
            head_norms = ah.float().pow(2).sum(dim=(0, 1, 3)).sqrt()  # [H]
            head_norms_np = head_norms.cpu().numpy()
            sorted_v = np.sort(head_norms_np)[::-1]
            total = float(sorted_v.sum())
            if total > 0:
                rec["attn_heads"] = {
                    "n_heads": int(n_heads),
                    "top1_share": float(sorted_v[0] / total),
                    "top4_share": float(sorted_v[: min(4, n_heads)].sum() / total),
                    "effective_n_heads": float((sorted_v.sum() ** 2) / (np.sum(sorted_v ** 2) + 1e-12)),
                    "values": [float(v) for v in head_norms_np],
                }
        result[bname] = rec
    for h in handles:
        h.remove()
    model.zero_grad(set_to_none=True)
    return result, float(loss.item())


def aggregate_dim(per_prompt: list[dict]) -> dict:
    """Average each numeric metric across prompts."""
    if not per_prompt:
        return {}
    keys = list(per_prompt[0].keys())
    out = {}
    for blk_name in keys:
        # Collect each metric across prompts
        agg = {}
        for sub in ("h_in_grad", "h_in_act", "mlp_out_act"):
            stats_list = [p[blk_name].get(sub) for p in per_prompt
                          if blk_name in p and sub in p[blk_name]]
            if not stats_list:
                continue
            keys_inner = stats_list[0].keys()
            agg[sub] = {k: float(np.mean([s[k] for s in stats_list]))
                        for k in keys_inner}
        # attn_heads
        ah_list = [p[blk_name].get("attn_heads") for p in per_prompt
                   if blk_name in p and "attn_heads" in p[blk_name]]
        if ah_list:
            ah = {
                "n_heads": ah_list[0]["n_heads"],
                "top1_share": float(np.mean([a["top1_share"] for a in ah_list])),
                "top4_share": float(np.mean([a["top4_share"] for a in ah_list])),
                "effective_n_heads": float(np.mean([a["effective_n_heads"] for a in ah_list])),
            }
            agg["attn_heads"] = ah
        out[blk_name] = agg
    return out


def run(ckpt_dir: str, data_dir: str, out_path: str,
        eval_file: str | None = None,
        max_prompts_per_kind: int = 15,
        max_seq_len: int = 1024) -> dict:
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
    model.train()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Path(data_dir) / "tokenizer"))
    ef = Path(eval_file) if eval_file else (Path(data_dir) / "eval_synth.json")
    examples = json.loads(Path(ef).read_text())

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_kind[ex["kind"]].append(ex)

    results = {
        "arch": arch_cfg.arch, "d_model": arch_cfg.d_model,
        "n_blocks": arch_cfg.n_blocks, "n_loops": arch_cfg.n_loops,
        "n_prelude": arch_cfg.n_prelude, "n_coda": arch_cfg.n_coda,
        "n_heads": arch_cfg.n_heads,
        "by_kind": {}, "wall_sec": 0.0,
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
            gold_ids = tokenizer.encode(gold, add_special_tokens=False)
            if not gold_ids:
                continue
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            if ids.size(1) > max_seq_len:
                ids = ids[:, -max_seq_len:]
            stats, loss = per_prompt_dim_flow(model, ids, gold_ids[0], device, dtype)
            per_prompt.append(stats)
            losses.append(loss)
        agg = aggregate_dim(per_prompt)
        results["by_kind"][kind] = {
            "n_prompts": len(per_prompt),
            "mean_loss": float(np.mean(losses)) if losses else None,
            "blocks": agg,
        }
        print(f"[{kind}] n={len(per_prompt)} mean_loss={results['by_kind'][kind]['mean_loss']:.4f}", flush=True)

    results["wall_sec"] = time.time() - t0
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="dim_flow.json")
    ap.add_argument("--max-prompts-per-kind", type=int, default=15)
    args = ap.parse_args()
    run(args.ckpt_dir, args.data_dir, args.out,
        max_prompts_per_kind=args.max_prompts_per_kind)
