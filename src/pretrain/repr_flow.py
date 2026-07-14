"""Representation-flow probe: what do Q/K/V aim for, and what do block outputs
"close by"?

For each block (in vanilla 8 distinct, in looped 1 shared block applied N times):
  - capture h_OUT: post-block residual (after attn+mlp) [B,T,d]
  - capture Q, K, V: attention projections [B,T,n_heads,head_dim]

Then compute:
  1. CROSS-BLOCK COSINE MATRIX of h_OUT — how similar are different blocks'
     outputs? Vanilla expectation: gradual progression; looped expectation:
     either contractive (later loops near each other) or translational (linear
     drift).
  2. CROSS-BLOCK Q/K/V SIMILARITY — for each head, mean Q-direction over
     batch+positions. Cosine across blocks tells us whether different blocks
     "look at" similar features.
  3. EFFECTIVE RANK of Q/K/V activations per block (SVD-based).
  4. Q-K SELF-SIMILARITY: how peaked are Q-K dot products (avg attention entropy
     across heads).

For looped models we capture each LOOP's output; the "block index" is the loop
step. For vanilla, the "block index" is the actual block number.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch


def _flatten_for_cos(t: torch.Tensor) -> torch.Tensor:
    """[B,T,d] -> [d_flat] = mean over batch+positions."""
    return t.float().mean(dim=(0, 1))


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())


def _eff_rank(M: torch.Tensor) -> float:
    """Effective rank via singular values: (sum)^2 / sum_of_squares."""
    M = M.reshape(M.shape[0], -1).float()
    if M.shape[0] > 1024:
        idx = torch.randperm(M.shape[0])[:1024]
        M = M[idx]
    try:
        s = torch.linalg.svdvals(M)
        return float((s.sum().item() ** 2) / (s.pow(2).sum().item() + 1e-12))
    except Exception:
        return 0.0


@torch.no_grad()
def per_prompt_repr_flow(model, x: torch.Tensor, device: str, dtype) -> dict:
    """Capture block outputs + QKV per block (or per loop) using forward hooks."""
    from .model import Block, Attention
    block_out_seq: list[tuple[str, torch.Tensor]] = []
    qkv_seq: list[tuple[str, torch.Tensor]] = []  # name, qkv [B,T,3*d]

    def make_block_hook(name):
        def hook(module, inputs, output):
            # xloop blocks return (out, kv) tuple; vanilla returns tensor
            o = output[0] if isinstance(output, tuple) else output
            block_out_seq.append((name, o.detach().float()))
        return hook

    def make_qkv_hook(name):
        def hook(module, inputs, output):
            # qkv is nn.Linear; output is plain tensor
            o = output[0] if isinstance(output, tuple) else output
            qkv_seq.append((name, o.detach().float()))
        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, Block):
            handles.append(mod.register_forward_hook(make_block_hook(name)))
            if hasattr(mod, "attn") and hasattr(mod.attn, "qkv"):
                handles.append(mod.attn.qkv.register_forward_hook(make_qkv_hook(name)))

    device_type = device if isinstance(device, str) else device.type
    with torch.amp.autocast(device_type=device_type, dtype=dtype,
                              enabled=(dtype != torch.float32)):
        out = model(x)
    for h in handles:
        h.remove()

    # Sequence of block outputs (in execution order). For looped, the same
    # name appears n_loops times. For vanilla, each block appears once.
    seq_names = [n for n, _ in block_out_seq]
    seq_outs = [t for _, t in block_out_seq]
    seq_qkv = [t for _, t in qkv_seq]

    # Cross-block cosine of mean-output vectors
    means = [_flatten_for_cos(t) for t in seq_outs]
    n = len(means)
    cos_mat = [[_cos(means[i], means[j]) for j in range(n)] for i in range(n)]

    # Effective rank of each block's output (sample of token-level vectors)
    eff_rank = []
    for t in seq_outs:
        B, T, D = t.shape
        flat = t.reshape(-1, D)  # [B*T, D]
        eff_rank.append(_eff_rank(flat))

    # Q/K/V analysis: split last dim into 3
    qkv_means = []
    qkv_eff_rank = {"q": [], "k": [], "v": []}
    qkv_cos_mat = {"q": [], "k": [], "v": []}
    for t in seq_qkv:
        B, T, three_d = t.shape
        d = three_d // 3
        q, k, v = t[:, :, :d], t[:, :, d:2*d], t[:, :, 2*d:]
        qkv_means.append((q.float().mean(dim=(0, 1)),
                          k.float().mean(dim=(0, 1)),
                          v.float().mean(dim=(0, 1))))
        for tag, m in zip(["q", "k", "v"], [q, k, v]):
            qkv_eff_rank[tag].append(_eff_rank(m.reshape(-1, d)))

    n_qkv = len(qkv_means)
    for tag_idx, tag in enumerate(["q", "k", "v"]):
        qkv_cos_mat[tag] = [
            [_cos(qkv_means[i][tag_idx], qkv_means[j][tag_idx])
             for j in range(n_qkv)] for i in range(n_qkv)
        ]

    return {
        "seq_names": seq_names,
        "block_out_eff_rank": eff_rank,
        "block_out_cos_mat": cos_mat,
        "qkv_eff_rank": qkv_eff_rank,
        "qkv_cos_mat": qkv_cos_mat,
    }


def aggregate_repr(per_prompt_list: list[dict]) -> dict:
    """Average matrices/lists across prompts."""
    if not per_prompt_list:
        return {}
    seq_names = per_prompt_list[0]["seq_names"]
    n = len(seq_names)
    avg_eff = np.mean([np.array(p["block_out_eff_rank"]) for p in per_prompt_list], axis=0).tolist()
    avg_cos = np.mean([np.array(p["block_out_cos_mat"]) for p in per_prompt_list], axis=0).tolist()
    avg_qkv_eff = {k: np.mean([np.array(p["qkv_eff_rank"][k]) for p in per_prompt_list], axis=0).tolist()
                   for k in ["q", "k", "v"]}
    avg_qkv_cos = {k: np.mean([np.array(p["qkv_cos_mat"][k]) for p in per_prompt_list], axis=0).tolist()
                   for k in ["q", "k", "v"]}
    return {
        "seq_names": seq_names,
        "block_out_eff_rank": avg_eff,
        "block_out_cos_mat": avg_cos,
        "qkv_eff_rank": avg_qkv_eff,
        "qkv_cos_mat": avg_qkv_cos,
        "n_prompts": len(per_prompt_list),
    }


def run(ckpt_dir: str, data_dir: str, out_path: str,
        eval_file: str | None = None,
        max_prompts_per_kind: int = 12,
        max_seq_len: int = 1024) -> dict:
    from .model import PretrainConfig, build_model
    from collections import defaultdict

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
    model.eval()

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
        for ex in exs[:max_prompts_per_kind]:
            prompt = ex["prompt"]
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            if ids.size(1) > max_seq_len:
                ids = ids[:, -max_seq_len:]
            stats = per_prompt_repr_flow(model, ids, device, dtype)
            per_prompt.append(stats)
        results["by_kind"][kind] = aggregate_repr(per_prompt)
        print(f"[{kind}] n={len(per_prompt)}", flush=True)

    results["wall_sec"] = time.time() - t0
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="repr_flow.json")
    args = ap.parse_args()
    run(args.ckpt_dir, args.data_dir, args.out)
