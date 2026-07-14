"""High-precision per-block Q/K/V investigation.

Three deeper probes beyond mean-direction cosine:

A. W_Q / W_K / W_V WEIGHT-MATRIX subspace overlap between blocks.
   Each block has W_qkv [d, 3d] -> W_Q [d, d], W_K [d, d], W_V [d, d]. Top-k
   left-singular vectors of each define a k-dim subspace. The PRINCIPAL ANGLES
   between subspaces tell us whether two blocks PROJECT into the same direction
   (independent of activations). This is the cleanest "what does Q aim for".

B. PER-HEAD specialization within each block. Each Q [B,T,n_heads,head_dim] is
   split per head; effective rank of head_h's Q activations across positions.
   Variation in eff_rank across heads = specialization (some heads narrow,
   some broad).

C. ANSWER-POSITION Q TRAJECTORY. Take the final token's Q vector at each block.
   Cosine matrix across blocks tells us how the "I'm asking about the answer"
   query evolves through depth. For vanilla we expect rotation; for looped we
   expect collapse.

D. Q-K alignment: cos(W_Q top-k vectors, W_K top-k vectors) per block. Does
   the block's Q "ask for" things that K "answers"? High alignment = retrieval-
   like, low = transformation-like.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _principal_angles(A: torch.Tensor, B: torch.Tensor, k: int = 16) -> dict:
    """Principal angles between column-spans of A and B. Both [d, K] (already
    orthonormal); returns the top angles in degrees + a single 'subspace
    distance' (sin of largest angle).
    """
    # SVD of A^T B; cosines = singular values
    M = A.T @ B
    s = torch.linalg.svdvals(M)
    s = torch.clamp(s, -1.0, 1.0)
    angles = torch.arccos(s).cpu().numpy() * (180.0 / np.pi)
    return {
        "max_angle_deg": float(angles.max()),
        "min_angle_deg": float(angles.min()),
        "mean_angle_deg": float(angles.mean()),
        "subspace_dist": float(np.sin(angles.max() * np.pi / 180.0)),
        "mean_cos": float(s.mean().item()),
        "top_cos": [float(v) for v in s[:k].cpu().numpy()],
    }


def _top_left_singular(W: torch.Tensor, k: int) -> torch.Tensor:
    """Return [d, k] orthonormal top-k left singular vectors of W."""
    U, _, _ = torch.linalg.svd(W, full_matrices=False)
    return U[:, :k]


def _eff_rank(M: torch.Tensor) -> float:
    s = torch.linalg.svdvals(M.float())
    return float((s.sum().item() ** 2) / (s.pow(2).sum().item() + 1e-12))


def block_weight_subspaces(model, k: int = 16) -> dict:
    """For each Block in execution order, compute (top-k left-singular vectors
    of W_Q / W_K / W_V). Return a dict {block_name: {q,k,v: tensor [d, k]}}.
    """
    from .model import Block
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, Block):
            W = mod.attn.qkv.weight.data  # [3d, d] (Linear is out, in)
            d = W.shape[1]
            Wq = W[0*d:1*d, :].T   # [d, d]
            Wk = W[1*d:2*d, :].T
            Wv = W[2*d:3*d, :].T
            out[name] = {
                "q": _top_left_singular(Wq, k),
                "k": _top_left_singular(Wk, k),
                "v": _top_left_singular(Wv, k),
                "q_eff_rank": _eff_rank(Wq),
                "k_eff_rank": _eff_rank(Wk),
                "v_eff_rank": _eff_rank(Wv),
            }
    return out


@torch.no_grad()
def per_prompt_block_probe(model, x: torch.Tensor, device: str, dtype) -> dict:
    """Capture per-block: full Q/K/V activation tensor for the FINAL position
    only (B,T,3*d) -> select t=-1 -> reshape to per-head."""
    from .model import Block
    captured: list[tuple[str, torch.Tensor]] = []  # name, qkv at last pos

    def make_qkv_hook(name):
        def hook(module, inputs, output):
            o = output[0] if isinstance(output, tuple) else output
            # output is [B, T, 3*d]
            captured.append((name, o[:, -1, :].detach().float()))  # [B, 3*d]
        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, Block):
            handles.append(mod.attn.qkv.register_forward_hook(make_qkv_hook(name)))

    device_type = device if isinstance(device, str) else device.type
    with torch.amp.autocast(device_type=device_type, dtype=dtype,
                              enabled=(dtype != torch.float32)):
        model(x)
    for h in handles:
        h.remove()

    seq_names = [n for n, _ in captured]
    seq_qkv = [t for _, t in captured]
    n = len(seq_qkv)
    if n == 0:
        return {"seq_names": [], "answer_pos": {}}

    d = seq_qkv[0].shape[-1] // 3
    Bs = seq_qkv[0].shape[0]

    # Per-block answer-position Q, K, V (mean over batch)
    q_vecs = [t[:, :d].mean(dim=0) for t in seq_qkv]   # each [d]
    k_vecs = [t[:, d:2*d].mean(dim=0) for t in seq_qkv]
    v_vecs = [t[:, 2*d:].mean(dim=0) for t in seq_qkv]

    def cos_mat(vecs):
        n_ = len(vecs)
        M = []
        for i in range(n_):
            row = []
            for j in range(n_):
                vi, vj = vecs[i], vecs[j]
                num = (vi * vj).sum().item()
                denom = vi.norm().item() * vj.norm().item() + 1e-9
                row.append(float(num / denom))
            M.append(row)
        return M

    return {
        "seq_names": seq_names,
        "answer_pos": {
            "q_cos_mat": cos_mat(q_vecs),
            "k_cos_mat": cos_mat(k_vecs),
            "v_cos_mat": cos_mat(v_vecs),
        },
    }


def aggregate_probe(per_prompt: list[dict]) -> dict:
    if not per_prompt:
        return {}
    seq_names = per_prompt[0]["seq_names"]
    keys = ["q_cos_mat", "k_cos_mat", "v_cos_mat"]
    out = {"seq_names": seq_names, "answer_pos": {}}
    for k in keys:
        arrs = [np.array(p["answer_pos"][k]) for p in per_prompt]
        out["answer_pos"][k] = np.mean(arrs, axis=0).tolist()
    out["n_prompts"] = len(per_prompt)
    return out


def run(ckpt_dir: str, data_dir: str, out_path: str,
        eval_file: str | None = None,
        max_prompts_per_kind: int = 12,
        max_seq_len: int = 1024,
        topk_subspace: int = 16) -> dict:
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
    model.eval()

    # ----- Probe A: weight-matrix subspaces (no forward needed) -----
    print(f"Probe A: extracting top-{topk_subspace} left singular vectors per block...", flush=True)
    subspaces = block_weight_subspaces(model, k=topk_subspace)

    # Cross-block subspace distances (Q vs Q, K vs K, V vs V) and Q vs K within block
    block_names = list(subspaces.keys())
    nb = len(block_names)
    subspace_results = {}
    for tag in ["q", "k", "v"]:
        mat = []
        for i in range(nb):
            row = []
            for j in range(nb):
                if i == j:
                    row.append({"max_angle_deg": 0.0, "subspace_dist": 0.0, "mean_cos": 1.0, "top_cos": [1.0]*topk_subspace})
                else:
                    A = subspaces[block_names[i]][tag]
                    B = subspaces[block_names[j]][tag]
                    row.append(_principal_angles(A, B, k=topk_subspace))
            mat.append(row)
        subspace_results[f"{tag}_subspace_mat"] = mat

    # Q-vs-K alignment within each block
    qk_alignment = {}
    for name in block_names:
        info = _principal_angles(subspaces[name]["q"], subspaces[name]["k"], k=topk_subspace)
        qk_alignment[name] = info

    # Per-block W eff_rank
    weight_eff_rank = {n: {tag: subspaces[n][f"{tag}_eff_rank"] for tag in ["q", "k", "v"]}
                       for n in block_names}

    # ----- Probe B/C: forward-time per-prompt Q/K/V at answer position -----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Path(data_dir) / "tokenizer"))
    ef = Path(eval_file) if eval_file else (Path(data_dir) / "eval_synth.json")
    examples = json.loads(Path(ef).read_text())

    by_kind: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_kind[ex["kind"]].append(ex)

    answer_pos_results = {}
    for kind, exs in by_kind.items():
        per_prompt = []
        for ex in exs[:max_prompts_per_kind]:
            ids = tokenizer.encode(ex["prompt"], return_tensors="pt").to(device)
            if ids.size(1) > max_seq_len:
                ids = ids[:, -max_seq_len:]
            stats = per_prompt_block_probe(model, ids, device, dtype)
            per_prompt.append(stats)
        answer_pos_results[kind] = aggregate_probe(per_prompt)
        print(f"[{kind}] n={len(per_prompt)}", flush=True)

    # ----- Output -----
    results = {
        "arch": arch_cfg.arch, "d_model": arch_cfg.d_model,
        "n_blocks": arch_cfg.n_blocks, "n_loops": arch_cfg.n_loops,
        "n_prelude": arch_cfg.n_prelude, "n_coda": arch_cfg.n_coda,
        "n_heads": arch_cfg.n_heads,
        "block_names": block_names,
        "weight_eff_rank": weight_eff_rank,
        "weight_subspace_mat": subspace_results,
        "qk_alignment_per_block": qk_alignment,
        "answer_pos_by_kind": answer_pos_results,
    }
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"Saved {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="block_probe.json")
    args = ap.parse_args()
    run(args.ckpt_dir, args.data_dir, args.out)
