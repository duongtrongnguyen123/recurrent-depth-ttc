"""Lyapunov-style loop sensitivity analysis.

For each loop r in 1..n_loops, perturb h_r with small Gaussian noise and
measure how the perturbation propagates to h_{r+1}, h_{r+2}, etc.

  λ_r = E[ log(||h'_{r+k} - h_{r+k}|| / ||eps||) / k ]   for k = 1..n_loops-r

If λ_r > 0: perturbation amplifies (chaotic dynamics).
If λ_r < 0: perturbation shrinks (contractive iteration).
λ_r ≈ 0: marginally stable.

Reasoning interpretation: contractive (λ<0) = converging-to-answer regime.
Chaotic (λ>0) = each loop expands sensitivity, model uses depth to amplify
small distinctions in input. PCC_AUX rank-stable behavior suggests
contractive; vanilla/SKIP rank-collapse suggests strongly contractive.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def lyapunov_per_loop(model, x: torch.Tensor, n_loops: int,
                       device: str, dtype, eps_scale: float = 1e-3,
                       n_perturbations: int = 4) -> dict:
    """Compute amplification ratio per perturbation loop.

    For each starting loop r in 1..n_loops, run forward up to r, perturb h_r
    with eps_scale·||h_r||·N(0,1)/sqrt(d), then continue forward to loop n_loops.
    Measure ||h_{r+k}(perturbed) - h_{r+k}(clean)|| / ||eps|| for k=1..n_loops-r.

    Returns dict: {start_loop r: {steps_ahead k: amplification_ratio}}.
    """
    T = x.size(1)
    device_type = device if isinstance(device, str) else device.type
    h0 = model.tok_emb(x)
    cos, sin = model._rope(T, h0.device, h0.dtype)
    with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(dtype != torch.float32)):
        for blk in model.prelude_blocks:
            h0 = blk(h0, cos, sin)

    # Run clean forward and save h at every loop.
    clean_hs: list = [h0.float()]
    h = h0
    with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(dtype != torch.float32)):
        for r in range(n_loops):
            for blk in model.core_blocks:
                h = blk(h, cos, sin)
            clean_hs.append(h.float())

    # For each starting loop r in 1..n_loops, perturb h_r and re-run.
    amp = {}
    d = h0.size(-1)
    for r in range(1, n_loops + 1):
        h_clean = clean_hs[r]
        per_k_ratios = {k: [] for k in range(1, n_loops - r + 1 + 1) if k > 0 and r + k <= n_loops}
        if not per_k_ratios:
            continue
        for _ in range(n_perturbations):
            # eps proportional to ||h_r||/sqrt(d)
            eps = torch.randn_like(h_clean) * (eps_scale * h_clean.norm(dim=-1, keepdim=True)
                                                  / (d ** 0.5))
            eps_norm_per_pos = eps.norm(dim=-1)         # [B, T]
            # Re-run from perturbed h_r forward
            h_pert = h_clean + eps
            with torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=(dtype != torch.float32)):
                for k in range(1, n_loops - r + 1):
                    for blk in model.core_blocks:
                        h_pert = blk(h_pert, cos, sin)
                    delta = (h_pert.float() - clean_hs[r + k]).norm(dim=-1)  # [B, T]
                    ratio = (delta / eps_norm_per_pos.clamp(min=1e-8)).mean().item()
                    per_k_ratios.setdefault(k, []).append(ratio)
        amp[r] = {k: round(float(np.mean(rs)), 4) for k, rs in per_k_ratios.items() if rs}
    return amp


def run(ckpt_dir: str, val_bin_path: str, out_path: str,
        n_batches: int = 2, batch_size: int = 2, block_size: int = 1024) -> dict:
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

    arr = np.memmap(val_bin_path, dtype=np.uint32, mode="r")
    rng = np.random.default_rng(0)
    aggregated = {r: {} for r in range(1, arch_cfg.n_loops + 1)}
    t0 = time.time()
    for _ in range(n_batches):
        x_buf = np.empty((batch_size, block_size), dtype=np.int64)
        for i in range(batch_size):
            start = int(rng.integers(0, len(arr) - block_size - 1))
            x_buf[i] = arr[start: start + block_size].astype(np.int64)
        x = torch.from_numpy(x_buf).to(device)
        amp = lyapunov_per_loop(model, x, arch_cfg.n_loops, device, dtype)
        for r, kvs in amp.items():
            for k, v in kvs.items():
                aggregated[r].setdefault(k, []).append(v)
    avg_amp = {r: {k: round(float(np.mean(vs)), 4) for k, vs in kvs.items()}
               for r, kvs in aggregated.items() if kvs}
    payload = {
        "arch": arch_cfg.arch, "d_model": arch_cfg.d_model, "n_loops": arch_cfg.n_loops,
        "amplification_per_start_loop": avg_amp,
        "wall_sec": time.time() - t0,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return payload
