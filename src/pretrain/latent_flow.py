"""Latent flow analysis: how does h_r evolve across loops?

Loads a saved aux-trained ckpt, runs forward on a batch of val data, captures
hidden states at every loop, then computes:

1. **Per-loop relative delta**: ||h_{r+1} - h_r|| / ||h_r||  → how much the
   latent changes per loop. Big at early loops, shrinking signals convergence.
2. **Cosine drift from h_0**: cos(h_r, h_0) — how far from the embedding the
   latent has moved.
3. **Per-channel variance ratio**: var(h_r[:,c]) / var(h_0[:,c]) for each
   channel c. Tells which dimensions amplify across loops.
4. **Per-position progress**: for each token position, ||h_{r+1} - h_r|| —
   which positions get most loop-progression (analogous to Q2 argmin).
5. **Cross-loop cosine matrix**: cos(h_i, h_j) for i, j ∈ 0..n_loops. Reveals
   whether loops are "linear" (geometric progression) or "cyclic".

Output: a JSON with these stats + sample positions for inspection.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def collect_loop_hiddens(model, x: torch.Tensor, n_loops: int, device: str,
                          dtype) -> list[torch.Tensor]:
    """Returns list of [B, T, d] for h_0, h_1, ..., h_{n_loops}.

    h_0 = embedding (after prelude if PCC).
    h_r = hidden state after r core loops.
    """
    with torch.amp.autocast(device_type=device, dtype=dtype,
                              enabled=(dtype != torch.float32)):
        out = model(x, n_loops=n_loops, collect_loops=True)
    # collect_loops gives loop_hiddens for r=1..n_loops. Need h_0 too.
    # Reconstruct h_0: pre-recurrence after prelude.
    T = x.size(1)
    h0 = model.tok_emb(x)
    cos_, sin_ = model._rope(T, h0.device, h0.dtype)
    with torch.amp.autocast(device_type=device, dtype=dtype,
                              enabled=(dtype != torch.float32)):
        for blk in model.prelude_blocks:
            h0 = blk(h0, cos_, sin_)
    hiddens = [h0.float()] + [h.float() for h in out["loop_hiddens"]]
    return hiddens


@torch.no_grad()
def collect_per_block_hiddens(model, x: torch.Tensor, device: str, dtype
                               ) -> tuple[list[torch.Tensor], list[str]]:
    """V3 capture: hidden state after EVERY block (prelude + every core block,
    every core loop, every coda block). Gives a fair per-block trajectory for
    vanilla so we can compare to recurrent's per-loop trajectory at matched
    granularity.

    Returns (hiddens, labels). hiddens[i] is [B, T, d], labels[i] describes
    what block produced it (e.g., 'pre_block_3' or 'core_loop_2_block_0').
    """
    T = x.size(1)
    h = model.tok_emb(x)
    cos_, sin_ = model._rope(T, h.device, h.dtype)
    hiddens = [h.float().clone()]
    labels = ["embed"]
    with torch.amp.autocast(device_type=device, dtype=dtype,
                              enabled=(dtype != torch.float32)):
        # Prelude blocks (one shot)
        for i, blk in enumerate(model.prelude_blocks):
            h = blk(h, cos_, sin_)
            hiddens.append(h.float().clone())
            labels.append(f"prelude_{i}")
        # Core blocks looped n_loops times
        n_loops = getattr(model.cfg, "n_loops", 1)
        for r in range(n_loops):
            for j, blk in enumerate(model.core_blocks):
                h = blk(h, cos_, sin_)
                hiddens.append(h.float().clone())
                labels.append(f"core_loop{r}_blk{j}")
        # Coda blocks (one shot)
        for k, blk in enumerate(model.coda_blocks):
            h = blk(h, cos_, sin_)
            hiddens.append(h.float().clone())
            labels.append(f"coda_{k}")
    return hiddens, labels


def analyze_flow(hiddens: list[torch.Tensor], n_sample_positions: int = 16):
    """Compute the suite of latent-flow stats from a list of [B, T, d] tensors."""
    L = len(hiddens) - 1                       # n_loops
    if L < 1:
        return {}
    B, T, d = hiddens[0].shape
    # Stack to [L+1, B, T, d]
    H = torch.stack(hiddens, dim=0)
    norms = H.norm(dim=-1)                     # [L+1, B, T]
    # Per-loop relative deltas: ||h_r - h_{r-1}|| / ||h_{r-1}||
    rel_deltas = []
    for r in range(1, L + 1):
        diff_norm = (H[r] - H[r - 1]).norm(dim=-1)        # [B, T]
        rel = (diff_norm / norms[r - 1].clamp(min=1e-8)).mean().item()
        rel_deltas.append(rel)
    # Cosine drift from h_0
    cos_drifts = []
    h0_flat = H[0].reshape(-1, d)
    for r in range(0, L + 1):
        hr_flat = H[r].reshape(-1, d)
        cos = torch.nn.functional.cosine_similarity(h0_flat, hr_flat, dim=-1)
        cos_drifts.append(cos.mean().item())
    # Per-channel variance ratio at last loop vs h_0
    var0 = H[0].reshape(-1, d).var(dim=0)              # [d]
    varL = H[L].reshape(-1, d).var(dim=0)              # [d]
    var_ratio = (varL / var0.clamp(min=1e-8))
    # Top channels by amplification
    top_amp_vals, top_amp_idx = var_ratio.topk(8)
    bot_amp_vals, bot_amp_idx = (-var_ratio).topk(8)
    # Per-position progress: mean ||h_{r+1} - h_r|| across loops, per (B,T)
    per_pos_total = torch.zeros_like(H[0][..., 0])     # [B, T]
    for r in range(1, L + 1):
        per_pos_total += (H[r] - H[r - 1]).norm(dim=-1)
    per_pos_mean = per_pos_total.mean().item()
    per_pos_top = per_pos_total.flatten().topk(n_sample_positions)
    per_pos_bot = (-per_pos_total).flatten().topk(n_sample_positions)
    # Cross-loop cosine matrix (averaged over B,T)
    M = []
    for i in range(L + 1):
        row = []
        for j in range(L + 1):
            cos = torch.nn.functional.cosine_similarity(
                H[i].reshape(-1, d), H[j].reshape(-1, d), dim=-1)
            row.append(round(cos.mean().item(), 4))
        M.append(row)

    # NEW: per-loop ||h_r|| stats (mean, std)
    h_norm_per_loop = [(round(norms[r].mean().item(), 3),
                          round(norms[r].std().item(), 3))
                         for r in range(L + 1)]

    # NEW: effective rank at each loop. h flattened over [B*T, d], then SVD.
    # Soft rank = (sum σ_i)² / sum σ_i² ∈ [1, d]. Tells us how many
    # dimensions are "active" in the latent at loop r.
    eff_ranks = []
    sing_top10_per_loop = []
    for r in range(L + 1):
        h_flat = H[r].reshape(-1, d)
        # Center to focus on variance, not mean shift.
        h_centered = h_flat - h_flat.mean(dim=0, keepdim=True)
        # SVD of [N, d]; cheaper to compute on the smaller side.
        # If N is large, use eig of covariance instead.
        try:
            with torch.no_grad():
                # Subsample N to keep SVD cheap if too many positions.
                N = h_centered.shape[0]
                if N > 4096:
                    idx = torch.randperm(N, device=h_centered.device)[:4096]
                    h_centered = h_centered[idx]
                # Compute singular values only.
                S = torch.linalg.svdvals(h_centered.float())
            sigma2 = (S ** 2)
            sigma2_sum = sigma2.sum().clamp(min=1e-10)
            eff_rank = (sigma2_sum ** 2) / (sigma2 ** 2).sum().clamp(min=1e-10)
            eff_ranks.append(round(eff_rank.item(), 1))
            sing_top10_per_loop.append([round(s.item(), 2) for s in S[:10]])
        except Exception as e:  # pragma: no cover
            eff_ranks.append(None)
            sing_top10_per_loop.append([])

    # NEW: per-channel std distribution at last loop (histogram in 8 buckets)
    chan_std_L = H[L].reshape(-1, d).std(dim=0)
    chan_std_0 = H[0].reshape(-1, d).std(dim=0)
    pct = lambda t, ps: [round(torch.quantile(t, p).item(), 3) for p in ps]
    chan_std_quantiles = pct(chan_std_L, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    chan_std_h0_quantiles = pct(chan_std_0, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

    return {
        "n_loops": L,
        "rel_deltas_per_loop": [round(x, 4) for x in rel_deltas],
        "cos_from_h0_per_loop": [round(x, 4) for x in cos_drifts],
        "var_ratio_top8": [(int(i), round(v.item(), 3))
                              for i, v in zip(top_amp_idx, top_amp_vals)],
        "var_ratio_bottom8": [(int(i), round(-v.item(), 3))
                                 for i, v in zip(bot_amp_idx, bot_amp_vals)],
        "per_position_total_delta_mean": round(per_pos_mean, 4),
        "per_position_top_delta_norm": [round(v.item(), 3) for v in per_pos_top.values],
        "cross_loop_cos_matrix": M,
        "h_norm_per_loop_mean_std": h_norm_per_loop,
        "effective_rank_per_loop": eff_ranks,
        "top10_singular_values_per_loop": sing_top10_per_loop,
        "channel_std_quantiles_h0": chan_std_h0_quantiles,
        "channel_std_quantiles_hL": chan_std_quantiles,
    }


def run(ckpt_dir: str, val_bin_path: str, out_path: str,
        n_batches: int = 4, batch_size: int = 4, block_size: int = 2048,
        per_block: bool = False) -> dict:
    """V3: per_block=True captures hidden state after EVERY block (prelude +
    every core block, every core loop, every coda). Gives fair per-block
    trajectory for vanilla matched to recurrent's per-loop granularity."""
    from .model import PretrainConfig, build_model

    ckpt_path = Path(ckpt_dir)
    cfg_path = ckpt_path.parent.parent / "config.json"
    cfg_data = json.loads(cfg_path.read_text())
    field_names = {f.name for f in PretrainConfig.__dataclass_fields__.values()}
    arch_cfg = PretrainConfig(**{k: v for k, v in cfg_data["arch"].items()
                                    if k in field_names})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32
    model = build_model(arch_cfg).to(device)
    state = torch.load(ckpt_path / "state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    arr = np.memmap(val_bin_path, dtype=np.uint32, mode="r")
    rng = np.random.default_rng(0)
    aggregated_stats = []
    block_labels = None
    t0 = time.time()
    for _ in range(n_batches):
        x_buf = np.empty((batch_size, block_size), dtype=np.int64)
        for i in range(batch_size):
            start = int(rng.integers(0, len(arr) - block_size - 1))
            x_buf[i] = arr[start: start + block_size].astype(np.int64)
        x = torch.from_numpy(x_buf).to(device)
        if per_block:
            hiddens, labels = collect_per_block_hiddens(model, x, device, dtype)
            block_labels = labels
        else:
            hiddens = collect_loop_hiddens(model, x, arch_cfg.n_loops, device, dtype)
        stats = analyze_flow(hiddens)
        aggregated_stats.append(stats)
    # Merge stats: average rel_deltas / cos / var_ratio across batches.
    L = aggregated_stats[0]["n_loops"]
    avg_rel = [float(np.mean([s["rel_deltas_per_loop"][r] for s in aggregated_stats]))
                for r in range(L)]
    avg_cos = [float(np.mean([s["cos_from_h0_per_loop"][r] for s in aggregated_stats]))
                for r in range(L + 1)]
    avg_M = [[float(np.mean([s["cross_loop_cos_matrix"][i][j]
                              for s in aggregated_stats]))
              for j in range(L + 1)] for i in range(L + 1)]
    # Aggregate new stats
    avg_eff_rank = [float(np.mean([s["effective_rank_per_loop"][r] for s in aggregated_stats
                                      if s["effective_rank_per_loop"][r] is not None]))
                       for r in range(L + 1)]
    avg_h_norm = [(float(np.mean([s["h_norm_per_loop_mean_std"][r][0] for s in aggregated_stats])),
                     float(np.mean([s["h_norm_per_loop_mean_std"][r][1] for s in aggregated_stats])))
                    for r in range(L + 1)]
    # First batch's quantiles + top channels (they shouldn't vary much across batches)
    s0 = aggregated_stats[0]
    payload = {
        "arch": arch_cfg.arch,
        "d_model": arch_cfg.d_model,
        "n_loops": L,
        "n_batches": n_batches,
        "per_block": per_block,
        "block_labels": block_labels,
        "rel_deltas_per_loop": [round(x, 4) for x in avg_rel],
        "cos_from_h0_per_loop": [round(x, 4) for x in avg_cos],
        "cross_loop_cos_matrix": [[round(x, 4) for x in row] for row in avg_M],
        "effective_rank_per_loop": [round(x, 1) for x in avg_eff_rank],
        "h_norm_per_loop_mean_std": [(round(m, 3), round(s, 3)) for m, s in avg_h_norm],
        "channel_std_quantiles_h0": s0.get("channel_std_quantiles_h0"),
        "channel_std_quantiles_hL": s0.get("channel_std_quantiles_hL"),
        "var_ratio_top8": s0.get("var_ratio_top8"),
        "top10_singular_values_h0": s0.get("top10_singular_values_per_loop", [None])[0],
        "top10_singular_values_hL": s0.get("top10_singular_values_per_loop", [None])[-1],
        "wall_sec": time.time() - t0,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return payload
