"""Training loop. Auto-resumes from /kaggle/input/checkpoint/ when present;
saves to /kaggle/working/ckpt/. Logs to JSON line per save.

Usage: invoked by the Kaggle notebook with config dict, runs until time budget elapses
or until target tokens reached, whichever first.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import LoopedTransformer, PretrainConfig, build_model


@dataclass
class TrainConfig:
    arch: str = "vanilla"
    target_tokens: int = 5_000_000_000     # 5B
    micro_batch: int = 8
    grad_accum: int = 64                   # global batch = micro_batch * grad_accum
    block_size: int = 2048
    lr_max: float = 6e-4
    lr_min: float = 6e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    save_every_steps: int = 250
    eval_every_steps: int = 250
    log_every_steps: int = 25
    time_budget_sec: int = 11 * 3600 + 30 * 60   # 11.5h, leave buffer to save
    n_loops: int = 8
    aux_min_loops: int = 2
    sd_min: int = 4                              # stochastic depth: random in [sd_min, n_loops]
    sd_max: int = 8
    # random_n_loops: Huginn-style per-step n_loops sampling for any recurrent arch
    # (pcc / looped / pcc_aux*). When True, each step samples n_loops from
    # Uniform[random_n_loops_min, n_loops]. Decouples depth from arch tag so we
    # can keep arch="pcc" but pretrain with depth variety — required for Branch 2
    # adaptive-depth deployment without NL forgetting (CC38 fix).
    random_n_loops: bool = False
    random_n_loops_min: int = 1
    curr_warmup_frac: float = 0.5                # curriculum: ramp k_max over this frac of training
    seed: int = 0
    # LR schedule: "cosine" (legacy, monotonic decay over full target)
    #              "wsd" (warmup-stable-decay; SmolLM2/OLMo2/Phi-3 modern recipe).
    # WSD: warmup → lr_max constant for (1-decay_frac) of post-warmup steps,
    # then linear decay from lr_max → lr_max*decay_min_ratio over the last
    # decay_frac. Per arXiv:2511.18903, moderate decay (~0.3×) beats aggressive
    # decay-to-near-zero when paired with high-quality decay-phase data.
    schedule: str = "cosine"
    decay_frac: float = 0.1                      # WSD: fraction of post-warmup tokens spent decaying
    decay_min_ratio: float = 0.3                 # WSD: lr_floor = lr_max * decay_min_ratio (moderate)
    # Iter-target multi-step horizon (Result AD ported to pretrain).
    # At loop r, position t's logits predict input[t+r]. Loss = mean_r CE.
    use_iter_target: bool = False
    # Data-recipe regional re-weighting. Empty list → default uniform sampling
    # (BinShardLoader). Non-empty → use RegionalShardLoader with these weights.
    # region_offsets are cumulative fractions of within-shard length;
    # region_weights are sampling probabilities for each region. For
    # reasoning-mix-mvp 50/30/20 layout, the offsets are [0.0, 0.5, 0.8, 1.0].
    region_offsets: tuple = ()
    region_weights: tuple = ()
    # torch.compile (Blackwell sm_120 typically 20-40% faster). Safe to disable
    # for debugging — compile masks traceback line numbers.
    use_compile: bool = False
    # Z-loss coefficient (PaLM default 1e-4). Penalizes large logsumexp →
    # better numerics + small loss improvement. Set 0 to disable.
    z_loss_coef: float = 1e-4
    # Background GPU sampler (util %, power W, VRAM GB) via pynvml. Adds
    # ~0 step overhead; appends to every log line so we can read MFU + power
    # without post-hoc inference.
    gpu_monitor: bool = True
    gpu_monitor_period_s: float = 5.0


@dataclass
class Paths:
    data_dir: str
    work_dir: str
    resume_dir: str = ""             # empty = fresh init
    eval_data_dir: str = ""          # for held-out perplexity


def _build_data(data_dir: str, block_size: int,
                region_offsets: tuple = (), region_weights: tuple = ()):
    from .data import BinShardLoader, RegionalShardLoader, discover_shards
    shards = discover_shards(data_dir, glob="train_*.bin")
    assert shards, f"no train_*.bin in {data_dir}"
    # All Phase 1 / 50B-canonical shards are uint16 (vocab=50257 < 65535).
    # Older reasoning-mix-mvp datasets were uint32 but shards happened to be
    # divisible by 4. Partial-shard layouts (e.g. 875M-per-part output) require
    # uint16 to avoid "size not multiple of data-type" assertion in memmap.
    if region_weights:
        return RegionalShardLoader(
            [str(s) for s in shards], dtype="uint16",
            block_size=block_size,
            region_offsets=list(region_offsets) or [0.0, 0.5, 0.8, 1.0],
            region_weights=list(region_weights),
        )
    return BinShardLoader([str(s) for s in shards], dtype="uint16",
                          block_size=block_size)


def _build_optimizer(model, cfg: TrainConfig):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim < 2 or n.endswith(".bias") or "ln" in n.lower() or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr_max, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
    )


def _lr(step: int, total_steps: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr_max * step / max(1, cfg.warmup_steps)
    post = total_steps - cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, post)
    progress = min(max(progress, 0.0), 1.0)
    if getattr(cfg, "schedule", "cosine") == "wsd":
        stable_frac = 1.0 - cfg.decay_frac
        if progress < stable_frac:
            return cfg.lr_max
        # Linear decay from lr_max → lr_max*decay_min_ratio over the last decay_frac
        d = (progress - stable_frac) / max(1e-6, cfg.decay_frac)
        floor = cfg.lr_max * cfg.decay_min_ratio
        return cfg.lr_max + (floor - cfg.lr_max) * d
    # Default: cosine
    return cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1 + math.cos(math.pi * progress))


def _save_checkpoint(model, optimizer, scheduler_step: int, tokens_seen: int,
                     log: list[dict], path: Path):
    """Save model+optimizer state. For models where state.pt > 1/2 of disk cap
    (1B vanilla on Kaggle: state.pt ≈ 14 GB, /kaggle/working cap = 20 GB), the
    old atomic-write pattern (state.pt.tmp → rename state.pt) doubled peak
    disk to 28 GB on the 2nd save and produced an iostream error in nguyen-
    duongtrong's 1B s1 at step 1000.

    Recipe now: delete state.pt FIRST, then write state.pt.tmp, then rename.
    Trade: a crash between the unlink and the rename loses this session's
    state.pt entirely. That's acceptable because chained kernel-source resume
    falls back to the previous session's output. The original atomicity goal
    (preventing 0-byte truncation) is preserved within a single write — we
    only write state.pt via rename of a fully-written tmp.

    For the log file: keep atomic write since it's tiny."""
    path.mkdir(parents=True, exist_ok=True)
    final_state = path / "state.pt"
    tmp_state = path / "state.pt.tmp"
    tmp_log = path / "log.json.tmp"
    # Delete old state.pt and any lingering tmp from a prior failed save so
    # the new tmp write has the full 20 GB cap available.
    if final_state.exists():
        final_state.unlink()
    if tmp_state.exists():
        tmp_state.unlink()
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": scheduler_step,
        "tokens_seen": tokens_seen,
    }, tmp_state)
    tmp_log.write_text(json.dumps(log, indent=2))
    tmp_state.replace(final_state)
    tmp_log.replace(path / "log.json")


def _load_checkpoint(path: Path):
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    log_path = path / "log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    return state, log


def _aux_loss(model, x, y, cfg: TrainConfig, n_loops: int, aux_min_loops: int,
              dtype: torch.dtype, z_loss_coef: float = 1e-4):
    with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
        out = model.forward_with_aux(x, n_loops=n_loops, aux_min_loops=aux_min_loops)
        losses = [F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1))
                  for lg in out["per_loop_logits"]]
        ce_mean = torch.stack(losses).mean()
        # Z-loss only on the FINAL loop's logits (cheapest + most informative).
        return ce_mean + _z_loss(out["per_loop_logits"][-1], z_loss_coef)


def _z_loss(logits: torch.Tensor, coef: float = 1e-4) -> torch.Tensor:
    """Auxiliary stability loss penalizing large logsumexp. Used in PaLM,
    Chinchilla, Gopher. Keeps softmax denominators bounded -> better numerics +
    small loss improvement. Coef 1e-4 is the PaLM default; safe across scales."""
    if coef <= 0:
        return logits.sum() * 0.0
    lse = torch.logsumexp(logits.float(), dim=-1)  # [B, T]
    return coef * (lse ** 2).mean()


def _vanilla_loss(model, x, y, n_loops: int, dtype: torch.dtype,
                   z_loss_coef: float = 1e-4):
    with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
        logits = model(x, n_loops=n_loops)["logits"]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        return ce + _z_loss(logits, z_loss_coef)


def _iter_target_loss(model, x, y, n_loops: int, dtype: torch.dtype):
    """Iter-target multi-step horizon (Result AD recipe ported to pretrain).

    At loop r in [1..n_loops], position t's logits should predict input[t+r].
    Per-loop r: valid positions are [0..T-r-1]; targets are x[r:T].
    Total loss = mean_r CE.

    Loop r=1 collapses to standard LM next-token.
    """
    T = x.size(1)
    with torch.amp.autocast(device_type="cuda", dtype=dtype,
                              enabled=(dtype != torch.float32)):
        out = model.forward_with_aux(x, n_loops=n_loops, aux_min_loops=1)
        per_loop_logits = out["per_loop_logits"]   # list of [B, T, V], len n_loops
        losses = []
        for r in range(1, n_loops + 1):
            logits_r = per_loop_logits[r - 1]
            if r >= T:
                continue                           # no targets at this horizon
            valid_logits = logits_r[:, : T - r, :].reshape(-1, logits_r.size(-1))
            target_r = x[:, r:T].reshape(-1)
            losses.append(F.cross_entropy(valid_logits, target_r))
        return torch.stack(losses).mean()


def _eval_loss(model, eval_loader, cfg: TrainConfig, n_eval_batches: int = 32) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_eval_batches):
            x, y = eval_loader.get_batch(cfg.micro_batch, device="cuda")
            logits = model(x, n_loops=cfg.n_loops)["logits"]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(arch_cfg: PretrainConfig, train_cfg: TrainConfig, paths: Paths) -> dict:
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    arch_cfg.arch = train_cfg.arch
    arch_cfg.n_loops = train_cfg.n_loops

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cap = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}  compute capability: sm_{cap[0]}{cap[1]}", flush=True)
        if cap[0] < 7:
            raise RuntimeError(
                f"GPU sm_{cap[0]}{cap[1]} ({gpu_name}) is too old for torch 2.10. "
                f"Need sm_70+ (V100/T4/A100/RTX 6000 Pro). Re-deploy with a different "
                f"machine_shape.")
        # bf16 needs sm_80+; fp16 works on sm_70+; fp32 works on sm_60+ (but we error on <70).
        if cap[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32
    print(f"device={device}  dtype={dtype}", flush=True)
    # Force flash attention via SDPA backend hint. On Blackwell sm_120 PyTorch
    # 2.5+ supports the flash kernel; this avoids falling back to the slower
    # math/efficient backend if flash availability is misdetected.
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        print(f"[sdpa] flash={torch.backends.cuda.flash_sdp_enabled()}  "
              f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}", flush=True)
    except Exception as e:
        print(f"[sdpa] backend hint failed: {e}", flush=True)

    # Keep model in fp32 master copy; let autocast handle low-precision compute.
    model = build_model(arch_cfg).to(device)

    # torch.compile speedup. On RTX 6000 Pro / Blackwell sm_120 typically gives
    # 20-40% throughput improvement. Set use_compile=False to disable for
    # debugging (compile hides traceback line numbers).
    if getattr(train_cfg, 'use_compile', False):
        # mode='default' (kernel fusion, no CUDA graphs). 'reduce-overhead'
        # conflicts with our RoPE cos/sin cache (CUDA graphs assume static
        # tensors; cos/sin gets recomputed/replaced per call).
        print(f"[compile] wrapping model with torch.compile(mode='default')", flush=True)
        try:
            model = torch.compile(model, mode='default', fullgraph=False)
            print(f"[compile] OK", flush=True)
        except Exception as e:
            print(f"[compile] FAILED: {e}; falling back to eager", flush=True)

    optimizer = _build_optimizer(model, train_cfg)
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    train_loader = _build_data(paths.data_dir, train_cfg.block_size,
                                 region_offsets=train_cfg.region_offsets,
                                 region_weights=train_cfg.region_weights)
    # Eval loader keeps the default uniform sampling for cross-run comparability.
    eval_loader = (_build_data(paths.eval_data_dir, train_cfg.block_size)
                   if paths.eval_data_dir else None)

    tokens_per_step = train_cfg.micro_batch * train_cfg.grad_accum * train_cfg.block_size
    total_steps = train_cfg.target_tokens // tokens_per_step
    print(f"target_tokens={train_cfg.target_tokens}  tokens/step={tokens_per_step}  "
          f"total_steps={total_steps}")

    state_step = 0
    tokens_seen = 0
    log: list[dict] = []
    if paths.resume_dir and Path(paths.resume_dir).exists():
        state, log = _load_checkpoint(Path(paths.resume_dir))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        state_step = state["step"]
        tokens_seen = state["tokens_seen"]
        print(f"resumed from step={state_step} tokens={tokens_seen}")

    rng = np.random.default_rng(train_cfg.seed + state_step)
    work = Path(paths.work_dir)
    ckpt_dir = work / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (work / "config.json").write_text(json.dumps({
        "arch": dataclasses.asdict(arch_cfg),
        "train": dataclasses.asdict(train_cfg),
        "paths": dataclasses.asdict(paths),
    }, indent=2))

    train_start = time.time()
    last_log_t = train_start
    use_aux = train_cfg.arch in ("looped_aux", "looped_aux_robust",
                                  "pcc_aux_robust", "curriculum")
    use_iter = train_cfg.use_iter_target

    gpu_sampler = None
    if getattr(train_cfg, 'gpu_monitor', False) and device == "cuda":
        from .gpu_monitor import GpuSampler
        gpu_sampler = GpuSampler(period_s=train_cfg.gpu_monitor_period_s)

    while state_step < total_steps:
        elapsed = time.time() - train_start
        if elapsed > train_cfg.time_budget_sec:
            print(f"time budget reached at step={state_step}, saving and exiting")
            break

        lr = _lr(state_step, total_steps, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Determine n_loops for this step (variant-specific).
        if train_cfg.arch == "stochastic_depth":
            n_loops = int(rng.integers(train_cfg.sd_min, train_cfg.sd_max + 1))
        elif train_cfg.arch == "curriculum":
            frac = state_step / max(1, total_steps * train_cfg.curr_warmup_frac)
            n_loops = max(2, min(train_cfg.n_loops,
                                  2 + int(frac * (train_cfg.n_loops - 2))))
        elif train_cfg.random_n_loops:
            n_loops = int(rng.integers(train_cfg.random_n_loops_min,
                                        train_cfg.n_loops + 1))
        else:
            n_loops = train_cfg.n_loops

        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(train_cfg.grad_accum):
            x, y = train_loader.get_batch(train_cfg.micro_batch, device=device, rng=rng)
            if use_iter:
                loss = _iter_target_loss(model, x, y, n_loops, dtype)
            elif use_aux:
                loss = _aux_loss(model, x, y, train_cfg, n_loops,
                                 train_cfg.aux_min_loops, dtype)
            else:
                loss = _vanilla_loss(model, x, y, n_loops, dtype)
            scaled = loss / train_cfg.grad_accum
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            loss_acc += loss.item()
        loss_acc /= train_cfg.grad_accum

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

        state_step += 1
        tokens_seen += tokens_per_step

        if state_step % train_cfg.log_every_steps == 0:
            now = time.time()
            tok_per_sec = tokens_per_step * train_cfg.log_every_steps / (now - last_log_t)
            vram_alloc = torch.cuda.memory_allocated(0) / 1024**3
            vram_peak = torch.cuda.max_memory_allocated(0) / 1024**3
            gpu_str = f"  {gpu_sampler.fmt()}" if gpu_sampler is not None else ""
            print(f"step {state_step:>5}  loss {loss_acc:.4f}  lr {lr:.2e}  "
                  f"n_loops {n_loops}  tok/s {tok_per_sec/1e3:.1f}K  "
                  f"vram {vram_alloc:.1f}/{vram_peak:.1f}GB{gpu_str}  "
                  f"elapsed {(now-train_start)/60:.1f}min", flush=True)
            last_log_t = now

        if state_step % train_cfg.eval_every_steps == 0 and eval_loader is not None:
            val_loss = _eval_loss(model, eval_loader, train_cfg)
            print(f"  [eval] step {state_step}  val_loss {val_loss:.4f}", flush=True)
            log.append({"step": state_step, "tokens": tokens_seen,
                        "train_loss": loss_acc, "val_loss": val_loss,
                        "lr": lr, "wall_sec": time.time() - train_start})

        if state_step % train_cfg.save_every_steps == 0:
            _save_checkpoint(model, optimizer, state_step, tokens_seen, log,
                             ckpt_dir / "latest")
            print(f"  [save] step {state_step}", flush=True)

    _save_checkpoint(model, optimizer, state_step, tokens_seen, log, ckpt_dir / "latest")
    print(f"final save: step={state_step} tokens={tokens_seen}")
    return {"step": state_step, "tokens": tokens_seen, "log": log}
