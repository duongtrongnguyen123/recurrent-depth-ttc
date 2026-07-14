"""Single entry point for Kaggle notebook. Auto-detects Kaggle paths, picks config,
launches training, saves results to /kaggle/working/.

Usage in notebook (after sys.path append):
    from pretrain.kaggle_run import run
    run(arch="looped_aux")
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from .train import Paths, train


# Possible names for the input datasets — we tolerate kaggle's slug renaming.
DATA_NAME_HINTS = ("reasoning-mvp", "reasoning-mix", "fineweb-edu-tokens", "fineweb_edu_tokens", "pretrain-data",
                   "pretrain_data", "fwe-tokens", "tokens",
                   "pretrain-50b", "pretrain_50b", "pretrain-50b-canonical")
CKPT_NAME_HINTS = ("pretrain-ckpt", "pretrain_ckpt", "ckpt-rolling",
                   "ckpt_rolling", "checkpoint", "pretrain-checkpoint",
                   # Match kernel-source slug patterns for chained sessions (s_i -> s_{i+1}).
                   # When deploy_kernel.py passes --kernel-source <user>/pretrain-<arch>-<scale>-s<i>,
                   # Kaggle mounts it at /kaggle/input/notebooks/<user>/pretrain-<arch>-...
                   # We match these slug prefixes so find_input() can locate the ckpt.
                   "pretrain-vanilla", "pretrain-xloop", "pretrain-pcc",
                   "pretrain-hr", "pretrain-hybrid", "pretrain-looped",
                   "pretrain-mol", "pretrain-two-stream", "pretrain-lti",
                   "pretrain-iter", "pretrain-aux", "pretrain-skip",
                   "pretrain-r9", "pretrain-r10", "pretrain-r11", "pretrain-r12")
CODE_NAME_HINTS = ("pretrain-code", "pretrain_code")


def find_input(hints: tuple[str, ...]) -> Path | None:
    """Look up Kaggle dataset by slug-name match.
    Restricted to top-level dirs (depth 1) so substring matches don't recurse
    into code-dataset subdirs (e.g. src/pretrain matching the 'pretrain' hint).
    Strict h-in-name only (not the reverse direction)."""
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    norm_hints = [h.lower().replace("-", "_") for h in hints]

    def matches(name: str) -> bool:
        n = name.lower().replace("-", "_")
        return any(h in n for h in norm_hints)

    candidates: list[Path] = []
    # Top-level dataset slugs only.
    for d in root.iterdir():
        if d.is_dir() and matches(d.name):
            candidates.append(d)
    # Modern Kaggle nests under /kaggle/input/datasets/<owner>/<slug>/.
    # Walk one extra level to find the slug-named dir.
    nested = root / "datasets"
    if nested.is_dir():
        for owner in nested.iterdir():
            if not owner.is_dir():
                continue
            if matches(owner.name):
                candidates.append(owner)
            for slug_d in owner.iterdir():
                if slug_d.is_dir() and matches(slug_d.name):
                    candidates.append(slug_d)
    # Kernel sources mount under /kaggle/input/notebooks/<user>/<slug>/.
    # When deploy_kernel.py passes --kernel-source <user>/<slug>, the previous
    # session's working dir (containing ckpt/) ends up here. Walk into it so
    # we can resume chained training (s_i -> s_{i+1}).
    nested_nb = root / "notebooks"
    if nested_nb.is_dir():
        for owner in nested_nb.iterdir():
            if not owner.is_dir():
                continue
            for slug_d in owner.iterdir():
                if slug_d.is_dir() and matches(slug_d.name):
                    candidates.append(slug_d)
    candidates.sort(key=lambda p: (len(p.parts), p.name))
    return candidates[0] if candidates else None


def run(arch: str = "vanilla", target_tokens: int | None = None) -> None:
    """Main entry. Discovers inputs, sets up paths, runs training."""
    print(f"=== pretrain.kaggle_run.run(arch={arch!r}) ===", flush=True)
    print(f"python: {sys.version}", flush=True)
    print(f"pwd: {os.getcwd()}", flush=True)

    try:
        import torch  # noqa: F401
        print(f"torch: {torch.__version__}  cuda: {torch.cuda.is_available()}",
              flush=True)
        if torch.cuda.is_available():
            print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    except Exception:
        traceback.print_exc()
        raise

    # Inputs.
    # Multi-source path: if `prep-*` kernel-sources are mounted, gather their
    # train_*.bin files into a single symlink farm so the existing loader sees
    # all shards across all sources. Falls back to single-dataset find_input
    # for legacy single-source training.
    nb_root = Path("/kaggle/input/notebooks")
    prep_dirs: list[Path] = []
    if nb_root.exists():
        for owner in nb_root.iterdir():
            if not owner.is_dir():
                continue
            for slug_d in owner.iterdir():
                if slug_d.is_dir() and slug_d.name.startswith("prep-"):
                    prep_dirs.append(slug_d)

    work_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./work")
    work_dir = work_dir / arch
    work_dir.mkdir(parents=True, exist_ok=True)

    if prep_dirs:
        farm = work_dir / "_data"
        farm.mkdir(exist_ok=True)
        n_links = 0
        val_src: Path | None = None
        for pd in sorted(prep_dirs):
            tag = pd.parent.name + "_" + pd.name  # owner_slug
            for shard in sorted(pd.glob("train_*.bin")):
                link = farm / f"train_{tag}_{shard.name[len('train_'):]}"
                if not link.exists():
                    try: link.symlink_to(shard)
                    except OSError:
                        link.hardlink_to(shard) if hasattr(link, 'hardlink_to') else None
                n_links += 1
            if val_src is None and (pd / "val.bin").exists():
                val_src = pd / "val.bin"
        data_dir = farm
        has_val = val_src is not None
        if has_val and not (farm / "val.bin").exists():
            try: (farm / "val.bin").symlink_to(val_src)
            except OSError: pass
        print(f"[multi-source] mounted {len(prep_dirs)} prep dirs, "
              f"{n_links} shards linked into {farm}", flush=True)
    else:
        data_dir = find_input(DATA_NAME_HINTS)
        if data_dir is None:
            avail = [p.name for p in Path("/kaggle/input").iterdir()] if Path("/kaggle/input").exists() else []
            raise FileNotFoundError(f"data dataset not found. /kaggle/input/: {avail}")
        has_val = (data_dir / "val.bin").exists()
    print(f"data_dir: {data_dir}", flush=True)
    print(f"val.bin present: {has_val}", flush=True)

    # Resume from previous session if a checkpoint dataset is mounted.
    ckpt_input = find_input(CKPT_NAME_HINTS)
    resume_dir = ""
    if ckpt_input is not None:
        # Look for latest/ subdirectory that has state.pt.
        # Path patterns observed in practice:
        #   <ckpt_input>/<arch>/ckpt/latest/state.pt   (kernel-source, s_i ran with this arch)
        #   <ckpt_input>/<arch>/latest/state.pt        (legacy structure)
        #   <ckpt_input>/ckpt/latest/state.pt          (kernel-source, single-arch)
        #   <ckpt_input>/latest/state.pt
        #   <ckpt_input>/state.pt                      (root-level)
        candidates = [ckpt_input / arch / "ckpt" / "latest",
                      ckpt_input / arch / "latest",
                      ckpt_input / "ckpt" / "latest",
                      ckpt_input / "latest",
                      ckpt_input]
        # Arch-agnostic fallback: a config/arch RENAME (e.g. base ->
        # *_t6h5) must NOT orphan a valid checkpoint written by the prior
        # stage under its own arch-named dir (this stalled the 1B chain at
        # s5). Discover any */state.pt in the mount; newest by mtime wins.
        # Appended AFTER the explicit arch-matched paths so an exact match
        # is still preferred when present.
        try:
            globbed = sorted(
                {p.parent for p in ckpt_input.rglob("state.pt")
                 if p.is_file() and p.stat().st_size > 0},
                key=lambda d: (d / "state.pt").stat().st_mtime, reverse=True)
            for g in globbed:
                if g not in candidates:
                    candidates.append(g)
        except Exception as _e:
            print(f"[resume] glob fallback skipped: {_e}", flush=True)
        print(f"ckpt_input found: {ckpt_input}", flush=True)
        print(f"ckpt resume candidates:", flush=True)
        for c in candidates:
            sp = c / "state.pt"
            sz = sp.stat().st_size if sp.exists() else 0
            print(f"  {c}/state.pt  exists={sp.exists()}  size={sz}", flush=True)
            if sp.exists() and sz > 0 and not resume_dir:
                resume_dir = str(c)
        if resume_dir:
            print(f"==> resuming from: {resume_dir}", flush=True)
        else:
            # CRITICAL: a kernel-source was mounted (deploy intent was to chain),
            # but no valid state.pt found. Refuse to silently fresh-start —
            # that has historically wasted multiple sessions of compute.
            # If you genuinely want a fresh run, deploy without --kernel-source.
            avail = []
            for owner in (Path("/kaggle/input/notebooks").iterdir()
                          if Path("/kaggle/input/notebooks").exists() else []):
                for slug_d in owner.iterdir() if owner.is_dir() else []:
                    avail.append(str(slug_d))
            raise FileNotFoundError(
                f"ckpt_input mounted at {ckpt_input} but no valid state.pt found "
                f"in any of: {[str(c) for c in candidates]}.\n"
                f"This usually means the prior session's checkpoint never saved "
                f"or the slug doesn't match. Investigate before re-deploying.\n"
                f"Available kernel-source mounts: {avail}"
            )

    work_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./work")
    work_dir = work_dir / arch
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build config.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "configs"))
    from architectures import get
    arch_cfg, train_cfg = get(arch)
    if target_tokens is not None:
        train_cfg.target_tokens = target_tokens

    # Eval data uses val.bin, not train shards. Trick: build a separate dataset path.
    eval_data_dir = ""
    if has_val:
        # Make a small directory that exposes val as a "shard".
        eval_alias = work_dir / "_eval_alias"
        eval_alias.mkdir(exist_ok=True)
        target = eval_alias / "train_0000.bin"
        if not target.exists():
            try:
                target.symlink_to(data_dir / "val.bin")
            except OSError:
                # symlink may not be allowed; copy a small slice instead
                import shutil
                shutil.copyfile(data_dir / "val.bin", target)
        eval_data_dir = str(eval_alias)

    paths = Paths(
        data_dir=str(data_dir),
        work_dir=str(work_dir),
        resume_dir=resume_dir,
        eval_data_dir=eval_data_dir,
    )
    print("paths:", json.dumps(asdict(paths), indent=2), flush=True)
    print("arch_cfg:", json.dumps(asdict(arch_cfg), indent=2), flush=True)

    # Save the launch config to working/ so it's in the notebook output.
    (work_dir / "launch.json").write_text(json.dumps({
        "arch": arch,
        "arch_cfg": asdict(arch_cfg),
        "train_cfg": asdict(train_cfg),
        "paths": asdict(paths),
    }, indent=2))

    # Run.
    try:
        result = train(arch_cfg, train_cfg, paths)
        (work_dir / "result.json").write_text(json.dumps({
            "step": result["step"],
            "tokens": result["tokens"],
            "log_tail": result["log"][-10:],
        }, indent=2))
        print("=== training complete ===", flush=True)
    except Exception:
        traceback.print_exc()
        # Even on error, drop a marker so monitoring can detect it.
        (work_dir / "ERROR.txt").write_text(traceback.format_exc())
        raise
