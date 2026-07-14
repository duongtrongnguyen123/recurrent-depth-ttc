from __future__ import annotations

import os
import sys
from pathlib import Path


def detect_kaggle() -> bool:
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None or \
           Path("/kaggle/input").exists()


def find_input_dir(name: str) -> Path:
    """Find a /kaggle/input/ subdirectory by name. Tolerates kaggle's slug substitution."""
    root = Path("/kaggle/input")
    if not root.exists():
        return Path(name)
    candidates = list(root.iterdir())
    for c in candidates:
        if c.name == name or name.replace("_", "-") in c.name or c.name in name:
            return c
    raise FileNotFoundError(f"input dataset '{name}' not found in {[c.name for c in candidates]}")


def info(*args, **kwargs):
    print(*args, **kwargs, flush=True)
