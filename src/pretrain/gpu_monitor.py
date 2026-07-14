"""Background GPU sampler (util, power, vram) for in-training efficiency telemetry.

Uses pynvml (preinstalled on Kaggle GPU envs). Samples once per period in a
daemon thread so the training step pays ~0 cost. Read `.last` to get the most
recent sample tuple (util_pct, watts, vram_used_gb), or `None` if pynvml is
unavailable / sampler not started.

Wired into pretrain.train via `cfg.gpu_monitor=True` (default off so legacy
runs and CPU sanity tests don't error).
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple


class GpuSampler:
    def __init__(self, period_s: float = 5.0, device_index: int = 0):
        self.period = period_s
        self._stop = False
        self.last: Optional[Tuple[float, float, float]] = None
        self._handle = None
        self._pynvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            # One synchronous sample so .last is populated before the thread loop.
            self._sample()
            t = threading.Thread(target=self._run, daemon=True)
            t.start()
            print(f"[gpu_monitor] sampler started (period={period_s}s)", flush=True)
        except Exception as e:
            print(f"[gpu_monitor] disabled: {e}", flush=True)

    def _sample(self) -> None:
        p = self._pynvml
        h = self._handle
        try:
            util = p.nvmlDeviceGetUtilizationRates(h).gpu
            watts = p.nvmlDeviceGetPowerUsage(h) / 1000.0
            mem = p.nvmlDeviceGetMemoryInfo(h)
            self.last = (float(util), float(watts), mem.used / 1e9)
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop:
            self._sample()
            time.sleep(self.period)

    def stop(self) -> None:
        self._stop = True

    def fmt(self) -> str:
        if self.last is None:
            return "util=?  power=?  vram=?"
        u, w, v = self.last
        return f"util {u:.0f}%  power {w:.0f}W  gpu_mem {v:.1f}GB"
