"""Memory-mapped binary token loader, nanoGPT pattern.

Format: data files are uint16/uint32 packed token ids in a single .bin per shard.
Sampling is uniform-random offsets, no formal epoch structure (acceptable at our scale).

For reasoning-mix-mvp shards, the WITHIN-shard layout is contiguous:
    [ math_tokens | synth_tokens | prose_tokens ]
where (math, synth, prose) = (100M, 60M, 40M) per 200M-token shard at the
default 50/30/20 mix. Because random-offset sampling weights by region size,
the effective batch mix matches the layout ratio. To CHANGE the mix without
re-creating shards, use ``RegionalShardLoader`` with custom region weights.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch


class BinShardLoader:
    def __init__(self, paths: list[str | Path], dtype: str = "uint32",
                 block_size: int = 2048):
        self.dtype = np.uint16 if dtype == "uint16" else np.uint32
        self.block_size = block_size
        self.shards: list[np.memmap] = []
        for p in paths:
            arr = np.memmap(p, dtype=self.dtype, mode="r")
            self.shards.append(arr)
        # Sample shards proportionally to size.
        self.lens = np.array([len(s) for s in self.shards], dtype=np.int64)
        self.weights = self.lens / self.lens.sum()
        self.total_tokens = int(self.lens.sum())

    def get_batch(self, batch_size: int, device: str = "cuda",
                  rng: np.random.Generator | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        rng = rng or np.random.default_rng()
        # Sample one shard per example.
        shard_idx = rng.choice(len(self.shards), size=batch_size, p=self.weights)
        x = np.empty((batch_size, self.block_size), dtype=np.int64)
        y = np.empty((batch_size, self.block_size), dtype=np.int64)
        for i, s in enumerate(shard_idx):
            arr = self.shards[s]
            start = int(rng.integers(0, len(arr) - self.block_size - 1))
            chunk = arr[start: start + self.block_size + 1].astype(np.int64)
            x[i] = chunk[:-1]
            y[i] = chunk[1:]
        return (torch.from_numpy(x).to(device, non_blocking=True),
                torch.from_numpy(y).to(device, non_blocking=True))


class RegionalShardLoader:
    """Like BinShardLoader, but re-weights sampling across within-shard regions.

    Each shard has a contiguous regional layout:
        [region_0_tokens | region_1_tokens | ... | region_R-1_tokens]
    of cumulative lengths defined by ``region_offsets`` (length R+1, fractions
    of the per-shard token count). For 50/30/20 the default layout fractions
    are [0.0, 0.5, 0.8, 1.0] (math 0-50%, synth 50-80%, prose 80-100%).

    ``region_weights`` controls how often each region is sampled. To convert
    to a 30/50/20 mix (synth-dominant): pass [0.3, 0.5, 0.2].
    """

    def __init__(self, paths: list[str | Path], dtype: str = "uint32",
                 block_size: int = 2048,
                 region_offsets: list[float] = (0.0, 0.5, 0.8, 1.0),
                 region_weights: list[float] = (0.5, 0.3, 0.2)):
        self.dtype = np.uint16 if dtype == "uint16" else np.uint32
        self.block_size = block_size
        self.shards: list[np.memmap] = []
        for p in paths:
            arr = np.memmap(p, dtype=self.dtype, mode="r")
            self.shards.append(arr)
        self.lens = np.array([len(s) for s in self.shards], dtype=np.int64)
        # Per-shard sampling: weight by shard size (so larger shards seen more).
        self.shard_weights = self.lens / self.lens.sum()
        self.total_tokens = int(self.lens.sum())
        # Region fractions: cumulative offset boundaries within each shard.
        ro = np.asarray(region_offsets, dtype=np.float64)
        assert ro[0] == 0.0 and ro[-1] == 1.0, "region_offsets must start at 0.0 and end at 1.0"
        assert (np.diff(ro) > 0).all(), "region_offsets must be strictly increasing"
        rw = np.asarray(region_weights, dtype=np.float64)
        assert len(rw) == len(ro) - 1, "region_weights length must equal n_regions"
        rw = rw / rw.sum()
        self.region_offsets = ro
        self.region_weights = rw

    def _region_bounds_in_shard(self, shard_len: int) -> list[tuple[int, int]]:
        """Return [(lo, hi), ...] absolute offset bounds for each region in
        a shard of size ``shard_len``. The hi bound excludes the last
        ``block_size`` tokens to keep chunks within the region."""
        bounds: list[tuple[int, int]] = []
        for i in range(len(self.region_offsets) - 1):
            lo = int(self.region_offsets[i] * shard_len)
            hi = int(self.region_offsets[i + 1] * shard_len) - self.block_size - 1
            if hi <= lo:
                hi = lo + 1  # tiny region; still sample at the start
            bounds.append((lo, hi))
        return bounds

    def get_batch(self, batch_size: int, device: str = "cuda",
                  rng: np.random.Generator | None = None
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        rng = rng or np.random.default_rng()
        shard_idx = rng.choice(len(self.shards), size=batch_size, p=self.shard_weights)
        region_idx = rng.choice(len(self.region_weights), size=batch_size,
                                p=self.region_weights)
        x = np.empty((batch_size, self.block_size), dtype=np.int64)
        y = np.empty((batch_size, self.block_size), dtype=np.int64)
        for i, (s, r) in enumerate(zip(shard_idx, region_idx)):
            arr = self.shards[s]
            bounds = self._region_bounds_in_shard(len(arr))
            lo, hi = bounds[r]
            start = int(rng.integers(lo, max(hi, lo + 1)))
            chunk = arr[start: start + self.block_size + 1].astype(np.int64)
            x[i] = chunk[:-1]
            y[i] = chunk[1:]
        return (torch.from_numpy(x).to(device, non_blocking=True),
                torch.from_numpy(y).to(device, non_blocking=True))


def discover_shards(root: str | Path, glob: str = "train_*.bin") -> list[Path]:
    return sorted(Path(root).glob(glob))
