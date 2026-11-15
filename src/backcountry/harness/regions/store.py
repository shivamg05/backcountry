"""Boundary store: saved region-edge tensors on disk, keyed by
region fingerprint, workload, and input-set index. Safetensors keys are
"a<array_id>" against the priced trace's ids; workers stream sets from disk so
nothing large stays resident between evaluations.

A region's weights are the same in every set and at every size, so they are
saved once, in the region's weights file under their model paths, and each
set names the ones it uses (metadata "weights": array id -> path). Saved with
every set, Qwen3-30B-A3B's expert weights were 88 of every 94 MB.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping
from urllib.parse import quote
import json
import os
import struct

import mlx.core as mx

from backcountry_runtime.exact import bitwise_equal


WEIGHTS_FILE = "weights.safetensors"  # a region's weights, by model path, beside its labels


class BoundaryMismatch(RuntimeError):
    """Saved evaluation data does not match the recorded region boundary."""


class BoundaryStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, fingerprint: str, workload: str, set_idx: int, kind: str) -> Path:
        # Labels are metadata, not paths. Keep distinct labels distinct even
        # when they contain slashes, percent signs or traversal components.
        label = quote(workload, safe="")
        if label in {".", ".."}:
            label = "%2E" * len(label)
        return self.root / fingerprint / label / f"set{set_idx}.{kind}.safetensors"

    def save(self, fingerprint: str, workload: str, set_idx: int, kind: str,
             arrays: dict[int, mx.array], weight_paths: Mapping[int, str] | None = None) -> Path:
        """kind is 'inputs' or 'outputs' (the library's reference values).
        weight_paths (array id -> model path) marks the arrays that are model
        weights: they go to the region's weights file once."""
        path = self._path(fingerprint, workload, set_idx, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        mx.eval(list(arrays.values()))
        weights = {aid: weight_paths[aid] for aid in arrays if weight_paths and aid in weight_paths}
        if weights:
            # shared only when bitwise what is already there under that path:
            # a state array reachable from the model can differ between passes
            shared = path.parent.parent / WEIGHTS_FILE
            have = mx.load(str(shared)) if shared.is_file() else {}
            weights = {aid: p for aid, p in weights.items() if p not in have or bitwise_equal(have[p], arrays[aid])}
            new = {p: arrays[aid] for aid, p in weights.items() if p not in have}
            if new:
                # have is read lazily from shared: write beside it, then swap
                part = shared.with_name(f"{shared.stem}.part.safetensors")
                mx.save_safetensors(str(part), {**have, **new})
                os.replace(part, shared)
        mx.save_safetensors(str(path), {f"a{aid}": arr for aid, arr in arrays.items() if aid not in weights},
                            metadata={"weights": json.dumps({str(aid): p for aid, p in weights.items()})}
                            if weights else None)
        return path

    def load(self, fingerprint: str, workload: str, set_idx: int, kind: str) -> dict[int, mx.array]:
        return self.load_sets(fingerprint, workload, 1, kind, first=set_idx)[0]

    def load_sets(self, fingerprint: str, workload: str, count: int, kind: str = "inputs",
                  first: int = 0) -> list[dict[int, mx.array]]:
        """count sets, sharing one copy of the region's large weights that
        only they keep alive: the job holds a 15 GB model within 4 GB of the
        GPU's limit on Qwen3.8-27B, so nothing may stay cached behind them."""
        try:
            return [load_set(self._path(fingerprint, workload, first + i, kind)) for i in range(count)]
        finally:
            _shared_weights.cache_clear()

    def set_count(self, fingerprint: str, workload: str) -> int:
        d = self._path(fingerprint, workload, 0, "inputs").parent
        if not d.exists():
            return 0
        return len(list(d.glob("set*.inputs.safetensors")))

    def validate(self, fingerprint: str, workload: str, set_idx: int, kind: str,
                 expected_ids: tuple[int, ...]) -> None:
        """Check the safetensors header without loading weights or using the GPU."""
        path = self._path(fingerprint, workload, set_idx, kind)
        try:
            size = path.stat().st_size
            with path.open("rb") as f:
                length, = struct.unpack("<Q", f.read(8))
                if length > size - 8:
                    raise ValueError("truncated tensor header")
                header = json.loads(f.read(length))
            expected = {f"a{a}" for a in expected_ids}
            weights = json.loads((header.get("__metadata__") or {}).get("weights", "{}"))
            actual = (set(header) - {"__metadata__"}) | {f"a{a}" for a in weights}
            if weights:
                shared = path.parent.parent / WEIGHTS_FILE
                missing = set(weights.values()) - set(_header(shared)) if shared.is_file() \
                    else set(weights.values())
                if missing:
                    raise ValueError(f"the region's weights file lacks {sorted(missing)[:3]}")
            if actual != expected:
                raise ValueError(f"missing arrays {sorted(expected - actual)}; "
                                 f"unexpected arrays {sorted(actual - expected)}")
            for key in set(header) - {"__metadata__"}:
                lo, hi = header[key]["data_offsets"]
                if not 0 <= lo <= hi <= size - 8 - length:
                    raise ValueError(f"truncated or invalid tensor data for {key}")
        except (OSError, ValueError, TypeError, KeyError, struct.error) as error:
            raise BoundaryMismatch(f"Saved boundary data mismatch at {path}: {error}") from error


def _header(path: Path) -> dict:
    """A safetensors file's header: tensor names and offsets, without the data."""
    with path.open("rb") as f:
        length, = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(length))


def load_set(path: str | Path) -> dict[int, mx.array]:
    loaded, metadata = mx.load(str(path), return_metadata=True)
    arrays = {int(k[1:]): v for k, v in loaded.items()}
    weights = json.loads(metadata.get("weights", "{}"))
    if weights:
        shared = _region_weights(Path(path).parent.parent / WEIGHTS_FILE)
        arrays.update({int(aid): shared[p] for aid, p in weights.items()})
    return arrays


def _region_weights(path: Path) -> dict[str, mx.array]:
    """A region's weights. Each set gets its own copy, which the cache-cold
    clocks rotate, unless one copy already outgrows the clocks' working set:
    then every set shares one. Qwen3.8-27B's vocabulary-sized regions carry
    0.7 GB of weights; a copy per set per recorded shape put about 13 GB on
    a 24 GB Mac during pricing, and the GPU timed out waiting on swap."""
    from ..measure.clocks import CACHE_DEFEAT_BYTES
    stat = path.stat()
    if stat.st_size < CACHE_DEFEAT_BYTES:
        return mx.load(str(path))
    return _shared_weights(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=1)  # shared while one region's sets load; the job's store clears it after
def _shared_weights(path: str, _mtime_ns: int, _size: int) -> dict[str, mx.array]:
    return mx.load(path)
