"""strife.weights — the model file, memory-mapped, decoded on demand.

The GGUF is mapped read-only once. A tensor's bytes are a view into that
map, so opening a 20 GB model costs address space, not 20 GB of RAM.

Decoding to float32 is where memory goes, so it is budgeted:
``cache_bytes`` (default: 60% of free RAM) holds decoded tensors,
least recently used first out. A tensor larger than the whole budget is
never cached; the backend multiplies it a slab of rows at a time instead
(:meth:`WeightStore.matmul`), so peak RAM stays bounded whatever the
model's size. ``stats()`` reports mapped bytes, cached bytes, hits and
misses separately.

Embedding lookups decode only the rows asked for.

With the native kernels (:mod:`.kernels`) a weight is multiplied while it
is still quantised, so it is never decoded whole and never cached: RAM is
the map plus the small tensors. Tensors of a type the kernels were not
verified on take the numpy path above, unchanged.
"""
from __future__ import annotations

import mmap
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from ..quant.gguf import GGUFFile, GGUFTensor, type_block_size, type_size_bytes
from .errors import StrifeError
from .quants import decode, quant_name

__all__ = ["WeightStore", "default_cache_bytes"]

#: Rows decoded at once when a tensor is too big to cache.
_SLAB_BYTES = 64 << 20


def default_cache_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available * 0.6)
    except Exception:
        return 2 << 30


class WeightStore:
    def __init__(self, gguf: GGUFFile, *, cache_bytes: int | None = None,
                 kernels: Any = None, native_matmul: bool = False, threads: int = 0) -> None:
        if gguf.path is None:
            raise StrifeError("strife.model.unreadable", "the model has no file behind it")
        self.gguf = gguf
        self.path = Path(gguf.path)
        self._file = open(self.path, "rb")  # noqa: SIM115 - kept open for the map's lifetime
        try:
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError) as exc:
            self._file.close()
            raise StrifeError("strife.model.unreadable", f"could not map {self.path}: {exc}") from exc
        self._view = memoryview(self._map)
        self._by_name = {t.name: t for t in gguf.tensors}
        self.cache_bytes = default_cache_bytes() if cache_bytes is None else max(0, int(cache_bytes))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cached = 0
        self._lock = threading.Lock()
        self.hits = self.misses = 0
        self.peak_cached = 0
        #: The native kernels (``None``: numpy only). Used to decode any type
        #: they were verified on; for matmul only when *native_matmul* is set,
        #: which the engine does when the kernels can carry the whole model.
        self.kernels = kernels
        self.native_matmul = bool(native_matmul and kernels is not None)
        self.threads = max(0, int(threads))
        self.native_matmuls = self.native_decodes = 0
        #: Matrices of the layers offloaded to a GPU (:mod:`.gpu`), or ``None``.
        self.device: Any = None

    # -- raw access --------------------------------------------------------

    def tensor(self, name: str) -> GGUFTensor:
        t = self._by_name.get(name)
        if t is None:
            raise StrifeError("strife.model.missing_tensor", f"{name} is not in the model",
                              details={"tensor": name})
        return t

    def has(self, name: str) -> bool:
        return name in self._by_name

    def raw(self, t: GGUFTensor) -> memoryview:
        start = self.gguf.data_start + t.offset
        return self._view[start:start + t.nbytes]

    def shape2d(self, t: GGUFTensor) -> tuple[int, int]:
        """``(rows, row_length)``: GGUF's ``ne[0]`` is the row length."""
        cols = int(t.shape[0])
        rows = 1
        for d in t.shape[1:]:
            rows *= int(d)
        return rows, cols

    # -- decoded access ------------------------------------------------------

    def _decoded_nbytes(self, t: GGUFTensor) -> int:
        rows, cols = self.shape2d(t)
        return rows * cols * 4

    def _decode(self, raw: memoryview, ggml_type: int, elements: int) -> np.ndarray:
        """float32 from *raw*: natively when the type has a verified kernel."""
        if self.kernels is not None and self.kernels.has(ggml_type):
            self.native_decodes += 1
            return self.kernels.decode(raw, ggml_type, elements, self.threads)
        return decode(raw, ggml_type)

    def offload(self, device: Any, layers: int) -> int:
        """Put the matrices of blocks ``0 .. layers-1`` on *device*. Returns the bytes placed.

        Each is decoded (natively when it can be), moved, and dropped from
        host memory; the map still holds its quantised form.
        """
        from .gpu import layer_of

        self.device = device
        for t in self.gguf.tensors:
            index = layer_of(t.name)
            if index is None or index >= layers or len(t.shape) < 2:
                continue
            rows, cols = self.shape2d(t)
            device.put(t.name, self._decode(self.raw(t), t.ggml_type, rows * cols).reshape(rows, cols))
        return int(device.bytes)

    def _native(self, t: GGUFTensor) -> bool:
        return self.native_matmul and self.kernels.has(t.ggml_type)

    def get(self, name: str) -> np.ndarray:
        """The whole tensor as float32, shaped ``(rows, row_length)`` (1-D for vectors)."""
        with self._lock:
            hit = self._cache.get(name)
            if hit is not None:
                self._cache.move_to_end(name)
                self.hits += 1
                return hit
        t = self.tensor(name)
        rows, cols = self.shape2d(t)
        array = self._decode(self.raw(t), t.ggml_type, rows * cols)
        if len(t.shape) > 1:
            array = array.reshape(rows, cols)
        size = array.nbytes
        with self._lock:
            self.misses += 1
            if size <= self.cache_bytes:
                while self._cached + size > self.cache_bytes and self._cache:
                    _, old = self._cache.popitem(last=False)
                    self._cached -= old.nbytes
                self._cache[name] = array
                self._cached += size
                self.peak_cached = max(self.peak_cached, self._cached)
        return array

    def cacheable(self, name: str) -> bool:
        return self._decoded_nbytes(self.tensor(name)) <= self.cache_bytes

    def rows(self, name: str, ids: np.ndarray) -> np.ndarray:
        """Decoded rows *ids* of a 2-D tensor, decoding only those rows."""
        t = self.tensor(name)
        n_rows, cols = self.shape2d(t)
        native = self._native(t)
        if not native and self.cacheable(name):
            return self.get(name)[ids]
        per_row = cols // type_block_size(t.ggml_type) * type_size_bytes(t.ggml_type)
        raw = self.raw(t)
        out = np.empty((len(ids), cols), dtype=np.float32)
        for i, rid in enumerate(np.asarray(ids).tolist()):
            if not 0 <= rid < n_rows:
                raise StrifeError("strife.runtime.bad_token", f"token id {rid} is out of range")
            out[i] = self._decode(raw[rid * per_row:(rid + 1) * per_row], t.ggml_type, cols)
        return out

    def matmul(self, x: np.ndarray, name: str, rows: tuple[int, int] | None = None) -> np.ndarray:
        """``x @ W.T`` for weight *name* stored ``(out, in)``: ``x`` is ``(n, in)``.

        *rows* takes a range of output rows only: one expert's slice of a
        3-D expert tensor, decoded alone when the whole is not cached.
        """
        t = self.tensor(name)
        n_rows, cols = self.shape2d(t)
        first, last = rows if rows is not None else (0, n_rows)
        if not 0 <= first < last <= n_rows:
            raise StrifeError("strife.runtime.bad_rows", f"rows {first}:{last} of {name}")
        if self.device is not None and x.ndim == 2 and self.device.has(name):
            return self.device.matmul(x, name, rows)
        per_row = cols // type_block_size(t.ggml_type) * type_size_bytes(t.ggml_type)
        if self._native(t) and x.ndim == 2:
            # The weight stays quantised: nothing is decoded whole or cached.
            self.native_matmuls += 1
            raw = self.raw(t)
            return self.kernels.matmul(raw[first * per_row:last * per_row], t.ggml_type,
                                       last - first, cols, x, self.threads)
        if self.cacheable(name):
            w = self.get(name)
            return x @ (w if rows is None else w[first:last]).T
        slab = max(1, _SLAB_BYTES // max(1, cols * 4))
        raw = self.raw(t)
        out = np.empty((x.shape[0], last - first), dtype=np.float32)
        for start in range(first, last, slab):
            stop = min(last, start + slab)
            block = self._decode(raw[start * per_row:stop * per_row], t.ggml_type,
                                 (stop - start) * cols).reshape(-1, cols)
            out[:, start - first:stop - first] = x @ block.T
        return out

    # -- reporting ---------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {"mapped_bytes": len(self._map), "cache_limit_bytes": self.cache_bytes,
                "cached_bytes": self._cached, "peak_cached_bytes": self.peak_cached,
                "cache_hits": self.hits, "cache_misses": self.misses,
                "tensors_cached": len(self._cache),
                "native_matmul": self.native_matmul, "native_matmuls": self.native_matmuls,
                "native_decodes": self.native_decodes,
                "device": self.device.stats() if self.device is not None else None}

    def describe(self, name: str) -> dict[str, Any]:
        t = self.tensor(name)
        return {"name": name, "shape": list(t.shape), "type": quant_name(t.ggml_type),
                "bytes": t.nbytes}

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
            self._cached = 0
        if self.device is not None:
            self.device.close()
            self.device = None
        try:
            self._view.release()
            self._map.close()
        except (BufferError, ValueError):
            pass
        self._file.close()
