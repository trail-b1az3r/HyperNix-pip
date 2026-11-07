"""strife.quants — the tensor types Strife can read, and how.

A type is supported only when it has a decoder that is tested against
its own encoder elsewhere in this package:

* F32, F16, BF16 -- read directly;
* the llama.cpp block types Q4_0 … Q8_0 and the K-quants Q2_K … Q6_K --
  :mod:`hypernix.quant.llamaquants`, bit-compatible with llama.cpp;
* the HyperNix sub-bit and low-bit types (type ids 200+) --
  :mod:`hypernix.quant.subbit` and :mod:`hypernix.quant.lowbit`, through
  the tier table in :mod:`hypernix.quant.hyprslug`, which is the one
  place a type id is tied to a packing.

Anything else -- the IQ types, Q8_1, a type id nobody allocated -- is
refused by name. It is never decoded as a neighbouring type: reading a
0.5-bit tensor as Q4_K produces confident noise, which is worse than an
error.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np

from ..quant.gguf import GGMLType

__all__ = ["decode", "quant_name", "supported_type", "supported_types", "row_bytes"]

_Decoder = Callable[[bytes], np.ndarray]


def _plain(dtype: str) -> _Decoder:
    def decode(raw: bytes) -> np.ndarray:
        return np.frombuffer(raw, dtype=dtype).astype(np.float32)
    return decode


def _bf16(raw: bytes) -> np.ndarray:
    halves = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return halves.view(np.float32)


def _llama(raw: bytes, *, ggml_type: int) -> np.ndarray:
    from ..quant import llamaquants

    return llamaquants.dequantize_array(raw, ggml_type)


def _subbit(raw: bytes, *, packing: str) -> np.ndarray:
    from ..quant import subbit

    return np.asarray(subbit.dequantize_array(raw, packing), dtype=np.float32)


def _lowbit(raw: bytes, *, packing: str) -> np.ndarray:
    from ..quant import lowbit

    return np.asarray(lowbit.dequantize_array(raw, packing), dtype=np.float32)


def _build() -> dict[int, tuple[str, _Decoder]]:
    from ..quant import llamaquants

    table: dict[int, tuple[str, _Decoder]] = {
        int(GGMLType.F32): ("F32", _plain("<f4")),
        int(GGMLType.F16): ("F16", _plain("<f2")),
        int(GGMLType.BF16): ("BF16", _bf16),
    }
    for fmt in llamaquants.FORMATS.values():
        if fmt.name == "Q8_K":  # an activation format, never a stored weight
            continue
        table[fmt.ggml_type] = (fmt.name, partial(_llama, ggml_type=fmt.ggml_type))
    try:
        from ..quant import lowbit, subbit
        from ..quant.hyprslug import TIER_TYPES
    except ImportError:  # pragma: no cover - the quant package is part of hypernix
        return table
    for tier, (type_id, packing) in TIER_TYPES.items():
        if packing in subbit.PACKINGS:
            table[type_id] = (_hnx(tier), partial(_subbit, packing=packing))
        elif lowbit.is_supported(packing):
            table[type_id] = (_hnx(tier), partial(_lowbit, packing=packing))
    return table


def _hnx(tier: str) -> str:
    return tier if tier.startswith("HNX_") else f"HNX_{tier}"


_TABLE: dict[int, tuple[str, _Decoder]] | None = None


def _table() -> dict[int, tuple[str, _Decoder]]:
    global _TABLE
    if _TABLE is None:
        _TABLE = _build()
    return _TABLE


def supported_type(ggml_type: int) -> bool:
    return int(ggml_type) in _table()


def supported_types() -> dict[int, str]:
    return {k: v[0] for k, v in sorted(_table().items())}


def quant_name(ggml_type: int) -> str:
    entry = _table().get(int(ggml_type))
    if entry:
        return entry[0]
    try:
        return GGMLType(int(ggml_type)).name
    except ValueError:
        return f"type{int(ggml_type)}"


def decode(raw: bytes | memoryview, ggml_type: int) -> np.ndarray:
    """Decode a whole number of blocks of *ggml_type* to float32."""
    entry = _table().get(int(ggml_type))
    if entry is None:
        from .errors import StrifeError

        raise StrifeError("strife.quant.unsupported",
                          f"{quant_name(ggml_type)} (type {ggml_type}) has no verified decoder",
                          details={"ggml_type": int(ggml_type)})
    return np.asarray(entry[1](bytes(raw)), dtype=np.float32)


def row_bytes(ggml_type: int, row_elements: int) -> int:
    from ..quant.gguf import type_block_size, type_size_bytes

    return row_elements // type_block_size(ggml_type) * type_size_bytes(ggml_type)
