"""hypernix.models.hnxtorch — decode the extension types on the accelerator.

The CPU decoders in :mod:`hypernix.quant.subbit` and
:mod:`hypernix.quant.lowbit` are numpy, and they are tuned for it — a
byte-to-signs lookup table, one gather, one in-place scale. On a GPU they
are the wrong shape entirely, and not because numpy is slow.

Why the obvious port is backwards
---------------------------------
The first version of CUDA support here was one line: decode with numpy as
before, then ``.to("cuda")`` on each chunk. That runs, and it is the
worst arrangement available. Every forward pass decodes on the CPU and
pushes **expanded float32** across PCIe — for a 0.9-bit tensor that is
34 times the bytes the tensor actually occupies, moved every token, to
save nothing.

The packed form is the small one. That is the entire premise of the
tier. So the packed bytes go to the device **once**, at load, and the
decode happens there: PCIe carries 0.9 bits per weight one time instead
of 32 bits per weight per token.

    7B at IQ0.9_L:  ~800 MB uploaded once
    the same model dequantised:  ~28 GB, per forward pass

Which is also why this is not just a speed question. Holding the packed
bytes in VRAM is what lets a 7B sub-bit model fit on a card that could
not hold its float16 weights at all.

What is here
------------
Bit-exact torch equivalents of the two decoders, in the ops that exist on
every backend torch supports — shifts, masks, gathers, reshapes. No
custom kernels and nothing CUDA-specific, so the same code runs on CUDA,
ROCm, MPS and XPU, and on CPU where it is used to test itself against
numpy.

The tests assert equality with the numpy decoders rather than closeness.
These are integer unpacking followed by one multiply; if the two ever
disagree it is a bug in one of them, not rounding.
"""
from __future__ import annotations

from typing import Any

__all__ = [
    "to_device_bytes",
    "stored_signs",
    "dequantize_subbit",
    "dequantize_lowbit",
    "decode",
    "supports",
]

#: Types decoded here, by GGML id. Kept as a lookup rather than a range
#: so an id added to one family and not the other cannot silently fall
#: into the wrong decoder.
_SUBBIT = {200: "sign_scale_l", 201: "pair_code_m", 202: "quad_code_xxxl",
           203: "quarter_code_uxl", 204: "int1_binary"}
_LOWBIT = {205: "INT4", 206: "FP2"}


def supports(ggml_type: int) -> bool:
    kind = int(ggml_type)
    return kind in _SUBBIT or kind in _LOWBIT


def to_device_bytes(raw: bytes, device: Any):
    """Packed bytes as a ``uint8`` tensor on *device*.

    ``frombuffer`` rather than ``tensor(list(raw))``: the second copies
    through a Python list of ints, which for a 7B model is several
    hundred million objects and takes minutes.
    """
    import numpy as np
    import torch

    array = np.frombuffer(raw, dtype=np.uint8)
    # .copy() because frombuffer is read-only and torch refuses to wrap a
    # read-only buffer without warning about it on every call.
    return torch.from_numpy(array.copy()).to(device)


def _scales(blocks):
    """The leading FP16 scale of each block, as float32.

    ``.view(torch.float16)`` reinterprets the two bytes rather than
    converting them, which is what the format stores. The
    ``.contiguous()`` is required: ``blocks[:, :2]`` is a stride into a
    wider row and torch refuses to reinterpret a non-contiguous tensor,
    with an error that does not mention slicing.
    """
    import torch

    return blocks[:, :2].contiguous().view(torch.float16).reshape(-1).to(torch.float32)


def _unpack(payload, width: int, count: int):
    """``count`` little-endian ``width``-bit codes per row of *payload*.

    The bit order matches :func:`numpy.unpackbits` with
    ``bitorder="little"``, which is what both packers emit. Getting it
    backwards produces a model that loads, runs, and is not the model in
    the file — so this is asserted against numpy rather than reasoned
    about.
    """
    import torch

    shifts = torch.arange(8, device=payload.device, dtype=torch.uint8)
    bits = (payload.unsqueeze(-1) >> shifts) & 1          # (rows, bytes, 8)
    bits = bits.reshape(payload.shape[0], -1)             # (rows, bytes * 8)
    if width == 1:
        return bits[:, :count].to(torch.int64)
    bits = bits[:, : count * width].reshape(payload.shape[0], count, width)
    weights = (1 << torch.arange(
        width, device=payload.device, dtype=torch.int64
    ))
    return (bits.to(torch.int64) * weights).sum(dim=2)


def stored_signs(packed, packing: str):
    """The stored signs, scaled, without expanding the dropped ones.

    Shape ``(blocks, codes_per_block, kept)`` — the same array
    :func:`hypernix.quant.subbit.stored_signs` returns, on whatever
    device *packed* lives on, so the folded matmul in
    :mod:`hypernix.models.hnxrun` works unchanged.
    """
    import torch

    from ..quant.subbit import PACKINGS

    spec = PACKINGS[packing]
    blocks = packed.reshape(-1, spec.block_bytes)
    if blocks.numel() == 0:
        return torch.zeros(
            (0, spec.codes_per_block, spec.kept),
            dtype=torch.float32, device=packed.device,
        )
    scales = _scales(blocks)
    stored = spec.codes_per_block * spec.kept
    bits = _unpack(blocks[:, 2:].contiguous(), 1, stored)
    head = bits.reshape(-1, spec.codes_per_block, spec.kept).to(torch.float32)
    if spec.has_sub_magnitude:
        # Sixteen magnitudes per block, not one. Same shape out, but the
        # scale varies along the block -- which is the whole reason the
        # tier exists, and the reason this branch cannot be folded into
        # the line below.
        signs = head.reshape(-1, spec.sub_blocks, spec.sub_size) * 2.0 - 1.0
        magnitudes = (scales[:, None] * _sub_indices(blocks, spec)) / spec.levels
        return (signs * magnitudes[:, :, None]).reshape(
            -1, spec.codes_per_block, spec.kept
        )
    # (2b - 1) * s, folded so the sign map and the scale are one pass.
    return head * (scales * 2.0)[:, None, None] - scales[:, None, None]


def _sub_indices(blocks, spec):
    """Each sub-block's magnitude index, on *blocks*' device.

    Mirrors ``subbit._sub_indices``: the signs are one bit per weight so
    they end on a byte boundary, and the indices start there.
    """
    first = 2 + (spec.codes_per_block * spec.code_bits) // 8
    raw = _unpack(
        blocks[:, first:].contiguous(), 1, spec.magnitude_bits
    ).reshape(-1, spec.sub_blocks, spec.sub_bits)
    import torch

    powers = (1 << torch.arange(spec.sub_bits, device=blocks.device)).to(torch.float32)
    return (raw.to(torch.float32) * powers).sum(dim=2)


def dequantize_subbit(packed, packing: str):
    """A whole sign-and-scale tensor, flat, on *packed*'s device."""
    import torch

    from ..quant.subbit import PACKINGS

    spec = PACKINGS[packing]
    head = stored_signs(packed, packing)
    if head.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=packed.device)
    # Every dropped position repeats its group's last stored sign, so the
    # expansion is one repeat_interleave rather than a gather.
    counts = torch.tensor(
        [1] * (spec.kept - 1) + [1 + spec.group - spec.kept],
        device=packed.device,
    )
    return torch.repeat_interleave(head, counts, dim=2).reshape(-1)


def dequantize_lowbit(packed, codec_name: str):
    """A whole fixed-codebook tensor, flat, on *packed*'s device."""
    import torch

    from ..quant.lowbit import BLOCK_SIZE, CODECS

    codec = CODECS[codec_name]
    blocks = packed.reshape(-1, codec.block_bytes)
    if blocks.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=packed.device)
    scales = _scales(blocks)
    codes = _unpack(blocks[:, 2:].contiguous(), codec.code_bits, BLOCK_SIZE)
    levels = torch.tensor(
        codec.levels, dtype=torch.float32, device=packed.device
    )
    return (levels[codes] * scales[:, None]).reshape(-1)


def decode(packed, ggml_type: int):
    """Dispatch to the right decoder for *ggml_type*."""
    kind = int(ggml_type)
    if kind in _SUBBIT:
        return dequantize_subbit(packed, _SUBBIT[kind])
    if kind in _LOWBIT:
        return dequantize_lowbit(packed, _LOWBIT[kind])
    raise ValueError(
        f"GGML type {kind} is not one hnxtorch decodes. It handles the "
        f"HyperNix extension types; everything else goes through the numpy "
        f"path in hypernix.models.hnxrun."
    )
