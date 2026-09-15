"""hypernix.quant.lowbit — the fixed-codebook types: INT8, INT4, INT2, INT1, FP2.

Three formats that name themselves after the width of one weight: a
block is 256 weights, one FP16 scale, and a code per weight indexing a
table that never changes. No importance weighting — at these widths
there is nothing to allocate.

The scale is searched, and that was not the original plan
---------------------------------------------------------
The first version fitted the scale to the block's largest magnitude, the
way ``Q4_0`` and ``Q8_0`` do, on the reasoning that a fixed codebook has
no budget for a search. Measured on Gaussian weights that gave:

======  ==========  =============
codec   peak-fit    17-step
======  ==========  =============
INT4         0.113          0.104
FP2          0.944          0.396
======  ==========  =============

(relative RMS error). FP2's peak fit is not merely worse, it is worse
than **one bit** — ``int1_binary`` scores 0.599 at half the size — because
with four levels and a scale pinned to a 3.5-sigma outlier, the levels
land at 1.75 and 3.5 sigma and almost every weight rounds to the larger
of two numbers that are both too big. A 2-bit format that loses to a
1-bit format is not a format.

So there is a 17-step search over shrink factors, per block, picking the
scale with the lowest squared error. It is the same shape as upstream's
``make_qx_quants`` and it is cheap here because the codebook is fixed:
the levels are sorted, so the nearest one is a :func:`numpy.searchsorted`
against their midpoints rather than an argmin over a broadcast.

What the names mean
-------------------
=======  =====  ====================  ============  =====
Name     bits   levels                block bytes   bpw
=======  =====  ====================  ============  =====
INT8         8  -128 .. 127                   258  8.062
INT4         4  -8 .. 7                       130  4.062
INT2         2  -2, -1, 0, +1                   66  2.062
FP2          2  -2, -1, +1, +2                  66  2.062
INT1         1  -1, +1                          34  1.062
=======  =====  ====================  ============  =====

The bpw column is the honest one and it is not the number in the name,
because the FP16 block scale has to live somewhere. This is the same
convention llama.cpp uses — ``Q4_0`` is 4.5 bits per weight, not 4 — and
it is stated here rather than left to be discovered from a file size.

``INT1`` is not implemented in this module. One bit per weight with a
block scale is exactly a sign-and-scale code with nothing dropped, which
is :mod:`hypernix.quant.subbit`'s ``int1_binary`` packing — the ``k == g``
case of machinery that already exists. Reimplementing it here would be a
second thing to keep correct for no gain.

FP2 is a float, not a small integer
-----------------------------------
Its four levels are ``±1`` and ``±2``: one sign bit and one exponent bit,
which is what two bits of float buys. There is no zero. A two-bit type
*with* a zero needs five levels and therefore three bits, and the version
that rounds a third of a normal distribution to zero is measurably worse
than the version that does not — so this stores the exponent instead.

INT2 is that other version, and it measured better than expected
----------------------------------------------------------------
``INT2`` is the two's-complement 2-bit range — ``-2, -1, 0, +1`` — so it
*does* round to zero, which the paragraph above predicts should cost it.
On 400k Gaussian weights over twelve seeds it does not:

======  ==========  ==========
codec   relative    weights
        RMS error   at zero
======  ==========  ==========
INT2         0.384       39.7%
FP2          0.397        0.0%
======  ==========  ==========

The prediction was made about a *symmetric* zero-inclusive codebook,
where the zero is bought by spending a level on it. INT2 does not buy it:
two's complement hands over ``-2, -1, 0, +1`` with an asymmetry already
in it, and the searched scale then shifts to suit a codebook that is not
centred. What comes out is a fourth level's worth of resolution near the
origin, where a Gaussian keeps most of its mass, in exchange for a
ceiling on the positive side — and on these weights that trade is very
slightly positive rather than negative.

Treat the 3% gap as a tie, not as a reason to switch. The reason to pick
INT2 is the second column: a zero level is one a sparse kernel can skip,
and a weight matrix that is 40% zeros is 40% of the multiply-accumulates
that never happen. FP2 remains the right default when the runtime does
dense arithmetic either way, which is most of them.

The asymmetry is kept rather than corrected for the same reason
``INT4``'s is: a signed 2-bit integer is what a kernel expecting one will
read, and a codebook that quietly shifted the range would be a different
format wearing the name.

INT8 is the wide end, and it is not Q8_0
-----------------------------------------
Both store 8-bit codes against a block scale; the difference is the block.
``Q8_0`` blocks 32 weights, so it pays a scale every 32 and lands at 8.5
bits per weight. ``INT8`` blocks 256, the same as everything else in this
module, and lands at 8.0625 — a smaller file, a coarser scale, and a
tensor whose row-length constraint matches the rest of HyperNix rather
than being the one exception. Prefer ``Q8_0`` when the file has to load
in stock llama.cpp, which will refuse this one by type id.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "BLOCK_SIZE",
    "LowBitError",
    "CODECS",
    "Codec",
    "quantize_array",
    "dequantize_array",
    "packed_block_bytes",
    "is_supported",
]

#: Weights per block, matching the K-quant family and :mod:`subbit`, so a
#: tensor that divides evenly for one divides evenly for all of them.
BLOCK_SIZE = 256

#: Shrink factors tried against the peak-fitted scale. 17 steps down to
#: a fifth of the peak: FP2's best is near 0.45 and INT4's near 0.90, so
#: the range has to cover both and the resolution has to be fine enough
#: that neither lands on an endpoint.
SEARCH_STEPS = 17
SEARCH_FLOOR = 0.20


class LowBitError(ValueError):
    """A block could not be packed or unpacked."""


@dataclass(frozen=True)
class Codec:
    """One fixed-codebook format: how many bits, and what they mean."""

    name: str
    #: Bits per weight, and so ``2 ** code_bits`` levels.
    code_bits: int
    #: The levels a code indexes, in code order. Multiplied by the
    #: block's FP16 scale to reconstruct.
    levels: tuple[float, ...]

    @property
    def payload_bytes(self) -> int:
        return BLOCK_SIZE * self.code_bits // 8

    @property
    def block_bytes(self) -> int:
        return 2 + self.payload_bytes      # FP16 scale + codes

    @property
    def bits_per_weight(self) -> float:
        return (self.block_bytes * 8) / BLOCK_SIZE

    @property
    def peak(self) -> float:
        """The largest level, which the scale search starts from."""
        return max(abs(level) for level in self.levels)

    @property
    def midpoints(self):
        """Boundaries between adjacent levels, for the nearest-level map.

        The levels are sorted, so rounding to the nearest is a binary
        search against these rather than an argmin over every level —
        which matters because the scale search evaluates the whole
        tensor once per step.
        """
        import numpy as np

        levels = np.asarray(self.levels, dtype=np.float32)
        return (levels[:-1] + levels[1:]) * 0.5


CODECS: dict[str, Codec] = {
    #: Signed 8-bit over a 256-weight block. Not Q8_0 -- same code width,
    #: eight times the block, so the file is smaller and the scale is
    #: coarser. See the module docstring.
    "INT8": Codec("INT8", 8, tuple(float(v) for v in range(-128, 128))),
    #: Signed 4-bit, the range a nibble holds. Asymmetric because two's
    #: complement is: -8 exists and +8 does not, and pretending otherwise
    #: wastes an eighth of the range on every block.
    "INT4": Codec("INT4", 4, tuple(float(v) for v in range(-8, 8))),
    #: Signed 2-bit two's complement, zero included. Worse than FP2 on
    #: reconstruction error and better than it on anything that skips
    #: zeros. See the module docstring.
    "INT2": Codec("INT2", 2, (-2.0, -1.0, 0.0, 1.0)),
    #: Sign and exponent, no mantissa, no zero. See the module docstring.
    "FP2": Codec("FP2", 2, (-2.0, -1.0, 1.0, 2.0)),
}


def is_supported(name: str) -> bool:
    return str(name).upper() in CODECS


def _codec(name: str) -> Codec:
    codec = CODECS.get(str(name).upper())
    if codec is None:
        raise LowBitError(
            f"Unknown low-bit codec {name!r}. This module writes: "
            f"{', '.join(CODECS)}"
        )
    return codec


def packed_block_bytes(name: str) -> int:
    """Bytes one 256-weight block occupies under *name*."""
    return _codec(name).block_bytes


def _pack_codes(codes, code_bits: int):
    """Codes to bytes, ``code_bits`` each, least-significant bit first.

    Done with :func:`numpy.packbits` over the expanded bits rather than
    with shifts, because a code width that does not divide 8 (there is
    none here yet, but ``code_bits`` is a parameter) makes the shift
    version quietly wrong at the block boundary.
    """
    import numpy as np

    flat = codes.reshape(codes.shape[0], -1).astype(np.uint8)
    bits = np.unpackbits(flat[..., None], axis=-1, bitorder="little")
    bits = bits[..., :code_bits].reshape(flat.shape[0], -1)
    return np.packbits(bits, axis=1, bitorder="little")


def _unpack_codes(payload, code_bits: int, count: int):
    """The inverse of :func:`_pack_codes`."""
    import numpy as np

    bits = np.unpackbits(payload, axis=1, bitorder="little")
    bits = bits[:, : count * code_bits].reshape(payload.shape[0], count, code_bits)
    weights = (1 << np.arange(code_bits, dtype=np.uint16))
    return (bits * weights).sum(axis=2)


def _encode_at(blocks, scales, codec):
    """``(codes, squared error)`` for one candidate scale per block."""
    import numpy as np

    # A zero scale means an all-zero block; every level is equally wrong,
    # so divide by one and let it round rather than dividing by zero.
    safe = np.where(scales > 0, scales, 1.0)[:, None]
    codes = np.searchsorted(codec.midpoints, blocks / safe).astype(np.uint8)
    levels = np.asarray(codec.levels, dtype=np.float32)
    error = ((levels[codes] * scales[:, None] - blocks) ** 2).sum(axis=1)
    return codes, error


def quantize_array(values, name: str) -> bytes:
    """Encode a flat sequence of floats into *name*'s blocks.

    The scale is searched, not assumed — see the module docstring for
    the measurement that made that necessary. A block of all zeros gets
    a zero scale, which reconstructs as zeros.
    """
    import numpy as np

    codec = _codec(name)
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size % BLOCK_SIZE:
        raise LowBitError(
            f"{x.size} values do not divide into {BLOCK_SIZE}-element blocks "
            f"for {codec.name}."
        )
    blocks = x.reshape(-1, BLOCK_SIZE)

    peak = np.abs(blocks).max(axis=1)
    # Guard the scale against inf/NaN reaching the file: one poisoned
    # weight would otherwise dequantise its whole block to NaN.
    peak = np.where(np.isfinite(peak), peak, 0.0) / codec.peak

    best_codes = None
    best_error = None
    best_scales = None
    for factor in np.linspace(SEARCH_FLOOR, 1.0, SEARCH_STEPS, dtype=np.float32):
        # Round-trip through FP16 inside the loop: the scale that wins
        # has to be the one that will actually be stored, or the search
        # optimises a number the file cannot hold.
        scales = (peak * factor).astype(np.float16).astype(np.float32)
        codes, error = _encode_at(blocks, scales, codec)
        if best_error is None:
            best_codes, best_error, best_scales = codes, error, scales
            continue
        take = error < best_error
        best_error = np.where(take, error, best_error)
        best_scales = np.where(take, scales, best_scales)
        best_codes = np.where(take[:, None], codes, best_codes)

    payload = _pack_codes(best_codes, codec.code_bits)
    scale_bytes = best_scales.astype(np.float16).view(np.uint8).reshape(-1, 2)
    return np.concatenate([scale_bytes, payload], axis=1).tobytes()


def dequantize_array(data: bytes, name: str):
    """Reconstruct a whole tensor as a flat float32 numpy array."""
    import numpy as np

    codec = _codec(name)
    if len(data) % codec.block_bytes:
        raise LowBitError(
            f"{len(data)} bytes is not a whole number of "
            f"{codec.block_bytes}-byte {codec.name} blocks"
        )
    blocks = np.frombuffer(data, dtype=np.uint8).reshape(-1, codec.block_bytes)
    if not blocks.size:
        return np.zeros(0, dtype=np.float32)

    # .copy() because frombuffer is read-only and .view needs an
    # alignment it cannot assume of someone else's bytes.
    scales = blocks[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)
    codes = _unpack_codes(
        np.ascontiguousarray(blocks[:, 2:]), codec.code_bits, BLOCK_SIZE
    )

    levels = np.asarray(codec.levels, dtype=np.float32)
    return (levels[codes] * scales[:, None]).reshape(-1)
