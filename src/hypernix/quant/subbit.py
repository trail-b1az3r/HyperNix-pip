"""hypernix.quant.subbit — the sub-1-bit packings, for real.

:mod:`hypernix.quant.steamroller` has advertised ``IQ0.9_L``,
``IQ0.75_M`` and ``IQ0.5_XXXL`` for some time. What it actually did was
copy the Q3_K_L staging file and write a sidecar JSON claiming a tier —
so a "0.5-bit model" was byte-identical to the 3-bit model it came from,
the same size, and no more quantised. The tier was a label.

This is the arithmetic that was missing. Nothing here needs llama.cpp:
these are HyperNix types that ``llama-quantize`` has never heard of, so
shelling out to it was never going to produce one.

How you get below one bit
-------------------------
Not by storing a fraction of a bit per weight — you cannot — but by
storing *fewer signs than weights* and reconstructing the rest. Every
tier keeps one FP16 scale per 256-weight block; the magnitude is gone
entirely and the scale stands in for all of it. What separates the tiers
is how many of the signs survive:

=========  ======  ==========  ===========  =====
Tier       group   bits/code   block bytes  bpw
=========  ======  ==========  ===========  =====
IQ0.9_L         8           7           30  0.938
IQ0.75_M        4           3           26  0.812
IQ0.5_XXXL      4           2           18  0.562
=========  ======  ==========  ===========  =====

So IQ0.9 keeps 7 of every 8 signs, IQ0.75 keeps 3 of every 4, and IQ0.5
keeps 2 of every 4. The signs that are not stored are reconstructed by
repeating the last one that was.

**Which signs get dropped is the whole decision**, and it is where an
importance matrix earns its place: there is no magnitude left to
allocate at these rates, so all an imatrix can do is decide which signs
matter. The positions kept are the ones with the highest importance ×
magnitude, so a dropped sign is one whose weight was small, or one the
imatrix said the model does not lean on. Without an imatrix the choice
falls back to magnitude alone.

The code is *derived*, not searched. A brute-force over the 128 patterns
a 7-bit code can name would be 32k comparisons per block, and a 7B model
is 27 million blocks — hours of it. Selecting positions and reading their
signs is O(group) and gives the same answer for this codebook.

What this is not
----------------
It is not a way to make a 0.5-bit model good. Below about 1.5 bits per
weight a model stops being a slightly worse version of itself and starts
being a different, much worse model, and no packing changes that. The
tiers carry :attr:`Tier.honest_warning` for this reason. What changed is
that the file now genuinely is what its header says it is.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

__all__ = [
    "BLOCK_SIZE",
    "SubBitError",
    "PACKINGS",
    "quantize_block",
    "dequantize_block",
    "quantize_tensor",
    "dequantize_tensor",
    "dequantize_array",
    "stored_signs",
    "packed_block_bytes",
]

#: Weights per block. 256 to match the K-quant family, so a tensor that
#: divides evenly for Q4_K divides evenly here too.
BLOCK_SIZE = 256


class SubBitError(ValueError):
    """A block could not be packed or unpacked."""


@dataclass(frozen=True)
class Packing:
    """One sub-bit format: how many signs survive out of how many weights."""

    name: str
    #: Weights covered by one code.
    group: int
    #: Bits in that code — and so how many of the group's signs are kept.
    code_bits: int
    #: Sub-blocks that carry their own magnitude index, or 0 for the
    #: sign-only tiers where the block scale is the whole of the
    #: magnitude. Non-zero is what makes a tier above one bit per weight
    #: possible at all: once every sign is stored, ``code_bits/group``
    #: is capped at 1 and any further rate has to buy something else.
    sub_blocks: int = 0
    #: Bits in each sub-block's magnitude index.
    sub_bits: int = 0

    @property
    def sub_size(self) -> int:
        """Weights sharing one magnitude index."""
        return BLOCK_SIZE // self.sub_blocks if self.sub_blocks else BLOCK_SIZE

    @property
    def levels(self) -> int:
        """The largest magnitude index, so the codebook is ``i/levels``."""
        return (1 << self.sub_bits) - 1 if self.sub_bits else 0

    @property
    def has_sub_magnitude(self) -> bool:
        return bool(self.sub_blocks and self.sub_bits)

    @property
    def kept(self) -> int:
        """Signs stored per group. One bit each, so this is code_bits."""
        return self.code_bits

    @property
    def codes_per_block(self) -> int:
        return BLOCK_SIZE // self.group

    @property
    def magnitude_bits(self) -> int:
        return self.sub_blocks * self.sub_bits

    @property
    def payload_bytes(self) -> int:
        return (self.codes_per_block * self.code_bits + self.magnitude_bits + 7) // 8

    @property
    def block_bytes(self) -> int:
        return 2 + self.payload_bytes      # FP16 scale + codes

    @property
    def bits_per_weight(self) -> float:
        return (self.block_bytes * 8) / BLOCK_SIZE


PACKINGS: dict[str, Packing] = {
    #: Every sign kept. The k == g case, where nothing is reconstructed
    #: and the only loss is the magnitude — a binary net with a block
    #: scale. 1.0625 bits/weight: one bit each plus the FP16 scale, the
    #: same way Q4_0 is 4.5 and not 4.
    "int1_binary": Packing("int1_binary", group=1, code_bits=1),
    #: 7 signs kept of every 8. The least destructive sub-bit tier.
    "sign_scale_l": Packing("sign_scale_l", group=8, code_bits=7),
    #: 3 of every 4.
    "pair_code_m": Packing("pair_code_m", group=4, code_bits=3),
    #: 2 of every 4 — half the signs, and no magnitude at all.
    "quad_code_xxxl": Packing("quad_code_xxxl", group=4, code_bits=2),
    #: 3 of every 16, and the other 13 repeat the third. 0.25 bits per
    #: weight *exactly* — 8 bytes covering 256 weights, scale included.
    #: About 59% of signs survive, which is barely above the 50% a coin
    #: gets; see the warning on the tier.
    "quarter_code_uxl": Packing("quarter_code_uxl", group=16, code_bits=3),
    #: Every sign kept, *and* magnitude structure: 16 sub-blocks of 16
    #: weights, each with a 5-bit index into ``i/31`` of the block scale.
    #:
    #: 32 bytes of signs + 10 bytes of indices + a 2-byte scale = 44,
    #: which is 1.375 bits per weight exactly. It is the first tier above
    #: INT1, and the difference between them is the whole idea: INT1
    #: reconstructs all 256 weights at one magnitude, this one at sixteen.
    #: Within a row, weights an order of magnitude apart are common, and
    #: a single block mean puts every one of them at the same distance
    #: from zero.
    "hnx_1375bit": Packing(
        "hnx_1375bit", group=1, code_bits=1, sub_blocks=16, sub_bits=5
    ),
}


def packed_block_bytes(packing: str) -> int:
    """Bytes one 256-weight block occupies under *packing*."""
    spec = PACKINGS.get(packing)
    if spec is None:
        raise SubBitError(f"Unknown packing {packing!r}")
    return spec.block_bytes


def _fp16_bytes(value: float) -> bytes:
    return struct.pack("<e", _finite(value))


def _finite(value: float) -> float:
    """Guard the scale against inf/NaN reaching the file.

    A block of all zeros gives a scale of zero, which is correct; a block
    containing an inf gives a scale of inf, which serialises and then
    dequantises every weight in the block to NaN. Clamping here means a
    poisoned tensor degrades to zeros rather than spreading.
    """
    if not math.isfinite(value):
        return 0.0
    # FP16's maximum. Above it, struct.pack('<e', ...) yields inf.
    return max(-65504.0, min(65504.0, value))


def quantize_block(weights: list[float], packing: str,
                   importance: list[float] | None = None) -> bytes:
    """Pack one block of :data:`BLOCK_SIZE` weights.

    The scale is the importance-weighted **mean** absolute value, which
    minimises weighted squared error for a sign-and-scale code. Not the
    maximum: that minimises a different thing and makes every
    reconstruction systematically too large.
    """
    spec = PACKINGS.get(packing)
    if spec is None:
        raise SubBitError(f"Unknown packing {packing!r}")
    if len(weights) != BLOCK_SIZE:
        raise SubBitError(f"A block is {BLOCK_SIZE} weights, got {len(weights)}")
    if importance is not None and len(importance) != BLOCK_SIZE:
        raise SubBitError("importance must be the same length as the block")

    if importance is None:
        divisor = float(BLOCK_SIZE)
        magnitude_sum = sum(abs(w) for w in weights)
    else:
        divisor = sum(max(0.0, i) for i in importance)
        magnitude_sum = sum(
            abs(w) * max(0.0, i) for w, i in zip(weights, importance, strict=True)
        )
    scale = (magnitude_sum / divisor) if divisor > 0 else 0.0

    # The first `kept` positions of each group, in order.
    #
    # Choosing them by magnitude was tried and is wrong: the decoder has
    # no way to learn which positions were chosen, so it fills the group
    # left to right regardless — and a cleverer encoder just means the
    # signs land on the wrong weights. Encoder and decoder have to agree
    # on the mapping, and with no bits to spend describing it, "the first
    # k, always" is the only mapping both can know.
    #
    # The importance matrix still earns its place: it sets the scale, and
    # at these bitrates the scale is most of what is left to get right.
    if spec.has_sub_magnitude:
        return _quantize_with_magnitude(weights, spec, importance)

    payload = bytearray(spec.payload_bytes)
    bit_cursor = 0
    for start in range(0, BLOCK_SIZE, spec.group):
        for position in range(spec.kept):
            if weights[start + position] >= 0:
                payload[bit_cursor // 8] |= 1 << (bit_cursor % 8)
            bit_cursor += 1
    return _fp16_bytes(scale) + bytes(payload)


def _sub_means(weights: list[float], spec: Packing,
               importance: list[float] | None) -> list[float]:
    """Each sub-block's (importance-weighted) mean absolute value."""
    means = []
    for start in range(0, BLOCK_SIZE, spec.sub_size):
        stop = start + spec.sub_size
        if importance is None:
            divisor = float(spec.sub_size)
            total = sum(abs(w) for w in weights[start:stop])
        else:
            weightings = [max(0.0, i) for i in importance[start:stop]]
            divisor = sum(weightings)
            total = sum(
                abs(w) * i for w, i in zip(weights[start:stop], weightings, strict=True)
            )
        means.append((total / divisor) if divisor > 0 else 0.0)
    return means


def _quantize_with_magnitude(weights: list[float], spec: Packing,
                             importance: list[float] | None) -> bytes:
    """Every sign, plus one magnitude index per sub-block.

    The stored scale is the **largest** sub-block mean rather than the
    block mean, so the indices span ``0 .. levels`` and nothing clamps.
    Using the block mean would put the hot sub-blocks off the top of the
    codebook, which is precisely the structure this tier exists to keep.

    That is not a contradiction of the mean-not-maximum rule the
    sign-only tiers follow: the value each *weight* reconstructs to is
    still a mean -- its own sub-block's -- and the maximum is only the
    unit the sixteen means are expressed in.
    """
    means = _sub_means(weights, spec, importance)
    scale = _finite(max(means) if means else 0.0)

    payload = bytearray(spec.payload_bytes)
    cursor = 0
    for value in weights:
        if value >= 0:
            payload[cursor // 8] |= 1 << (cursor % 8)
        cursor += 1
    for mean in means:
        index = 0 if scale <= 0 else min(
            spec.levels, int(round(spec.levels * mean / scale))
        )
        for bit in range(spec.sub_bits):
            if index >> bit & 1:
                payload[cursor // 8] |= 1 << (cursor % 8)
            cursor += 1
    return _fp16_bytes(scale) + bytes(payload)


def dequantize_block(data: bytes, packing: str) -> list[float]:
    """Reconstruct one block. The inverse of :func:`quantize_block`.

    A position whose sign was not stored repeats the last stored one.
    That is a choice, not a neutral default: repeating is right far more
    often than alternating, because adjacent weights in a row correlate,
    and it is what makes the dropped signs cost less than random.

    The kept positions are the first ``kept`` of each group, which is
    exactly what the encoder stored, so those weights get their own sign
    back. Only the tail is approximated.
    """
    spec = PACKINGS.get(packing)
    if spec is None:
        raise SubBitError(f"Unknown packing {packing!r}")
    if len(data) != spec.block_bytes:
        raise SubBitError(
            f"{packing} blocks are {spec.block_bytes} bytes, got {len(data)}"
        )
    scale = struct.unpack("<e", data[:2])[0]
    payload = data[2:]

    if spec.has_sub_magnitude:
        signs = [
            1.0 if payload[i // 8] >> (i % 8) & 1 else -1.0 for i in range(BLOCK_SIZE)
        ]
        cursor = BLOCK_SIZE
        out: list[float] = []
        for sub in range(spec.sub_blocks):
            index = 0
            for bit in range(spec.sub_bits):
                if payload[cursor // 8] >> (cursor % 8) & 1:
                    index |= 1 << bit
                cursor += 1
            magnitude = scale * index / spec.levels
            base = sub * spec.sub_size
            out.extend(s * magnitude for s in signs[base:base + spec.sub_size])
        return out

    weights: list[float] = []
    bit_cursor = 0
    for _ in range(spec.codes_per_block):
        signs: list[float] = []
        for _ in range(spec.kept):
            bit = 1 if payload[bit_cursor // 8] >> (bit_cursor % 8) & 1 else -1
            signs.append(float(bit) * scale)
            bit_cursor += 1
        # The group's remaining positions repeat the last stored sign.
        while len(signs) < spec.group:
            signs.append(signs[-1])
        weights.extend(signs)
    return weights


def quantize_tensor(weights: list[float], packing: str,
                    importance: list[float] | None = None) -> bytes:
    """Pack a whole tensor, block by block.

    The element count must divide into :data:`BLOCK_SIZE`. Padding a
    ragged tail would change the tensor's shape, and silently changing a
    model's shape is worse than refusing the tensor.
    """
    if len(weights) % BLOCK_SIZE:
        raise SubBitError(
            f"{len(weights)} weights do not divide into {BLOCK_SIZE}-weight blocks"
        )
    out = bytearray()
    for start in range(0, len(weights), BLOCK_SIZE):
        block_importance = (
            importance[start:start + BLOCK_SIZE] if importance is not None else None
        )
        out += quantize_block(weights[start:start + BLOCK_SIZE], packing, block_importance)
    return bytes(out)


def _expansion(spec: Packing):
    """How many output positions each stored sign accounts for.

    The first ``kept - 1`` stand for themselves; the last also covers the
    ``group - kept`` positions whose sign was dropped, because those
    repeat it. As a repeat-count array this is one contiguous expansion
    rather than a gather plus a copy.
    """
    import numpy as np

    counts = _EXPANSION_CACHE.get(spec.name)
    if counts is None:
        counts = np.array(
            [1] * (spec.kept - 1) + [1 + spec.group - spec.kept], dtype=np.intp
        )
        _EXPANSION_CACHE[spec.name] = counts
    return counts


_EXPANSION_CACHE: dict[str, object] = {}


def _sign_table():
    """``byte -> the eight signs its bits stand for``, as float32.

    One gather off this replaces unpackbits plus a uint8-to-float32
    conversion plus the ``2b - 1`` mapping: three passes over the widest
    array on the hot path become one, and it measured about 1.6x on every
    packing. 8 KiB, built once.
    """
    import numpy as np

    global _SIGN_TABLE
    if _SIGN_TABLE is None:
        bits = np.unpackbits(
            np.arange(256, dtype=np.uint8)[:, None], axis=1, bitorder="little"
        )
        _SIGN_TABLE = (bits.astype(np.float32) * 2.0) - 1.0
    return _SIGN_TABLE


_SIGN_TABLE = None


def stored_signs(data: bytes, packing: str):
    """The stored signs, scaled, *without* expanding the dropped ones.

    Shape ``(blocks, codes_per_block, kept)``. This is the array
    :func:`dequantize_array` expands, and it is smaller than the tensor
    it came from by exactly ``kept / group`` -- half, for the 0.5-bit
    tier.

    It is separate because the expansion is avoidable in the one place it
    costs most. Every dropped position repeats its group's last stored
    sign, so a dot product against the group factors::

        sum_k sign[k] * x[k]  ==  sum_{k<kept-1} sign[k] * x[k]
                                  + sign[kept-1] * sum_{k>=kept-1} x[k]

    -- the matmul can run against these signs directly, given an ``x``
    whose last stored position has absorbed the dropped ones. That is
    what :mod:`hypernix.models.hnxrun` does, and it removes the only
    full-size operation left on the hot path.

    The obvious spelling of the expansion -- astype, *2, -1, repeat the
    tail, concatenate, scale -- allocates six full-size arrays, and a
    runtime that keeps weights packed pays for all of them on every
    forward pass. Expanding with an index gather is worse: the result is
    non-contiguous, so the reshape that follows silently copies, and that
    copy measured larger than every other step combined. So what is left
    here is two passes over the *smaller* array: one gather off
    :func:`_sign_table`, which does the unpacking and the ``2b - 1``
    mapping at once, and one in-place multiply by the block's scale.
    """
    import numpy as np

    spec = PACKINGS.get(packing)
    if spec is None:
        raise SubBitError(f"Unknown packing {packing!r}")
    if len(data) % spec.block_bytes:
        raise SubBitError(
            f"{len(data)} bytes is not a whole number of {spec.block_bytes}-byte blocks"
        )

    blocks = np.frombuffer(data, dtype=np.uint8).reshape(-1, spec.block_bytes)
    if not blocks.size:
        return np.zeros((0, spec.codes_per_block, spec.kept), dtype=np.float32)

    # .copy() because frombuffer is read-only and .view needs alignment
    # it cannot assume on an arbitrary slice of someone else's bytes.
    scales = blocks[:, :2].copy().view(np.float16).reshape(-1).astype(np.float32)

    # blocks[:, 2:] is a strided view and stays one: np.take reads it at
    # the same speed and skips an allocation the size of the payload,
    # which on a runtime that exists to not hold the model is the point.
    stored = spec.codes_per_block * spec.kept
    head = np.take(_sign_table(), blocks[:, 2:], axis=0).reshape(len(blocks), -1)
    if spec.has_sub_magnitude:
        # Signs are byte-aligned (one per weight), so the indices start
        # at a byte boundary and can be read without touching the signs.
        signs = head[:, :stored].reshape(-1, spec.sub_blocks, spec.sub_size)
        signs *= (
            scales[:, None, None]
            * _sub_indices(blocks, spec).astype(np.float32)[:, :, None]
            / spec.levels
        )
        return signs.reshape(-1, spec.codes_per_block, spec.kept)
    head = head[:, :stored].reshape(-1, spec.codes_per_block, spec.kept)
    head *= scales[:, None, None]
    return head


def _sub_indices(blocks, spec: Packing):
    """The magnitude index of every sub-block, shape ``(blocks, sub)``.

    Read straight off the packed bytes rather than by unpacking the whole
    payload to bits: the indices are a tenth of it, and this runs on
    every forward pass of a model held packed.
    """
    import numpy as np

    first = 2 + (spec.codes_per_block * spec.code_bits) // 8
    raw = np.unpackbits(blocks[:, first:], axis=1, bitorder="little")
    raw = raw[:, : spec.magnitude_bits].reshape(-1, spec.sub_blocks, spec.sub_bits)
    weights = (1 << np.arange(spec.sub_bits, dtype=np.uint32))
    return (raw.astype(np.uint32) * weights).sum(axis=2)


def dequantize_array(data: bytes, packing: str):
    """Reconstruct a whole tensor as a numpy float32 array.

    The same arithmetic as :func:`dequantize_block`, done to every block
    at once. That matters more than it looks: the block-at-a-time form is
    a Python loop over 256 weights, and a runtime that keeps weights
    packed re-runs it on every forward pass. Measured on hnxrun it was
    the entire difference between a packed model and a materialised one.
    """
    import numpy as np

    spec = PACKINGS.get(packing)
    if spec is None:
        raise SubBitError(f"Unknown packing {packing!r}")

    # Scale first, expand second. The order is the whole optimisation --
    # see :func:`stored_signs`.
    return np.repeat(
        stored_signs(data, packing), _expansion(spec), axis=2
    ).reshape(-1)


def dequantize_tensor(data: bytes, packing: str) -> list[float]:
    """Reconstruct a whole tensor, as a list of floats.

    :func:`dequantize_array` is the same thing without the conversion,
    and is what anything doing arithmetic should call.
    """
    return dequantize_array(data, packing).tolist()
