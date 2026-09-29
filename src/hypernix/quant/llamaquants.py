"""hypernix.quant.llamaquants — the llama.cpp block formats, in Python.

:mod:`hypernix.quant.subbit` gave hyprslug the HyperNix sub-bit tiers.
This gives it the other half: ``Q4_0``, ``Q4_1``, ``Q5_0``, ``Q5_1``,
``Q8_0`` and the K-quants ``Q2_K`` through ``Q6_K`` — the formats every
existing GGUF is already in, encoded and decoded here rather than by
shelling out to ``llama-quantize``.

Why this exists
---------------
"hyprslug quantises without llama.cpp" is only true if it can produce
the quantisations people actually want. A tool that can write a 0.5-bit
HyperNix type but not a Q4_K_M is not a quantiser, it is a curiosity;
and the machine that cannot build llama.cpp is exactly the machine that
needs to make a Q4_K_M.

What is faithful, and what is not
---------------------------------
The **block layouts are exact**. Every struct here matches
``ggml-common.h`` field for field and byte for byte — the byte counts
are asserted against :data:`hypernix.quant.gguf._BLOCK_SHAPE`, and the
dequantisers are the reference ``dequantize_row_*`` arithmetic, so a
round trip through this module reproduces what llama.cpp would read out
of the same bytes.

The **encoders follow the reference search**: :func:`make_qx_quants`
(symmetric, for Q3_K/Q6_K) and :func:`make_qkx2_quants` (asymmetric,
for Q2_K/Q4_K/Q5_K) are ports of the functions of those names in
``ggml-quants.c``, including the 19-step and 21-step scale searches that
do most of the work. What is *not* here is llama.cpp's per-tensor
type-mixing policy — the rule that a "Q4_K_M" file stores attention
output at Q6_K and everything else at Q4_K. That is a policy about which
tensor gets which type, it lives in :mod:`hypernix.quant.hyprslug`, and
conflating it with the block encoding is how you end up unable to
express either one.

Speed
-----
Vectorised over blocks with numpy. A 7B model is about 27 million
256-element super-blocks; a scalar Python loop over those is hours, and
an hour of Python is not a quantiser either. Every search step here is
one array operation across every block at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gguf import GGMLType, type_size_bytes

__all__ = [
    "QK_K",
    "LlamaQuantError",
    "BlockFormat",
    "FORMATS",
    "is_supported",
    "quantize_array",
    "dequantize_array",
    "make_qx_quants",
    "make_qkx2_quants",
]

#: Elements in a K-quant super-block. Upstream's ``QK_K``, and the same
#: 256 that :mod:`hypernix.quant.subbit` uses, so a tensor that divides
#: for one divides for the other.
QK_K = 256

#: Elements in a legacy (non-K) block.
QK_LEGACY = 32

#: Below this the whole block is zero and no scale is worth storing.
#: Upstream calls it GROUP_MAX_EPS.
_GROUP_MAX_EPS = 1e-15


class LlamaQuantError(ValueError):
    """A tensor could not be encoded or decoded in the requested format."""


def _f16(values: np.ndarray) -> np.ndarray:
    """Round to FP16, as the stored scale will be."""
    return values.astype(np.float16)


def _nearest_int(values: np.ndarray) -> np.ndarray:
    """``round()`` the way ``nearest_int`` in ggml-quants.c does it.

    ggml adds 12582912.f and takes the low bits, which is round-half-away
    -from-zero. numpy's ``rint`` is round-half-to-even, and the two
    disagree on exactly the values a quantiser hits most often — a
    weight sitting on x.5 of a step. ``floor(x + 0.5)`` for positives
    and its mirror for negatives is the same rule ggml uses.
    """
    return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5))


# ---------------------------------------------------------------------------
# The two scale searches, ported from ggml-quants.c
# ---------------------------------------------------------------------------


def make_qx_quants(
    x: np.ndarray,
    nmax: int,
    *,
    rmse_type: int = 1,
    qw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric quantisation of each row of *x* to ``[-nmax, nmax-1]``.

    Returns ``(scale, L)`` where ``L`` is biased by ``nmax`` (so it is
    non-negative and packable) and ``scale * (L - nmax)`` reconstructs.

    This is ``make_qx_quants`` from ggml-quants.c: an initial scale from
    the largest magnitude, a weighted least-squares refit, then 18 more
    candidate scales either side of it, keeping whichever maximises
    ``sumlx^2 / suml2``. Q3_K and Q6_K both call it, per 16-element
    group; here every group in the tensor is searched at once.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise LlamaQuantError("make_qx_quants wants a 2-D array of groups")

    amax_index = np.argmax(np.abs(x), axis=1)
    rows = np.arange(x.shape[0])
    max_value = x[rows, amax_index]
    amax = np.abs(max_value)

    if qw is not None:
        weights = np.asarray(qw, dtype=np.float64)
    elif rmse_type == 1:
        weights = x * x
    elif rmse_type == 2:
        weights = np.ones_like(x)
    elif rmse_type == 3:
        weights = np.abs(x)
    else:
        weights = np.sqrt(np.abs(x))

    # A block of (near) zeros has no scale worth storing; upstream
    # returns 0 and leaves L at the mid-point.
    dead = amax < _GROUP_MAX_EPS
    safe_max = np.where(dead, 1.0, max_value)

    iscale = -nmax / safe_max
    quantised = np.clip(_nearest_int(iscale[:, None] * x), -nmax, nmax - 1)
    sumlx = np.sum(weights * x * quantised, axis=1)
    suml2 = np.sum(weights * quantised * quantised, axis=1)
    scale = np.where(suml2 > 0, sumlx / np.where(suml2 > 0, suml2, 1.0), 0.0)
    best = scale * sumlx
    L = quantised + nmax

    for step in range(-9, 10):
        if step == 0:
            continue
        trial_iscale = -(nmax + 0.1 * step) / safe_max
        trial = np.clip(_nearest_int(trial_iscale[:, None] * x), -nmax, nmax - 1)
        trial_sumlx = np.sum(weights * x * trial, axis=1)
        trial_suml2 = np.sum(weights * trial * trial, axis=1)
        better = (trial_suml2 > 0) & (
            trial_sumlx * trial_sumlx > best * trial_suml2
        )
        if not better.any():
            continue
        new_scale = np.where(
            trial_suml2 > 0, trial_sumlx / np.where(trial_suml2 > 0, trial_suml2, 1.0), 0.0
        )
        scale = np.where(better, new_scale, scale)
        best = np.where(better, new_scale * trial_sumlx, best)
        L = np.where(better[:, None], trial + nmax, L)

    scale = np.where(dead, 0.0, scale)
    L = np.where(dead[:, None], nmax, L)
    return scale, L.astype(np.int32)


def make_qkx2_quants(
    x: np.ndarray,
    weights: np.ndarray,
    nmax: int,
    *,
    rmin: float = -1.0,
    rdelta: float = 0.1,
    nstep: int = 20,
    use_mad: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric quantisation of each row of *x* to ``[0, nmax]``.

    Returns ``(scale, the_min, L)``; ``scale * L - the_min``
    reconstructs. This is ``make_qkx2_quants`` from ggml-quants.c, which
    is what makes the K-quants worth their bit budget: rather than take
    min and max and call it a day, it refits scale *and* offset by
    weighted least squares at each of ``nstep`` candidate scales and
    keeps the best. Q2_K, Q4_K and Q5_K all use it.
    """
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2:
        raise LlamaQuantError("make_qkx2_quants wants a 2-D array of groups")

    minimum = np.minimum(np.min(x, axis=1), 0.0)
    maximum = np.max(x, axis=1)
    sum_w = np.sum(weights, axis=1)
    sum_x = np.sum(weights * x, axis=1)

    flat = maximum == minimum
    span = np.where(flat, 1.0, maximum - minimum)

    iscale = nmax / span
    scale = 1.0 / iscale
    L = np.clip(_nearest_int(iscale[:, None] * (x - minimum[:, None])), 0, nmax)
    diff = scale[:, None] * L + minimum[:, None] - x
    diff = np.abs(diff) if use_mad else diff * diff
    best_mad = np.sum(weights * diff, axis=1)

    for step in range(nstep + 1):
        trial_iscale = (rmin + rdelta * step + nmax) / span
        trial_L = np.clip(
            _nearest_int(trial_iscale[:, None] * (x - minimum[:, None])), 0, nmax
        )
        sum_l = np.sum(weights * trial_L, axis=1)
        sum_l2 = np.sum(weights * trial_L * trial_L, axis=1)
        sum_xl = np.sum(weights * trial_L * x, axis=1)
        determinant = sum_w * sum_l2 - sum_l * sum_l
        usable = determinant > 0
        if not usable.any():
            continue
        safe_det = np.where(usable, determinant, 1.0)
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / safe_det
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / safe_det
        # A positive offset means the fit wants to start above zero,
        # which this format cannot say: it stores -min as unsigned. Pin
        # it at zero and refit the scale alone.
        positive = this_min > 0
        safe_l2 = np.where(sum_l2 > 0, sum_l2, 1.0)
        this_scale = np.where(positive, sum_xl / safe_l2, this_scale)
        this_min = np.where(positive, 0.0, this_min)

        trial_diff = this_scale[:, None] * trial_L + this_min[:, None] - x
        trial_diff = np.abs(trial_diff) if use_mad else trial_diff * trial_diff
        mad = np.sum(weights * trial_diff, axis=1)

        better = usable & (mad < best_mad)
        if not better.any():
            continue
        L = np.where(better[:, None], trial_L, L)
        best_mad = np.where(better, mad, best_mad)
        scale = np.where(better, this_scale, scale)
        minimum = np.where(better, this_min, minimum)

    scale = np.where(flat, 0.0, scale)
    L = np.where(flat[:, None], 0, L)
    return scale, -minimum, L.astype(np.int32)


# ---------------------------------------------------------------------------
# Legacy 32-element blocks
# ---------------------------------------------------------------------------


def _q8_0_encode(x: np.ndarray) -> bytes:
    amax = np.max(np.abs(x), axis=1)
    d = amax / 127.0
    d16 = _f16(d)
    inverse = np.where(d16 > 0, 1.0 / np.where(d16 > 0, d16.astype(np.float64), 1.0), 0.0)
    q = np.clip(_nearest_int(x * inverse[:, None]), -128, 127).astype(np.int8)
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:34] = q.view(np.uint8)
    return out.tobytes()


def _q8_0_decode(raw: np.ndarray) -> np.ndarray:
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    q = raw[:, 2:34].view(np.int8).astype(np.float32)
    return q * d[:, None]


def _q8_k_encode(x: np.ndarray) -> bytes:
    """``quantize_row_q8_K_ref``, in float32 the way ggml computes it.

    Q8_K is what llama.cpp quantises *activations* to before a K-quant
    dot product, so it is exact where it has to be: an FP32 scale rather
    than FP16, and the scale taken from the signed value of largest
    magnitude so that value lands on -127 exactly. ``bsums`` are the
    sixteen 16-code partial sums the K-quant kernels use to fold in a
    block minimum without a second pass.
    """
    xf = x.astype(np.float32)
    idx = np.argmax(np.abs(xf), axis=1)
    peak = xf[np.arange(xf.shape[0]), idx]
    nonzero = peak != 0
    iscale = np.where(nonzero, np.float32(-127.0) / np.where(nonzero, peak, 1), 0).astype(np.float32)
    q = _nearest_int((xf * iscale[:, None]).astype(np.float32))
    q = np.minimum(q, 127).astype(np.int8)
    d = np.where(nonzero, np.float32(1.0) / np.where(nonzero, iscale, 1), 0).astype(np.float32)
    bsums = q.reshape(-1, 16, 16).astype(np.int16).sum(axis=2).astype("<i2")
    out = np.empty((x.shape[0], 292), dtype=np.uint8)
    out[:, 0:4] = d.astype("<f4").view(np.uint8).reshape(-1, 4)
    out[:, 4:260] = q.view(np.uint8)
    out[:, 260:292] = bsums.view(np.uint8).reshape(-1, 32)
    return out.tobytes()


def _q8_k_decode(raw: np.ndarray) -> np.ndarray:
    d = raw[:, 0:4].copy().view("<f4").astype(np.float32).reshape(-1)
    q = raw[:, 4:260].view(np.int8).astype(np.float32)
    return q * d[:, None]


def _sym_lowbits(x: np.ndarray, nmax_half: int) -> tuple[np.ndarray, np.ndarray]:
    """The Q4_0/Q5_0 quantiser: one signed scale, values offset to unsigned.

    *nmax_half* is 8 for Q4_0 (4 bits) and 16 for Q5_0 (5 bits).
    """
    amax_index = np.argmax(np.abs(x), axis=1)
    rows = np.arange(x.shape[0])
    max_value = x[rows, amax_index]
    d = max_value / -float(nmax_half)
    d16 = _f16(d)
    d64 = d16.astype(np.float64)
    inverse = np.where(d64 != 0, 1.0 / np.where(d64 != 0, d64, 1.0), 0.0)
    q = np.clip(
        np.floor(x * inverse[:, None] + nmax_half + 0.5), 0, 2 * nmax_half - 1
    ).astype(np.int32)
    return d16, q


def _asym_lowbits(x: np.ndarray, nmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Q4_1/Q5_1 quantiser: scale plus offset, values unsigned."""
    minimum = np.min(x, axis=1)
    maximum = np.max(x, axis=1)
    d = (maximum - minimum) / float(nmax)
    d16 = _f16(d)
    m16 = _f16(minimum)
    d64 = d16.astype(np.float64)
    inverse = np.where(d64 != 0, 1.0 / np.where(d64 != 0, d64, 1.0), 0.0)
    q = np.clip(
        np.floor((x - m16.astype(np.float64)[:, None]) * inverse[:, None] + 0.5), 0, nmax
    ).astype(np.int32)
    return d16, m16, q


def _pack_nibbles(q: np.ndarray) -> np.ndarray:
    """32 values, low 4 bits each, into 16 bytes — j paired with j+16."""
    low = q[:, :16] & 0x0F
    high = q[:, 16:] & 0x0F
    return (low | (high << 4)).astype(np.uint8)


def _pack_fifth_bits(q: np.ndarray) -> np.ndarray:
    """Bit 4 of each of 32 values, as a little-endian uint32 in 4 bytes."""
    bits = ((q >> 4) & 1).astype(np.uint32)
    # Upstream writes value j at bit j and value j+16 at bit j+16, which
    # for a straight 0..31 walk is simply bit j.
    weights = (1 << np.arange(32)).astype(np.uint32)
    packed = np.sum(bits * weights, axis=1, dtype=np.uint32)
    return packed.astype("<u4").view(np.uint8).reshape(-1, 4)


def _q4_0_encode(x: np.ndarray) -> bytes:
    d16, q = _sym_lowbits(x, 8)
    out = np.empty((x.shape[0], 18), dtype=np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:18] = _pack_nibbles(q)
    return out.tobytes()


def _q4_0_decode(raw: np.ndarray) -> np.ndarray:
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = raw[:, 2:18].astype(np.int32)
    low = (qs & 0x0F) - 8
    high = (qs >> 4) - 8
    return np.concatenate([low, high], axis=1).astype(np.float32) * d[:, None]


def _q4_1_encode(x: np.ndarray) -> bytes:
    d16, m16, q = _asym_lowbits(x, 15)
    out = np.empty((x.shape[0], 20), dtype=np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = m16.view(np.uint8).reshape(-1, 2)
    out[:, 4:20] = _pack_nibbles(q)
    return out.tobytes()


def _q4_1_decode(raw: np.ndarray) -> np.ndarray:
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    m = raw[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1)
    qs = raw[:, 4:20].astype(np.int32)
    low = qs & 0x0F
    high = qs >> 4
    q = np.concatenate([low, high], axis=1).astype(np.float32)
    return q * d[:, None] + m[:, None]


def _q5_0_encode(x: np.ndarray) -> bytes:
    d16, q = _sym_lowbits(x, 16)
    out = np.empty((x.shape[0], 22), dtype=np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:6] = _pack_fifth_bits(q)
    out[:, 6:22] = _pack_nibbles(q)
    return out.tobytes()


def _fifth_bits(raw4: np.ndarray) -> np.ndarray:
    packed = raw4.copy().view("<u4").reshape(-1)
    shifts = np.arange(32, dtype=np.uint32)
    return ((packed[:, None] >> shifts) & 1).astype(np.int32)


def _q5_0_decode(raw: np.ndarray) -> np.ndarray:
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    high_bit = _fifth_bits(raw[:, 2:6])
    qs = raw[:, 6:22].astype(np.int32)
    low = np.concatenate([qs & 0x0F, qs >> 4], axis=1)
    q = low | (high_bit << 4)
    return (q - 16).astype(np.float32) * d[:, None]


def _q5_1_encode(x: np.ndarray) -> bytes:
    d16, m16, q = _asym_lowbits(x, 31)
    out = np.empty((x.shape[0], 24), dtype=np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = m16.view(np.uint8).reshape(-1, 2)
    out[:, 4:8] = _pack_fifth_bits(q)
    out[:, 8:24] = _pack_nibbles(q)
    return out.tobytes()


def _q5_1_decode(raw: np.ndarray) -> np.ndarray:
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    m = raw[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1)
    high_bit = _fifth_bits(raw[:, 4:8])
    qs = raw[:, 8:24].astype(np.int32)
    low = np.concatenate([qs & 0x0F, qs >> 4], axis=1)
    q = (low | (high_bit << 4)).astype(np.float32)
    return q * d[:, None] + m[:, None]


# ---------------------------------------------------------------------------
# K-quants: 256-element super-blocks
# ---------------------------------------------------------------------------


def _sigma_weights(x: np.ndarray, group: int, importance: np.ndarray | None) -> np.ndarray:
    """The per-element weights the reference K-quant encoders use.

    Without an imatrix: ``av_x + |x|``, where ``av_x`` is the RMS of the
    whole super-block — a floor under the weight of a small value, so a
    group of near-zeros still gets fitted rather than ignored. With one:
    ``imatrix * sqrt(sigma2 + x^2)``, the form ``quantize_q4_K_impl``
    uses — the importance decides *which* weights the fit protects, and
    the magnitude term stops it protecting a channel carrying nothing.

    ``sigma2`` and ``av_x`` are per super-block, so each is repeated
    across that block's groups.
    """
    blocks = x.reshape(-1, QK_K)
    sigma2 = 2.0 * np.sum(blocks * blocks, axis=1) / QK_K
    grouped = x.reshape(-1, group)
    groups_per_block = QK_K // group
    if importance is None:
        av_x = np.repeat(np.sqrt(sigma2), groups_per_block).reshape(-1, 1)
        return av_x + np.abs(grouped)
    spread = np.repeat(sigma2, groups_per_block).reshape(-1, 1)
    return importance.reshape(-1, group) * np.sqrt(spread + grouped * grouped)


def _put_scale_min_k4(ls: np.ndarray, lm: np.ndarray) -> np.ndarray:
    """Eight 6-bit scales and eight 6-bit mins into 12 bytes.

    The inverse of ``get_scale_min_k4``. The first four of each pair go
    in whole; the last four are split, four bits in their own byte and
    the top two borrowed into the spare bits of an earlier one. Getting
    this wrong does not corrupt the file's structure — it silently
    scales a quarter of every tensor wrong.
    """
    n = ls.shape[0]
    out = np.zeros((n, 12), dtype=np.uint8)
    for j in range(4):
        out[:, j] = (ls[:, j] & 63).astype(np.uint8)
        out[:, j + 4] = (lm[:, j] & 63).astype(np.uint8)
    for j in range(4, 8):
        out[:, j + 4] = ((ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)).astype(np.uint8)
        out[:, j - 4] |= (((ls[:, j] >> 4) & 3) << 6).astype(np.uint8)
        out[:, j] |= (((lm[:, j] >> 4) & 3) << 6).astype(np.uint8)
    return out


def _get_scale_min_k4(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read back what :func:`_put_scale_min_k4` wrote."""
    n = packed.shape[0]
    scales = np.zeros((n, 8), dtype=np.int32)
    mins = np.zeros((n, 8), dtype=np.int32)
    q = packed.astype(np.int32)
    for j in range(4):
        scales[:, j] = q[:, j] & 63
        mins[:, j] = q[:, j + 4] & 63
    for j in range(4, 8):
        scales[:, j] = (q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4)
        mins[:, j] = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
    return scales, mins


def _q4_k_encode(x: np.ndarray, importance: np.ndarray | None) -> bytes:
    n = x.shape[0]
    groups = x.reshape(-1, 32)
    weights = _sigma_weights(x, 32, importance)
    scale, minimum, L = make_qkx2_quants(groups, weights, 15, nstep=20, use_mad=False)
    scale = scale.reshape(n, 8)
    minimum = minimum.reshape(n, 8)

    max_scale = np.max(scale, axis=1)
    max_min = np.max(minimum, axis=1)
    inv_scale = np.where(max_scale > 0, 63.0 / np.where(max_scale > 0, max_scale, 1.0), 0.0)
    inv_min = np.where(max_min > 0, 63.0 / np.where(max_min > 0, max_min, 1.0), 0.0)
    ls = np.minimum(63, _nearest_int(inv_scale[:, None] * scale)).astype(np.int32)
    lm = np.minimum(63, _nearest_int(inv_min[:, None] * minimum)).astype(np.int32)
    packed_scales = _put_scale_min_k4(ls, lm)

    d16 = _f16(max_scale / 63.0)
    dmin16 = _f16(max_min / 63.0)

    read_scales, read_mins = _get_scale_min_k4(packed_scales)
    d = d16.astype(np.float64)[:, None] * read_scales
    dm = dmin16.astype(np.float64)[:, None] * read_mins
    live = d != 0
    safe_d = np.where(live, d, 1.0)
    requantized = np.clip(
        _nearest_int((x.reshape(n, 8, 32) + dm[:, :, None]) / safe_d[:, :, None]), 0, 15
    ).astype(np.int32)
    requantized = np.where(live[:, :, None], requantized, L.reshape(n, 8, 32))
    flat = requantized.reshape(n, QK_K)

    qs = np.empty((n, 128), dtype=np.uint8)
    for start in range(0, QK_K, 64):
        chunk = flat[:, start:start + 64]
        qs[:, start // 2:start // 2 + 32] = (
            (chunk[:, :32] & 0xF) | ((chunk[:, 32:] & 0xF) << 4)
        ).astype(np.uint8)

    out = np.empty((n, 144), dtype=np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = dmin16.view(np.uint8).reshape(-1, 2)
    out[:, 4:16] = packed_scales
    out[:, 16:144] = qs
    return out.tobytes()


def _q4_k_decode(raw: np.ndarray) -> np.ndarray:
    n = raw.shape[0]
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    dmin = raw[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1)
    scales, mins = _get_scale_min_k4(raw[:, 4:16])
    qs = raw[:, 16:144].astype(np.int32)

    values = np.empty((n, QK_K), dtype=np.float32)
    for start in range(0, QK_K, 64):
        chunk = qs[:, start // 2:start // 2 + 32]
        values[:, start:start + 32] = chunk & 0xF
        values[:, start + 32:start + 64] = chunk >> 4
    per_group = values.reshape(n, 8, 32)
    out = per_group * (d[:, None] * scales)[:, :, None] - (dmin[:, None] * mins)[:, :, None]
    return out.reshape(n, QK_K).astype(np.float32)


def _q5_k_encode(x: np.ndarray, importance: np.ndarray | None) -> bytes:
    n = x.shape[0]
    groups = x.reshape(-1, 32)
    weights = _sigma_weights(x, 32, importance)
    scale, minimum, L = make_qkx2_quants(groups, weights, 31, nstep=20, use_mad=False)
    scale = scale.reshape(n, 8)
    minimum = minimum.reshape(n, 8)

    max_scale = np.max(scale, axis=1)
    max_min = np.max(minimum, axis=1)
    inv_scale = np.where(max_scale > 0, 63.0 / np.where(max_scale > 0, max_scale, 1.0), 0.0)
    inv_min = np.where(max_min > 0, 63.0 / np.where(max_min > 0, max_min, 1.0), 0.0)
    ls = np.minimum(63, _nearest_int(inv_scale[:, None] * scale)).astype(np.int32)
    lm = np.minimum(63, _nearest_int(inv_min[:, None] * minimum)).astype(np.int32)
    packed_scales = _put_scale_min_k4(ls, lm)

    d16 = _f16(max_scale / 63.0)
    dmin16 = _f16(max_min / 63.0)
    read_scales, read_mins = _get_scale_min_k4(packed_scales)
    d = d16.astype(np.float64)[:, None] * read_scales
    dm = dmin16.astype(np.float64)[:, None] * read_mins
    live = d != 0
    safe_d = np.where(live, d, 1.0)
    requantized = np.clip(
        _nearest_int((x.reshape(n, 8, 32) + dm[:, :, None]) / safe_d[:, :, None]), 0, 31
    ).astype(np.int32)
    requantized = np.where(live[:, :, None], requantized, L.reshape(n, 8, 32))
    flat = requantized.reshape(n, QK_K)

    qh = np.zeros((n, 32), dtype=np.uint8)
    qs = np.empty((n, 128), dtype=np.uint8)
    m1, m2 = 1, 2
    for index, start in enumerate(range(0, QK_K, 64)):
        first = flat[:, start:start + 32]
        second = flat[:, start + 32:start + 64]
        qh |= np.where(first > 15, m1, 0).astype(np.uint8)
        qh |= np.where(second > 15, m2, 0).astype(np.uint8)
        low1 = np.where(first > 15, first - 16, first)
        low2 = np.where(second > 15, second - 16, second)
        qs[:, index * 32:(index + 1) * 32] = (low1 | (low2 << 4)).astype(np.uint8)
        m1 <<= 2
        m2 <<= 2

    out = np.empty((n, 176), dtype=np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = dmin16.view(np.uint8).reshape(-1, 2)
    out[:, 4:16] = packed_scales
    out[:, 16:48] = qh
    out[:, 48:176] = qs
    return out.tobytes()


def _q5_k_decode(raw: np.ndarray) -> np.ndarray:
    n = raw.shape[0]
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
    dmin = raw[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(-1)
    scales, mins = _get_scale_min_k4(raw[:, 4:16])
    qh = raw[:, 16:48].astype(np.int32)
    qs = raw[:, 48:176].astype(np.int32)

    values = np.empty((n, QK_K), dtype=np.int32)
    m1, m2 = 1, 2
    for index, start in enumerate(range(0, QK_K, 64)):
        chunk = qs[:, index * 32:(index + 1) * 32]
        values[:, start:start + 32] = (chunk & 0xF) + np.where(qh & m1, 16, 0)
        values[:, start + 32:start + 64] = (chunk >> 4) + np.where(qh & m2, 16, 0)
        m1 <<= 2
        m2 <<= 2
    per_group = values.reshape(n, 8, 32).astype(np.float32)
    out = per_group * (d[:, None] * scales)[:, :, None] - (dmin[:, None] * mins)[:, :, None]
    return out.reshape(n, QK_K).astype(np.float32)


def _q2_k_encode(x: np.ndarray, importance: np.ndarray | None) -> bytes:
    n = x.shape[0]
    groups = x.reshape(-1, 16)
    # Q2_K's reference weights are plain |x| -- not the av_x floor the
    # wider K-quants use, because at two bits the fit has nothing to
    # spend on a value that is already almost zero.
    weights = np.abs(groups) if importance is None else _sigma_weights(x, 16, importance)
    scale, minimum, L = make_qkx2_quants(
        groups, weights, 3, rmin=-0.5, rdelta=0.1, nstep=15, use_mad=True
    )
    scale = scale.reshape(n, 16)
    minimum = minimum.reshape(n, 16)

    max_scale = np.max(scale, axis=1)
    max_min = np.max(minimum, axis=1)
    inv_scale = np.where(max_scale > 0, 15.0 / np.where(max_scale > 0, max_scale, 1.0), 0.0)
    inv_min = np.where(max_min > 0, 15.0 / np.where(max_min > 0, max_min, 1.0), 0.0)
    ls = np.minimum(15, _nearest_int(inv_scale[:, None] * scale)).astype(np.int32)
    lm = np.minimum(15, _nearest_int(inv_min[:, None] * minimum)).astype(np.int32)
    packed = (ls | (lm << 4)).astype(np.uint8)

    d16 = _f16(np.where(max_scale > 0, max_scale / 15.0, 0.0))
    dmin16 = _f16(np.where(max_min > 0, max_min / 15.0, 0.0))
    d = d16.astype(np.float64)[:, None] * (packed & 0xF)
    dm = dmin16.astype(np.float64)[:, None] * (packed >> 4)
    live = d != 0
    safe_d = np.where(live, d, 1.0)
    requantized = np.clip(
        _nearest_int((x.reshape(n, 16, 16) + dm[:, :, None]) / safe_d[:, :, None]), 0, 3
    ).astype(np.int32)
    requantized = np.where(live[:, :, None], requantized, L.reshape(n, 16, 16))
    flat = requantized.reshape(n, QK_K)

    qs = np.empty((n, 64), dtype=np.uint8)
    for index, start in enumerate(range(0, QK_K, 128)):
        chunk = flat[:, start:start + 128]
        qs[:, index * 32:(index + 1) * 32] = (
            chunk[:, 0:32]
            | (chunk[:, 32:64] << 2)
            | (chunk[:, 64:96] << 4)
            | (chunk[:, 96:128] << 6)
        ).astype(np.uint8)

    out = np.empty((n, 84), dtype=np.uint8)
    out[:, 0:16] = packed
    out[:, 16:80] = qs
    out[:, 80:82] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 82:84] = dmin16.view(np.uint8).reshape(-1, 2)
    return out.tobytes()


def _q2_k_decode(raw: np.ndarray) -> np.ndarray:
    n = raw.shape[0]
    packed = raw[:, 0:16].astype(np.int32)
    qs = raw[:, 16:80].astype(np.int32)
    d = raw[:, 80:82].copy().view(np.float16).astype(np.float32).reshape(-1)
    dmin = raw[:, 82:84].copy().view(np.float16).astype(np.float32).reshape(-1)

    values = np.empty((n, QK_K), dtype=np.int32)
    for index, start in enumerate(range(0, QK_K, 128)):
        chunk = qs[:, index * 32:(index + 1) * 32]
        values[:, start + 0:start + 32] = chunk & 3
        values[:, start + 32:start + 64] = (chunk >> 2) & 3
        values[:, start + 64:start + 96] = (chunk >> 4) & 3
        values[:, start + 96:start + 128] = (chunk >> 6) & 3
    per_group = values.reshape(n, 16, 16).astype(np.float32)
    scale = (d[:, None] * (packed & 0xF))[:, :, None]
    offset = (dmin[:, None] * (packed >> 4))[:, :, None]
    return (per_group * scale - offset).reshape(n, QK_K).astype(np.float32)


def _put_q3_k_scales(levels: np.ndarray) -> np.ndarray:
    """Sixteen 6-bit scales into 12 bytes, Q3_K's own packing.

    Not the same layout as :func:`_put_scale_min_k4`: here the low four
    bits of the sixteen scales fill the first eight bytes two to a byte,
    and the top two bits go into the last four bytes, four scales to a
    byte.
    """
    n = levels.shape[0]
    out = np.zeros((n, 12), dtype=np.uint8)
    for j in range(16):
        value = levels[:, j].astype(np.uint8)
        if j < 8:
            out[:, j] |= (value & 0xF)
        else:
            out[:, j - 8] |= ((value & 0xF) << 4)
        out[:, j % 4 + 8] |= (((value >> 4) & 3) << (2 * (j // 4)))
    return out


def _get_q3_k_scales(packed: np.ndarray) -> np.ndarray:
    n = packed.shape[0]
    q = packed.astype(np.int32)
    scales = np.empty((n, 16), dtype=np.int32)
    for j in range(16):
        low = (q[:, j] & 0xF) if j < 8 else (q[:, j - 8] >> 4)
        high = (q[:, 8 + j % 4] >> (2 * (j // 4))) & 3
        scales[:, j] = (low | (high << 4)) - 32
    return scales


def _q3_k_encode(x: np.ndarray, importance: np.ndarray | None) -> bytes:
    n = x.shape[0]
    groups = x.reshape(-1, 16)
    qw = None if importance is None else importance.reshape(-1, 16)
    scale, _ = make_qx_quants(groups, 4, rmse_type=1, qw=qw)
    scale = scale.reshape(n, 16)

    amax_index = np.argmax(np.abs(scale), axis=1)
    rows = np.arange(n)
    max_scale = scale[rows, amax_index]
    dead = np.abs(max_scale) < _GROUP_MAX_EPS
    safe_max = np.where(dead, 1.0, max_scale)
    iscale = -32.0 / safe_max
    ls = np.clip(_nearest_int(iscale[:, None] * scale), -32, 31).astype(np.int32) + 32
    ls = np.where(dead[:, None], 32, ls)
    packed_scales = _put_q3_k_scales(ls)
    d16 = _f16(np.where(dead, 0.0, 1.0 / iscale))

    read_scales = _get_q3_k_scales(packed_scales)
    d = d16.astype(np.float64)[:, None] * read_scales
    live = d != 0
    safe_d = np.where(live, d, 1.0)
    quantised = np.clip(
        _nearest_int(x.reshape(n, 16, 16) / safe_d[:, :, None]), -4, 3
    ).astype(np.int32) + 4
    quantised = np.where(live[:, :, None], quantised, 4)
    flat = quantised.reshape(n, QK_K)

    # The third bit of every value lives in its own plane, walked in
    # 32-value strides with the bit position advancing every 32.
    hmask = np.zeros((n, 32), dtype=np.uint8)
    high = (flat > 3).astype(np.uint8)
    for j in range(QK_K):
        hmask[:, j % 32] |= high[:, j] << (j // 32)
    low = np.where(flat > 3, flat - 4, flat)

    qs = np.empty((n, 64), dtype=np.uint8)
    for index, start in enumerate(range(0, QK_K, 128)):
        chunk = low[:, start:start + 128]
        qs[:, index * 32:(index + 1) * 32] = (
            chunk[:, 0:32]
            | (chunk[:, 32:64] << 2)
            | (chunk[:, 64:96] << 4)
            | (chunk[:, 96:128] << 6)
        ).astype(np.uint8)

    out = np.empty((n, 110), dtype=np.uint8)
    out[:, 0:32] = hmask
    out[:, 32:96] = qs
    out[:, 96:108] = packed_scales
    out[:, 108:110] = d16.view(np.uint8).reshape(-1, 2)
    return out.tobytes()


def _q3_k_decode(raw: np.ndarray) -> np.ndarray:
    n = raw.shape[0]
    hmask = raw[:, 0:32].astype(np.int32)
    qs = raw[:, 32:96].astype(np.int32)
    scales = _get_q3_k_scales(raw[:, 96:108])
    d = raw[:, 108:110].copy().view(np.float16).astype(np.float32).reshape(-1)

    low = np.empty((n, QK_K), dtype=np.int32)
    for index, start in enumerate(range(0, QK_K, 128)):
        chunk = qs[:, index * 32:(index + 1) * 32]
        low[:, start + 0:start + 32] = chunk & 3
        low[:, start + 32:start + 64] = (chunk >> 2) & 3
        low[:, start + 64:start + 96] = (chunk >> 4) & 3
        low[:, start + 96:start + 128] = (chunk >> 6) & 3

    values = np.empty((n, QK_K), dtype=np.int32)
    for j in range(QK_K):
        bit = (hmask[:, j % 32] >> (j // 32)) & 1
        values[:, j] = low[:, j] + (bit << 2) - 4

    per_group = values.reshape(n, 16, 16).astype(np.float32)
    return (per_group * (d[:, None] * scales)[:, :, None]).reshape(n, QK_K).astype(np.float32)


def _q6_k_encode(x: np.ndarray, importance: np.ndarray | None) -> bytes:
    n = x.shape[0]
    groups = x.reshape(-1, 16)
    qw = None if importance is None else importance.reshape(-1, 16)
    scale, _ = make_qx_quants(groups, 32, rmse_type=1, qw=qw)
    scale = scale.reshape(n, 16)

    amax_index = np.argmax(np.abs(scale), axis=1)
    rows = np.arange(n)
    max_scale = scale[rows, amax_index]
    dead = np.abs(max_scale) < _GROUP_MAX_EPS
    safe_max = np.where(dead, 1.0, max_scale)
    iscale = -128.0 / safe_max
    scales = np.minimum(127, _nearest_int(iscale[:, None] * scale)).astype(np.int8)
    scales = np.where(dead[:, None], 0, scales).astype(np.int8)
    d16 = _f16(np.where(dead, 0.0, 1.0 / iscale))

    d = d16.astype(np.float64)[:, None] * scales.astype(np.float64)
    live = d != 0
    safe_d = np.where(live, d, 1.0)
    quantised = np.clip(
        _nearest_int(x.reshape(n, 16, 16) / safe_d[:, :, None]), -32, 31
    ).astype(np.int32) + 32
    quantised = np.where(live[:, :, None], quantised, 32)
    flat = quantised.reshape(n, QK_K)

    ql = np.empty((n, 128), dtype=np.uint8)
    qh = np.empty((n, 64), dtype=np.uint8)
    for index, start in enumerate(range(0, QK_K, 128)):
        chunk = flat[:, start:start + 128]
        q1, q2 = chunk[:, 0:32] & 0xF, chunk[:, 32:64] & 0xF
        q3, q4 = chunk[:, 64:96] & 0xF, chunk[:, 96:128] & 0xF
        ql[:, index * 64:index * 64 + 32] = (q1 | (q3 << 4)).astype(np.uint8)
        ql[:, index * 64 + 32:index * 64 + 64] = (q2 | (q4 << 4)).astype(np.uint8)
        qh[:, index * 32:(index + 1) * 32] = (
            (chunk[:, 0:32] >> 4)
            | ((chunk[:, 32:64] >> 4) << 2)
            | ((chunk[:, 64:96] >> 4) << 4)
            | ((chunk[:, 96:128] >> 4) << 6)
        ).astype(np.uint8)

    out = np.empty((n, 210), dtype=np.uint8)
    out[:, 0:128] = ql
    out[:, 128:192] = qh
    out[:, 192:208] = scales.view(np.uint8)
    out[:, 208:210] = d16.view(np.uint8).reshape(-1, 2)
    return out.tobytes()


def _q6_k_decode(raw: np.ndarray) -> np.ndarray:
    n = raw.shape[0]
    ql = raw[:, 0:128].astype(np.int32)
    qh = raw[:, 128:192].astype(np.int32)
    scales = raw[:, 192:208].view(np.int8).astype(np.float32)
    d = raw[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(-1)

    values = np.empty((n, QK_K), dtype=np.int32)
    for index, start in enumerate(range(0, QK_K, 128)):
        low = ql[:, index * 64:index * 64 + 64]
        high = qh[:, index * 32:(index + 1) * 32]
        values[:, start + 0:start + 32] = (low[:, 0:32] & 0xF) | (((high >> 0) & 3) << 4)
        values[:, start + 32:start + 64] = (low[:, 32:64] & 0xF) | (((high >> 2) & 3) << 4)
        values[:, start + 64:start + 96] = (low[:, 0:32] >> 4) | (((high >> 4) & 3) << 4)
        values[:, start + 96:start + 128] = (low[:, 32:64] >> 4) | (((high >> 6) & 3) << 4)
    per_group = (values - 32).reshape(n, 16, 16).astype(np.float32)
    return (per_group * (d[:, None] * scales)[:, :, None]).reshape(n, QK_K).astype(np.float32)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockFormat:
    """One llama.cpp block type: its shape, and how to get in and out."""

    name: str
    ggml_type: int
    block: int
    block_bytes: int
    encode: object
    decode: object
    takes_imatrix: bool = False

    @property
    def bits_per_weight(self) -> float:
        return self.block_bytes * 8 / self.block


def _plain(fn):
    """Adapt an encoder that has no use for an importance matrix."""

    def call(x, importance=None):  # noqa: ARG001 - signature parity
        return fn(x)

    return call


FORMATS: dict[str, BlockFormat] = {
    "Q4_0": BlockFormat("Q4_0", int(GGMLType.Q4_0), 32, 18, _plain(_q4_0_encode), _q4_0_decode),
    "Q4_1": BlockFormat("Q4_1", int(GGMLType.Q4_1), 32, 20, _plain(_q4_1_encode), _q4_1_decode),
    "Q5_0": BlockFormat("Q5_0", int(GGMLType.Q5_0), 32, 22, _plain(_q5_0_encode), _q5_0_decode),
    "Q5_1": BlockFormat("Q5_1", int(GGMLType.Q5_1), 32, 24, _plain(_q5_1_encode), _q5_1_decode),
    "Q8_0": BlockFormat("Q8_0", int(GGMLType.Q8_0), 32, 34, _plain(_q8_0_encode), _q8_0_decode),
    "Q2_K": BlockFormat("Q2_K", int(GGMLType.Q2_K), QK_K, 84, _q2_k_encode, _q2_k_decode, True),
    "Q3_K": BlockFormat("Q3_K", int(GGMLType.Q3_K), QK_K, 110, _q3_k_encode, _q3_k_decode, True),
    "Q4_K": BlockFormat("Q4_K", int(GGMLType.Q4_K), QK_K, 144, _q4_k_encode, _q4_k_decode, True),
    "Q5_K": BlockFormat("Q5_K", int(GGMLType.Q5_K), QK_K, 176, _q5_k_encode, _q5_k_decode, True),
    "Q6_K": BlockFormat("Q6_K", int(GGMLType.Q6_K), QK_K, 210, _q6_k_encode, _q6_k_decode, True),
    # Upstream uses Q8_K only for activations, so stock llama.cpp has no
    # CPU kernel that takes it as a *weight*; ggml-hnx's patch adds one.
    "Q8_K": BlockFormat("Q8_K", int(GGMLType.Q8_K), QK_K, 292, _plain(_q8_k_encode), _q8_k_decode),
}

#: ``GGML type id -> BlockFormat``, for callers that have a type not a name.
BY_TYPE: dict[int, BlockFormat] = {fmt.ggml_type: fmt for fmt in FORMATS.values()}

# The gguf module sizes every tensor from its own table; a block byte
# count that drifted from the packer would produce a file whose offsets
# are wrong from the first tensor and whose second tensor is noise.
for _fmt in FORMATS.values():
    if type_size_bytes(_fmt.ggml_type) != _fmt.block_bytes:
        raise AssertionError(
            f"{_fmt.name}: gguf says {type_size_bytes(_fmt.ggml_type)} bytes per block, "
            f"llamaquants writes {_fmt.block_bytes}"
        )


def is_supported(name_or_type: str | int) -> bool:
    """True if this module can encode and decode *name_or_type*."""
    if isinstance(name_or_type, str):
        return name_or_type.upper() in FORMATS
    return int(name_or_type) in BY_TYPE


def _format_for(name_or_type: str | int) -> BlockFormat:
    if isinstance(name_or_type, str):
        try:
            return FORMATS[name_or_type.upper()]
        except KeyError:
            raise LlamaQuantError(
                f"Unknown llama.cpp quant {name_or_type!r}. Known: {', '.join(FORMATS)}"
            ) from None
    try:
        return BY_TYPE[int(name_or_type)]
    except KeyError:
        raise LlamaQuantError(
            f"GGML type {name_or_type} is not one this module encodes."
        ) from None


def quantize_array(
    values,
    name_or_type: str | int,
    importance=None,
) -> bytes:
    """Encode a flat sequence of floats into *name_or_type*'s blocks.

    *importance* is one weight per element, from an imatrix. The
    K-quants use it to decide what the fit protects; the legacy types
    have no place to put it and ignore it, which is said here rather
    than discovered later.
    """
    fmt = _format_for(name_or_type)
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size % fmt.block:
        raise LlamaQuantError(
            f"{x.size} values do not divide into {fmt.block}-element blocks for "
            f"{fmt.name}."
        )
    blocks = x.reshape(-1, fmt.block)
    weights = None
    if importance is not None and fmt.takes_imatrix:
        weights = np.asarray(importance, dtype=np.float64).reshape(-1)
        if weights.size != x.size:
            raise LlamaQuantError(
                f"imatrix has {weights.size} weights for {x.size} values."
            )
        weights = weights.reshape(-1, fmt.block)
    raw = fmt.encode(blocks, weights)
    expected = blocks.shape[0] * fmt.block_bytes
    if len(raw) != expected:
        raise LlamaQuantError(
            f"{fmt.name} encoder produced {len(raw)} bytes, not {expected}."
        )
    return raw


def dequantize_array(raw: bytes, name_or_type: str | int) -> np.ndarray:
    """Decode *raw* back to float32, as llama.cpp would read it."""
    fmt = _format_for(name_or_type)
    if len(raw) % fmt.block_bytes:
        raise LlamaQuantError(
            f"{len(raw)} bytes is not a whole number of {fmt.name} blocks "
            f"({fmt.block_bytes} bytes each)."
        )
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, fmt.block_bytes)
    return fmt.decode(blocks).reshape(-1)
