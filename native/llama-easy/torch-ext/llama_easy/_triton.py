"""Triton kernels for the Llama hot path.

One program per row. Every kernel loads in the tensor's dtype, computes
in fp32 and rounds once on the way out, with the same rounding points as
the reference where the reference has them (RMSNorm rounds the
normalised value to the input dtype before scaling by the weight, so
this does too; the outputs then agree bit-for-bit in fp32).
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    stride_x,
    stride_out,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    variance = tl.sum(xf * xf, axis=0) / n_cols
    normed = (xf * tl.math.rsqrt(variance + eps)).to(x.dtype)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    out = w.to(tl.float32) * normed.to(tl.float32)
    tl.store(out_ptr + row * stride_out + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    stride_x,
    stride_out,
    d,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.program_id(1) * BLOCK
    cols = start + tl.arange(0, BLOCK)
    mask = cols < d

    gate = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * stride_x + d + cols, mask=mask, other=0.0).to(tl.float32)
    out = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + row * stride_out + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rotary_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    # x: [A, B, C, D] with arbitrary strides (a transposed q is fine).
    sxa, sxb, sxc, sxd,
    # cos/sin: broadcast to x's shape; a broadcast dim has stride 0.
    sca, scb, scc, scd,
    ssa, ssb, ssc, ssd,
    n_b,
    n_c,
    head_dim,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % n_c
    b = (pid // n_c) % n_b
    a = pid // (n_c * n_b)

    cols = tl.arange(0, BLOCK)
    mask = cols < head_dim
    half = head_dim // 2
    # rotate_half: the first half reads -x[i + half], the second x[i - half].
    partner = tl.where(cols < half, cols + half, cols - half)
    sign = tl.where(cols < half, -1.0, 1.0)

    x_row = x_ptr + a * sxa + b * sxb + c * sxc
    x = tl.load(x_row + cols * sxd, mask=mask, other=0.0)
    rotated = tl.load(x_row + partner * sxd, mask=mask, other=0.0).to(tl.float32) * sign
    cos = tl.load(cos_ptr + a * sca + b * scb + c * scc + cols * scd, mask=mask, other=0.0)
    sin = tl.load(sin_ptr + a * ssa + b * ssb + c * ssc + cols * ssd, mask=mask, other=0.0)

    out = x.to(tl.float32) * cos.to(tl.float32) + rotated * sin.to(tl.float32)
    # out is contiguous [A, B, C, D].
    tl.store(out_ptr + pid * head_dim + cols, out.to(out_ptr.dtype.element_ty), mask=mask)
