"""The ops, registered as Torch custom ops in this build's own namespace.

Registered through ``add_op_namespace_prefix`` so that several versions of
the kernel can be loaded in one process, and with fake implementations so
``torch.compile`` traces through them without a graph break.

Each op runs its Triton kernel on a GPU tensor (CUDA, or ROCm, which Torch
also calls ``cuda``) and the plain-PyTorch definition otherwise. Setting
``TRITON_INTERPRET=1`` runs the Triton kernels on CPU tensors through
Triton's interpreter, which is how they are tested without a GPU.
"""

from __future__ import annotations

import os

import torch

from . import _reference
from ._ops import add_op_namespace_prefix


def _use_triton(*tensors: torch.Tensor) -> bool:
    if all(t.is_cuda for t in tensors):
        return True
    return os.environ.get("TRITON_INTERPRET") == "1" and all(t.device.type == "cpu" for t in tensors)


def _block(n: int) -> int:
    import triton

    return triton.next_power_of_2(n)


# -- RMSNorm ---------------------------------------------------------------


@torch.library.custom_op(add_op_namespace_prefix("rms_norm"), mutates_args={"out"})
def _rms_norm(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    hidden = x.shape[-1]
    if out.numel() == 0:
        return
    if not _use_triton(out, x, weight) or weight.shape != (hidden,):
        out.copy_(_reference.rms_norm(x, weight, eps))
        return

    from ._triton import _rms_norm_kernel

    rows_in = x.reshape(-1, hidden)
    if rows_in.stride(-1) != 1:
        rows_in = rows_in.contiguous()
    rows_out = out.view(-1, hidden)
    _rms_norm_kernel[(rows_in.shape[0],)](
        rows_in, weight.contiguous(), rows_out,
        rows_in.stride(0), rows_out.stride(0), hidden, eps,
        BLOCK=_block(hidden),
    )


@_rms_norm.register_fake
def _(out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    pass


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``LlamaRMSNorm``: ``weight * normalise(x)`` over the last dimension."""
    out = torch.empty(x.shape, dtype=torch.promote_types(weight.dtype, x.dtype), device=x.device)
    _rms_norm(out, x, weight, float(eps))
    return out


# -- SwiGLU gate -----------------------------------------------------------


_SILU_BLOCK = 1024


@torch.library.custom_op(add_op_namespace_prefix("silu_and_mul"), mutates_args={"out"})
def _silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    d = x.shape[-1] // 2
    if out.numel() == 0:
        return
    if not _use_triton(out, x) or x.shape[-1] % 2:
        out.copy_(_reference.silu_and_mul(x))
        return

    from ._triton import _silu_and_mul_kernel

    rows_in = x.reshape(-1, 2 * d)
    if rows_in.stride(-1) != 1:
        rows_in = rows_in.contiguous()
    rows_out = out.view(-1, d)
    grid = (rows_in.shape[0], (d + _SILU_BLOCK - 1) // _SILU_BLOCK)
    _silu_and_mul_kernel[grid](
        rows_in, rows_out, rows_in.stride(0), rows_out.stride(0), d,
        BLOCK=min(_SILU_BLOCK, _block(d)),
    )


@_silu_and_mul.register_fake
def _(out: torch.Tensor, x: torch.Tensor) -> None:
    pass


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``silu(x[..., :d]) * x[..., d:]`` -- the gate of a Llama MLP."""
    out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), dtype=x.dtype, device=x.device)
    _silu_and_mul(out, x)
    return out


# -- Rotary position embedding --------------------------------------------


@torch.library.custom_op(add_op_namespace_prefix("rotary"), mutates_args={"out"})
def _rotary(out: torch.Tensor, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    head_dim = x.shape[-1]
    if out.numel() == 0:
        return
    fits = (
        x.dim() <= 4
        and head_dim % 2 == 0
        and cos.shape[-1] == head_dim
        and sin.shape[-1] == head_dim
        and torch.broadcast_shapes(x.shape, cos.shape, sin.shape) == x.shape
    )
    if not _use_triton(out, x, cos, sin) or not fits:
        out.copy_(_reference.rotary(x, cos, sin))
        return

    from ._triton import _rotary_kernel

    # Up to four dims, padded at the front; expand() gives a broadcast
    # dimension stride 0, so cos/sin are never materialised at q's size.
    while x.dim() < 4:
        x, cos, sin, out = x.unsqueeze(0), cos.unsqueeze(0), sin.unsqueeze(0), out.unsqueeze(0)
    cos = cos.expand(x.shape)
    sin = sin.expand(x.shape)
    n_a, n_b, n_c, _ = x.shape
    _rotary_kernel[(n_a * n_b * n_c,)](
        x, cos, sin, out,
        *x.stride(), *cos.stride(), *sin.stride(),
        n_b, n_c, head_dim,
        BLOCK=_block(head_dim),
    )


@_rotary.register_fake
def _(out: torch.Tensor, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    pass


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE, ``x * cos + rotate_half(x) * sin``; *cos*/*sin* broadcast to *x*."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _rotary(out, x, cos, sin)
    return out


def apply_rotary_transformers(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for Transformers' ``apply_rotary_pos_emb``, the
    ``rotary_pos_emb`` hook every Llama-architecture model exposes."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return rotary(q, cos, sin), rotary(k, cos, sin)


# kernels reads these to decide whether a hooked function may be used in
# a compiled or a training model; there is no backward pass.
apply_rotary_transformers.can_torch_compile = True  # type: ignore[attr-defined]
apply_rotary_transformers.has_backward = False  # type: ignore[attr-defined]
