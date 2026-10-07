"""The definition of every op, in plain PyTorch.

These are the Llama layers exactly as Transformers' ``modeling_llama``
computes them. They are what the Triton kernels are tested against, and
what runs where Triton cannot: on the CPU, and for shapes the kernels do
not cover (see ``op.py``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``LlamaRMSNorm.forward``: normalise in fp32, scale in the input dtype."""
    input_dtype = x.dtype
    hidden = x.to(torch.float32)
    variance = hidden.pow(2).mean(-1, keepdim=True)
    hidden = hidden * torch.rsqrt(variance + eps)
    return weight * hidden.to(input_dtype)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU's gate: ``silu(x[..., :d]) * x[..., d:]``."""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE for one tensor; *cos* and *sin* already broadcast against *x*."""
    return (x * cos) + (_rotate_half(x) * sin)


def apply_rotary_transformers(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``apply_rotary_pos_emb`` from ``modeling_llama``."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return rotary(q, cos, sin), rotary(k, cos, sin)
