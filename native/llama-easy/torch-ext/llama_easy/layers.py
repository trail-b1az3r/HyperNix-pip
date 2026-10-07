"""Layers that replace the ``forward`` of a model's own modules.

The ``kernels`` library swaps these in with ``kernelize()``. They must be
pure -- no constructor, no state, no members beyond ``forward`` and the
two flags -- and their ``forward`` must take the arguments of the layer it
replaces; ``kernels`` checks all of it when it loads them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .op import apply_rotary_transformers, rms_norm, silu_and_mul


class RMSNorm(nn.Module):
    """For ``LlamaRMSNorm`` (Transformers' ``RMSNorm`` hook), and any
    RMSNorm that keeps its scale in ``weight`` and epsilon in
    ``variance_epsilon``."""

    weight: torch.Tensor
    variance_epsilon: float

    can_torch_compile: bool = True
    has_backward: bool = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


class SiluAndMul(nn.Module):
    """For a ``SiluAndMul`` hook: ``silu(x[..., :d]) * x[..., d:]``."""

    can_torch_compile: bool = True
    has_backward: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x)


class ApplyRotary(nn.Module):
    """For Transformers' ``rotary_pos_emb`` hook (``apply_rotary_pos_emb``),
    as a layer, which is what a Transformers ``KernelConfig`` names."""

    can_torch_compile: bool = True
    has_backward: bool = False

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rotary_transformers(q, k, cos, sin, unsqueeze_dim)
