"""llama-essir: Triton kernels for Llama-architecture models.

RMSNorm, the SwiGLU gate and rotary position embeddings -- the three
element-wise hot spots of every Llama block -- with layers that drop into
a Transformers model through ``kernels.kernelize``.
"""

from . import layers
from .op import apply_rotary_transformers, rms_norm, rotary, silu_and_mul

__all__ = [
    "apply_rotary_transformers",
    "layers",
    "rms_norm",
    "rotary",
    "silu_and_mul",
]
