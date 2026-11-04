"""strife.model — the transformer, on the CPU, from GGUF weights.

One decoder-only layout covers every architecture Strife runs, with the
differences stated in :data:`~.inspect.ARCHITECTURES`:

* RMSNorm before attention and before the feed-forward;
* grouped-query attention with a causal mask, scaled by 1/sqrt(head_dim);
* rotary position embedding -- ``norm`` (adjacent pairs, llama: the GGUF
  converter permutes q and k for it) or ``neox`` (first half with second
  half, qwen) -- with linear rope scaling;
* qwen2's q/k/v biases and qwen3's per-head q/k RMSNorm;
* a SwiGLU feed-forward;
* output projection, or the token embedding when it is tied.

Arithmetic is float32 throughout. Weights are decoded through
:class:`~.weights.WeightStore`, which bounds the RAM that costs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import StrifeError
from .inspect import ARCHITECTURES, ModelReport
from .weights import WeightStore

__all__ = ["Hyper", "KVCache", "Transformer"]


@dataclass(frozen=True)
class Hyper:
    arch: str
    n_layers: int
    n_embd: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_ff: int
    n_vocab: int
    rope: str
    rope_base: float
    rope_scale: float
    eps: float
    qkv_bias: bool
    qk_norm: bool
    tied: bool

    @classmethod
    def from_report(cls, r: ModelReport) -> Hyper:
        spec = ARCHITECTURES[r.architecture]
        return cls(r.architecture, r.n_layers, r.n_embd, r.n_heads, r.n_kv_heads, r.head_dim,
                   r.n_ff, r.n_vocab, spec.rope, r.rope_freq_base, r.rope_scale, r.rms_eps,
                   spec.qkv_bias, spec.qk_norm, r.tied_output)


class KVCache:
    """Keys and values for one sequence, bounded by ``n_ctx``."""

    def __init__(self, hp: Hyper, n_ctx: int) -> None:
        if n_ctx <= 0:
            raise StrifeError("strife.config.bad_context", "the context must be at least 1 token")
        shape = (hp.n_layers, n_ctx, hp.n_kv_heads, hp.head_dim)
        self.k = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.n_ctx = n_ctx
        self.tokens: list[int] = []

    @property
    def n(self) -> int:
        return len(self.tokens)

    @property
    def nbytes(self) -> int:
        return self.k.nbytes + self.v.nbytes

    def truncate(self, n: int) -> None:
        """Forget everything from position *n* on (the values are overwritten later)."""
        del self.tokens[n:]

    @staticmethod
    def nbytes_for(hp: Hyper, n_ctx: int) -> int:
        return 2 * hp.n_layers * n_ctx * hp.n_kv_heads * hp.head_dim * 4


def _rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    scale = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x * scale) * w


class Transformer:
    def __init__(self, weights: WeightStore, report: ModelReport) -> None:
        self.w = weights
        self.hp = Hyper.from_report(report)
        hp = self.hp
        half = hp.head_dim // 2
        self._inv_freq = (hp.rope_base ** (-np.arange(0, half, dtype=np.float64) * 2.0
                                           / hp.head_dim)).astype(np.float64)
        self._out_name = "token_embd.weight" if hp.tied else "output.weight"

    # -- pieces --------------------------------------------------------------

    def _rope(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Rotate ``x`` of shape ``(T, heads, head_dim)`` by position."""
        hp = self.hp
        angles = (positions.astype(np.float64)[:, None] / hp.rope_scale) * self._inv_freq[None, :]
        cos = np.cos(angles).astype(np.float32)[:, None, :]
        sin = np.sin(angles).astype(np.float32)[:, None, :]
        out = np.empty_like(x)
        if hp.rope == "neox":
            half = hp.head_dim // 2
            a, b = x[..., :half], x[..., half:]
            out[..., :half] = a * cos - b * sin
            out[..., half:] = a * sin + b * cos
        else:
            a, b = x[..., 0::2], x[..., 1::2]
            out[..., 0::2] = a * cos - b * sin
            out[..., 1::2] = a * sin + b * cos
        return out

    def _proj(self, h: np.ndarray, layer: int, part: str, bias: bool = False) -> np.ndarray:
        y = self.w.matmul(h, f"blk.{layer}.{part}.weight")
        if bias:
            y = y + self.w.get(f"blk.{layer}.{part}.bias")
        return y

    # -- the pass ------------------------------------------------------------

    def forward(self, tokens: list[int], cache: KVCache, *, all_logits: bool = False,
                cancelled: Any = None) -> np.ndarray:
        """Run *tokens* after what *cache* holds; return logits for the last
        token (or every token with ``all_logits``)."""
        hp = self.hp
        T = len(tokens)
        start = cache.n
        if T == 0:
            raise StrifeError("strife.runtime.empty", "nothing to run")
        if start + T > cache.n_ctx:
            raise StrifeError("strife.runtime.context_full",
                              f"{start + T} tokens do not fit a context of {cache.n_ctx}",
                              remedy="Raise --ctx, or start a new session.",
                              details={"needed": start + T, "n_ctx": cache.n_ctx})
        ids = np.asarray(tokens, dtype=np.int64)
        if ids.min() < 0 or ids.max() >= hp.n_vocab:
            raise StrifeError("strife.runtime.bad_token", "a token id is outside the vocabulary")
        positions = np.arange(start, start + T)
        x = self.w.rows("token_embd.weight", ids).astype(np.float32, copy=False)
        group = hp.n_heads // hp.n_kv_heads
        scale = 1.0 / math.sqrt(hp.head_dim)
        total = start + T
        # Causal mask over the whole visible span: query i sits at start + i.
        mask = np.where(np.arange(total)[None, :] > positions[:, None], -np.inf, 0.0
                        ).astype(np.float32)

        for layer in range(hp.n_layers):
            if cancelled is not None and cancelled():
                raise StrifeError("strife.runtime.cancelled", "cancelled")
            h = _rms(x, self.w.get(f"blk.{layer}.attn_norm.weight"), hp.eps)
            q = self._proj(h, layer, "attn_q", hp.qkv_bias).reshape(T, hp.n_heads, hp.head_dim)
            k = self._proj(h, layer, "attn_k", hp.qkv_bias).reshape(T, hp.n_kv_heads, hp.head_dim)
            v = self._proj(h, layer, "attn_v", hp.qkv_bias).reshape(T, hp.n_kv_heads, hp.head_dim)
            if hp.qk_norm:
                q = _rms(q, self.w.get(f"blk.{layer}.attn_q_norm.weight"), hp.eps)
                k = _rms(k, self.w.get(f"blk.{layer}.attn_k_norm.weight"), hp.eps)
            q = self._rope(q, positions)
            k = self._rope(k, positions)
            cache.k[layer, start:total] = k
            cache.v[layer, start:total] = v
            keys = cache.k[layer, :total]           # (S, kv, d)
            values = cache.v[layer, :total]
            qg = q.reshape(T, hp.n_kv_heads, group, hp.head_dim)
            scores = np.einsum("tkgd,skd->kgts", qg, keys, optimize=True) * scale
            scores = scores + mask[None, None]
            scores -= scores.max(axis=-1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= scores.sum(axis=-1, keepdims=True)
            attn = np.einsum("kgts,skd->tkgd", scores, values, optimize=True)
            x = x + self.w.matmul(attn.reshape(T, hp.n_heads * hp.head_dim),
                                  f"blk.{layer}.attn_output.weight")
            h = _rms(x, self.w.get(f"blk.{layer}.ffn_norm.weight"), hp.eps)
            gate = self.w.matmul(h, f"blk.{layer}.ffn_gate.weight")
            up = self.w.matmul(h, f"blk.{layer}.ffn_up.weight")
            act = gate / (1.0 + np.exp(-gate)) * up
            x = x + self.w.matmul(act, f"blk.{layer}.ffn_down.weight")

        cache.tokens.extend(int(t) for t in tokens)
        x = _rms(x if all_logits else x[-1:], self.w.get("output_norm.weight"), hp.eps)
        logits = self.w.matmul(x, self._out_name)
        return logits if all_logits else logits[0]
