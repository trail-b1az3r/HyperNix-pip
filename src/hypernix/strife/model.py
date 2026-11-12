"""strife.model — the forward pass, on the CPU, for every architecture in
:mod:`.archs`.

One loop over the plan's layers. Each layer has a *mixer* -- attention,
LFM's gated short convolution, or Qwen3.5's gated delta net -- and a
feed-forward -- dense SwiGLU or a mixture of experts -- arranged in one of
three norm topologies:

* ``pre``: ``x += mix(norm(x)); x += ffn(ffn_norm(x))``
* ``sandwich`` (LFM3): both blocks' outputs normalised again before the
  residual add
* ``post_attn_ffn`` (Qwen3.5, GLM-4.5): the norm GGUF calls
  ``post_attention_norm`` is the feed-forward's input norm

A sequence's state is a :class:`SeqState`: a KV cache for each attention
layer (bounded by ``n_ctx``), and for conv and delta-net layers the last
inputs their convolution needs plus, for delta nets, the recurrent
matrix. Attention can be cut back to any earlier position; recurrent
state cannot, so the state keeps snapshots at the ends of prompts
(:meth:`SeqState.checkpoint`) and :meth:`SeqState.rewind` returns to the
latest one at or before the position asked for.

Float32 throughout. Weights come through
:class:`~.weights.WeightStore`, which bounds the RAM that costs.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .archs import AttnSpec, LayerSpec, Plan, build_plan
from .errors import StrifeError
from .weights import WeightStore

__all__ = ["KVCache", "SeqState", "Transformer", "state_bytes"]

_GDN_L2_EPS = 1e-6
_MAX_CHECKPOINTS = 2


def _rms(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    scale = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x * scale) * w


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def state_bytes(plan: Plan, n_ctx: int) -> dict[str, int]:
    """What one sequence's state costs, by kind."""
    kv = conv = rec = 0
    for layer in plan.layers:
        if layer.attn is not None:
            kv += 2 * n_ctx * layer.attn.n_kv_heads * layer.attn.head_dim * 4
        elif layer.conv is not None:
            conv += (layer.conv.kernel - 1) * plan.n_embd * 4
        elif layer.gdn is not None:
            g = layer.gdn
            conv += (g.kernel - 1) * g.conv_dim * 4
            rec += g.n_v_heads * g.head_k * g.head_v * 4
    return {"kv_cache": kv, "conv_state": conv, "recurrent_state": rec,
            "total": kv + conv + rec, "checkpoints_max": (conv + rec) * _MAX_CHECKPOINTS}


class SeqState:
    """Everything one sequence carries between forward passes."""

    def __init__(self, plan: Plan, n_ctx: int) -> None:
        if n_ctx <= 0:
            raise StrifeError("strife.config.bad_context", "the context must be at least 1 token")
        self.plan = plan
        self.n_ctx = n_ctx
        self.tokens: list[int] = []
        self.k: dict[int, np.ndarray] = {}
        self.v: dict[int, np.ndarray] = {}
        self.conv: dict[int, np.ndarray] = {}
        self.rec: dict[int, np.ndarray] = {}
        for layer in plan.layers:
            i = layer.index
            if layer.attn is not None:
                shape = (n_ctx, layer.attn.n_kv_heads, layer.attn.head_dim)
                self.k[i] = np.zeros(shape, dtype=np.float32)
                self.v[i] = np.zeros(shape, dtype=np.float32)
            elif layer.conv is not None:
                self.conv[i] = np.zeros((layer.conv.kernel - 1, plan.n_embd), dtype=np.float32)
            elif layer.gdn is not None:
                g = layer.gdn
                self.conv[i] = np.zeros((g.kernel - 1, g.conv_dim), dtype=np.float32)
                self.rec[i] = np.zeros((g.n_v_heads, g.head_k, g.head_v), dtype=np.float32)
        self._checkpoints: dict[int, tuple[dict[int, np.ndarray], dict[int, np.ndarray]]] = {}

    @property
    def n(self) -> int:
        return len(self.tokens)

    @property
    def recurrent(self) -> bool:
        return bool(self.conv or self.rec)

    @property
    def nbytes(self) -> int:
        arrays = [*self.k.values(), *self.v.values(), *self.conv.values(), *self.rec.values()]
        snaps = sum(a.nbytes for c, r in self._checkpoints.values() for a in [*c.values(), *r.values()])
        return sum(a.nbytes for a in arrays) + snaps

    def checkpoint(self) -> None:
        """Remember the recurrent state as it is now, at position :attr:`n`."""
        if not self.recurrent:
            return
        self._checkpoints[self.n] = ({i: a.copy() for i, a in self.conv.items()},
                                     {i: a.copy() for i, a in self.rec.items()})
        while len(self._checkpoints) > _MAX_CHECKPOINTS:
            del self._checkpoints[min(self._checkpoints)]

    def _reset_recurrent(self) -> None:
        for a in [*self.conv.values(), *self.rec.values()]:
            a.fill(0.0)

    def rewind(self, n: int, *, force: bool = False) -> int:
        """Go back to position *n* or the nearest earlier one this state can
        reach; return the position it is now at.

        *force* restores even at the current position: after a forward pass
        was cancelled partway, recurrent layers it reached have already
        moved on, and only a snapshot is trustworthy.
        """
        n = max(0, min(n, self.n))
        if n == self.n and not (force and self.recurrent):
            return n
        if not self.recurrent:
            del self.tokens[n:]
            return n
        usable = [p for p in self._checkpoints if p <= n]
        if not usable:
            self._reset_recurrent()
            self.tokens.clear()
            self._checkpoints.clear()
            return 0
        at = max(usable)
        conv, rec = self._checkpoints[at]
        for i, a in conv.items():
            self.conv[i][...] = a
        for i, a in rec.items():
            self.rec[i][...] = a
        del self.tokens[at:]
        for p in [p for p in self._checkpoints if p > at]:
            del self._checkpoints[p]
        return at

    def truncate(self, n: int) -> None:
        """Forget everything from position *n* on; refuses what cannot be undone."""
        wanted = min(n, self.n)
        if self.rewind(n) != wanted:
            raise StrifeError("strife.runtime.no_rewind",
                              f"this model's recurrent state cannot go back to position {n}")


#: The name the first Strife alpha used; kept for callers that build one.
KVCache = SeqState


class Transformer:
    def __init__(self, weights: WeightStore, report: Any = None) -> None:
        from .inspect import shapes_of

        self.w = weights
        self.plan: Plan = build_plan(weights.gguf.metadata, shapes_of(weights.gguf))
        self._out_name = "token_embd.weight" if self.plan.tied else "output.weight"
        self._freq_cache: dict[tuple[Any, ...], np.ndarray] = {}

    @property
    def hp(self) -> Plan:  # the alpha-1 attribute name
        return self.plan

    def new_state(self, n_ctx: int) -> SeqState:
        return SeqState(self.plan, n_ctx)

    # -- rope ----------------------------------------------------------------

    def _inv_freq(self, a: AttnSpec) -> np.ndarray:
        key = (a.n_rot, a.rope_base, a.rope_factors)
        cached = self._freq_cache.get(key)
        if cached is None:
            half = a.n_rot // 2
            cached = a.rope_base ** (-np.arange(half, dtype=np.float64) * 2.0 / a.n_rot)
            if a.rope_factors:
                factors = self.w.get(a.rope_factors).astype(np.float64)[:half]
                cached = cached / factors
            self._freq_cache[key] = cached
        return cached

    def _rope(self, x: np.ndarray, positions: np.ndarray, a: AttnSpec) -> np.ndarray:
        """Rotate the first ``n_rot`` dims of ``x`` ``(T, heads, head_dim)``."""
        rot = a.n_rot
        angles = (positions.astype(np.float64)[:, None] / a.rope_scale) * self._inv_freq(a)[None, :]
        cos = np.cos(angles).astype(np.float32)[:, None, :]
        sin = np.sin(angles).astype(np.float32)[:, None, :]
        out = x.copy()
        part = x[..., :rot]
        if a.rope == "neox":
            h = rot // 2
            p1, p2 = part[..., :h], part[..., h:]
            out[..., :h] = p1 * cos - p2 * sin
            out[..., h:rot] = p1 * sin + p2 * cos
        else:
            p1, p2 = part[..., 0::2], part[..., 1::2]
            out[..., 0:rot:2] = p1 * cos - p2 * sin
            out[..., 1:rot:2] = p1 * sin + p2 * cos
        return out

    # -- mixers --------------------------------------------------------------

    def _lin(self, h: np.ndarray, name: str, bias: str | None = None) -> np.ndarray:
        y = self.w.matmul(h, name)
        if bias is not None:
            y = y + self.w.get(bias)
        return y

    def _attention(self, layer: LayerSpec, h: np.ndarray, positions: np.ndarray,
                   state: SeqState) -> np.ndarray:
        a = layer.attn
        assert a is not None
        p, eps = f"blk.{layer.index}.", self.plan.eps
        T, H, KV, D = h.shape[0], a.n_heads, a.n_kv_heads, a.head_dim
        q = self._lin(h, p + "attn_q.weight", p + "attn_q.bias" if a.qkv_bias else None)
        gate = None
        if a.gate == "q":
            qg = q.reshape(T, H, 2 * D)
            q, gate = qg[..., :D], qg[..., D:]
        q = q.reshape(T, H, D)
        k = self._lin(h, p + "attn_k.weight", p + "attn_k.bias" if a.qkv_bias else None).reshape(T, KV, D)
        v = self._lin(h, p + "attn_v.weight", p + "attn_v.bias" if a.qkv_bias else None).reshape(T, KV, D)
        if a.qk_norm:
            q = _rms(q, self.w.get(p + "attn_q_norm.weight"), eps)
            k = _rms(k, self.w.get(p + "attn_k_norm.weight"), eps)
        q = self._rope(q, positions, a)
        k = self._rope(k, positions, a)
        start = int(positions[0])
        total = start + T
        state.k[layer.index][start:total] = k
        state.v[layer.index][start:total] = v
        keys, values = state.k[layer.index][:total], state.v[layer.index][:total]
        key_pos = np.arange(total)
        hidden = key_pos[None, :] > positions[:, None]
        if a.window:
            hidden |= (positions[:, None] - key_pos[None, :]) >= a.window
        mask = np.where(hidden, -np.inf, 0.0).astype(np.float32)
        group = H // KV
        qg4 = q.reshape(T, KV, group, D)
        scores = np.einsum("tkgd,skd->kgts", qg4, keys, optimize=True) * (1.0 / math.sqrt(D))
        scores = scores + mask[None, None]
        scores -= scores.max(axis=-1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= scores.sum(axis=-1, keepdims=True)
        out = np.einsum("kgts,skd->tkgd", scores, values, optimize=True).reshape(T, H, D)
        if a.gate == "q" and gate is not None:
            out = out * _sigmoid(gate)
        elif a.gate == "head":
            out = out * _sigmoid(self.w.matmul(h, p + "attn_gate.weight"))[:, :, None]
        return self.w.matmul(out.reshape(T, H * D), p + "attn_output.weight")

    @staticmethod
    def _causal_conv(x: np.ndarray, weight: np.ndarray, conv_state: np.ndarray) -> np.ndarray:
        """Depthwise causal convolution over time; updates *conv_state* in place.

        ``weight`` is ``(channels, K)`` with ``K - 1`` the current step;
        ``conv_state`` holds the last ``K - 1`` inputs.
        """
        K = weight.shape[1]
        seq = np.concatenate([conv_state, x], axis=0) if K > 1 else x
        T = x.shape[0]
        out = np.zeros_like(x)
        for j in range(K):
            out += seq[j:j + T] * weight[:, j]
        if K > 1:
            conv_state[...] = seq[-(K - 1):]
        return out

    def _shortconv(self, layer: LayerSpec, h: np.ndarray, state: SeqState) -> np.ndarray:
        p, E = f"blk.{layer.index}.", self.plan.n_embd
        bcx = self.w.matmul(h, p + "shortconv.in_proj.weight")
        b, c, x = bcx[:, :E], bcx[:, E:2 * E], bcx[:, 2 * E:]
        conv = self._causal_conv(b * x, self.w.get(p + "shortconv.conv.weight"), state.conv[layer.index])
        return self.w.matmul(c * conv, p + "shortconv.out_proj.weight")

    def _gdn(self, layer: LayerSpec, h: np.ndarray, state: SeqState) -> np.ndarray:
        g = layer.gdn
        assert g is not None
        p, T = f"blk.{layer.index}.", h.shape[0]
        qkv = self.w.matmul(h, p + "attn_qkv.weight")
        z = self.w.matmul(h, p + "attn_gate.weight").reshape(T, g.n_v_heads, g.head_v)
        beta = _sigmoid(self.w.matmul(h, p + "ssm_beta.weight"))
        decay = self.w.get(p + "ssm_a") * _softplus(
            self.w.matmul(h, p + "ssm_alpha.weight") + self.w.get(p + "ssm_dt.bias"))
        mixed = _silu(self._causal_conv(qkv, self.w.get(p + "ssm_conv1d.weight"), state.conv[layer.index]))
        kd = g.key_dim
        q = mixed[:, :kd].reshape(T, g.n_k_heads, g.head_k)
        k = mixed[:, kd:2 * kd].reshape(T, g.n_k_heads, g.head_k)
        v = mixed[:, 2 * kd:].reshape(T, g.n_v_heads, g.head_v)
        q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + _GDN_L2_EPS)
        k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + _GDN_L2_EPS)
        if g.n_v_heads != g.n_k_heads:  # tiled: value head h reads key head h mod n_k
            reps = g.n_v_heads // g.n_k_heads
            q, k = np.tile(q, (1, reps, 1)), np.tile(k, (1, reps, 1))
        q = q * (1.0 / math.sqrt(g.head_k))
        s = state.rec[layer.index]
        out = np.empty((T, g.n_v_heads, g.head_v), dtype=np.float32)
        gate = np.exp(decay).astype(np.float32)
        for t in range(T):
            s *= gate[t][:, None, None]
            remembered = np.einsum("hkv,hk->hv", s, k[t])
            delta = (v[t] - remembered) * beta[t][:, None]
            s += k[t][:, :, None] * delta[:, None, :]
            out[t] = np.einsum("hkv,hk->hv", s, q[t])
        normed = _rms(out, self.w.get(p + "ssm_norm.weight"), self.plan.eps) * _silu(z)
        return self.w.matmul(normed.reshape(T, g.value_dim), p + "ssm_out.weight")

    # -- feed-forward ----------------------------------------------------------

    def _dense(self, h: np.ndarray, prefix: str, suffix: str = "") -> np.ndarray:
        gate = self.w.matmul(h, f"{prefix}ffn_gate{suffix}.weight")
        up = self.w.matmul(h, f"{prefix}ffn_up{suffix}.weight")
        return self.w.matmul(_silu(gate) * up, f"{prefix}ffn_down{suffix}.weight")

    def _moe(self, layer: LayerSpec, h: np.ndarray) -> np.ndarray:
        m = layer.moe
        assert m is not None
        p, T, E = f"blk.{layer.index}.", h.shape[0], self.plan.n_embd
        logits = self.w.matmul(h, p + "ffn_gate_inp.weight")
        if m.gating == "softmax":
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
        else:
            probs = _sigmoid(logits)
        choose = probs + self.w.get(p + "exp_probs_b.bias") if m.probs_bias else probs
        picked = np.argpartition(-choose, m.n_used - 1, axis=-1)[:, : m.n_used]
        weights = np.take_along_axis(probs, picked, axis=-1)
        if m.norm:
            weights = weights / np.maximum(weights.sum(axis=-1, keepdims=True), 6.103515625e-5)
        weights = weights * m.scale
        out = np.zeros((T, E), dtype=np.float32)
        for expert in np.unique(picked):
            rows, slot = np.nonzero(picked == expert)
            x = h[rows]
            e = int(expert)
            if m.fused_gate_up:
                gu = self.w.matmul(x, p + "ffn_gate_up_exps.weight", (e * 2 * m.n_ff, (e + 1) * 2 * m.n_ff))
                gate, up = gu[:, : m.n_ff], gu[:, m.n_ff:]
            else:
                gate = self.w.matmul(x, p + "ffn_gate_exps.weight", (e * m.n_ff, (e + 1) * m.n_ff))
                up = self.w.matmul(x, p + "ffn_up_exps.weight", (e * m.n_ff, (e + 1) * m.n_ff))
            y = self.w.matmul(_silu(gate) * up, p + "ffn_down_exps.weight", (e * E, (e + 1) * E))
            out[rows] += y * weights[rows, slot][:, None]
        if m.shared_ff:
            shared = self._dense(h, p, "_shexp")
            if m.shared_gate:
                shared = shared * _sigmoid(h @ self.w.get(p + "ffn_gate_inp_shexp.weight"))[:, None]
            out += shared
        return out

    # -- the pass ------------------------------------------------------------

    def forward(self, tokens: list[int], cache: SeqState, *, all_logits: bool = False,
                cancelled: Any = None) -> np.ndarray:
        """Run *tokens* after what *cache* holds; return logits for the last
        token (or every token with ``all_logits``)."""
        plan = self.plan
        T, start = len(tokens), cache.n
        if T == 0:
            raise StrifeError("strife.runtime.empty", "nothing to run")
        if start + T > cache.n_ctx:
            raise StrifeError("strife.runtime.context_full",
                              f"{start + T} tokens do not fit a context of {cache.n_ctx}",
                              remedy="Raise --ctx, or start a new session.",
                              details={"needed": start + T, "n_ctx": cache.n_ctx})
        ids = np.asarray(tokens, dtype=np.int64)
        if ids.min() < 0 or ids.max() >= plan.n_vocab:
            raise StrifeError("strife.runtime.bad_token", "a token id is outside the vocabulary")
        positions = np.arange(start, start + T)
        x = self.w.rows("token_embd.weight", ids).astype(np.float32, copy=False)
        eps = plan.eps
        for layer in plan.layers:
            if cancelled is not None and cancelled():
                raise StrifeError("strife.runtime.cancelled", "cancelled")
            p = f"blk.{layer.index}."
            h = _rms(x, self.w.get(p + "attn_norm.weight"), eps)
            if layer.mixer == "attn":
                y = self._attention(layer, h, positions, cache)
            elif layer.mixer == "conv":
                y = self._shortconv(layer, h, cache)
            else:
                y = self._gdn(layer, h, cache)
            if plan.norm_style == "sandwich":
                y = _rms(y, self.w.get(p + "post_attention_norm.weight"), eps)
            x = x + y
            ffn_norm = "post_attention_norm" if plan.norm_style == "post_attn_ffn" else "ffn_norm"
            h = _rms(x, self.w.get(f"{p}{ffn_norm}.weight"), eps)
            f = self._moe(layer, h) if layer.ffn == "moe" else self._dense(h, p)
            if plan.norm_style == "sandwich":
                f = _rms(f, self.w.get(p + "post_ffw_norm.weight"), eps)
            x = x + f
        cache.tokens.extend(int(t) for t in tokens)
        x = _rms(x if all_logits else x[-1:], self.w.get("output_norm.weight"), eps)
        logits = self.w.matmul(x, self._out_name)
        return logits if all_logits else logits[0]
