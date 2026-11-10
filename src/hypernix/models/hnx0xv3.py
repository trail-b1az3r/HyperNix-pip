"""hypernix.models.hnx0xv3 — the HyperNix0xV3 architecture (0.73.0, experimental).

``hnx brew new --preset small -E`` builds one; without ``-E`` brewer builds
HyperNix0xV2, as before. V3 is an architecture to train and evaluate, not
yet one with trained checkpoints: nothing here claims it beats V2 until
the ablations in ``wiki/HyperNix0xV3.md`` have been run on real data.

V2 is a strong conventional decoder: GQA attention (sliding on alternate
layers), RoPE, SwiGLU. Its cost is the usual one -- every token keeps
keys and values for every earlier token it may attend to, so memory and
compute grow with the whole context. V3 replaces that with two new layers
that split the work V2's attention does alone.

Tidepool attention — the efficient context layer
-----------------------------------------------
Each query attends, in **one softmax**, to two things:

* the last ``W`` tokens exactly (keys rotated by RoPE: precise, local);
* one *summary* per completed chunk of ``C`` tokens that lies wholly
  before that window. A summary is a salience-weighted pool of its
  chunk's keys and values (content only, no RoPE: far context is looked
  up by what it is, not where it is)::

      s_t      = k_t · u_h                        (salience, per kv head)
      a_t      = softmax over the chunk of s_t
      K̃_c      = Σ_t a_t k_t ,  Ṽ_c = Σ_t a_t v_t
      bias_c   = γ_h · (logsumexp_c s_t − log C)  (chunks that matter, louder)

The salience vector ``u_h`` is learned, so the model decides what deserves
to survive compression; ``bias_c`` lets a chunk whose content scored high
win more of the shared softmax. Causal by construction: a chunk becomes
visible only after its last token, and only to queries whose window no
longer covers it, so nothing is counted twice. With ``max_summaries`` set,
only that many summaries are kept, chosen by ``bias_c``: past it, context
degrades by forgetting the least salient chunks rather than by failing.

Cost per query: ``O((W + T/C) · d_h)`` instead of ``O(T · d_h)``. Cache
per layer: ``W + C`` raw keys/values plus ``T/C`` summaries, instead of
``T``. With the defaults (``W=256``, ``C=64``) a 32k context keeps 832
entries per layer instead of 32 768, about 39× less.

Keel state — the adaptive state layer
-------------------------------------
A small set of ``S`` latent slots per layer (``S × d_s``) that persists
across the sequence. Tokens in chunk ``c`` *read* the state as it was
after chunk ``c−1`` (cross-attention, token → slots); at the end of each
chunk the slots *write* (cross-attention, slot → tokens) and a gate
decides, per slot dimension, what to keep::

      r_t   = softmax(q_t K(S_{c−1})ᵀ / √d_s) V(S_{c−1})       read
      w     = softmax_t(Q(S_{c−1}) k̂_tᵀ / √d_s + log p_t) v̂_t  write
      p_t   = σ(x_t · w_p)                         is this token worth keeping?
      z     = σ([S_{c−1}; w] W_z + b_z),  b_z = +2  (start by remembering)
      S_c   = RMSNorm(z ⊙ S_{c−1} + (1 − z) ⊙ w)

``p_t`` separates transient tokens from persistent ones; ``z`` separates
what the state keeps from what it overwrites; RMSNorm keeps the state
bounded however long the sequence. The recurrence runs once per chunk
(``T/C`` steps), never per token, and everything inside a chunk is
parallel, so it trains like a transformer and cannot become a token-by-
token RNN bottleneck. Overhead: ``O(T · S · d_s)``.

The block
---------
::

    x → RMSNorm → Tidepool → gated residual
      → RMSNorm → Keel     → gated residual
      → RMSNorm → SwiGLU (or top-k MoE) → residual

Tidepool first, because Keel's reads are most useful once each token has
gathered its local and summarised context; the MLP last, as in V2, to
transform what both gathered. The residual gates are per-channel and
start small for Keel (``0.1``) so a new block begins close to V2's
attention-plus-MLP and learns how much state to use, which keeps early
training stable.

Initialisation: normal(0, 0.02) everywhere, residual output projections
scaled by ``1/√(2L)`` (as V2), Keel's slots normal(0, 1/√d_s), the write
gate bias +2, Tidepool's ``γ`` 0.5. Gradients reach the slots through the
gated chunk recurrence; ``z`` near 1 at the start keeps that path short.

Scaling
-------
:data:`PRESETS` runs from ``cpu-nano`` (a few million parameters, CPU) to
``xl`` (mixture of experts). Every config reports its parameter count,
memory needs and recommended threads (:meth:`HyperNix0xV3Config.estimates`);
none assumes the model fits on one GPU.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ARCH", "HyperNix0xV3Config", "HyperNix0xV3Model", "PRESETS", "TidepoolAttention",
           "KeelState", "preset", "is_v3_config"]

ARCH = "hypernix0x-v3"


@dataclass
class HyperNix0xV3Config:
    vocab_size: int = 32_000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 0               # 0: d_model // n_heads
    d_ff: int = 0                   # 0: 8/3 · d_model, rounded up to 64
    max_seq_len: int = 8192
    rope_theta: float = 500_000.0
    # Tidepool
    window: int = 256
    chunk: int = 64
    max_summaries: int = 0          # 0: keep every summary
    # Keel
    n_slots: int = 16
    slot_dim: int = 0               # 0: d_model // 2
    # Mixture of experts (0: dense SwiGLU)
    n_experts: int = 0
    experts_active: int = 2
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    precision: str = "bf16"
    name: str = "hypernix0x-v3-custom"
    arch: str = field(default=ARCH)

    def __post_init__(self) -> None:
        self.arch = ARCH
        if self.head_dim == 0:
            self.head_dim = self.d_model // self.n_heads
        if self.d_ff == 0:
            self.d_ff = (int(self.d_model * 8 / 3) + 63) // 64 * 64
        if self.slot_dim == 0:
            self.slot_dim = max(16, self.d_model // 2)
        if self.n_heads % self.n_kv_heads:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even (RoPE rotates pairs)")
        if self.window < 1 or self.chunk < 1:
            raise ValueError("window and chunk must be at least 1")
        if self.n_experts and not 1 <= self.experts_active <= self.n_experts:
            raise ValueError("experts_active must be between 1 and n_experts")

    # -- serialisation (same shape as BrewerConfig, so brewer tools read it) --

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HyperNix0xV3Config:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> HyperNix0xV3Config:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- sizing ---------------------------------------------------------------

    def approx_params(self) -> int:
        """The exact parameter count, from a model built on the meta device
        (no memory allocated, so this works for the largest preset too)."""
        with torch.device("meta"):
            model = HyperNix0xV3Model(self)
        return model.num_params()

    def active_params(self) -> int:
        if not self.n_experts:
            return self.approx_params()
        per_expert = 3 * self.d_model * self.d_ff
        return self.approx_params() - (self.n_experts - self.experts_active) * per_expert * self.n_layers

    def cache_entries(self, context: int | None = None) -> int:
        """Keys/values kept per layer per sequence at *context* tokens."""
        t = context or self.max_seq_len
        summaries = t // self.chunk
        if self.max_summaries:
            summaries = min(summaries, self.max_summaries)
        return min(t, self.window + self.chunk) + summaries

    def estimates(self) -> dict[str, Any]:
        bytes_per = {"fp32": 4, "bf16": 2, "fp16": 2, "int8": 1}.get(self.precision, 2)
        params = self.approx_params()
        weights = params * bytes_per
        cache = (self.n_layers * self.cache_entries() * 2 * self.n_kv_heads * self.head_dim
                 * bytes_per + self.n_layers * self.n_slots * self.slot_dim * 4)
        full_kv = self.n_layers * self.max_seq_len * 2 * self.n_kv_heads * self.head_dim * bytes_per
        threads = max(1, min(os.cpu_count() or 1, 4 if params < 50e6 else 8 if params < 1e9 else 32))
        return {
            "parameters": params, "active_parameters": self.active_params(),
            "weights_bytes": weights,
            "inference_vram_bytes": int(weights + cache),
            "training_vram_bytes": int(params * (bytes_per + 4 + 8 + 4)),  # weights, master, Adam, grads
            "ram_bytes": int(weights * 1.2 + cache),
            "cache_bytes_per_sequence": int(cache),
            "full_attention_kv_bytes": int(full_kv),
            "cache_saving": round(full_kv / max(1, cache - self.n_layers * self.n_slots * self.slot_dim * 4), 1),
            "recommended_cpu_threads": threads,
            "recommended_gpu_utilisation": 0.9,
            "precision": self.precision,
        }


def is_v3_config(data: dict[str, Any]) -> bool:
    return str(data.get("arch") or "").lower() == ARCH


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def _rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """NeoX-style RoPE over the last dim of ``x`` ``(B, H, T, D)``."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = positions.to(torch.float32)[:, None] * inv[None, :]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., : d // 2], x[..., d // 2:]
    return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1)


class TidepoolAttention(nn.Module):
    """Exact local attention plus salience-pooled chunk summaries, in one softmax."""

    def __init__(self, cfg: HyperNix0xV3Config) -> None:
        super().__init__()
        self.cfg = cfg
        d, hd = cfg.d_model, cfg.head_dim
        self.q_proj = nn.Linear(d, cfg.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * hd, d, bias=False)
        self.salience = nn.Parameter(torch.zeros(cfg.n_kv_heads, hd))
        self.gamma = nn.Parameter(torch.full((cfg.n_heads,), 0.5))
        self.drop = nn.Dropout(cfg.dropout)

    def summaries(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``(K̃, Ṽ, lse)`` for every complete chunk of ``k``/``v`` ``(B, KV, T, D)``."""
        cfg = self.cfg
        B, KV, T, D = k.shape
        n = T // cfg.chunk
        if n == 0:
            empty = k.new_zeros(B, KV, 0, D)
            return empty, empty, k.new_zeros(B, KV, 0)
        kc = k[:, :, : n * cfg.chunk].reshape(B, KV, n, cfg.chunk, D)
        vc = v[:, :, : n * cfg.chunk].reshape(B, KV, n, cfg.chunk, D)
        s = torch.einsum("bkncd,kd->bknc", kc, self.salience.to(k.dtype))
        a = s.softmax(-1)
        lse = torch.logsumexp(s.float(), -1).to(k.dtype) - math.log(cfg.chunk)
        return (torch.einsum("bknc,bkncd->bknd", a, kc), torch.einsum("bknc,bkncd->bknd", a, vc), lse)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        cfg = self.cfg
        B, T, _ = x.shape
        H, KV, D = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        group = H // KV
        pos = positions if positions is not None else torch.arange(T, device=x.device)
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, KV, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, KV, D).transpose(1, 2)
        k_sum, v_sum, lse = self.summaries(k, v)          # content keys, no RoPE
        q_rot, k_rot = _rope(q, pos, cfg.rope_theta), _rope(k, pos, cfg.rope_theta)
        scale = 1.0 / math.sqrt(D)

        # Local band: query i sees keys j with i - W < j <= i.
        idx = torch.arange(T, device=x.device)
        local_mask = (idx[None, :] <= idx[:, None]) & (idx[None, :] > idx[:, None] - cfg.window)
        k_rep = k_rot.repeat_interleave(group, dim=1)
        v_rep = v.repeat_interleave(group, dim=1)
        local = torch.einsum("bhtd,bhsd->bhts", q_rot, k_rep) * scale
        local = local.masked_fill(~local_mask, float("-inf"))

        n = k_sum.shape[2]
        if n:
            ks = k_sum.repeat_interleave(group, dim=1)
            vs = v_sum.repeat_interleave(group, dim=1)
            bias = lse.repeat_interleave(group, dim=1) * self.gamma.view(1, H, 1).to(lse.dtype)
            far = torch.einsum("bhtd,bhnd->bhtn", q, ks) * scale + bias[:, :, None, :]
            # Chunk c is visible once it lies wholly before the query's window.
            chunk_end = (torch.arange(n, device=x.device) + 1) * cfg.chunk - 1
            visible = chunk_end[None, :] <= (idx[:, None] - cfg.window)
            if cfg.max_summaries and n > cfg.max_summaries:
                keep = torch.zeros(B, H, n, dtype=torch.bool, device=x.device)
                top = bias.topk(cfg.max_summaries, dim=-1).indices
                keep.scatter_(-1, top, True)
                visible = visible[None, None] & keep[:, :, None, :]
            far = far.masked_fill(~visible, float("-inf"))
            scores = torch.cat([local, far], dim=-1)
            weights = self.drop(torch.softmax(scores.float(), -1).to(x.dtype))
            out = (torch.einsum("bhts,bhsd->bhtd", weights[..., :T], v_rep)
                   + torch.einsum("bhtn,bhnd->bhtd", weights[..., T:], vs))
        else:
            weights = self.drop(torch.softmax(local.float(), -1).to(x.dtype))
            out = torch.einsum("bhts,bhsd->bhtd", weights, v_rep)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))


class KeelState(nn.Module):
    """A few latent slots, read by every token, rewritten once per chunk."""

    def __init__(self, cfg: HyperNix0xV3Config) -> None:
        super().__init__()
        self.cfg = cfg
        d, ds = cfg.d_model, cfg.slot_dim
        self.init_slots = nn.Parameter(torch.randn(cfg.n_slots, ds) / math.sqrt(ds))
        self.read_q = nn.Linear(d, ds, bias=False)
        self.read_k = nn.Linear(ds, ds, bias=False)
        self.read_v = nn.Linear(ds, ds, bias=False)
        self.read_o = nn.Linear(ds, d, bias=False)
        self.write_q = nn.Linear(ds, ds, bias=False)
        self.write_k = nn.Linear(d, ds, bias=False)
        self.write_v = nn.Linear(d, ds, bias=False)
        self.persist = nn.Linear(d, 1)
        self.gate = nn.Linear(2 * ds, ds)
        self.norm = RMSNorm(ds, cfg.norm_eps)
        nn.init.constant_(self.gate.bias, 2.0)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        B, T, _ = x.shape
        ds, C = cfg.slot_dim, cfg.chunk
        s = state if state is not None else self.init_slots.to(x.dtype).expand(B, -1, -1)
        scale = 1.0 / math.sqrt(ds)
        q_all = self.read_q(x)                                      # (B, T, ds)
        wk_all, wv_all = self.write_k(x), self.write_v(x)
        logp_all = F.logsigmoid(self.persist(x)).squeeze(-1)       # (B, T)
        outs = []
        for start in range(0, T, C):
            stop = min(T, start + C)
            att = torch.softmax((q_all[:, start:stop] @ self.read_k(s).transpose(1, 2)) * scale, -1)
            outs.append(att @ self.read_v(s))
            if stop - start == C:                                    # a complete chunk writes
                wl = (self.write_q(s) @ wk_all[:, start:stop].transpose(1, 2)) * scale
                wl = wl + logp_all[:, None, start:stop]
                write = torch.softmax(wl, -1) @ wv_all[:, start:stop]
                z = torch.sigmoid(self.gate(torch.cat([s, write], -1)))
                s = self.norm(z * s + (1 - z) * write)
        return self.read_o(torch.cat(outs, dim=1)), s


class _SwiGLU(nn.Module):
    def __init__(self, d: int, ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, ff, bias=False)
        self.up_proj = nn.Linear(d, ff, bias=False)
        self.down_proj = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _MoE(nn.Module):
    """Top-k routed SwiGLU experts with a load-balancing loss on ``aux_loss``."""

    def __init__(self, cfg: HyperNix0xV3Config) -> None:
        super().__init__()
        self.k = cfg.experts_active
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(_SwiGLU(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_experts))
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        probs = self.router(flat).float().softmax(-1)
        weights, chosen = probs.topk(self.k, dim=-1)
        weights = (weights / weights.sum(-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            rows, slot = (chosen == e).nonzero(as_tuple=True)
            if rows.numel():
                out[rows] += expert(flat[rows]) * weights[rows, slot, None]
        load = torch.zeros(len(self.experts), device=x.device).scatter_add_(
            0, chosen.reshape(-1), torch.ones(chosen.numel(), device=x.device)) / chosen.numel()
        self.aux_loss = len(self.experts) * (load * probs.mean(0)).sum()
        return out.reshape(B, T, D)


class HyperNix0xV3Block(nn.Module):
    def __init__(self, cfg: HyperNix0xV3Config) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.tidepool = TidepoolAttention(cfg)
        self.gate1 = nn.Parameter(torch.ones(cfg.d_model))
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.keel = KeelState(cfg)
        self.gate2 = nn.Parameter(torch.full((cfg.d_model,), 0.1))
        self.norm3 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn: nn.Module = _MoE(cfg) if cfg.n_experts else _SwiGLU(cfg.d_model, cfg.d_ff)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None,
                state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.gate1 * self.drop(self.tidepool(self.norm1(x), positions))
        y, state = self.keel(self.norm2(x), state)
        x = x + self.gate2 * self.drop(y)
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x, state


class HyperNix0xV3Model(nn.Module):
    """The whole V3 decoder. ``forward(input_ids)`` returns logits ``(B, T, V)``,
    the same convention as V2's ``BrewerModel``, so brewer trains and serves it."""

    def __init__(self, cfg: HyperNix0xV3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(HyperNix0xV3Block(cfg) for _ in range(cfg.n_layers))
        self.norm_out = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, std)
        scale = (2 * self.cfg.n_layers) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.tidepool.o_proj.weight, 0.0, std * scale)
            nn.init.normal_(block.keel.read_o.weight, 0.0, std * scale)
            nn.init.constant_(block.keel.gate.bias, 2.0)
            ffns = block.ffn.experts if isinstance(block.ffn, _MoE) else [block.ffn]
            for ffn in ffns:
                nn.init.normal_(ffn.down_proj.weight, 0.0, std * scale)

    def tie_weights(self) -> None:
        if self.cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_ids.shape[1] > self.cfg.max_seq_len:
            input_ids = input_ids[:, -self.cfg.max_seq_len:]
        x = self.embed(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        for block in self.blocks:
            x, _ = block(x, positions)
        return self.lm_head(self.norm_out(x))

    @property
    def aux_loss(self) -> torch.Tensor:
        losses = [b.ffn.aux_loss for b in self.blocks if isinstance(b.ffn, _MoE)]
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)

    def num_params(self, trainable_only: bool = False) -> int:
        params = (p for p in self.parameters() if p.requires_grad) if trainable_only else self.parameters()
        seen, total = set(), 0
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 64,
                 temperature: float = 0.0) -> torch.Tensor:
        """Greedy or sampled continuation (recomputes; caching comes with Strife support)."""
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = self(ids[:, -self.cfg.max_seq_len:])[:, -1]
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


# ---------------------------------------------------------------------------
# Presets: the same size names as V2, so -E swaps the architecture only.
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict[str, Any]] = {
    "cpu-nano": dict(vocab_size=8_000, d_model=128, n_layers=4, n_heads=4, n_kv_heads=2,
                     max_seq_len=1024, window=128, chunk=32, n_slots=8, slot_dim=64,
                     rope_theta=10_000.0),
    "cpu-tiny": dict(vocab_size=8_000, d_model=256, n_layers=6, n_heads=4, n_kv_heads=2,
                     max_seq_len=2048, window=128, chunk=32, n_slots=8, slot_dim=96,
                     rope_theta=10_000.0),
    "cpu-small": dict(vocab_size=16_000, d_model=384, n_layers=8, n_heads=6, n_kv_heads=2,
                      max_seq_len=4096, window=256, chunk=64, n_slots=12, slot_dim=128),
    "33m": dict(vocab_size=32_000, d_model=512, n_layers=6, n_heads=8, n_kv_heads=2,
                max_seq_len=8192, window=256, chunk=64, n_slots=16, slot_dim=192,
                rope_theta=100_000.0),
    "small": dict(vocab_size=32_000, d_model=1024, n_layers=9, n_heads=16, n_kv_heads=4,
                  max_seq_len=32_768, window=512, chunk=64, n_slots=24, slot_dim=384),
    "medium": dict(vocab_size=32_000, d_model=1280, n_layers=18, n_heads=20, n_kv_heads=4,
                   max_seq_len=65_536, window=1024, chunk=128, n_slots=32, slot_dim=512),
    "large": dict(vocab_size=32_000, d_model=2048, n_layers=36, n_heads=32, n_kv_heads=8,
                  max_seq_len=131_072, window=2048, chunk=128, n_slots=48, slot_dim=768,
                  max_summaries=4096),
    "xl": dict(vocab_size=64_000, d_model=4096, n_layers=48, n_heads=32, n_kv_heads=8,
               max_seq_len=262_144, window=4096, chunk=256, n_slots=64, slot_dim=1024,
               max_summaries=8192, n_experts=16, experts_active=2),
}
PRESETS["micro"] = PRESETS["33m"]


def preset(name: str) -> HyperNix0xV3Config:
    key = name.lower().removeprefix("hypernix0x-v3-")
    if key not in PRESETS:
        raise ValueError(f"Unknown V3 preset {name!r}. Choose: {', '.join(PRESETS)}")
    return HyperNix0xV3Config(name=f"hypernix0x-v3-{key}", **PRESETS[key])
