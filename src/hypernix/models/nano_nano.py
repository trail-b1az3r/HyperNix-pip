"""Minimal loader for the custom ``nano-nano`` architecture.

Used by ``ray0rf1re/nano-nano-927-v3``. The upstream repo ships a
``modeling_nano_nano.py`` that requires ``trust_remote_code=True`` with
``transformers``; we reimplement it here (about 100 LOC) so users can
load + run the model through HyperNix's :class:`CodeOven` with no remote
code execution and no mandatory ``transformers`` dependency.

Tensor-name layout (matches the upstream repo so safetensors load
cleanly without remapping)::

    tok_embeddings.weight
    layers.{N}.attn_norm.weight
    layers.{N}.attn.q_proj.weight
    layers.{N}.attn.k_proj.weight
    layers.{N}.attn.v_proj.weight
    layers.{N}.attn.o_proj.weight
    layers.{N}.ffn_norm.weight
    layers.{N}.ffn.gate_proj.weight
    layers.{N}.ffn.up_proj.weight
    layers.{N}.ffn.down_proj.weight
    norm.weight
    output.weight                   (tied to tok_embeddings.weight)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypernix.system.torch_compat import RMSNorm as _RMSNorm
from hypernix.system.torch_compat import scaled_dot_product_attention as _sdpa


@dataclass
class NanoNanoConfig:
    """Config for the nano-nano arch. Field names mirror the upstream repo.

    The properties below (``hidden_size``, ``head_dim``, etc.) expose a
    HyperNix-compatible view so generic code paths in :class:`CodeOven`
    that peek at the config don't need nano-nano-specific branches.
    """

    model_type: str = "nano-nano"
    vocab_size: int = 2048
    dim: int = 120
    num_layers: int = 12
    num_heads: int = 4
    num_kv_heads: int = 2
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0
    # Hidden ratio used to size the gated MLP: round_up(dim * hidden_ratio, 8).
    mlp_hidden_ratio: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NanoNanoConfig:
        fields = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**fields)

    @classmethod
    def from_json(cls, path: Path | str) -> NanoNanoConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # --- HyperNix-style compatibility accessors -------------------------

    @property
    def hidden_size(self) -> int:
        return self.dim

    @property
    def num_attention_heads(self) -> int:
        return self.num_heads

    @property
    def num_hidden_layers(self) -> int:
        return self.num_layers

    @property
    def num_key_value_heads(self) -> int:
        return self.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


def _mlp_hidden_dim(dim: int, ratio: float) -> int:
    raw = int(dim * ratio)
    return ((raw + 7) // 8) * 8


class _Rotary(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # [1, T, 1, D/2] — matches upstream exactly (including the
        # dim-order quirk the upstream apply_rotary_emb relies on).
        return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Upstream uses x.shape[1] as the slice length — which, because x has
    # been transposed to [B, H, T, D], is actually the number of heads. We
    # reproduce that behavior literally so logits match the published
    # checkpoint. Do NOT "fix" it here; fixing it would diverge from
    # what the model was trained with.
    slice_len = x.shape[1]
    cos = cos[:, :slice_len, :, :]
    sin = sin[:, :slice_len, :, :]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class _NanoAttn(nn.Module):
    def __init__(self, cfg: NanoNanoConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.num_groups = cfg.num_heads // cfg.num_kv_heads
        self.q_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.rotary = _Rotary(self.head_dim, theta=cfg.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(x, T)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        out = _sdpa(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class _NanoMLP(nn.Module):
    def __init__(self, cfg: NanoNanoConfig) -> None:
        super().__init__()
        hidden = _mlp_hidden_dim(cfg.dim, cfg.mlp_hidden_ratio)
        self.gate_proj = nn.Linear(cfg.dim, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _NanoBlock(nn.Module):
    def __init__(self, cfg: NanoNanoConfig) -> None:
        super().__init__()
        self.attn_norm = _RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.attn = _NanoAttn(cfg)
        self.ffn_norm = _RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.ffn = _NanoMLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class NanoNanoModel(nn.Module):
    """Minimal standalone port of the upstream ``NanoNanoModel``.

    Forward signature matches :class:`hypernix.train.HyperNixModel` so
    :class:`CodeOven` can treat it uniformly: ``forward(input_ids,
    labels=None) -> {"logits": ..., "loss": optional}``.
    """

    def __init__(self, cfg: NanoNanoConfig) -> None:
        super().__init__()
        self.config = cfg
        self.vocab_size = cfg.vocab_size
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([_NanoBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = _RMSNorm(cfg.dim, eps=cfg.rms_norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.tok_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output(x)
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100,
            )
        return out
