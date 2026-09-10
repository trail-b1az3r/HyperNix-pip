"""HyperNix Brewer — train custom models from scratch with no base architecture.

The ``hyperNix0x-v2`` preset family is a fully custom PyTorch transformer
architecture inspired by LFM2.5, GPT-NeoX, Gemma4, and Qwen3.5 (text-only).
It uses RMSNorm, RoPE embeddings, Grouped Query Attention (GQA), SwiGLU FFNs,
and optional sliding-window attention.

Quick-start::

    from hypernix.brewer import Brewer, hypernix0x_v2_small

    cfg = hypernix0x_v2_small()
    brewer = Brewer(cfg, name="my-model")
    model = brewer.build()
    brewer.train(data_path="corpus.txt", steps=1000)
    brewer.export(out_path="my-model.gguf", fmt="gguf")

CLI::

    python -m hypernix brew new  --preset small --name my-model --save-dir ./models
    python -m hypernix brew list
    python -m hypernix brew train --name my-model --data corpus.txt --steps 2000
    python -m hypernix brew export --name my-model --format gguf --out my-model.gguf
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

#: Maps registered model name → (BrewerConfig, Path | None) tuples.
BREWER_REGISTRY: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# BrewerConfig
# ---------------------------------------------------------------------------

@dataclass
class BrewerConfig:
    """All hyperparameters for a ``hyperNix0x-v2`` style model.

    Args:
        vocab_size:         Vocabulary size (default 32 000).
        n_layers:           Number of transformer blocks (9 – 36).
        n_heads:            Number of query attention heads.
        n_kv_heads:         Number of key/value heads for GQA.  Must divide
                            ``n_heads`` evenly.  Set equal to ``n_heads`` to
                            disable GQA (standard MHA).
        d_model:            Hidden / embedding dimension.
        d_ff:               Inner dimension of the SwiGLU FFN.  If 0, defaults
                            to ``int(d_model * 8 / 3)`` rounded to the nearest
                            multiple of 256.
        max_seq_len:        Maximum sequence length / context window.
        rope_theta:         RoPE base frequency (default 500 000.0, à la Llama3).
        norm_eps:           Epsilon for RMSNorm (default 1e-5).
        dropout:            Dropout probability applied in attention and FFN.
        tie_embeddings:     Share input-embedding weights with the output LM head.
        use_sliding_window: Enable sliding-window attention on odd-numbered layers.
        sliding_window_size:Local window size when sliding-window attention is on.
        attention_type:     ``"gqa"`` (default) or ``"mha"`` (forces n_kv_heads =
                            n_heads).
        name:               Human-readable label stored in checkpoints.
    """

    vocab_size: int = 32_000
    n_layers: int = 9
    n_heads: int = 32
    n_kv_heads: int = 8
    d_model: int = 1024
    d_ff: int = 0           # computed lazily (see __post_init__)
    max_seq_len: int = 20_482
    rope_theta: float = 500_000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    use_sliding_window: bool = True
    sliding_window_size: int = 4096
    attention_type: str = "gqa"   # "gqa" | "mha"
    name: str = "hypernix0x-v2-custom"

    def __post_init__(self) -> None:
        if self.attention_type == "mha":
            self.n_kv_heads = self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        if self.d_ff == 0:
            raw = int(self.d_model * 8 / 3)
            self.d_ff = (raw + 255) & ~255   # round up to multiple of 256

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BrewerConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BrewerConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def approx_params(self) -> int:
        """Rough parameter count estimate (millions would be /1e6)."""
        head_dim = self.d_model // self.n_heads
        embed = self.vocab_size * self.d_model
        attn = self.d_model * (
            self.n_heads * head_dim          # Q
            + self.n_kv_heads * head_dim     # K
            + self.n_kv_heads * head_dim     # V
            + self.n_heads * head_dim        # O
        )
        ffn = 3 * self.d_model * self.d_ff  # gate + up + down
        norms = 4 * self.d_model            # 2 norms per block + final
        per_block = attn + ffn + norms
        total = embed + self.n_layers * per_block
        if not self.tie_embeddings:
            total += self.vocab_size * self.d_model
        return total


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class BrewerRMSNorm(nn.Module):
    """Root-mean-square layer normalisation (no bias, no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine / sine tables for RoPE up to *seq_len* positions."""
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)          # (seq_len, half)
    emb = torch.cat([freqs, freqs], dim=-1)           # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k shape: (B, n_heads, T, head_dim)
    # cos/sin shape: (T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# BrewerEmbedding
# ---------------------------------------------------------------------------

class BrewerEmbedding(nn.Module):
    """Token embedding layer.  RoPE is applied inside :class:`BrewerAttention`."""

    def __init__(self, cfg: BrewerConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embed(token_ids))


# ---------------------------------------------------------------------------
# BrewerAttention
# ---------------------------------------------------------------------------

class BrewerAttention(nn.Module):
    """Multi-head / Grouped Query Attention with RoPE and optional sliding window.

    When ``use_sliding_window=True`` and this layer index is odd, local
    attention is applied using a causal sliding-window mask of size
    ``sliding_window_size``.
    """

    def __init__(self, cfg: BrewerConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.n_rep = self.n_heads // self.n_kv_heads   # GQA repetition factor
        self.use_sliding_window = cfg.use_sliding_window and (layer_idx % 2 == 1)
        self.sliding_window_size = cfg.sliding_window_size
        self.dropout_p = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.rope_theta = cfg.rope_theta
        self._rope_cache: dict[tuple, tuple] = {}

    # ------------------------------------------------------------------

    def _get_rope(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (seq_len, device, dtype)
        if key not in self._rope_cache:
            self._rope_cache[key] = _build_rope_cache(
                seq_len, self.head_dim, self.rope_theta, device, dtype
            )
        return self._rope_cache[key]

    def _sliding_mask(
        self, T: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return an additive causal sliding-window mask of shape (T, T).

        ``dist[i][j] = i - j`` is how far key *j* lies in the past of query
        *i*: positive is the past, zero is the token itself, negative is
        the future. A causal local window keeps ``0 <= dist <= win - 1``
        and masks everything else.

        This used to read ``(dist >= 0) | (dist < -(win - 1))``, which
        masks out the past and the token itself and *keeps* the strictly
        future positions inside the window -- an anti-causal band. On its
        own that is only half the story; see :meth:`forward` for what the
        two masks did together.
        """
        win = self.sliding_window_size
        idx = torch.arange(T, device=device)
        dist = idx.unsqueeze(1) - idx.unsqueeze(0)   # (T, T); + is the past
        mask = (dist < 0) | (dist > win - 1)         # True = mask out
        return mask.to(dtype) * torch.finfo(dtype).min

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Project
        q = self.q_proj(x)                                # (B, T, n_heads*hd)
        k = self.k_proj(x)                                # (B, T, n_kv_heads*hd)
        v = self.v_proj(x)

        # Reshape → (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self._get_rope(T, x.device, q.dtype)
        q, k = _apply_rope(q, k, cos, sin)

        # GQA: expand KV heads to match Q heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # A plain causal layer needs no mask tensor at all: is_causal lets
        # SDPA take its fused path instead of materialising a B*H*T*T
        # score matrix to add a mask to. is_causal and attn_mask are
        # mutually exclusive, so this is only the no-window, no-padding
        # case -- which is every even layer.
        if not self.use_sliding_window and attn_mask is None:
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        else:
            causal_mask = torch.full(
                (T, T), torch.finfo(q.dtype).min, device=x.device, dtype=q.dtype
            )
            causal_mask = causal_mask.triu(diagonal=1)  # upper-triangular = -inf

            if self.use_sliding_window:
                sw_mask = self._sliding_mask(T, x.device, q.dtype)
                # Mask if EITHER is masked. These are *additive* masks --
                # 0 allows, finfo.min forbids -- so "either" is the
                # elementwise **minimum**. torch.maximum keeps the less
                # masked of the two, which is "allow if either allows",
                # and it let every query on an odd layer attend to its own
                # future: with the default window of 4096 and any sequence
                # no longer than that, the layer was fully bidirectional.
                # The model trains to a good loss and generates nothing.
                #
                # The two halves of this bug are coupled: minimum against
                # the old anti-causal window mask masks every position of
                # every row, and fixing that mask while keeping maximum
                # makes the window a silent no-op equal to plain causal.
                causal_mask = torch.minimum(causal_mask, sw_mask)

            if attn_mask is not None:
                causal_mask = causal_mask + attn_mask

            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=causal_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

        # Merge heads → (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# BrewerFFN — SwiGLU
# ---------------------------------------------------------------------------

class BrewerFFN(nn.Module):
    """SwiGLU feed-forward network.

    Computes ``gate * silu(x) → down``, where the gate and up projections are
    separate linear layers (no bias), matching the Llama / Gemma / Qwen style.
    """

    def __init__(self, cfg: BrewerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj   = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout   = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))


# ---------------------------------------------------------------------------
# BrewerBlock
# ---------------------------------------------------------------------------

class BrewerBlock(nn.Module):
    """Single Pre-Norm transformer block: RMSNorm → Attention → RMSNorm → FFN."""

    def __init__(self, cfg: BrewerConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm1   = BrewerRMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn    = BrewerAttention(cfg, layer_idx)
        self.norm2   = BrewerRMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn     = BrewerFFN(cfg)
        self.drop    = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention residual
        x = x + self.drop(self.attn(self.norm1(x), attn_mask))
        # FFN residual
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# BrewerModel
# ---------------------------------------------------------------------------

class BrewerModel(nn.Module):
    """Full decoder-only transformer stack for the ``hyperNix0x-v2`` family.

    This is the raw PyTorch model.  Use :class:`Brewer` for the high-level API.
    """

    def __init__(self, cfg: BrewerConfig) -> None:
        super().__init__()
        self.cfg        = cfg
        self.embed      = BrewerEmbedding(cfg)
        self.blocks     = nn.ModuleList(
            [BrewerBlock(cfg, i) for i in range(cfg.n_layers)]
        )
        self.norm_out   = BrewerRMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head    = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.embed.weight

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Scaled initialisation following the GPT-NeoX / LLaMA approach."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Scale residual projections by 1/sqrt(2*n_layers) (GPT-NeoX style)
        scale = (2 * self.cfg.n_layers) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=std * scale)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=std * scale)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids:  LongTensor of shape (B, T).
            attn_mask:  Optional additive float mask of shape (T, T) or
                        (B, 1, T, T).

        Returns:
            Logits tensor of shape (B, T, vocab_size).
        """
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x, attn_mask)
        x = self.norm_out(x)
        return self.lm_head(x)

    def num_params(self, trainable_only: bool = False) -> int:
        params = (p for p in self.parameters() if p.requires_grad) if trainable_only else self.parameters()
        return sum(p.numel() for p in params)


# ---------------------------------------------------------------------------
# Preset factory functions
# ---------------------------------------------------------------------------

def hypernix0x_v2_33m() -> BrewerConfig:
    """``hyperNix0x-v2-33m`` — 6 layers, ~33.6429M params (33,642,900 parameters), ctx=4096.

    Designed for lightweight edge devices, fast inference, and image text-to-text models.
    """
    return BrewerConfig(
        name="hypernix0x-v2-33m",
        vocab_size=32_000,
        n_layers=6,
        n_heads=16,
        n_kv_heads=4,
        d_model=512,
        d_ff=1444,            # tuned for 33.6429M total parameters
        max_seq_len=4096,
        rope_theta=100_000.0,
        norm_eps=1e-5,
        dropout=0.0,
        tie_embeddings=True,
        use_sliding_window=True,
        sliding_window_size=1024,
        attention_type="gqa",
    )


def hypernix0x_v2_micro() -> BrewerConfig:
    """Alias for ``hypernix0x-v2-33m`` (33.6429M parameters)."""
    return hypernix0x_v2_33m()


def hypernix0x_v2_small() -> BrewerConfig:
    """``hyperNix0x-v2-small`` — 9 layers, ~458 M params, ctx=20 482.

    Inspired by LFM2.5 / GPT-NeoX / Gemma4 / Qwen3.5 design choices.
    """
    return BrewerConfig(
        name="hypernix0x-v2-small",
        vocab_size=32_000,
        n_layers=9,
        n_heads=32,
        n_kv_heads=8,
        d_model=1024,
        d_ff=0,            # auto: ~2816
        max_seq_len=20_482,
        rope_theta=500_000.0,
        norm_eps=1e-5,
        dropout=0.0,
        tie_embeddings=True,
        use_sliding_window=True,
        sliding_window_size=4096,
        attention_type="gqa",
    )


def hypernix0x_v2_medium() -> BrewerConfig:
    """``hyperNix0x-v2-medium`` — 18 layers, ~918 M params, ctx=40 964."""
    return BrewerConfig(
        name="hypernix0x-v2-medium",
        vocab_size=32_000,
        n_layers=18,
        n_heads=32,
        n_kv_heads=8,
        d_model=1280,
        d_ff=0,            # auto: ~3328
        max_seq_len=40_964,
        rope_theta=500_000.0,
        norm_eps=1e-5,
        dropout=0.0,
        tie_embeddings=True,
        use_sliding_window=True,
        sliding_window_size=8192,
        attention_type="gqa",
    )


def hypernix0x_v2_large() -> BrewerConfig:
    """``hyperNix0x-v2-large`` — 36 layers, ~3.5 B params, ctx=103 724."""
    return BrewerConfig(
        name="hypernix0x-v2-large",
        vocab_size=32_000,
        n_layers=36,
        n_heads=32,
        n_kv_heads=8,
        d_model=2048,
        d_ff=0,            # auto: ~5376
        max_seq_len=103_724,
        rope_theta=500_000.0,
        norm_eps=1e-5,
        dropout=0.0,
        tie_embeddings=True,
        use_sliding_window=True,
        sliding_window_size=16_384,
        attention_type="gqa",
    )


# ---------------------------------------------------------------------------
# CPU-oriented presets — noticeably smaller than hypernix0x-v2-33m (the
# smallest of the presets above, itself pitched at "lightweight edge
# devices" rather than CPU specifically). Plain MHA (no GQA) and no
# sliding window on all three: at these sizes and context lengths neither
# buys real speed, just extra code paths — simplicity was prioritized over
# squeezing out marginal throughput a CPU won't meaningfully benefit from
# anyway. Parameter counts below are measured (BrewerModel(cfg).num_params()),
# not estimated, matching how hypernix0x_v2_33m documents its own count.
# ---------------------------------------------------------------------------

def hypernix0x_v2_cpu_nano() -> BrewerConfig:
    """``hyperNix0x-v2-cpu-nano`` — 4 layers, ~2.0737M params (2,073,728 parameters), ctx=512.

    Smallest preset in the family. Sized for CPU-only training and
    inference — smoke tests, quick architecture prototyping, low-power
    devices — not production output quality.
    """
    return BrewerConfig(
        name="hypernix0x-v2-cpu-nano",
        vocab_size=8_000,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        d_model=128,
        d_ff=0,            # auto: 512
        max_seq_len=512,
        rope_theta=10_000.0,
        norm_eps=1e-5,
        dropout=0.0,
        tie_embeddings=True,
        use_sliding_window=False,
        attention_type="mha",
    )


def hypernix0x_v2_cpu_tiny() -> BrewerConfig:
    """``hyperNix0x-v2-cpu-tiny`` — 6 layers, ~9.2111M params (9,211,136 parameters), ctx=1024.

    A step up from cpu-nano — still comfortably trainable on CPU, with
    more capacity and context than a pure smoke test needs.
    """
    return BrewerConfig(
        name="hypernix0x-v2-cpu-tiny",
        vocab_size=16_000,
        n_layers=6,
        n_heads=8,
        n_kv_heads=8,
        d_model=256,
        d_ff=0,            # auto: 768
        max_seq_len=1024,
        rope_theta=10_000.0,
        norm_eps=1e-5,
        dropout=0.0,
        tie_embeddings=True,
        use_sliding_window=False,
        attention_type="mha",
    )


def hypernix0x_v2_cpu_small() -> BrewerConfig:
    """``hyperNix0x-v2-cpu-small`` — 8 layers, ~26.4503M params (26,450,304 parameters), ctx=2048.

    The largest CPU-oriented preset — still noticeably smaller than
    ``hypernix0x-v2-33m``/``micro`` (the smallest GPU-oriented preset above)
    so CPU training stays practical, while having enough capacity for more
    than a smoke test.
    """
    return BrewerConfig(
        name="hypernix0x-v2-cpu-small",
        vocab_size=32_000,
        n_layers=8,
        n_heads=8,
        n_kv_heads=8,
        d_model=384,
        d_ff=0,            # auto: 1024
        max_seq_len=2048,
        rope_theta=10_000.0,
        norm_eps=1e-5,
        dropout=0.0,
        tie_embeddings=True,
        use_sliding_window=False,
        attention_type="mha",
    )


def custom_arch(**kwargs) -> BrewerConfig:
    """Create a fully user-defined :class:`BrewerConfig` by keyword argument.

    Any field of :class:`BrewerConfig` can be overridden.  Unspecified fields
    take their default values.

    Example::

        cfg = custom_arch(n_layers=12, d_model=512, max_seq_len=8192)
    """
    return BrewerConfig(**kwargs)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class _SimpleTextDataset:
    """Character-level text dataset backed by a single UTF-8 file.

    Chunks the file into non-overlapping windows of ``seq_len`` tokens.
    All characters are mapped to a compact integer vocabulary.
    """

    def __init__(self, path: str | Path, seq_len: int) -> None:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self.itos: dict[int, str] = {i: c for c, i in self.stoi.items()}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        # Trim to full chunks
        n_chunks = (len(data) - 1) // seq_len
        if n_chunks < 1:
            raise ValueError(
                f"Data file too short for seq_len={seq_len}. "
                f"Need at least {seq_len + 1} characters, got {len(data)}."
            )
        self.data = data[: n_chunks * seq_len + 1]
        self.seq_len = seq_len
        self.n_chunks = n_chunks

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y

    def random_batch(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, len(self), (batch_size,))
        xs, ys = zip(*[self[i] for i in indices.tolist()], strict=True)
        return torch.stack(xs).to(device), torch.stack(ys).to(device)


def train_model(
    model: BrewerModel,
    config: BrewerConfig,
    data_path: str | Path,
    steps: int,
    lr: float = 3e-4,
    batch_size: int = 4,
    device: str = "auto",
    checkpoint_dir: str | Path | None = None,
    log_callback: Callable[[int, float], None] | None = None,
) -> None:
    """Train *model* in-place on a plain-text corpus.

    This is a minimal but fully functional training loop suitable for
    character-level experiments.  For production workloads you should
    use a proper tokeniser and data pipeline.

    Args:
        model:          A :class:`BrewerModel` (already initialised).
        config:         The :class:`BrewerConfig` used to build *model*.
        data_path:      Path to a UTF-8 text file used as the training corpus.
        steps:          Total number of gradient update steps.
        lr:             AdamW learning rate.
        batch_size:     Sequences per micro-batch.
        device:         ``"auto"`` selects CUDA → MPS → CPU automatically.
        checkpoint_dir: Directory to save ``step_*.pt`` checkpoints every 500
                        steps.  Created if it doesn't exist.  Skipped if None.
        log_callback:   Optional ``(step, loss) → None`` callable invoked at
                        every logging step (every 100 steps).
    """
    # ---- Device selection -------------------------------------------------
    if device == "auto":
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    else:
        _device = torch.device(device)

    model = model.to(_device)
    model.train()

    # ---- Dataset ----------------------------------------------------------
    seq_len = min(config.max_seq_len, 512)   # cap training length for RAM
    dataset = _SimpleTextDataset(data_path, seq_len)

    # Warn if model vocab_size > dataset vocab_size (fine) or vice-versa (bad).
    if dataset.vocab_size > config.vocab_size:
        print(
            f"[brewer] WARNING: dataset has {dataset.vocab_size} unique chars but "
            f"model vocab_size={config.vocab_size}.  Indices will overflow — "
            f"consider increasing vocab_size.",
            file=sys.stderr,
        )

    # ---- Optimizer --------------------------------------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )

    # Linear warm-up over the first 10% of steps, then cosine decay.
    warmup_steps = max(1, steps // 10)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ---- Checkpoint dir ---------------------------------------------------
    ckpt_dir: Path | None = None
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop ----------------------------------------------------
    print(
        f"[brewer] Training '{config.name}' — "
        f"{model.num_params():,} params on {_device} "
        f"for {steps} steps (seq_len={seq_len}, batch={batch_size}).",
        flush=True,
    )
    t0 = time.perf_counter()
    running_loss = 0.0

    for step in range(1, steps + 1):
        x, y = dataset.random_batch(batch_size, _device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)                     # (B, T, vocab_size)
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            y.view(-1),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % 100 == 0:
            avg_loss = running_loss / 100
            running_loss = 0.0
            elapsed = time.perf_counter() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"[brewer] step {step:>6}/{steps}  "
                f"loss={avg_loss:.4f}  lr={lr_now:.2e}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            if log_callback is not None:
                log_callback(step, avg_loss)

        if ckpt_dir is not None and step % 500 == 0:
            ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config.to_dict(),
                    "loss": avg_loss if step % 100 == 0 else None,
                },
                ckpt_path,
            )
            print(f"[brewer] checkpoint saved → {ckpt_path}", flush=True)

    total_time = time.perf_counter() - t0
    print(f"[brewer] Training complete in {total_time:.1f}s.", flush=True)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _export_pt(model: BrewerModel, config: BrewerConfig, out_path: Path) -> None:
    """Save model as a standard PyTorch state-dict bundle."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "arch": "hypernix0x-v2",
            "version": 2,
        },
        out_path,
    )
    print(f"[brewer] Exported PyTorch model → {out_path}")


def _export_gguf(model: BrewerModel, config: BrewerConfig, out_path: Path) -> None:
    """Export a minimal GGUF-like flat binary (F32 tensors, JSON header).

    This produces a self-describing binary that downstream tools can parse.
    For full GGUF compatibility use llama.cpp's convert scripts on the
    intermediate PyTorch export.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import struct

    header = json.dumps(
        {"arch": "hypernix0x-v2", "config": config.to_dict(), "version": 2},
        separators=(",", ":"),
    ).encode("utf-8")

    state = model.state_dict()
    tensor_index: list[dict] = []
    blobs: list[bytes] = []
    offset = 0

    for name, tensor in state.items():
        data = tensor.cpu().float().numpy().tobytes()
        tensor_index.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": "F32",
                "offset": offset,
                "size": len(data),
            }
        )
        blobs.append(data)
        offset += len(data)

    index_bytes = json.dumps(tensor_index, separators=(",", ":")).encode("utf-8")

    with open(out_path, "wb") as f:
        # Magic + version
        f.write(b"HNXG")                              # HyperNiX GGuf
        f.write(struct.pack("<I", 2))                  # format version
        # Header length + content
        f.write(struct.pack("<I", len(header)))
        f.write(header)
        # Index length + content
        f.write(struct.pack("<I", len(index_bytes)))
        f.write(index_bytes)
        # Tensor data
        for blob in blobs:
            f.write(blob)

    print(f"[brewer] Exported GGUF-style model → {out_path} ({out_path.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# Brewer — high-level API
# ---------------------------------------------------------------------------

class Brewer:
    """High-level API for building, training, and exporting ``hyperNix0x-v2`` models.

    Example::

        from hypernix.brewer import Brewer, hypernix0x_v2_small

        brewer = Brewer(hypernix0x_v2_small(), name="my-small")
        model  = brewer.build()
        brewer.train(data_path="wiki.txt", steps=5000)
        brewer.export("my-small.gguf", fmt="gguf")
    """

    def __init__(
        self,
        config: BrewerConfig,
        name: str | None = None,
        save_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.name = name or config.name
        self.save_dir = Path(save_dir) if save_dir else Path.cwd() / "brewer_models" / self.name
        self._model: BrewerModel | None = None
        self._register()

    # ------------------------------------------------------------------

    def _register(self) -> None:
        BREWER_REGISTRY[self.name] = {
            "config": self.config.to_dict(),
            "save_dir": str(self.save_dir),
            "built": False,
            "trained": False,
        }
        # Optionally insert a stub entry into download.KNOWN_MODELS
        try:
            from hypernix.models.download import KNOWN_MODELS, ModelInfo  # type: ignore
            key = self.name.lower()
            if key not in KNOWN_MODELS:
                KNOWN_MODELS[key] = ModelInfo(
                    repo_id=f"local/brewer/{self.name}",
                    arch="hypernix0x-v2",
                    notes=f"Locally brewed model '{self.name}'.",
                    family="brewer",
                )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------

    def build(self, device: str = "cpu") -> BrewerModel:
        """Instantiate and return the :class:`BrewerModel`.

        The model is also stored as ``self.model`` for subsequent calls to
        :meth:`train` and :meth:`export`.
        """
        self._model = BrewerModel(self.config)
        self._model = self._model.to(torch.device(device))
        BREWER_REGISTRY[self.name]["built"] = True
        params = self._model.num_params()
        print(
            f"[brewer] Built '{self.name}' — "
            f"{params:,} params ({params / 1e6:.1f} M)  "
            f"n_layers={self.config.n_layers}  "
            f"d_model={self.config.d_model}  "
            f"ctx={self.config.max_seq_len:,}",
            flush=True,
        )
        return self._model

    @property
    def model(self) -> BrewerModel:
        if self._model is None:
            raise RuntimeError("Model not built yet — call .build() first.")
        return self._model

    # ------------------------------------------------------------------

    def train(
        self,
        data_path: str | Path,
        steps: int = 1000,
        lr: float = 3e-4,
        batch_size: int = 4,
        device: str = "auto",
        log_callback: Callable[[int, float], None] | None = None,
    ) -> None:
        """Train the model on a text corpus.  Builds if not yet built."""
        if self._model is None:
            self.build()
        train_model(
            model=self.model,
            config=self.config,
            data_path=data_path,
            steps=steps,
            lr=lr,
            batch_size=batch_size,
            device=device,
            checkpoint_dir=self.save_dir / "checkpoints",
            log_callback=log_callback,
        )
        BREWER_REGISTRY[self.name]["trained"] = True

    # ------------------------------------------------------------------

    def export(
        self,
        out_path: str | Path | None = None,
        fmt: str = "pt",
    ) -> Path:
        """Export the model to a file.

        Args:
            out_path: Destination file.  Defaults to ``<save_dir>/<name>.<fmt>``.
            fmt:      ``"pt"`` (PyTorch state-dict) or ``"gguf"``.

        Returns:
            The resolved output path.
        """
        if out_path is None:
            ext = "gguf" if fmt == "gguf" else "pt"
            out_path = self.save_dir / f"{self.name}.{ext}"
        out = Path(out_path)

        fmt_lower = fmt.lower()
        if fmt_lower == "gguf":
            _export_gguf(self.model, self.config, out)
        elif fmt_lower in ("pt", "pytorch"):
            _export_pt(self.model, self.config, out)
        else:
            raise ValueError(f"Unknown export format: {fmt!r}. Choose 'pt' or 'gguf'.")
        return out

    # ------------------------------------------------------------------

    def save_config(self) -> Path:
        """Persist the :class:`BrewerConfig` to ``<save_dir>/config.json``."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.save_dir / "config.json"
        self.config.save(cfg_path)
        print(f"[brewer] Config saved → {cfg_path}")
        return cfg_path

    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, name: str | None = None) -> Brewer:
        """Restore a :class:`Brewer` (with built model) from a ``.pt`` checkpoint."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = BrewerConfig.from_dict(ckpt["config"])
        brewer = cls(cfg, name=name or cfg.name)
        brewer.build()
        brewer.model.load_state_dict(ckpt["model_state_dict"])
        print(f"[brewer] Loaded checkpoint from {ckpt_path}")
        return brewer

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        built = self._model is not None
        trained = BREWER_REGISTRY.get(self.name, {}).get("trained", False)
        return (
            f"Brewer(name={self.name!r}, "
            f"preset={self.config.name!r}, "
            f"built={built}, trained={trained})"
        )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

_PRESET_MAP: dict[str, Callable[[], BrewerConfig]] = {
    "cpu-nano":  hypernix0x_v2_cpu_nano,
    "cpu-tiny":  hypernix0x_v2_cpu_tiny,
    "cpu-small": hypernix0x_v2_cpu_small,
    "33m":    hypernix0x_v2_33m,
    "micro":  hypernix0x_v2_micro,
    "small":  hypernix0x_v2_small,
    "medium": hypernix0x_v2_medium,
    "large":  hypernix0x_v2_large,
}


def _cmd_new(args: argparse.Namespace) -> None:
    preset_fn = _PRESET_MAP.get(args.preset.lower())
    if preset_fn is None:
        print(
            f"[brewer] Unknown preset {args.preset!r}. "
            f"Available: {', '.join(_PRESET_MAP)}",
            file=sys.stderr,
        )
        sys.exit(1)
    cfg = preset_fn()
    if args.name:
        cfg.name = args.name
    save_dir = Path(args.save_dir) / cfg.name if args.save_dir else None
    brewer = Brewer(cfg, name=cfg.name, save_dir=save_dir)
    brewer.build()
    brewer.save_config()
    print(f"[brewer] Model '{cfg.name}' created at {brewer.save_dir}")


def _cmd_list(_args: argparse.Namespace) -> None:
    if not BREWER_REGISTRY:
        print("[brewer] No models registered in this session.")
        return
    print(f"[brewer] Registered models ({len(BREWER_REGISTRY)}):")
    for name, info in BREWER_REGISTRY.items():
        cfg_info = info.get("config", {})
        print(
            f"  • {name!r:30s}  "
            f"layers={cfg_info.get('n_layers', '?')}  "
            f"d_model={cfg_info.get('d_model', '?')}  "
            f"built={info.get('built')}  "
            f"trained={info.get('trained')}"
        )


def _cmd_train(args: argparse.Namespace) -> None:
    if args.name not in BREWER_REGISTRY:
        print(
            f"[brewer] Model '{args.name}' not found in registry. "
            "Did you run 'brew new' first?",
            file=sys.stderr,
        )
        sys.exit(1)
    info = BREWER_REGISTRY[args.name]
    cfg = BrewerConfig.from_dict(info["config"])
    brewer = Brewer(cfg, name=args.name, save_dir=info.get("save_dir"))
    brewer.build()
    brewer.train(
        data_path=args.data,
        steps=int(args.steps),
    )


def _cmd_export(args: argparse.Namespace) -> None:
    if args.name not in BREWER_REGISTRY:
        print(
            f"[brewer] Model '{args.name}' not found in registry.",
            file=sys.stderr,
        )
        sys.exit(1)
    info = BREWER_REGISTRY[args.name]
    cfg = BrewerConfig.from_dict(info["config"])
    brewer = Brewer(cfg, name=args.name, save_dir=info.get("save_dir"))
    brewer.build()
    out = brewer.export(out_path=args.out, fmt=args.format)
    print(f"[brewer] Exported → {out}")


def cli_main(argv: list[str] | None = None) -> None:
    """Entry point for the ``brew`` sub-command family.

    Can be wired into the main HyperNix CLI or invoked standalone::

        python -m hypernix.brewer brew new --preset small --name my-model
    """
    parser = argparse.ArgumentParser(
        prog="brew",
        description="HyperNix Brewer — build and train custom models from scratch.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- brew new ---------------------------------------------------------
    p_new = sub.add_parser("new", help="Scaffold a new model from a preset.")
    p_new.add_argument(
        "--preset", required=True,
        choices=list(_PRESET_MAP.keys()),
        help="Architecture preset: cpu-nano | cpu-tiny | cpu-small | 33m | micro | small | medium | large",
    )
    p_new.add_argument("--name", default=None, help="Model name (overrides preset name).")
    p_new.add_argument("--save-dir", default=None, help="Root directory to save model files.")

    # ---- brew list --------------------------------------------------------
    sub.add_parser("list", help="List all registered brewer models.")

    # ---- brew train -------------------------------------------------------
    p_train = sub.add_parser("train", help="Train a registered model.")
    p_train.add_argument("--name", required=True, help="Registered model name.")
    p_train.add_argument("--data", required=True, help="Path to training text file.")
    p_train.add_argument("--steps", default=1000, type=int, help="Training steps.")
    p_train.add_argument("--lr", default=3e-4, type=float, help="Learning rate.")
    p_train.add_argument("--batch-size", default=4, type=int, help="Batch size.")
    p_train.add_argument("--device", default="auto", help="Device: auto | cpu | cuda | mps.")

    # ---- brew export ------------------------------------------------------
    p_export = sub.add_parser("export", help="Export a registered model.")
    p_export.add_argument("--name", required=True, help="Registered model name.")
    p_export.add_argument(
        "--format", default="gguf", choices=["gguf", "pt"],
        help="Export format: gguf | pt.",
    )
    p_export.add_argument("--out", default=None, help="Output file path.")

    ns = parser.parse_args(argv)

    dispatch = {
        "new":    _cmd_new,
        "list":   _cmd_list,
        "train":  _cmd_train,
        "export": _cmd_export,
    }
    dispatch[ns.command](ns)


# ---------------------------------------------------------------------------
# Module-level __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    "BrewerConfig",
    # Model components
    "BrewerRMSNorm",
    "BrewerEmbedding",
    "BrewerAttention",
    "BrewerFFN",
    "BrewerBlock",
    "BrewerModel",
    # High-level API
    "Brewer",
    # Preset factories
    "hypernix0x_v2_33m",
    "hypernix0x_v2_micro",
    "hypernix0x_v2_small",
    "hypernix0x_v2_medium",
    "hypernix0x_v2_large",
    "custom_arch",
    # Training
    "train_model",
    # Registry
    "BREWER_REGISTRY",
    # CLI
    "cli_main",
]


if __name__ == "__main__":
    cli_main()
