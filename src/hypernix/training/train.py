"""Train or scale up HyperNix models.

Two common flows are supported:

* **Scratch training** — build a new HyperNix-style causal LM at an
  arbitrary size (same as v1, wider, deeper, or a custom shape) and
  pretrain from a raw-text corpus.
* **Model expansion** — take an existing HyperNix checkpoint and grow it
  wider and/or deeper, warm-starting the new weights from the small-model
  weights so you keep the pretraining signal.

Both flows write a HuggingFace-style snapshot directory
(``config.json`` + ``model.safetensors`` + optional tokenizer files), so
the output feeds straight back into :func:`hypernix.convert_to_gguf` or
the ``hypernix convert`` CLI.

The goal is a small but runnable scaffold — not a replacement for a
full-featured trainer. DeepSpeed / FSDP / multi-node are out of scope.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from hypernix.system.torch_compat import scaled_dot_product_attention as _sdpa

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HyperNixConfig:
    """HyperNix model shape. Same layout as v1, parametric in every axis."""

    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    # Qwen2/Qwen2.5 put a bias on q_proj/k_proj/v_proj (but never o_proj);
    # HyperNix-native and Llama-style configs leave this False. The HF
    # Qwen2 model expects this to be True.
    attention_bias: bool = False
    model_type: str = "hypernix"
    # RoPE convention:
    #   "interleaved" - GPT-NeoX / HyperNix-native (cos/sin pairs over ::2/1::2).
    #   "half-rotate" - HuggingFace Llama / Qwen2 (first half vs second half).
    # Loading an HF Llama checkpoint (model_type="llama") uses "half-rotate";
    # loading a HyperNix snapshot uses "interleaved".
    rope_style: str = "interleaved"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["num_key_value_heads"] is None:
            d["num_key_value_heads"] = self.num_attention_heads
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HyperNixConfig:
        # Normalize HF Llama "rope_parameters": {"rope_theta": ..., ...} dict
        # to the flat rope_theta our dataclass expects.
        d = dict(d)
        if "rope_parameters" in d and isinstance(d["rope_parameters"], dict):
            rp = d["rope_parameters"]
            if "rope_theta" in rp and "rope_theta" not in d:
                d["rope_theta"] = rp["rope_theta"]
        # Some multimodal/wrapped configs nest the actual LM config under a
        # "text_config" key instead of putting it at the top level. If the
        # top level is missing every shape-defining field but a nested
        # dict has them, unwrap it rather than silently defaulting.
        _SHAPE_FIELDS = ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads")
        if not any(f in d for f in _SHAPE_FIELDS) and isinstance(d.get("text_config"), dict):
            nested = d["text_config"]
            if any(f in nested for f in _SHAPE_FIELDS):
                d = {**d, **nested}
        # Infer rope_style from model_type when the caller didn't supply one.
        if "rope_style" not in d:
            d["rope_style"] = _default_rope_style(d.get("model_type", "hypernix"))
        fields = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        cfg = cls(**fields)
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        """Catch the common shape errors at config-load time, not deep in forward."""
        if self.hidden_size <= 0 or self.num_attention_heads <= 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} / num_attention_heads="
                f"{self.num_attention_heads} must both be positive"
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"num_attention_heads={self.num_attention_heads}"
            )
        head_dim = self.hidden_size // self.num_attention_heads
        if head_dim % 2 != 0:
            # RoPE pairs adjacent dims — odd head_dim breaks both conventions.
            raise ValueError(
                f"head_dim={head_dim} must be even for RoPE "
                f"(derived from hidden_size={self.hidden_size} / "
                f"num_attention_heads={self.num_attention_heads})"
            )
        n_kv = self.num_key_value_heads or self.num_attention_heads
        if self.num_attention_heads % n_kv != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} must be divisible by "
                f"num_key_value_heads={n_kv} (grouped-query attention constraint)"
            )
        if self.rope_style not in {"interleaved", "half-rotate"}:
            raise ValueError(
                f"rope_style must be 'interleaved' or 'half-rotate', got {self.rope_style!r}"
            )

    def __post_init__(self) -> None:
        # Validate defaults too — a programmatic HyperNixConfig(hidden_size=1025)
        # should fail fast, not silently produce garbage.
        self._validate()

    @classmethod
    def from_json(cls, path: Path | str) -> HyperNixConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def n_kv_head(self) -> int:
        return self.num_key_value_heads or self.num_attention_heads


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm).to(dtype)


def _rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _default_rope_style(model_type: str) -> str:
    """Pick the RoPE convention a given HF ``model_type`` trains with."""
    if model_type in {"llama", "qwen2", "mistral"}:
        return "half-rotate"
    return "interleaved"


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, style: str = "interleaved",
) -> torch.Tensor:
    # x: [B, H, T, D] ; cos/sin: [T, D/2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    if style == "half-rotate":
        # HF Llama / Qwen2: rotate first half against second half.
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    # "interleaved" (default): GPT-NeoX style, pairs (0,1), (2,3), ...
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: HyperNixConfig) -> None:
        super().__init__()
        self.n_head = cfg.num_attention_heads
        self.n_kv = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.rope_style = cfg.rope_style
        hidden = cfg.hidden_size
        qkv_bias = cfg.attention_bias
        self.q_proj = nn.Linear(hidden, self.n_head * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden, self.n_kv * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden, self.n_kv * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin, style=self.rope_style)
        k = _apply_rope(k, cos, sin, style=self.rope_style)
        # Repeat KV heads for grouped-query attention.
        if self.n_kv != self.n_head:
            repeat = self.n_head // self.n_kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        out = _sdpa(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: HyperNixConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: HyperNixConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class HyperNixModel(nn.Module):
    """Llama-shaped HyperNix causal LM. The tensor names match HF conventions
    so the existing architecture-agnostic converter in
    :mod:`hypernix.arch` picks them up without any special casing."""

    def __init__(self, cfg: HyperNixConfig) -> None:
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.embed_tokens(input_ids)
        cos, sin = _rope_cache(input_ids.size(1), self.config.head_dim, self.config.rope_theta, x.device, x.dtype)
        for block in self.layers:
            x = block(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        out = {"logits": logits}
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out


# ---------------------------------------------------------------------------
# Checkpoint I/O (HuggingFace-compatible snapshot layout)
# ---------------------------------------------------------------------------

def save_snapshot(
    model: HyperNixModel,
    out_dir: Path | str,
    tokenizer_source: Path | str | None = None,
) -> Path:
    """Write ``out_dir`` in HuggingFace snapshot layout.

    Produces::

        <out_dir>/config.json
        <out_dir>/model.safetensors
        <out_dir>/tokenizer.json        (copied from tokenizer_source if given)
        <out_dir>/tokenizer.model       (ditto)
        <out_dir>/special_tokens_map.json
        <out_dir>/tokenizer_config.json
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2))
    state = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    # When weights are tied, `embed_tokens.weight` and `lm_head.weight` share
    # memory, which safetensors rejects. Drop the redundant lm_head tensor —
    # the model re-ties on load via HyperNixModel.__init__.
    if model.config.tie_word_embeddings and "lm_head.weight" in state:
        if "embed_tokens.weight" in state and state["lm_head.weight"].data_ptr() == state["embed_tokens.weight"].data_ptr():
            del state["lm_head.weight"]
    save_file(state, str(out / "model.safetensors"))
    if tokenizer_source is not None:
        src = Path(tokenizer_source)
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "merges.txt",
            "added_tokens.json",
        ):
            candidate = src / name
            if candidate.exists():
                shutil.copy2(candidate, out / name)
    return out


def _load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load the full state dict from a snapshot (single file or sharded)."""
    weights = model_dir / "model.safetensors"
    if weights.exists():
        return load_file(str(weights))
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        state.update(load_file(str(shard)))
    return state


def _strip_hf_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop the ``model.`` prefix HF LlamaForCausalLM puts on its tensors so
    they line up with :class:`HyperNixModel`'s flat naming."""
    if not any(k.startswith("model.") for k in state):
        return state
    remapped: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("model."):
            remapped[k[len("model.") :]] = v
        else:
            remapped[k] = v
    return remapped


# Model types our native HyperNixModel handles correctly via shape-compatible
# tensor names (Llama / Qwen2 / Mistral all ship as LlamaForCausalLM-like state
# dicts). Everything else falls through to the transformers AutoModel path.
_NATIVE_MODEL_TYPES: frozenset[str] = frozenset({"hypernix", "llama", "qwen2", "mistral"})


class _HFCausalLMWrapper(nn.Module):
    """Wraps ``transformers.AutoModelForCausalLM`` to match the HyperNixModel
    contract (``forward(input_ids, labels=None) -> {"logits": ..., "loss": ...}``
    and a ``.config`` with ``max_position_embeddings`` + ``model_type``).

    This is what powers broad-model support: any architecture transformers
    can load — Gemma, Phi, DeepSeek, GLM, GPT-OSS, Nemotron, Llama 3+, etc.
    — becomes usable via :func:`hypernix.oven.preheat` without hypernix
    itself having to implement each arch.
    """

    def __init__(self, hf_model, hf_config) -> None:
        super().__init__()
        self.hf_model = hf_model
        self.hf_config = hf_config
        # Oven/generate rely on these two attributes.
        self.config = _HFConfigShim(hf_config)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        out = self.hf_model(input_ids=input_ids, labels=labels)
        result: dict[str, torch.Tensor] = {"logits": out.logits}
        if labels is not None and getattr(out, "loss", None) is not None:
            result["loss"] = out.loss
        return result


class _HFConfigShim:
    """Minimal config facade hypernix's inference code expects.

    Exposes ``max_position_embeddings`` and ``model_type`` (plus a ``to_dict``
    for debugging / save-snapshot). Backing store is the real HF config so
    any other attribute access still works for callers that peek inside.
    """

    def __init__(self, hf_config) -> None:
        self._hf = hf_config

    def __getattr__(self, name: str):
        return getattr(self._hf, name)

    @property
    def max_position_embeddings(self) -> int:
        # HF's Gemma uses max_position_embeddings; some configs use
        # n_positions (GPT-2-ish) or ctx_size. Fall back through them.
        for attr in ("max_position_embeddings", "n_positions", "ctx_size"):
            v = getattr(self._hf, attr, None)
            if v:
                return int(v)
        return 2048

    @property
    def model_type(self) -> str:
        return getattr(self._hf, "model_type", "unknown")

    def to_dict(self) -> dict[str, Any]:
        if hasattr(self._hf, "to_dict"):
            return self._hf.to_dict()
        return dict(self._hf.__dict__)


def _try_hf_automodel(model_dir: Path, model_type: str) -> tuple[Any, Any] | None:
    """Last-resort loader: use transformers.AutoModelForCausalLM for any arch.

    Returns ``(wrapped_model, wrapped_config)`` on success, ``None`` if
    transformers isn't installed or the load fails for any reason. We try
    to auto-install transformers first via ``hypernix.deps.ensure`` so a
    fresh env doesn't silently drop to byte-tokenizer + random-weights land.
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ModuleNotFoundError:
        from hypernix.system import deps
        if not deps.ensure(["transformers>=4.44", "tokenizers>=0.20"]):
            return None
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
        except ModuleNotFoundError:
            return None

    try:
        hf_cfg = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=False)
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), trust_remote_code=False
        )
    except Exception as exc:  # noqa: BLE001 — HF can throw any exception
        import sys as _sys
        print(
            f"[hypernix] AutoModelForCausalLM could not load model_type={model_type!r}: {exc}",
            file=_sys.stderr,
        )
        return None
    wrapped = _HFCausalLMWrapper(hf_model, hf_cfg)
    return wrapped, wrapped.config


_SHAPE_FIELDS = ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads")


def _check_real_shape_fields(cfg_raw: dict[str, Any], model_dir: Path) -> None:
    """Guard against silently building a wrong-shaped model from a real
    downloaded checkpoint.

    ``HyperNixConfig.from_dict`` is a general-purpose classmethod other
    callers legitimately invoke with intentionally-partial dicts (tests,
    fresh-config construction) where falling back to its dataclass
    defaults is exactly what's wanted — so the check can't live there. But
    here in ``load_snapshot``, ``cfg_raw`` is always a real downloaded
    ``config.json`` that's supposed to describe the real checkpoint about
    to be loaded; if literally none of the shape-defining fields resolved
    (e.g. a non-standard config schema — GGUF/llama.cpp-style
    "n_embd"/"n_vocab" naming, or a shape this loader hasn't seen yet),
    silently defaulting produces a model whose dimensions don't match the
    weights, which used to surface 1000+ lines later as a cryptic
    ``load_state_dict`` size-mismatch instead of here, at the actual point
    where the mismatch was introduced.
    """
    text_config = cfg_raw.get("text_config")
    present_top = any(f in cfg_raw for f in _SHAPE_FIELDS)
    present_nested = isinstance(text_config, dict) and any(f in text_config for f in _SHAPE_FIELDS)
    if present_top or present_nested:
        return
    raise ValueError(
        f"load_snapshot({model_dir}): none of the expected shape fields "
        f"({', '.join(_SHAPE_FIELDS)}) were found in config.json (or its "
        f"text_config) — refusing to silently build a model at "
        f"HyperNixConfig's bare defaults (vocab_size={HyperNixConfig.vocab_size}, "
        f"hidden_size={HyperNixConfig.hidden_size}) for a real checkpoint, since "
        f"that would load successfully but produce a wrong-shaped model. "
        f"model_type={cfg_raw.get('model_type')!r}. Raw config.json top-level keys: "
        f"{sorted(cfg_raw.keys())}. If this model uses a renamed/non-standard config "
        f"schema, HyperNixConfig.from_dict needs an alias for it."
    )


def load_snapshot(model_dir: Path | str):
    """Load any supported snapshot from ``model_dir``.

    Dispatches on the ``model_type`` declared in ``config.json``:

    * ``"hypernix"`` / ``"llama"`` / ``"qwen2"`` / ``"mistral"`` — returns a
      native :class:`HyperNixModel` (our parametric Llama-shape); the loader
      automatically strips the HF ``model.`` prefix and picks the correct
      RoPE convention.
    * ``"nano-nano"`` — returns a :class:`hypernix.nano_nano.NanoNanoModel`
      (custom tiny arch used by ``ray0rf1re/nano-nano-927-v3``).
    * **Anything else** — gemma, gemma2, gemma3, phi, phi3, phi4, mistral
      variants, qwen3, llama3+, deepseek, glm4, gpt-oss, nemotron, etc. —
      falls through to ``transformers.AutoModelForCausalLM`` wrapped to
      match the HyperNix inference contract. Requires ``transformers`` to
      be installed (auto-installed on first use unless ``HYPERNIX_AUTO_INSTALL=0``).

    Returns ``(model, config)``.
    """
    model_dir = Path(model_dir)
    cfg_raw = json.loads((model_dir / "config.json").read_text())
    model_type = cfg_raw.get("model_type", "hypernix")

    if model_type == "nano-nano":
        # Custom arch — delegate to the dedicated module so we don't pollute
        # the HyperNix code path with nano-nano-specific tensor remapping.
        from hypernix.models.nano_nano import NanoNanoConfig, NanoNanoModel

        cfg = NanoNanoConfig.from_dict(cfg_raw)
        model = NanoNanoModel(cfg)
        state = _strip_hf_prefix(_load_state_dict(model_dir))
        model.load_state_dict(state, strict=False)
        return model, cfg

    if model_type in _NATIVE_MODEL_TYPES:
        _check_real_shape_fields(cfg_raw, model_dir)
        cfg = HyperNixConfig.from_dict(cfg_raw)
        model = HyperNixModel(cfg)
        state = _strip_hf_prefix(_load_state_dict(model_dir))
        model.load_state_dict(state, strict=False)
        # Re-assert the weight tie: HF checkpoints that stripped lm_head on
        # save leave lm_head at its freshly-initialized random values if
        # load_state_dict ran in strict=False mode. When embeddings are tied,
        # force the pointer identity again so the two heads stay in sync.
        if cfg.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight
        return model, cfg

    # Unknown model_type — try transformers AutoModel before giving up.
    result = _try_hf_automodel(model_dir, model_type)
    if result is not None:
        return result

    # Final fallback: load as HyperNix and warn. This is how the old code
    # behaved implicitly; we now surface the choice to the user.
    import sys as _sys
    print(
        f"[hypernix] WARNING: unknown model_type={model_type!r} and transformers "
        "could not load it; falling back to HyperNixModel which may produce "
        "garbage if the weights aren't Llama-shaped.",
        file=_sys.stderr,
    )
    _check_real_shape_fields(cfg_raw, model_dir)
    cfg = HyperNixConfig.from_dict(cfg_raw)
    model = HyperNixModel(cfg)
    state = _strip_hf_prefix(_load_state_dict(model_dir))
    model.load_state_dict(state, strict=False)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    return model, cfg


# ---------------------------------------------------------------------------
# Model expansion (warm-start a bigger model from a smaller one)
# ---------------------------------------------------------------------------

def _pad_tensor(src: torch.Tensor, dst_shape: tuple[int, ...], init_std: float = 0.02) -> torch.Tensor:
    """Copy ``src`` into a new tensor of ``dst_shape``; newly added slots are
    initialized from ``N(0, init_std)``. Works for 1D and 2D tensors."""
    if tuple(src.shape) == tuple(dst_shape):
        return src.clone()
    dst = torch.randn(*dst_shape, dtype=src.dtype) * init_std
    slices = tuple(slice(0, min(s, d)) for s, d in zip(src.shape, dst_shape, strict=False))
    dst[slices] = src[slices]
    return dst


def expand_checkpoint(
    src_dir: Path | str,
    dst_dir: Path | str,
    *,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    num_hidden_layers: int | None = None,
    num_attention_heads: int | None = None,
    vocab_size: int | None = None,
    init_std: float = 0.02,
    tokenizer_source: Path | str | None = None,
    seed: int | None = None,
) -> Path:
    """Warm-start a bigger HyperNix model from a smaller snapshot.

    Any dimension left ``None`` is inherited from the source. Widening
    copies existing rows/columns into the top-left of the new tensors and
    fills the rest with small random init. Depth expansion duplicates the
    final block weights into the newly-added blocks (a safe starting
    point — the residual path keeps the network functional from step 0).

    Returns the path of ``dst_dir`` (suitable for feeding back into
    :func:`hypernix.convert_to_gguf`).
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    if seed is not None:
        torch.manual_seed(seed)
    old_model, old_cfg = load_snapshot(src)

    new_cfg = HyperNixConfig(
        vocab_size=vocab_size or old_cfg.vocab_size,
        hidden_size=hidden_size or old_cfg.hidden_size,
        intermediate_size=intermediate_size or old_cfg.intermediate_size,
        num_hidden_layers=num_hidden_layers or old_cfg.num_hidden_layers,
        num_attention_heads=num_attention_heads or old_cfg.num_attention_heads,
        num_key_value_heads=old_cfg.num_key_value_heads,
        max_position_embeddings=old_cfg.max_position_embeddings,
        rope_theta=old_cfg.rope_theta,
        rms_norm_eps=old_cfg.rms_norm_eps,
        tie_word_embeddings=old_cfg.tie_word_embeddings,
    )
    if new_cfg.hidden_size % new_cfg.num_attention_heads != 0:
        raise ValueError(
            f"hidden_size={new_cfg.hidden_size} must be divisible by "
            f"num_attention_heads={new_cfg.num_attention_heads}"
        )

    new_model = HyperNixModel(new_cfg)
    old_state = old_model.state_dict()
    new_state = new_model.state_dict()

    # Copy overlapping portions of per-block weights for the first
    # min(old_n_layers, new_n_layers) blocks; duplicate the last old block
    # into any extra blocks.
    old_nl = old_cfg.num_hidden_layers
    for k, new_t in new_state.items():
        if k.startswith("layers."):
            _, idx, *rest = k.split(".")
            src_idx = min(int(idx), old_nl - 1)
            src_key = ".".join(["layers", str(src_idx), *rest])
            if src_key in old_state:
                new_state[k] = _pad_tensor(old_state[src_key], tuple(new_t.shape), init_std)
        elif k in old_state:
            new_state[k] = _pad_tensor(old_state[k], tuple(new_t.shape), init_std)

    new_model.load_state_dict(new_state, strict=True)
    save_snapshot(new_model, dst, tokenizer_source=tokenizer_source or src)
    return dst


# ---------------------------------------------------------------------------
# Minimal training loop
# ---------------------------------------------------------------------------

def _iter_chunks(path: Path, tokenizer, ctx_len: int, bos_id: int | None = None):
    """Stream-chunk a raw-text file into fixed-length token blocks."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    ids = tokenizer.encode(text)
    if bos_id is not None:
        ids = [bos_id, *ids]
    for i in range(0, len(ids) - ctx_len - 1, ctx_len):
        yield torch.tensor(ids[i : i + ctx_len + 1], dtype=torch.long)


class _FusedScheduler:
    """Cosine schedule across per-parameter optimizers.

    ``fuse_optimizer_into_backward`` produces one optimizer per
    parameter, and ``CosineAnnealingLR`` drives exactly one. Rather than
    hold N schedulers that would each advance on the same call, this
    computes the cosine directly and writes it across every param group —
    the same lr every one of them would have arrived at, from one place.
    """

    def __init__(self, handle: Any, total_steps: int) -> None:
        self._optimizers = list(handle.optimizers.values())
        self._base_lrs = [
            [group["lr"] for group in opt.param_groups]
            for opt in self._optimizers
        ]
        self._total = max(1, total_steps)
        self._step = 0

    def step(self) -> None:
        self._step += 1
        # CosineAnnealingLR's closed form, with eta_min=0.
        scale = 0.5 * (1.0 + math.cos(math.pi * min(self._step, self._total) / self._total))
        for opt, bases in zip(self._optimizers, self._base_lrs, strict=True):
            for group, base in zip(opt.param_groups, bases, strict=True):
                group["lr"] = base * scale

    def get_last_lr(self) -> list[float]:
        if not self._optimizers:
            return []
        return [group["lr"] for group in self._optimizers[0].param_groups]


def train(
    model_dir: Path | str,
    dataset_path: Path | str,
    out_dir: Path | str,
    *,
    steps: int = 1000,
    batch_size: int = 2,
    context_length: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    device: str | None = None,
    dtype: str = "float32",
    log_every: int = 10,
    save_every: int = 500,
    seed: int | None = None,
    use_abbicus: bool = False,
    use_turbo_abbicus: bool = False,
    use_stml: bool = False,
    untrained_max_context: int = 8192,
    segment_length: int = 512,
    gradient_checkpointing: bool = False,
    checkpoint_every: int = 1,
    fuse_optimizer: bool = False,
    tune_allocator: bool = False,
    run_id: str | None = None,
) -> Path:
    """Minimal causal-LM training loop.

    This is intentionally barebones (single-GPU/CPU, no sharding, no
    mixed-precision) so it runs anywhere `torch` runs. Use it to smoke-test
    a freshly-expanded model or to run short continue-pretraining jobs;
    anything serious should go through a real trainer.

    Args:
        use_abbicus: Enable Abbicus linear curriculum context scaling.
        use_turbo_abbicus: Enable TurboAbbicus exponential context scaling with
            sine wave oscillation at the hard cap.
        use_stml: Enable STML context segment folding for efficient memory use.
        untrained_max_context: Absolute maximum token count before truncation (STML).
        segment_length: Segment size for STML folding (must be <= context_length).
        gradient_checkpointing: Recompute activations in the backward pass
            instead of storing them. Trades roughly 30% more compute for
            most of the activation memory, which is what a long-context
            run actually runs out of. See :mod:`hypernix.system.vram`.
        checkpoint_every: Checkpoint every Nth block. 1 for all of them,
            2 for half the saving at half the extra compute.
        fuse_optimizer: Apply and free each gradient the moment it is
            ready, instead of holding every gradient between ``backward``
            and ``step`` — one full copy of the model, in gradient dtype,
            at exactly the moment activations peak. Requires
            ``grad_clip=0``: a global norm cannot be computed from one
            gradient at a time, so passing both is an error rather than a
            silently unclipped run.
        tune_allocator: Set ``expandable_segments`` on the CUDA caching
            allocator, so a long run with varying sequence lengths
            fragments less. Off by default because it edits the process
            environment; it has no effect once CUDA is initialized, and
            says so rather than pretending.
        run_id: Publish progress under this id, readable by
            ``GET /training/runs`` and by HyperLink. Defaults to
            ``$HNX_RUN_ID``, which ``hypernix-t1 launch-script`` sets, so
            a detached run is observable without the caller doing
            anything. Reporting is on the ``log_every`` cadence, not
            per-step: the status file is written on exactly the ticks
            that already print a line.
    """
    model_dir = Path(model_dir)
    dataset_path = Path(dataset_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if fuse_optimizer and grad_clip:
        raise ValueError(
            "fuse_optimizer=True steps and frees each gradient as it is "
            "produced, so there is never a moment when every gradient "
            f"exists to take a global norm over. Pass grad_clip=0 to accept "
            f"that, or leave fuse_optimizer off. (grad_clip={grad_clip})"
        )

    if tune_allocator:
        from hypernix.system import vram

        report = vram.configure_allocator()
        print(f"[hypernix.train] {report.report()}", flush=True)

    if seed is not None:
        torch.manual_seed(seed)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tdtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

    model, cfg = load_snapshot(model_dir)
    model.to(dev, dtype=tdtype)
    model.train()

    try:
        from transformers import AutoTokenizer  # lazy import - optional dep
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "hypernix.train needs the `transformers` tokenizer at runtime. "
            "Install it with:  pip install 'hypernix[train]'  "
            "(or `pip install transformers`)."
        ) from exc
    if not (model_dir / "tokenizer.json").exists() and not (model_dir / "tokenizer.model").exists():
        raise FileNotFoundError(
            f"No tokenizer files under {model_dir}. Re-init with "
            "`hypernix train init --tokenizer-source <snapshot_with_tokenizer>` "
            "or copy tokenizer.json/tokenizer.model into the model dir."
        )
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    chunks = list(_iter_chunks(dataset_path, tok, context_length))
    if not chunks:
        raise RuntimeError(f"dataset {dataset_path} produced no training chunks (too short?)")

    if gradient_checkpointing:
        from hypernix.system import vram

        ckpt = vram.checkpoint_blocks(model, every=checkpoint_every)
        print(f"[hypernix.train] {ckpt.report()}", flush=True)

    if fuse_optimizer:
        from hypernix.system import vram

        opt = vram.fuse_optimizer_into_backward(
            model,
            lambda params: torch.optim.AdamW(
                params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
            ),
            grad_clip=None,
        )
        # The scheduler needs something with param_groups to walk. Every
        # per-parameter optimizer shares the same schedule, so stepping
        # one is stepping the schedule; the rest are driven the same way
        # by writing the resulting lr across all of them.
        sched = _FusedScheduler(opt, steps)
        print(
            f"[hypernix.train] optimizer-in-backward over "
            f"{opt.parameter_count} parameters",
            flush=True,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # Build curriculum regulator (Abbicus or TurboAbbicus)
    regulator = None
    if use_turbo_abbicus:
        from hypernix.training.abbicus import TurboAbbicus, TurboAbbicusConfig
        regulator = TurboAbbicus(TurboAbbicusConfig(
            base_context_length=context_length,
            hard_cap=untrained_max_context,
        ))
    elif use_abbicus:
        from hypernix.training.abbicus import Abbicus, AbbicusConfig
        regulator = Abbicus(AbbicusConfig(base_context_length=context_length))

    # Build STML context manager
    stml_mgr = None
    if use_stml:
        from hypernix.models.stml import STML
        stml_mgr = STML(
            trained_context=context_length,
            untrained_max_context=untrained_max_context,
            segment_length=segment_length,
            regulator=regulator,
        )

    reporter = _progress_reporter(run_id, total_steps=steps, model=str(model_dir))

    try:
        step = 0
        while step < steps:
            batch_tensors = [chunks[(step * batch_size + i) % len(chunks)] for i in range(batch_size)]
            batch = torch.stack(batch_tensors).to(dev)
            inputs = batch[:, :-1]
            labels = batch[:, 1:]

            # Apply context regulation (STML wraps the regulator, or regulator alone)
            if stml_mgr is not None:
                if regulator is not None:
                    regulator.step(step)
                batch_dict = {"input_ids": inputs, "labels": labels}
                batch_dict = stml_mgr.regulate(batch_dict)
                inputs = batch_dict["input_ids"]
                labels = batch_dict["labels"]
            elif regulator is not None:
                regulator.step(step)
                batch_dict = {"input_ids": inputs, "labels": labels}
                batch_dict = regulator.regulate(batch_dict)
                inputs = batch_dict["input_ids"]
                labels = batch_dict["labels"]

            out_dict = model(inputs, labels=labels)
            loss = out_dict["loss"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            step += 1

            if step % log_every == 0:
                ctx_len = inputs.shape[1] if hasattr(inputs, "shape") else context_length
                print(f"[hypernix.train] step {step}/{steps}  loss={loss.item():.4f}  ppl={math.exp(min(loss.item(), 20)):.2f}  ctx={ctx_len}")
                if reporter is not None:
                    reporter.update(
                        step=step,
                        loss=loss.item(),
                        lr=(sched.get_last_lr() or [lr])[0],
                        context_length=ctx_len,
                    )
            if save_every and step % save_every == 0:
                save_snapshot(model, out, tokenizer_source=model_dir)
                if reporter is not None:
                    reporter.checkpoint(out)

    except BaseException as exc:
        # Including KeyboardInterrupt and SystemExit: a run that was
        # interrupted must not be left reporting "running" forever,
        # and Ctrl-C is the most common way a run ends.
        if reporter is not None:
            reporter.failed(f"{type(exc).__name__}: {exc}")
        raise

    save_snapshot(model, out, tokenizer_source=model_dir)
    if reporter is not None:
        reporter.checkpoint(out)
        reporter.finished()
    return out



def _progress_reporter(run_id: str | None, *, total_steps: int, model: str):
    """A :class:`~hypernix.training.monitor.ProgressReporter`, or None.

    None whenever no id was asked for *and* nothing can be built — the
    monitor is an observability feature, and a training run must not
    fail to start because the directory it would report into is not
    writable. Any failure here is printed once and then forgotten.
    """
    resolved = run_id or os.environ.get("HNX_RUN_ID", "")
    if not resolved:
        return None
    try:
        from hypernix.training.monitor import ProgressReporter

        return ProgressReporter(
            resolved,
            name=os.environ.get("HNX_JOB_NAME", resolved),
            total_steps=total_steps,
            model=model,
        )
    except Exception as exc:  # noqa: BLE001 - never fail a run over reporting
        print(f"[hypernix.train] progress reporting is off: {exc}", flush=True)
        return None

# ---------------------------------------------------------------------------
# Fresh-init helper
# ---------------------------------------------------------------------------

def init_from_scratch(
    out_dir: Path | str,
    cfg: HyperNixConfig,
    tokenizer_source: Path | str | None = None,
    init_std: float = 0.02,
    seed: int | None = None,
) -> Path:
    """Create a new randomly-initialized HyperNix snapshot at ``out_dir``.

    Pass ``seed`` to make initialization deterministic (useful when a user
    wants to reproduce a bigger-sibling model later via ``expand_checkpoint``).
    """
    if seed is not None:
        torch.manual_seed(seed)
    model = HyperNixModel(cfg)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, std=init_std)
    return save_snapshot(model, out_dir, tokenizer_source=tokenizer_source)
