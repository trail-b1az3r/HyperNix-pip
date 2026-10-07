"""Fetch the HyperNix model snapshot from the HuggingFace Hub.

The downloader pulls every file a HuggingFace-style checkpoint needs to
round-trip through :mod:`hypernix.convert`:

* model weights (safetensors / bin / loose .pt)
* config.json and generation_config.json
* tokenizer files (tokenizer.json, tokenizer.model, vocab*, merges*,
  special_tokens_map.json, tokenizer_config.json, added_tokens.json,
  chat_template.json / chat_template.jinja)
* any index.json sharded-weight manifests

After the download completes we run :func:`verify_snapshot` to make sure
the resulting directory can actually be consumed by ``hypernix convert``
— i.e. it contains at least one weight file *and* a ``config.json``.
"""
from __future__ import annotations

# PEP 810: on Python 3.15+ these imports are deferred to first use; on
# 3.12-3.14 the list is an ordinary variable and they run as before.
# hyped imports KNOWN_MODELS from here; only a download needs huggingface_hub.
__lazy_modules__ = ["huggingface_hub"]

import sys
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a known model on the HuggingFace Hub.

    ``arch`` is a short tag indicating which code path loads this model:

    * ``"hypernix"`` — HyperNix-native Llama-shape; loads via
      :class:`HyperNixModel` with interleaved RoPE and no q/k/v bias.
    * ``"llama"`` / ``"qwen2"`` / ``"mistral"`` — HF-shaped state dict with
      the ``model.`` prefix; loaded natively by :class:`HyperNixModel`.
    * ``"nano-nano"`` — custom ``NanoNanoModel`` (tiny toy arch).
    * ``"auto"`` — anything else (gemma, phi, deepseek, glm4, gpt-oss,
      nemotron, qwen3, llama3+ MoE, etc.). Loaded via
      ``transformers.AutoModelForCausalLM``.
    """

    repo_id: str
    arch: str
    notes: str = ""
    family: str = ""


# Registry of known HyperNix-family repos on the HuggingFace Hub. Users can
# pass either a short name (``"nano-nano-v4"``) or a full repo id
# (``"ray0rf1re/Nano-nano-v4"``) to :func:`download_model` / :func:`preheat`;
# short names resolve via this dict. Keys are case-insensitive.
KNOWN_MODELS: dict[str, ModelInfo] = {
    "hyper-nix.1": ModelInfo(
        "ray0rf1re/hyper-nix.1", "hypernix",
        "HyperNix v1 — Llama-shaped native HyperNix.",
    ),
    "hyper-nix.2": ModelInfo(
        "ray0rf1re/hyper-Nix.2", "hypernix",
        "HyperNix v2 — chat-tuned, Llama-shaped native HyperNix. Ships a "
        "ChatML-style chat template (see hypernix.cookbook.HYPER_NIX_2). "
        "Use hypernix.countertop for multi-turn sessions.",
    ),
    "hyper-nix2": ModelInfo(
        "ray0rf1re/hyper-Nix.2", "hypernix", "Alias for hyper-nix.2.",
    ),
    "hypernix2": ModelInfo(
        "ray0rf1re/hyper-Nix.2", "hypernix", "Alias for hyper-nix.2.",
    ),
    "hyper-nix": ModelInfo(
        "ray0rf1re/hyper-Nix.2", "hypernix",
        "Alias for hyper-nix.2 (latest HyperNix release).",
    ),
    "hypernix": ModelInfo(
        "ray0rf1re/hyper-Nix.2", "hypernix",
        "Alias for hyper-nix.2 (latest HyperNix release).",
    ),
    "nano-nano-v4": ModelInfo(
        "ray0rf1re/Nano-nano-v4", "llama",
        "HF LlamaForCausalLM, 896d / 14L / head_dim=64.",
    ),
    "nano-nano": ModelInfo(
        "ray0rf1re/Nano-nano-v4", "llama", "Alias for nano-nano-v4.",
    ),
    "nano-mini-6.99-v2": ModelInfo(
        "ray0rf1re/Nano-mini-6.99-v2", "llama",
        "HF LlamaForCausalLM, 768d / 12L / head_dim=64.",
    ),
    "nano-mini": ModelInfo(
        "ray0rf1re/Nano-mini-6.99-v2", "llama", "Alias for nano-mini-6.99-v2.",
    ),
    "nano-nano-927-v3": ModelInfo(
        "ray0rf1re/nano-nano-927-v3", "nano-nano",
        "Custom NanoNanoModel; dim=120, 12L, vocab=2048.",
    ),
    "nano-nano-927": ModelInfo(
        "ray0rf1re/nano-nano-927-v3", "nano-nano",
        "Alias for nano-nano-927-v3.",
    ),
    # ---- Nix family (ray0rf1re/nix collection) ----------------------------
    # Qwen2-shape with attention_bias off + tied embeddings. All versions
    # load natively through our qwen2 path; no AutoModel needed.
    "nix2.5": ModelInfo(
        "ray0rf1re/Nix2.5", "qwen2",
        "Nix 2.5 — 3B Qwen2-shape, tied embeddings, no qkv bias.",
    ),
    "nix": ModelInfo(
        "Nix-ai/Nix-2.7a", "qwen2",
        "Alias for the latest Nix release; download_model falls back "
        "through 2.7a → 2.6-mm → 2.5 if earlier choices are unreachable.",
    ),
    # Nix-ai org: pretrained weights (GGUF variants live under
    # mradermacher/*-GGUF; we point at the upstream HF repos here).
    "nix2.6-m": ModelInfo(
        "Nix-ai/Nix2.6-m", "qwen2",
        "Nix 2.6-m (Nix-ai) — 2B Qwen2-shape.",
    ),
    "nix2.6-mm": ModelInfo(
        "Nix-ai/Nix2.6-mm", "qwen2",
        "Nix 2.6-mm (Nix-ai) — 3B Qwen2-shape.",
    ),
    "nix-2.7a": ModelInfo(
        "Nix-ai/Nix-2.7a", "qwen2",
        "Nix 2.7a (Nix-ai) — 2B Qwen2-shape.",
    ),
    "nix2.7": ModelInfo(
        "Nix-ai/Nix-2.7a", "qwen2", "Alias for nix-2.7a.",
    ),
    "nix2.6": ModelInfo(
        "Nix-ai/Nix2.6-mm", "qwen2", "Alias for nix2.6-mm.",
    ),
    # ---- Llama 3 family ---------------------------------------------------
    "llama-3.1-8b": ModelInfo(
        "meta-llama/Llama-3.1-8B", "llama",
        "Llama 3.1 8B base (gated repo — HF token required).",
    ),
    "llama-3.1-8b-instruct": ModelInfo(
        "meta-llama/Llama-3.1-8B-Instruct", "llama",
        "Llama 3.1 8B instruction-tuned (gated repo).",
    ),
    "llama-3.2-1b": ModelInfo(
        "meta-llama/Llama-3.2-1B", "llama",
        "Llama 3.2 1B base (gated repo).",
    ),
    "llama-3.2-3b": ModelInfo(
        "meta-llama/Llama-3.2-3B", "llama",
        "Llama 3.2 3B base (gated repo).",
    ),
    "llama-3.3-70b-instruct": ModelInfo(
        "meta-llama/Llama-3.3-70B-Instruct", "llama",
        "Llama 3.3 70B instruct (gated repo).",
    ),
    # ---- Qwen 2.5 / 3 -----------------------------------------------------
    "qwen2.5-0.5b": ModelInfo(
        "Qwen/Qwen2.5-0.5B", "qwen2",
        "Qwen2.5 0.5B base.",
    ),
    "qwen2.5-7b": ModelInfo(
        "Qwen/Qwen2.5-7B", "qwen2",
        "Qwen2.5 7B base.",
    ),
    "qwen2.5-7b-instruct": ModelInfo(
        "Qwen/Qwen2.5-7B-Instruct", "qwen2",
        "Qwen2.5 7B instruct.",
    ),
    "qwen2.5-coder-7b": ModelInfo(
        "Qwen/Qwen2.5-Coder-7B", "qwen2",
        "Qwen2.5-Coder 7B.",
    ),
    "qwen3-0.6b": ModelInfo(
        "Qwen/Qwen3-0.6B", "auto",
        "Qwen3 0.6B — loads via AutoModel.",
    ),
    "qwen3-8b": ModelInfo(
        "Qwen/Qwen3-8B", "auto",
        "Qwen3 8B — loads via AutoModel.",
    ),
    # ---- Qwen 3.5 (model_type "qwen3_5") ---------------------------------
    # All Qwen3.5 checkpoints use a novel hybrid linear/full attention that's
    # implemented in transformers.Qwen3_5ForConditionalGeneration; they always
    # load via AutoModel.
    "qwen3.5-0.8b": ModelInfo(
        "Qwen/Qwen3.5-0.8B", "auto", "Qwen3.5 0.8B — AutoModel.",
    ),
    "qwen3.5-2b": ModelInfo(
        "Qwen/Qwen3.5-2B", "auto", "Qwen3.5 2B — AutoModel.",
    ),
    "qwen3.5-4b": ModelInfo(
        "Qwen/Qwen3.5-4B", "auto", "Qwen3.5 4B — AutoModel.",
    ),
    "qwen3.5-9b": ModelInfo(
        "Qwen/Qwen3.5-9B", "auto", "Qwen3.5 9B — AutoModel.",
    ),
    "qwen3.5-27b": ModelInfo(
        "Qwen/Qwen3.5-27B", "auto", "Qwen3.5 27B — AutoModel.",
    ),
    "qwen3.5-35b-a3b": ModelInfo(
        "Qwen/Qwen3.5-35B-A3B", "auto",
        "Qwen3.5 35B A3B MoE — AutoModel.",
    ),
    "qwen3.5-122b-a10b": ModelInfo(
        "Qwen/Qwen3.5-122B-A10B", "auto",
        "Qwen3.5 122B A10B MoE — AutoModel.",
    ),
    "qwen3.5-397b-a17b": ModelInfo(
        "Qwen/Qwen3.5-397B-A17B", "auto",
        "Qwen3.5 397B A17B MoE — AutoModel.",
    ),
    # ---- Qwen 3.6 (model_type "qwen3_5_moe") -----------------------------
    "qwen3.6-35b-a3b": ModelInfo(
        "Qwen/Qwen3.6-35B-A3B", "auto",
        "Qwen3.6 35B A3B MoE — AutoModel.",
    ),
    # ---- Qwen 3.8 (v0.71.5rc2) -------------------------------------------
    "qwen3.8-27b": ModelInfo(
        "Qwen/Qwen3.8-27B", "auto",
        "Qwen3.8 27B dense — AutoModel.", family="qwen",
    ),
    # ---- Gemma 2 / 3 / 4 --------------------------------------------------
    "gemma-2-2b": ModelInfo(
        "google/gemma-2-2b", "auto",
        "Gemma 2 2B (gated).",
    ),
    "gemma-2-9b": ModelInfo(
        "google/gemma-2-9b", "auto",
        "Gemma 2 9B (gated).",
    ),
    "gemma-2-27b": ModelInfo(
        "google/gemma-2-27b", "auto",
        "Gemma 2 27B (gated).",
    ),
    "gemma-3-1b": ModelInfo(
        "google/gemma-3-1b-it", "auto",
        "Gemma 3 1B instruction-tuned.",
    ),
    "gemma-3-4b": ModelInfo(
        "google/gemma-3-4b-it", "auto",
        "Gemma 3 4B instruction-tuned.",
    ),
    # Gemma 4 (model_type "gemma4"): hybrid full/sliding attention +
    # per-layer embeddings on the E-series. AutoModel-only; Gemma4 support
    # lives in transformers.Gemma4ForConditionalGeneration.
    "gemma-4-e2b": ModelInfo(
        "google/gemma-4-E2B-it", "auto",
        "Gemma 4 E2B it (per-layer embeddings) — AutoModel.",
    ),
    "gemma-4-e4b": ModelInfo(
        "google/gemma-4-E4B-it", "auto",
        "Gemma 4 E4B it (per-layer embeddings) — AutoModel.",
    ),
    "gemma-4-26b-a4b": ModelInfo(
        "google/gemma-4-26B-A4B-it", "auto",
        "Gemma 4 26B A4B MoE — AutoModel.",
    ),
    "gemma-4-31b": ModelInfo(
        "google/gemma-4-31B-it", "auto",
        "Gemma 4 31B it dense — AutoModel.",
    ),
    # ---- Phi --------------------------------------------------------------
    "phi-3-mini": ModelInfo(
        "microsoft/Phi-3-mini-4k-instruct", "auto",
        "Phi-3 mini 4k instruct.",
    ),
    "phi-3.5-mini": ModelInfo(
        "microsoft/Phi-3.5-mini-instruct", "auto",
        "Phi-3.5 mini instruct.",
    ),
    "phi-4": ModelInfo(
        "microsoft/phi-4", "auto",
        "Phi-4 14B.",
    ),
    # ---- DeepSeek ---------------------------------------------------------
    "deepseek-r1-distill-llama-8b": ModelInfo(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "llama",
        "R1-distilled Llama 8B (llama-shaped).",
    ),
    "deepseek-r1-distill-qwen-7b": ModelInfo(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "qwen2",
        "R1-distilled Qwen 7B.",
    ),
    "deepseek-v2-lite": ModelInfo(
        "deepseek-ai/DeepSeek-V2-Lite", "auto",
        "DeepSeek V2 Lite (MoE — AutoModel).",
    ),
    "deepseek-v3": ModelInfo(
        "deepseek-ai/DeepSeek-V3", "auto",
        "DeepSeek V3 671B MoE (AutoModel).",
    ),
    # ---- GLM --------------------------------------------------------------
    "glm-4-9b-chat": ModelInfo(
        "THUDM/glm-4-9b-chat", "auto",
        "GLM-4 9B chat.",
    ),
    "glm-4.1v": ModelInfo(
        "THUDM/GLM-4.1V-9B-Thinking", "auto",
        "GLM-4.1V-9B Thinking (VLM — AutoModel).",
    ),
    # GLM-5 / 5.1 (zai-org): MoE with dynamic sparse attention
    # (model_type "glm_moe_dsa"). 754B params; AutoModel-only.
    "glm-5": ModelInfo(
        "zai-org/GLM-5", "auto",
        "GLM-5 754B MoE-DSA — AutoModel.",
    ),
    "glm-5.1": ModelInfo(
        "zai-org/GLM-5.1", "auto",
        "GLM-5.1 754B MoE-DSA — AutoModel.",
    ),
    "glm-5.1-fp8": ModelInfo(
        "zai-org/GLM-5.1-FP8", "auto",
        "GLM-5.1 754B FP8 — AutoModel.",
    ),
    # ---- Nvidia -----------------------------------------------------------
    "nemotron-4-15b": ModelInfo(
        "nvidia/Nemotron-4-15B-Base", "auto",
        "Nemotron-4 15B base.",
    ),
    "llama-3.1-nemotron-70b-instruct": ModelInfo(
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", "llama",
        "Nemotron 70B Instruct (llama-shaped).",
    ),
    "mistral-nemo-12b": ModelInfo(
        "nvidia/Mistral-NeMo-12B-Base", "mistral",
        "Mistral-NeMo 12B base.",
    ),
    # ---- GPT-OSS (OpenAI) -------------------------------------------------
    "gpt-oss-20b": ModelInfo(
        "openai/gpt-oss-20b", "auto",
        "OpenAI gpt-oss 20B (AutoModel).",
    ),
    "gpt-oss-120b": ModelInfo(
        "openai/gpt-oss-120b", "auto",
        "OpenAI gpt-oss 120B (AutoModel).",
    ),
    # ---- Mistral ----------------------------------------------------------
    "mistral-7b-instruct": ModelInfo(
        "mistralai/Mistral-7B-Instruct-v0.3", "mistral",
        "Mistral 7B Instruct v0.3.",
    ),
    "mixtral-8x7b-instruct": ModelInfo(
        "mistralai/Mixtral-8x7B-Instruct-v0.1", "auto",
        "Mixtral 8x7B Instruct (MoE — AutoModel).",
    ),
    # ---- 7 More Models ----------------------------------------------------
    "qwen3.5-72b": ModelInfo(
        "Qwen/Qwen3.5-72B", "auto",
        "Qwen3.5 72B — AutoModel.",
    ),
    "qwen3.6-7b": ModelInfo(
        "Qwen/Qwen3.6-7B", "auto",
        "Qwen3.6 7B — AutoModel.",
    ),
    "gemma-4-9b": ModelInfo(
        "google/gemma-4-9b-it", "auto",
        "Gemma 4 9B instruction-tuned.",
    ),
    "llama-4-8b": ModelInfo(
        "meta-llama/Llama-4-8b", "llama",
        "Llama 4 8B base.",
    ),
    "mistral-large": ModelInfo(
        "mistralai/Mistral-Large-Instruct", "mistral",
        "Mistral Large.",
    ),
    "phi-4-mini": ModelInfo(
        "microsoft/Phi-4-mini", "auto",
        "Phi-4 mini.",
    ),
    # ---- New 0.70.5a2 additions -------------------------------------------
    "glm-5.2": ModelInfo(
        "THUDM/glm-5.2", "auto", "GLM 5.2 model.", family="glm"
    ),
    "nex-n2": ModelInfo(
        "NEX/Nex-N2", "auto", "Nex-N2 model.", family="nex"
    ),
    "nemo-12b-instruct": ModelInfo(
        "nvidia/Mistral-NeMo-12B-Instruct", "mistral", "Nemo 12B Instruct.", family="nemo"
    ),
    "lfm-2.5": ModelInfo(
        "liquidai/lfm-2.5", "auto", "Liquid Foundation Model 2.5.", family="lfm"
    ),
    "smollm-3": ModelInfo(
        "HuggingFaceTB/SmolLM-3", "auto", "SmolLM 3.", family="smollm"
    ),
    "z-image": ModelInfo(
        "Z-Image/z-image-v1", "auto", "Z-Image Vision Model.", family="vision"
    ),
    "whisper-tiny": ModelInfo(
        "openai/whisper-tiny", "asr", "Whisper Tiny.", family="whisper"
    ),
    "whisper-base": ModelInfo(
        "openai/whisper-base", "asr", "Whisper Base.", family="whisper"
    ),
    "whisper-small": ModelInfo(
        "openai/whisper-small", "asr", "Whisper Small.", family="whisper"
    ),
    "whisper-medium": ModelInfo(
        "openai/whisper-medium", "asr", "Whisper Medium.", family="whisper"
    ),
    "whisper-large-v3": ModelInfo(
        "openai/whisper-large-v3", "asr", "Whisper Large V3.", family="whisper"
    ),
    "deepseek-v4": ModelInfo(
        "deepseek-ai/DeepSeek-V4", "auto", "DeepSeek V4.", family="deepseek"
    ),
    "kimi-k2.5+": ModelInfo(
        "MoonshotAI/Kimi-K2.5-Plus", "auto", "Kimi K2.5+.", family="kimi"
    ),
    "gemma-4-27b": ModelInfo(
        "google/gemma-4-27b-it", "auto", "Gemma 4 27B.", family="gemma4"
    ),
    "qwen3.6-14b": ModelInfo(
        "Qwen/Qwen3.6-14B", "auto", "Qwen3.6 14B.", family="qwen"
    ),
    "qwen3.6-32b": ModelInfo(
        "Qwen/Qwen3.6-32B", "auto", "Qwen3.6 32B.", family="qwen"
    ),
    "mimo": ModelInfo(
        "Mimo/Mimo-1", "auto", "Mimo model.", family="mimo"
    ),
    # ---- 0.71.4b2 Additions -----------------------------------------------
    "kimi-k3": ModelInfo(
        "MoonshotAI/Kimi-K3", "auto", "Kimi K3 reasoning model.", family="kimi"
    ),
    "kimi k3": ModelInfo(
        "MoonshotAI/Kimi-K3", "auto", "Alias for kimi-k3.", family="kimi"
    ),
    "claude-sonnet-4.6": ModelInfo(
        "anthropic/claude-sonnet-4-6", "auto", "Claude Sonnet 4.6.", family="anthropic"
    ),
    "claude sonnet 4.6": ModelInfo(
        "anthropic/claude-sonnet-4-6", "auto", "Alias for claude-sonnet-4.6.", family="anthropic"
    ),
    "sonnet 5": ModelInfo(
        "anthropic/claude-sonnet-5", "auto", "Claude Sonnet 5.", family="anthropic"
    ),
    "claude-sonnet-5": ModelInfo(
        "anthropic/claude-sonnet-5", "auto", "Alias for sonnet 5.", family="anthropic"
    ),
    "opus 4.8": ModelInfo(
        "anthropic/claude-opus-4-8", "auto", "Claude Opus 4.8.", family="anthropic"
    ),
    "claude-opus-4.8": ModelInfo(
        "anthropic/claude-opus-4-8", "auto", "Alias for opus 4.8.", family="anthropic"
    ),
    "fable 5": ModelInfo(
        "fable-ai/fable-5", "auto", "Fable 5 model.", family="fable"
    ),
    "fable-5": ModelInfo(
        "fable-ai/fable-5", "auto", "Alias for fable 5.", family="fable"
    ),
    "gpt4o": ModelInfo(
        "openai/gpt-4o", "auto", "GPT-4o multimodal model.", family="openai"
    ),
    "gpt-4o": ModelInfo(
        "openai/gpt-4o", "auto", "Alias for gpt4o.", family="openai"
    ),
    "gpt5.6terra": ModelInfo(
        "openai/gpt-5.6-terra", "auto", "GPT-5.6 Terra model.", family="openai"
    ),
    "5.6sol": ModelInfo(
        "openai/gpt-5.6-sol", "auto", "GPT-5.6 Sol model.", family="openai"
    ),
    "5.5": ModelInfo(
        "openai/gpt-5.5", "auto", "GPT-5.5 model.", family="openai"
    ),
    "deepseek-r1": ModelInfo(
        "deepseek-ai/DeepSeek-R1", "auto", "DeepSeek R1 reasoning agent.", family="deepseek"
    ),
    "deekseek-v4flash": ModelInfo(
        "deepseek-ai/DeepSeek-V4-Flash", "auto", "DeepSeek V4 Flash model.", family="deepseek"
    ),
    "deepseek-v4flash": ModelInfo(
        "deepseek-ai/DeepSeek-V4-Flash", "auto", "Alias for deekseek-v4flash.", family="deepseek"
    ),
    "qwen 3.7 plus": ModelInfo(
        "Qwen/Qwen3.7-Plus", "auto", "Qwen 3.7 Plus model.", family="qwen"
    ),
    "qwen-3.7-plus": ModelInfo(
        "Qwen/Qwen3.7-Plus", "auto", "Alias for qwen 3.7 plus.", family="qwen"
    ),
    "gemma 4": ModelInfo(
        "google/gemma-4-27b-it", "auto", "Gemma 4 27B instruction-tuned.", family="gemma"
    ),
    "gemma-4": ModelInfo(
        "google/gemma-4-27b-it", "auto", "Alias for gemma 4.", family="gemma"
    ),
    "claude haiku 4.5": ModelInfo(
        "anthropic/claude-haiku-4-5", "auto", "Claude Haiku 4.5.", family="anthropic"
    ),
    "claude-haiku-4.5": ModelInfo(
        "anthropic/claude-haiku-4-5", "auto", "Alias for claude haiku 4.5.", family="anthropic"
    ),
}


#: Short names that should try multiple repos in order until one is
#: reachable.  Each tuple is an ordered fallback chain — entry 0 is
#: the preferred target, later entries are mirrors / older versions
#: used only when the preferred repo 404s, is gated, or hits a
#: network error.  ``resolve_repo_id`` still returns the first
#: entry; the chain is consulted by :func:`download_model` alone.
FALLBACK_CHAINS: dict[str, tuple[str, ...]] = {
    "nix": (
        "Nix-ai/Nix-2.7a",
        "Nix-ai/Nix2.6-mm",
        "ray0rf1re/Nix2.5",
    ),
}


def resolve_repo_id(name_or_repo_id: str) -> str:
    """Resolve a short name (``"nano-mini"``) to a full ``org/repo`` id.

    Anything that already contains a ``/`` is returned unchanged — users
    can keep passing full HF repo ids and nothing breaks.
    """
    if "/" in name_or_repo_id:
        return name_or_repo_id
    key = name_or_repo_id.lower()
    info = KNOWN_MODELS.get(key)
    return info.repo_id if info is not None else name_or_repo_id


def resolve_model_info(name_or_repo_id: str) -> ModelInfo | None:
    """Return the :class:`ModelInfo` for a known short-or-full name, else None."""
    key = name_or_repo_id.lower()
    if key in KNOWN_MODELS:
        return KNOWN_MODELS[key]
    # Match full repo_id against the registry values.
    for info in KNOWN_MODELS.values():
        if info.repo_id.lower() == key:
            return info
    return None


# Files we always try to pull. The list is intentionally explicit so the
# reader can see exactly what's considered "needed", and the glob fallbacks
# (``*.json`` etc.) still cover anything unusual.
REQUIRED_PATTERNS: list[str] = [
    # Every JSON in the repo root (config.json, generation_config.json,
    # tokenizer.json, tokenizer_config.json, special_tokens_map.json,
    # added_tokens.json, preprocessor_config.json, *.index.json, ...).
    "*.json",
    # Tokenizer flavours.
    "tokenizer.*",                   # tokenizer.json / tokenizer.model
    "tokenizer_config.*",
    "special_tokens_map.*",
    "added_tokens.*",
    "vocab.*",                       # vocab.json / vocab.txt
    "merges.*",                      # merges.txt
    "spiece.model",                  # T5-style SentencePiece
    "sentencepiece.bpe.model",
    "chat_template.*",               # chat_template.json / .jinja
    "*.tiktoken",
    # Weights — safetensors (single + sharded) and pickled variants.
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model.bin.index.json",
    "*.pt",
    "*.pth",
    # Misc.
    "*.txt",
    "*.md",
    "*.model",
    "LICENSE*",
    "README*",
]

# Files that MUST exist after a successful download for `hypernix convert` to
# work. Weights are checked via a glob in :func:`verify_snapshot` — any of the
# accepted weight patterns satisfies the requirement.
_REQUIRED_FILES = ("config.json",)
_WEIGHT_GLOBS = ("*.safetensors", "pytorch_model*.bin", "*.pt", "*.pth", "*.bin")


def verify_snapshot(model_dir: Path | str) -> list[str]:
    """Verify a downloaded snapshot contains everything `convert` needs.

    Returns a sorted list of the actual filenames present. Raises
    ``FileNotFoundError`` with a specific message if ``config.json`` or any
    weight file is missing — that's almost always the reason downstream
    conversion fails.
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"snapshot directory does not exist: {model_dir}")
    present = sorted(p.name for p in model_dir.iterdir() if p.is_file())

    missing_required = [f for f in _REQUIRED_FILES if not (model_dir / f).exists()]
    has_weights = any(next(model_dir.glob(pat), None) for pat in _WEIGHT_GLOBS)

    problems: list[str] = []
    if missing_required:
        problems.append("missing required metadata: " + ", ".join(missing_required))
    if not has_weights:
        problems.append(
            "no weight files found (looked for: " + ", ".join(_WEIGHT_GLOBS) + ")"
        )
    if problems:
        raise FileNotFoundError(
            f"snapshot at {model_dir} is incomplete: "
            + "; ".join(problems)
            + f". Got {len(present)} file(s): {present[:20]}"
            + ("..." if len(present) > 20 else "")
        )
    return present


def download_model(
    repo_id: str = "ray0rf1re/hyper-nix.1",
    revision: str | None = None,
    cache_dir: str | None = None,
    local_dir: str | None = None,
    model_dir_name: str | None = None,
    token: str | None = None,
    quiet: bool = False,
    verify: bool = True,
) -> Path:
    """Download a full HuggingFace model snapshot and return its directory.

    Args:
        repo_id: HuggingFace repo, defaults to ``ray0rf1re/hyper-nix.1``.
        revision: Git revision / branch / tag.
        cache_dir: Override the HF cache directory.
        local_dir: If set, download directly to this directory instead of
            the blob-store cache.
        model_dir_name: Optional name for the model cache directory under
            ``$HOME/.cache/hypernix/models``. If omitted, the repository name is used.
        token: HF access token (or reads ``HF_TOKEN`` /
            ``HUGGING_FACE_HUB_TOKEN``).
        quiet: Suppress the per-file listing.
        verify: After download, raise if ``config.json`` or any weight file
            is missing.
    """
    def log(msg: str) -> None:
        if not quiet:
            print(f"[hypernix] {msg}", file=sys.stderr)

    # v0.61.1: MAJOR warning when the user resolves to hyper-Nix.2.
    try:
        from hypernix.system.utils import warn_hyper_nix_2
        warn_hyper_nix_2(repo_id)
    except Exception:  # noqa: BLE001
        pass

    if token is None:
        try:
            from hypernix.system.config import get_hf_token
            token = get_hf_token()
        except Exception:  # noqa: BLE001
            pass

    # Determine the local directory where the snapshot will be placed.
    # If the caller supplied ``local_dir`` we honour it (used for explicit paths).
    # Otherwise we compute a directory under the unified models directory.
    if local_dir is not None:
        target_dir = Path(local_dir).expanduser().resolve()
    else:
        try:
            from hypernix.system.config import get_models_dir
            base_dir = get_models_dir()
        except Exception:  # noqa: BLE001
            base_dir = Path.home() / ".hypernix" / "models"
        model_name = model_dir_name or repo_id.split('/')[-1]
        target_dir = base_dir / model_name
    target_dir.mkdir(parents=True, exist_ok=True)
    # Build the candidate list.  When the short name has an entry in
    # FALLBACK_CHAINS (e.g. "nix" → 2.7a → 2.6-mm → 2.5), each repo
    # is tried in order until snapshot_download succeeds.  Otherwise
    # there's just the single resolved repo.
    short_key = repo_id.lower() if "/" not in repo_id else None
    chain: list[str] = list(FALLBACK_CHAINS.get(short_key or "", ()))
    if not chain:
        resolved = resolve_repo_id(repo_id)
        if resolved != repo_id:
            log(f"resolved short name {repo_id!r} -> {resolved}")
        chain = [resolved]
    else:
        log(f"resolved short name {repo_id!r} -> fallback chain {chain}")

    last_exc: Exception | None = None
    path: Path | None = None
    for attempt, candidate in enumerate(chain):
        try:
            log(f"downloading {candidate} ...")
            path = Path(
                snapshot_download(
                    repo_id=candidate,
                    revision=revision,
                    cache_dir=cache_dir,
                    local_dir=str(target_dir),
                    token=token,
                    allow_patterns=REQUIRED_PATTERNS,
                )
            )
            repo_id = candidate
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            remaining = chain[attempt + 1 :]
            if remaining:
                log(f"WARNING: {candidate} failed ({exc}); trying {remaining[0]}")
            else:
                log(f"ERROR: exhausted fallback chain; last failure: {exc}")
    if path is None:
        raise RuntimeError(
            f"download_model: all candidates in the fallback chain "
            f"failed for {short_key or chain[0]!r}: {chain}",
        ) from last_exc

    # Safety net: some repos put ``config.json`` behind a non-default branch
    # or embed it only in a README; if it wasn't picked up by the glob,
    # try a single-file fetch as a last resort.
    if not (path / "config.json").exists():
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                local_dir=str(target_dir),
                token=token,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: config.json missing and single-file fetch failed: {exc}")

    if verify:
        present = verify_snapshot(path)
    else:
        present = sorted(p.name for p in path.iterdir() if p.is_file())

    log(f"snapshot at {path} ({len(present)} files)")
    for name in present:
        log(f"  - {name}")
    return path
