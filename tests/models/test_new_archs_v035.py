"""Coverage for the v0.35 arch/known-model additions.

Gemma 4, Qwen 3.5 / 3.6, GLM 5.x go through the AutoModel fallback (new
HF-side architectures we don't replicate natively). The Nix family from
``ray0rf1re/nix`` is Qwen2-shaped and resolves to the native ``qwen2``
code path, so hand-building a fresh Nix-style oven with ``new_oven`` must
produce a model with the expected config knobs (no qkv bias, tied
embeddings, qwen2 model_type).
"""
from __future__ import annotations

from pathlib import Path

import pytest

NEW_PRESETS = ["gemma4", "glm5", "glm5.1", "qwen3.5", "qwen3.6", "nix", "nix2"]


@pytest.mark.parametrize("name", NEW_PRESETS)
def test_new_preset_registered(name: str) -> None:
    from hypernix import ARCH_PRESETS

    preset = ARCH_PRESETS[name]
    # Every preset must supply the four knobs new_oven inspects.
    assert "attention_bias" in preset
    assert "model_type" in preset
    assert "rope_theta" in preset
    assert "rms_norm_eps" in preset
    assert "tie_word_embeddings" in preset


def test_gemma4_preset_matches_google_config() -> None:
    """google/gemma-4-*-it configs use rope_theta=1e6 on full-attention layers,
    rms_norm_eps=1e-6, tied embeddings, no attention_bias.
    """
    from hypernix import ARCH_PRESETS

    g4 = ARCH_PRESETS["gemma4"]
    assert g4["rope_theta"] == 1_000_000.0
    assert g4["rms_norm_eps"] == 1e-6
    assert g4["tie_word_embeddings"] is True
    assert g4["attention_bias"] is False


def test_qwen35_preset_matches_qwen_config() -> None:
    """Qwen/Qwen3.5-* configs use rope_theta=1e7 and tied embeddings."""
    from hypernix import ARCH_PRESETS

    q = ARCH_PRESETS["qwen3.5"]
    assert q["rope_theta"] == 10_000_000.0
    assert q["rms_norm_eps"] == 1e-6
    assert q["tie_word_embeddings"] is True
    assert q["attention_bias"] is False


def test_qwen36_preset_untied() -> None:
    """Qwen3.6 MoE variants set tie_word_embeddings=False."""
    from hypernix import ARCH_PRESETS

    q = ARCH_PRESETS["qwen3.6"]
    assert q["rope_theta"] == 10_000_000.0
    assert q["tie_word_embeddings"] is False


def test_glm5_preset_matches_zai_config() -> None:
    """zai-org/GLM-5(.1) uses rope_theta=1e6 and untied embeddings."""
    from hypernix import ARCH_PRESETS

    g = ARCH_PRESETS["glm5"]
    assert g["rope_theta"] == 1_000_000.0
    assert g["rms_norm_eps"] == 1e-5
    assert g["tie_word_embeddings"] is False
    assert ARCH_PRESETS["glm5.1"] == g


def test_nix_preset_is_qwen2_without_bias() -> None:
    """ray0rf1re/Nix2.5 is Qwen2-shaped but with attention_bias=False."""
    from hypernix import ARCH_PRESETS

    n = ARCH_PRESETS["nix"]
    assert n["model_type"] == "qwen2"
    assert n["attention_bias"] is False
    assert n["tie_word_embeddings"] is True
    # nix / nix2 are the same preset.
    assert ARCH_PRESETS["nix2"] == n


def test_new_oven_nix_builds_without_qkv_bias(tmp_path: Path) -> None:
    """`new_oven(arch="nix")` must produce a qwen2-typed model with no qkv bias."""
    from hypernix import new_oven

    oven = new_oven(
        tmp_path / "nix", arch="nix",
        vocab_size=256, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=16, device="cpu", seed=0,
    )
    cfg = oven.model.config
    assert cfg.model_type == "qwen2"
    assert cfg.attention_bias is False
    assert cfg.tie_word_embeddings is True
    attn = oven.model.layers[0].self_attn
    assert attn.q_proj.bias is None
    assert attn.k_proj.bias is None
    assert attn.v_proj.bias is None


@pytest.mark.parametrize(
    ("short_name", "expected_repo_id"),
    [
        # Nix family (ray0rf1re/nix collection) — "nix" alias now resolves
        # to the newest release (2.7a); older versions keep their explicit
        # short names.
        ("nix", "Nix-ai/Nix-2.7a"),
        ("nix2.5", "ray0rf1re/Nix2.5"),
        ("nix2.6-m", "Nix-ai/Nix2.6-m"),
        ("nix2.6-mm", "Nix-ai/Nix2.6-mm"),
        ("nix-2.7a", "Nix-ai/Nix-2.7a"),
        # Gemma 4
        ("gemma-4-e2b", "google/gemma-4-E2B-it"),
        ("gemma-4-e4b", "google/gemma-4-E4B-it"),
        ("gemma-4-31b", "google/gemma-4-31B-it"),
        ("gemma-4-26b-a4b", "google/gemma-4-26B-A4B-it"),
        # Qwen 3.5 / 3.6
        ("qwen3.5-4b", "Qwen/Qwen3.5-4B"),
        ("qwen3.5-35b-a3b", "Qwen/Qwen3.5-35B-A3B"),
        ("qwen3.6-35b-a3b", "Qwen/Qwen3.6-35B-A3B"),
        # GLM 5.x
        ("glm-5", "zai-org/GLM-5"),
        ("glm-5.1", "zai-org/GLM-5.1"),
        ("glm-5.1-fp8", "zai-org/GLM-5.1-FP8"),
    ],
)
def test_known_model_resolves(short_name: str, expected_repo_id: str) -> None:
    from hypernix import resolve_repo_id

    assert resolve_repo_id(short_name) == expected_repo_id
