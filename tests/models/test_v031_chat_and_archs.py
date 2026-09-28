"""Tests for v0.31.0 additions: KNOWN_MODELS, rope_style, HF-prefix stripping,
nano-nano arch, and CodeOven.chat()."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# download.py — KNOWN_MODELS registry + short-name resolution
# ---------------------------------------------------------------------------

def test_known_models_covers_all_three_new_repos() -> None:
    from hypernix import KNOWN_MODELS

    want = {
        "ray0rf1re/Nano-nano-v4",
        "ray0rf1re/Nano-mini-6.99-v2",
        "ray0rf1re/nano-nano-927-v3",
    }
    assert want <= {info.repo_id for info in KNOWN_MODELS.values()}


def test_resolve_repo_id_short_name_maps_to_full() -> None:
    from hypernix import resolve_repo_id

    assert resolve_repo_id("nano-nano-v4") == "ray0rf1re/Nano-nano-v4"
    assert resolve_repo_id("nano-mini") == "ray0rf1re/Nano-mini-6.99-v2"
    assert resolve_repo_id("nano-nano-927") == "ray0rf1re/nano-nano-927-v3"


def test_resolve_repo_id_passes_through_full_id() -> None:
    from hypernix import resolve_repo_id

    # Anything with a slash is returned verbatim, even if unknown.
    assert resolve_repo_id("someone/else") == "someone/else"


def test_resolve_model_info_returns_arch() -> None:
    from hypernix import resolve_model_info

    nano_v4 = resolve_model_info("nano-nano-v4")
    assert nano_v4 is not None and nano_v4.arch == "llama"

    nano_mini = resolve_model_info("nano-mini")
    assert nano_mini is not None and nano_mini.arch == "llama"

    nano_927 = resolve_model_info("nano-nano-927-v3")
    assert nano_927 is not None and nano_927.arch == "nano-nano"


def test_resolve_model_info_by_full_repo_id() -> None:
    from hypernix import resolve_model_info

    info = resolve_model_info("ray0rf1re/Nano-mini-6.99-v2")
    assert info is not None and info.repo_id == "ray0rf1re/Nano-mini-6.99-v2"


def test_resolve_model_info_unknown_returns_none() -> None:
    from hypernix import resolve_model_info

    assert resolve_model_info("definitely/not-a-real-repo") is None


# ---------------------------------------------------------------------------
# HyperNixConfig — rope_style inference and rope_parameters normalization
# ---------------------------------------------------------------------------

def test_rope_style_inferred_from_model_type_llama() -> None:
    from hypernix import HyperNixConfig

    cfg = HyperNixConfig.from_dict({
        "model_type": "llama", "hidden_size": 16, "num_attention_heads": 2,
    })
    assert cfg.rope_style == "half-rotate"


def test_rope_style_inferred_from_model_type_qwen2() -> None:
    from hypernix import HyperNixConfig

    cfg = HyperNixConfig.from_dict({
        "model_type": "qwen2", "hidden_size": 16, "num_attention_heads": 2,
    })
    assert cfg.rope_style == "half-rotate"


def test_rope_style_default_is_interleaved_for_hypernix() -> None:
    from hypernix import HyperNixConfig

    cfg = HyperNixConfig.from_dict({"model_type": "hypernix"})
    assert cfg.rope_style == "interleaved"


def test_rope_parameters_dict_unpacked_to_flat_theta() -> None:
    """HF Llama checkpoints nest rope_theta under ``rope_parameters``."""
    from hypernix import HyperNixConfig

    cfg = HyperNixConfig.from_dict({
        "model_type": "llama",
        "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    })
    assert cfg.rope_theta == 500000.0


def test_explicit_rope_style_overrides_inference() -> None:
    from hypernix import HyperNixConfig

    cfg = HyperNixConfig.from_dict({
        "model_type": "llama", "rope_style": "interleaved",
    })
    assert cfg.rope_style == "interleaved"


# ---------------------------------------------------------------------------
# HF ``model.`` prefix stripping on load
# ---------------------------------------------------------------------------

def test_hf_model_prefix_stripped_on_load(tmp_path: Path) -> None:
    """A config/state_dict with HF-style ``model.`` prefix loads cleanly."""
    from safetensors.torch import save_file

    from hypernix import HyperNixConfig, HyperNixModel, load_snapshot

    cfg = HyperNixConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=16, model_type="llama", rope_style="half-rotate",
    )
    # Build a reference model; snapshot uses flat names.
    ref = HyperNixModel(cfg)
    flat_state = {k: v.detach().cpu().contiguous() for k, v in ref.state_dict().items()}

    snap = tmp_path / "hf-style"
    snap.mkdir()
    # Real HF Llama configs don't include rope_style; drop it so inference kicks in.
    hf_like = {k: v for k, v in cfg.to_dict().items() if k != "rope_style"}
    (snap / "config.json").write_text(json.dumps(hf_like))
    # Prefix every non-lm_head weight with "model." to mimic HF Llama.
    prefixed = {
        (f"model.{k}" if not k.startswith("lm_head") else k): v
        for k, v in flat_state.items()
    }
    save_file(prefixed, str(snap / "model.safetensors"))

    model, loaded_cfg = load_snapshot(snap)
    assert loaded_cfg.model_type == "llama"
    assert loaded_cfg.rope_style == "half-rotate"
    # Embedding should match reference (proves prefix strip succeeded).
    assert torch.equal(
        model.embed_tokens.weight.detach().cpu(),
        ref.embed_tokens.weight.detach().cpu(),
    )


# ---------------------------------------------------------------------------
# nano_nano.py — custom architecture
# ---------------------------------------------------------------------------

def test_nano_nano_config_exposes_hypernix_compat_fields() -> None:
    from hypernix.nano_nano import NanoNanoConfig

    cfg = NanoNanoConfig(dim=120, num_layers=12, num_heads=4, num_kv_heads=2)
    # HyperNix-style accessors are properties on the config.
    assert cfg.hidden_size == 120
    assert cfg.num_hidden_layers == 12
    assert cfg.num_attention_heads == 4
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 30


def test_nano_nano_forward_runs_and_returns_loss() -> None:
    from hypernix.nano_nano import NanoNanoConfig, NanoNanoModel

    cfg = NanoNanoConfig(
        vocab_size=64, dim=16, num_layers=2, num_heads=2, num_kv_heads=1,
        max_position_embeddings=32,
    )
    model = NanoNanoModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)
    out = model(ids, labels=ids)
    assert out["logits"].shape == (1, 8, cfg.vocab_size)
    assert "loss" in out and out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])


def test_nano_nano_ties_output_to_embedding() -> None:
    from hypernix.nano_nano import NanoNanoConfig, NanoNanoModel

    cfg = NanoNanoConfig(vocab_size=32, dim=8, num_layers=1, num_heads=2, num_kv_heads=1)
    model = NanoNanoModel(cfg)
    assert model.output.weight.data_ptr() == model.tok_embeddings.weight.data_ptr()


def test_load_snapshot_dispatches_to_nano_nano(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from hypernix import load_snapshot
    from hypernix.nano_nano import NanoNanoConfig, NanoNanoModel

    cfg = NanoNanoConfig(
        vocab_size=64, dim=16, num_layers=2, num_heads=2, num_kv_heads=1,
        max_position_embeddings=32,
    )
    model = NanoNanoModel(cfg)

    snap = tmp_path / "nano"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps(cfg.to_dict()))
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    # Drop the tied output tensor — safetensors refuses shared memory.
    if state.get("output.weight") is not None and state["output.weight"].data_ptr() == state["tok_embeddings.weight"].data_ptr():
        del state["output.weight"]
    save_file(state, str(snap / "model.safetensors"))

    loaded, loaded_cfg = load_snapshot(snap)
    assert loaded_cfg.model_type == "nano-nano"
    assert type(loaded).__name__ == "NanoNanoModel"
    # Forward pass works after load.
    ids = torch.randint(0, cfg.vocab_size, (1, 4), dtype=torch.long)
    assert loaded(ids)["logits"].shape == (1, 4, cfg.vocab_size)


# ---------------------------------------------------------------------------
# CodeOven.chat()
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_oven(tmp_path: Path):
    from hypernix import new_oven

    return new_oven(
        tmp_path / "chat-oven", arch="hypernix",
        vocab_size=256, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, device="cpu", seed=0,
    )


def test_chat_runs_without_hf_template(tiny_oven) -> None:
    """Byte-tokenizer oven has no chat template -> plain role:content fallback."""
    out = tiny_oven.chat(
        [{"role": "user", "content": "hello"}],
        max_new_tokens=4, temperature=0.0,
    )
    assert isinstance(out, str)


def test_chat_is_deterministic_at_temperature_zero(tiny_oven) -> None:
    messages = [
        {"role": "system", "content": "you are terse"},
        {"role": "user", "content": "hi"},
    ]
    a = tiny_oven.chat(messages, max_new_tokens=6, temperature=0.0, seed=0)
    b = tiny_oven.chat(messages, max_new_tokens=6, temperature=0.0, seed=0)
    assert a == b


def test_chat_format_byte_fallback_includes_assistant_suffix(tiny_oven) -> None:
    """The fallback transcript ends with 'assistant:' so the model continues the turn."""
    ids = tiny_oven._format_chat([{"role": "user", "content": "hi"}])
    decoded = tiny_oven._decode(ids)
    assert decoded.rstrip().endswith("assistant:")


def test_chat_uses_hf_apply_chat_template_when_available(tiny_oven) -> None:
    """When the tokenizer exposes apply_chat_template, the fast path fires."""
    called: dict[str, object] = {}

    class _FakeTokenizer:
        chat_template = "{% for m in messages %}{{m.role}}:{{m.content}}\n{% endfor %}"

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            called["messages"] = messages
            called["add_gen"] = add_generation_prompt
            return [1, 2, 3, 4]

    tiny_oven.tokenizer = _FakeTokenizer()
    tiny_oven.tokenizer_kind = "hf"

    ids = tiny_oven._format_chat([{"role": "user", "content": "hi"}])
    assert ids == [1, 2, 3, 4]
    assert called["add_gen"] is True
    assert called["messages"] == [{"role": "user", "content": "hi"}]
