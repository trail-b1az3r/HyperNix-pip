"""Tests for hypernix.old_oven (code-generation wrapper around HyperNix).

Every import here reaches `hypernix.models.old_oven` directly rather
than through `hypernix.new_oven` / `hypernix.preheat`. Those shortcuts
resolved here until 0.72.4.post9 and now resolve to `neo_oven`, which
would silently turn this file into a second NeoOven test suite --
passing, and covering nothing it was written to cover.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch


@pytest.fixture
def tiny_snapshot_dir(tmp_path: Path) -> Path:
    from hypernix import HyperNixConfig, init_from_scratch

    # vocab_size=256 so the byte-tokenizer (which emits UTF-8 byte ids 0..255)
    # fits within the embedding table.
    cfg = HyperNixConfig(
        vocab_size=256, hidden_size=8, intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32,
    )
    init_from_scratch(str(tmp_path / "tiny"), cfg, tokenizer_source=None, seed=42)
    return tmp_path / "tiny"


def test_preheat_from_local_snapshot(tiny_snapshot_dir: Path) -> None:
    from hypernix.models import old_oven

    oven = old_oven.preheat(local_dir=tiny_snapshot_dir, device="cpu")
    assert oven.tokenizer_kind == "byte"  # no tokenizer.json => byte fallback
    assert oven.model_dir == tiny_snapshot_dir
    assert oven.device.type == "cpu"


def test_complete_is_deterministic_with_temperature_zero(tiny_snapshot_dir: Path) -> None:
    from hypernix.models import old_oven

    oven = old_oven.preheat(local_dir=tiny_snapshot_dir, device="cpu")
    out_a = oven.complete(
        "def add(a, b):", max_new_tokens=4, temperature=0.0, stop=(), seed=0,
    )
    out_b = oven.complete(
        "def add(a, b):", max_new_tokens=4, temperature=0.0, stop=(), seed=0,
    )
    assert isinstance(out_a, str)
    assert out_a == out_b


def test_fill_falls_back_without_fim_tokens(tiny_snapshot_dir: Path) -> None:
    """Byte tokenizer has no FIM tokens => .fill() continues the prefix."""
    from hypernix.models import old_oven

    oven = old_oven.preheat(local_dir=tiny_snapshot_dir, device="cpu")
    out = oven.fill(
        prefix="def add(a, b):\n    return ",
        suffix="\n\nresult = add(1, 2)",
        max_new_tokens=4, temperature=0.0,
    )
    assert isinstance(out, str)


def test_trim_at_stop_cuts_at_first_match() -> None:
    from hypernix.old_oven import _trim_at_stop

    text = "hello world\nclass Foo:\n    pass"
    assert _trim_at_stop(text, ("\nclass ",)) == "hello world"
    assert _trim_at_stop(text, ()) == text  # empty stops -> no trim


def test_save_pt_and_load_pt_roundtrip(tiny_snapshot_dir: Path, tmp_path: Path) -> None:
    from hypernix.models import old_oven

    oven = old_oven.preheat(local_dir=tiny_snapshot_dir, device="cpu")
    pt_path = tmp_path / "oven.pt"
    oven.save_pt(pt_path)
    assert pt_path.exists() and pt_path.stat().st_size > 0

    restored = old_oven.load_pt(pt_path, device="cpu")
    # Config survives the round trip.
    assert restored.model.config.vocab_size == oven.model.config.vocab_size
    assert restored.model.config.hidden_size == oven.model.config.hidden_size
    # Weights match (first parameter only, as a spot-check).
    orig_sd = oven.model.state_dict()
    new_sd = restored.model.state_dict()
    assert orig_sd.keys() == new_sd.keys()
    for k in orig_sd:
        if orig_sd[k].numel() == 0:
            continue
        assert torch.equal(orig_sd[k], new_sd[k]), f"weight mismatch on {k}"


def test_bake_code_accepts_snapshot_path(tiny_snapshot_dir: Path) -> None:
    """`bake_code` should accept a raw path and preheat internally."""
    from hypernix.models import old_oven

    out = old_oven.bake_code(
        tiny_snapshot_dir, "def fib(n):",
        max_new_tokens=2, temperature=0.0, stop=(),
    )
    assert isinstance(out, str)


# ---------------------------------------------------------------------------
# new_oven: architecture presets
# ---------------------------------------------------------------------------

def test_new_oven_hypernix_arch_has_no_qkv_bias(tmp_path: Path) -> None:
    from hypernix.models.old_oven import new_oven

    oven = new_oven(
        tmp_path / "hn", arch="hypernix",
        vocab_size=256, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=16, device="cpu", seed=0,
    )
    assert oven.model.config.model_type == "hypernix"
    assert oven.model.config.attention_bias is False
    attn = oven.model.layers[0].self_attn
    assert attn.q_proj.bias is None
    assert attn.k_proj.bias is None
    assert attn.v_proj.bias is None
    assert attn.o_proj.bias is None  # o_proj never has bias


def test_new_oven_qwen2_arch_has_qkv_bias_but_not_oproj(tmp_path: Path) -> None:
    from hypernix.models.old_oven import new_oven

    oven = new_oven(
        tmp_path / "qw", arch="qwen2",
        vocab_size=256, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=16, device="cpu", seed=0,
    )
    cfg = oven.model.config
    assert cfg.model_type == "qwen2"
    assert cfg.attention_bias is True
    assert cfg.rope_theta == 1000000.0
    assert cfg.tie_word_embeddings is True
    attn = oven.model.layers[0].self_attn
    # hidden_size=8, num_attention_heads=2 -> head_dim=4.
    # q_proj: 2 heads * 4 head_dim = 8 ; k/v: 1 kv_head * 4 = 4.
    assert attn.q_proj.bias is not None and attn.q_proj.bias.shape == (8,)
    assert attn.k_proj.bias is not None and attn.k_proj.bias.shape == (4,)
    assert attn.v_proj.bias is not None and attn.v_proj.bias.shape == (4,)
    assert attn.o_proj.bias is None


def test_new_oven_qwen25_alias_matches_qwen2(tmp_path: Path) -> None:
    from hypernix import ARCH_PRESETS

    assert ARCH_PRESETS["qwen2.5"] == ARCH_PRESETS["qwen2"]


def test_new_oven_rejects_unknown_arch(tmp_path: Path) -> None:
    from hypernix.models.old_oven import new_oven

    with pytest.raises(ValueError, match="unknown arch"):
        new_oven(tmp_path / "x", arch="does-not-exist", device="cpu")


# ---------------------------------------------------------------------------
# CodeOven.train
# ---------------------------------------------------------------------------

def _write_dataset(path: Path) -> Path:
    # ~3 KB of ASCII so byte tokenizer yields well over 512 tokens.
    path.write_text(("abcdefghij" * 64 + "\n") * 10, encoding="utf-8")
    return path


@pytest.mark.parametrize("arch", ["hypernix", "qwen2"])
def test_train_drives_loss_down(tmp_path: Path, arch: str) -> None:
    """Quick smoke training: loss at the end should be below initial loss."""
    from hypernix.models.old_oven import new_oven

    oven = new_oven(
        tmp_path / f"src-{arch}", arch=arch,
        vocab_size=256, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, device="cpu", seed=7,
    )
    dataset = _write_dataset(tmp_path / "text.txt")

    # Capture initial loss by running one no-op evaluation step.
    initial = _eval_loss(oven, dataset, context_length=32)
    out = oven.train(
        dataset, tmp_path / f"trained-{arch}",
        steps=20, batch_size=2, context_length=32,
        lr=1e-3, log_every=100, save_every=0, seed=7, quiet=True,
    )
    final = _eval_loss(oven, dataset, context_length=32)

    assert out.exists()
    assert (out / "config.json").exists()
    assert (out / "model.safetensors").exists()
    assert final < initial, f"training did not reduce loss: {initial:.4f} -> {final:.4f}"


def _eval_loss(oven, dataset_path: Path, context_length: int) -> float:
    """Helper: compute mean loss on the first two training chunks."""
    from hypernix.train import _iter_chunks  # noqa: PLC2701

    oven.model.eval()
    chunks = list(_iter_chunks(dataset_path, oven.tokenizer, context_length))[:2]
    batch = torch.stack(chunks).to(oven.device)
    with torch.no_grad():
        out = oven.model(batch[:, :-1], labels=batch[:, 1:])
    return float(out["loss"].item())


def test_train_reloadable_and_bias_survives_save_roundtrip(tmp_path: Path) -> None:
    """After .train() a qwen2 oven can be re-preheated with its bias intact."""
    from hypernix.models import old_oven
    from hypernix.models.old_oven import new_oven

    oven = new_oven(
        tmp_path / "qw", arch="qwen2",
        vocab_size=256, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=16, device="cpu", seed=0,
    )
    dataset = _write_dataset(tmp_path / "text.txt")
    trained = oven.train(
        dataset, tmp_path / "trained",
        steps=3, batch_size=1, context_length=8,
        lr=1e-3, log_every=100, save_every=0, seed=0, quiet=True,
    )
    # Re-preheat from the trained snapshot and verify the config survived.
    reloaded = old_oven.preheat(local_dir=trained, device="cpu")
    assert reloaded.model.config.model_type == "qwen2"
    assert reloaded.model.config.attention_bias is True
    assert reloaded.model.layers[0].self_attn.q_proj.bias is not None
