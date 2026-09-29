"""End-to-end test: init a tiny model -> expand -> convert to GGUF -> verify."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tiny_snapshot_dir(tmp_path: Path) -> Path:
    from hypernix import HyperNixConfig, init_from_scratch

    cfg = HyperNixConfig(
        vocab_size=64, hidden_size=8, intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=32,
    )
    init_from_scratch(str(tmp_path / "tiny"), cfg, tokenizer_source=None, seed=42)
    return tmp_path / "tiny"


def test_init_produces_snapshot(tiny_snapshot_dir: Path) -> None:
    names = sorted(p.name for p in tiny_snapshot_dir.iterdir())
    assert "config.json" in names
    assert "model.safetensors" in names


def test_verify_snapshot_ok(tiny_snapshot_dir: Path) -> None:
    from hypernix import verify_snapshot

    present = verify_snapshot(tiny_snapshot_dir)
    assert "config.json" in present
    assert "model.safetensors" in present


def test_verify_snapshot_missing_config_raises(tmp_path: Path) -> None:
    from hypernix import verify_snapshot

    # Empty dir — should raise.
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "empty")


def test_verify_snapshot_missing_weights_raises(tmp_path: Path) -> None:
    from hypernix import verify_snapshot

    d = tmp_path / "config-only"
    d.mkdir()
    (d / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="no weight files"):
        verify_snapshot(d)


def test_seeded_init_is_deterministic(tmp_path: Path) -> None:
    """Same seed -> identical weights; different seed -> different weights."""
    import torch
    from safetensors.torch import load_file

    from hypernix import HyperNixConfig, init_from_scratch

    cfg = HyperNixConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=16,
    )
    init_from_scratch(str(tmp_path / "a"), cfg, seed=7)
    init_from_scratch(str(tmp_path / "b"), cfg, seed=7)
    init_from_scratch(str(tmp_path / "c"), cfg, seed=8)
    a = load_file(str(tmp_path / "a" / "model.safetensors"))
    b = load_file(str(tmp_path / "b" / "model.safetensors"))
    c = load_file(str(tmp_path / "c" / "model.safetensors"))
    # a == b (same seed), a != c (different seed).
    for k in a:
        assert torch.equal(a[k], b[k]), f"seed determinism broken on {k}"
    any_diff = any(not torch.equal(a[k], c[k]) for k in a if a[k].numel() > 0)
    assert any_diff, "different seeds produced identical weights"


def test_expand_checkpoint_grows_and_roundtrips(tiny_snapshot_dir: Path, tmp_path: Path) -> None:
    from hypernix import expand_checkpoint, load_snapshot

    dst = tmp_path / "bigger"
    expand_checkpoint(
        str(tiny_snapshot_dir), str(dst),
        hidden_size=16, num_hidden_layers=4,
    )
    # Re-load and sanity check the new shape.
    model, cfg = load_snapshot(dst)
    assert cfg.hidden_size == 16
    assert cfg.num_hidden_layers == 4
    # forward pass should not crash on the new shape.
    import torch

    ids = torch.zeros((1, 4), dtype=torch.long)
    out = model(ids)
    assert out["logits"].shape == (1, 4, cfg.vocab_size)


def test_convert_tiny_to_gguf(tiny_snapshot_dir: Path, tmp_path: Path) -> None:
    from gguf import GGUFReader

    from hypernix import convert_to_gguf

    gguf_path = tmp_path / "tiny.gguf"
    convert_to_gguf(
        model_dir=tiny_snapshot_dir, output=gguf_path,
        dtype="fp16", arch_name="hypernix", name="Tiny",
    )
    assert gguf_path.exists() and gguf_path.stat().st_size > 0
    reader = GGUFReader(str(gguf_path), "r")
    assert len(reader.tensors) > 0
    assert len(reader.fields) > 0


def test_generate_with_byte_fallback(tiny_snapshot_dir: Path) -> None:
    """Snapshot has no tokenizer -> generate_text falls back to byte tokenizer."""
    from hypernix import generate_text

    out = generate_text(
        model_dir=tiny_snapshot_dir,
        prompt="",
        max_new_tokens=4,
        temperature=0.0,  # greedy, fully deterministic
        seed=0,
        device="cpu",
    )
    assert isinstance(out, str)
