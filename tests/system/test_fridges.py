"""Tests for old_fridge / mediocre_fridge / new_fridge."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn

# ---------------------------------------------------------------------------
# old_fridge
# ---------------------------------------------------------------------------

def _toy_model() -> nn.Module:
    m = nn.Module()
    m.embed_tokens = nn.Embedding(8, 4)
    m.head = nn.Linear(4, 8)
    return m


def test_freeze_matches_substring() -> None:
    from hypernix import old_fridge

    m = _toy_model()
    n = old_fridge.freeze(m, patterns=("embed_tokens",))
    assert n == 8 * 4  # embedding weights
    assert m.embed_tokens.weight.requires_grad is False
    assert m.head.weight.requires_grad is True


def test_freeze_is_idempotent() -> None:
    from hypernix import old_fridge

    m = _toy_model()
    old_fridge.freeze(m, patterns=("embed_tokens",))
    assert old_fridge.freeze(m, patterns=("embed_tokens",)) == 0


def test_unfreeze_restores() -> None:
    from hypernix import old_fridge

    m = _toy_model()
    old_fridge.freeze(m, patterns=("*",))
    n = old_fridge.unfreeze(m, patterns=("head*",))
    assert n > 0
    assert m.head.weight.requires_grad is True
    assert m.embed_tokens.weight.requires_grad is False


def test_parameter_stats_counts() -> None:
    from hypernix import old_fridge

    m = _toy_model()
    stats = old_fridge.parameter_stats(m)
    assert stats.total == stats.trainable
    assert stats.frozen == 0
    assert stats.megabytes > 0


def test_unwrap_model_ddp_like() -> None:
    from hypernix import old_fridge

    inner = _toy_model()

    class Wrapper(nn.Module):
        def __init__(self, mod: nn.Module) -> None:
            super().__init__()
            self.module = mod

    wrapped = Wrapper(inner)
    assert old_fridge.unwrap_model(wrapped) is inner
    old_fridge.freeze(wrapped, patterns=("embed_tokens",))
    stats = old_fridge.parameter_stats(wrapped)
    assert stats.frozen == 8 * 4


def test_chill_cache_never_raises_on_cpu() -> None:
    from hypernix import old_fridge

    old_fridge.chill_cache()


# ---------------------------------------------------------------------------
# mediocre_fridge
# ---------------------------------------------------------------------------

def test_synthesize_judge_corpus_writes_file(tmp_path: Path) -> None:
    from hypernix import mediocre_fridge

    out = tmp_path / "judge.txt"
    p = mediocre_fridge.synthesize_judge_corpus(n=20, out_path=out, seed=0)
    assert p == out
    text = out.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln]
    assert len(lines) == 20
    for line in lines:
        assert mediocre_fridge.JUDGE_PROMPT in line
        assert mediocre_fridge.JUDGE_RESPONSE in line
        assert mediocre_fridge.JUDGE_LABEL in line
        # Label must be one of the canonical strings.
        label = line.rsplit(mediocre_fridge.JUDGE_LABEL, 1)[1]
        assert label in (mediocre_fridge.LABEL_GOOD, mediocre_fridge.LABEL_BAD)


def test_synthesize_judge_corpus_is_deterministic(tmp_path: Path) -> None:
    from hypernix import mediocre_fridge

    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    mediocre_fridge.synthesize_judge_corpus(n=30, out_path=a, seed=42)
    mediocre_fridge.synthesize_judge_corpus(n=30, out_path=b, seed=42)
    assert a.read_text() == b.read_text()


def test_judge_example_format() -> None:
    from hypernix import mediocre_fridge

    ex = mediocre_fridge.JudgeExample("Q?", "A", mediocre_fridge.LABEL_GOOD)
    s = ex.format()
    assert s.startswith(mediocre_fridge.JUDGE_PROMPT + "Q?")
    assert s.endswith(mediocre_fridge.LABEL_GOOD)


def test_good_ratio_all_good(tmp_path: Path) -> None:
    from hypernix import mediocre_fridge

    out = tmp_path / "g.txt"
    mediocre_fridge.synthesize_judge_corpus(n=10, out_path=out, seed=1, good_ratio=1.0)
    text = out.read_text()
    assert text.count(mediocre_fridge.LABEL_GOOD) == 10
    assert text.count(mediocre_fridge.LABEL_BAD) == 0


# ---------------------------------------------------------------------------
# new_fridge
# ---------------------------------------------------------------------------

_SAMPLE_LOG = """\
[hypernix.train] step 10/100  loss=2.3456  ppl=10.44
[hypernix.train] step 20/100  loss=1.9870  ppl=7.30
[hypernix.train] step 30/100  loss=1.5012  ppl=4.49
"""


def test_parse_training_log_extracts_pairs() -> None:
    from hypernix import new_fridge

    pairs = new_fridge.parse_training_log(_SAMPLE_LOG)
    assert pairs == [(10, 2.3456), (20, 1.987), (30, 1.5012)]


def test_parse_training_log_ignores_noise() -> None:
    from hypernix import new_fridge

    assert new_fridge.parse_training_log("some unrelated text\n") == []


def test_plot_loss_curve_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib", reason="matplotlib not installed")
    from hypernix import new_fridge

    out = tmp_path / "loss.png"
    p = new_fridge.plot_loss_curve([(1, 3.0), (2, 2.0), (3, 1.0)], out)
    assert p.exists()
    assert p.stat().st_size > 0
    # PNG magic bytes.
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# End-to-end: example script smoke (runs the full orchestration in-process).
# ---------------------------------------------------------------------------

def test_evaluator_example_runs_end_to_end(tmp_path: Path) -> None:
    """Mini version of examples/train_hypernix_0_1_5_evaluator.py."""
    import contextlib
    import io

    from hypernix import (
        HyperNixConfig,
        init_from_scratch,
        mediocre_fridge,
        new_fridge,
        old_fridge,
        old_oven,
    )

    root = tmp_path / "eval-demo"
    root.mkdir()

    snap = root / "scratch"
    cfg = HyperNixConfig(
        vocab_size=256, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, model_type="hypernix",
    )
    init_from_scratch(snap, cfg, tokenizer_source=None, seed=0)

    dataset = root / "judge.txt"
    mediocre_fridge.synthesize_judge_corpus(n=40, out_path=dataset, seed=0)

    oven = old_oven.preheat(local_dir=snap, device="cpu")
    old_fridge.freeze(oven.model, patterns=("embed_tokens",))
    stats = old_fridge.parameter_stats(oven.model)
    assert stats.frozen > 0
    assert stats.trainable > 0
    old_fridge.chill_cache()

    trained = root / "trained"
    log_buf = io.StringIO()
    with contextlib.redirect_stdout(log_buf):
        oven.train(
            dataset, trained,
            steps=5, batch_size=1, context_length=32,
            lr=3e-4, log_every=1, save_every=0, seed=0, quiet=False,
        )
    assert (trained / "config.json").exists()

    judge = old_oven.preheat(local_dir=trained, device="cpu")
    continuation = judge.complete(
        f"{mediocre_fridge.JUDGE_PROMPT}Capital of France?"
        f"{mediocre_fridge.JUDGE_RESPONSE}Paris"
        f"{mediocre_fridge.JUDGE_LABEL}",
        max_new_tokens=2, temperature=0.0, stop=(), seed=0,
    )
    assert isinstance(continuation, str)

    pairs = new_fridge.parse_training_log(log_buf.getvalue())
    assert len(pairs) >= 1
