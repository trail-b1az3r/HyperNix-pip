"""Basic tests for neo_oven module.

Tests cover:
  - NeoOven construction
  - Memory management functions (freeze/unfreeze/parameter_stats)
  - JudgeCorpus creation, serialization, and deserialization
  - TrainingMetrics recording and JSONL output
  - parse_training_log parsing
  - synthesize_judge_corpus backward-compat shim
  - preheat / new_oven are importable without errors
"""
from __future__ import annotations

import json
import random

import pytest
import torch
import torch.nn as nn

from hypernix.neo_oven import (
    ARCH_PRESETS,
    LABEL_BAD,
    LABEL_GOOD,
    JudgeCorpus,
    JudgeExample,
    ParamStats,
    TrainingMetrics,
    _mangle_response,
    chill_cache,
    freeze,
    offload_to_cpu,
    parameter_stats,
    parse_training_log,
    plot_loss_curve,
    plot_score_distribution,
    synthesize_judge_corpus,
    unfreeze,
    unwrap_model,
    vram_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_model() -> nn.Module:
    """A tiny two-layer MLP for testing — no HyperNix deps needed."""
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


# ---------------------------------------------------------------------------
# Memory management (freeze / unfreeze / parameter_stats)
# ---------------------------------------------------------------------------

class TestMemoryManagement:
    def test_parameter_stats_all_trainable(self):
        m = _tiny_model()
        stats = parameter_stats(m)
        assert isinstance(stats, ParamStats)
        assert stats.total > 0
        assert stats.trainable == stats.total
        assert stats.frozen == 0
        assert stats.megabytes > 0

    def test_freeze_by_name_and_unfreeze(self):
        m = _tiny_model()
        # The Sequential has layers named "0", "2"
        frozen_count = freeze(m, ["0"])
        assert frozen_count > 0
        stats = parameter_stats(m)
        assert stats.frozen == frozen_count

        unfrozen_count = unfreeze(m, ["0"])
        assert unfrozen_count == frozen_count
        stats2 = parameter_stats(m)
        assert stats2.frozen == 0

    def test_unwrap_model_no_wrapper(self):
        m = _tiny_model()
        assert unwrap_model(m) is m

    def test_offload_to_cpu_noop_on_sequential(self):
        m = _tiny_model()
        # "embed_tokens" doesn't exist — should return 0 moved
        moved = offload_to_cpu(m, ["embed_tokens"])
        assert moved == 0

    def test_chill_cache_runs(self):
        chill_cache()  # Should never raise

    def test_vram_stats_cpu(self):
        stats = vram_stats()
        if not torch.cuda.is_available():
            assert stats == {}
        else:
            assert "allocated_mb" in stats

    def test_param_stats_str(self):
        m = _tiny_model()
        s = str(parameter_stats(m))
        assert "total=" in s
        assert "MB" in s


# ---------------------------------------------------------------------------
# JudgeCorpus
# ---------------------------------------------------------------------------

class TestJudgeCorpus:
    _PAIRS = [
        ("What is 1 + 1?", "2"),
        ("Capital of Italy?", "Rome"),
        ("Smallest prime?", "2"),
    ]

    def test_from_pairs_size(self):
        corpus = JudgeCorpus.from_pairs(self._PAIRS, n=9, seed=42)
        assert len(corpus) == 9

    def test_from_pairs_labels(self):
        corpus = JudgeCorpus.from_pairs(self._PAIRS, n=20, seed=0, good_ratio=1.0)
        assert all(e.label == LABEL_GOOD for e in corpus.examples)

        corpus_bad = JudgeCorpus.from_pairs(self._PAIRS, n=20, seed=0, good_ratio=0.0)
        assert all(e.label == LABEL_BAD for e in corpus_bad.examples)

    def test_save_load_jsonl(self, tmp_path):
        corpus = JudgeCorpus.from_pairs(self._PAIRS, n=6, seed=1)
        out = corpus.save(tmp_path / "corpus.jsonl")
        loaded = JudgeCorpus.load(out)
        assert len(loaded) == len(corpus)
        for a, b in zip(corpus.examples, loaded.examples, strict=True):
            assert a.prompt == b.prompt
            assert a.response == b.response
            assert a.label == b.label

    def test_save_load_text(self, tmp_path):
        corpus = JudgeCorpus.from_pairs(self._PAIRS, n=6, seed=2)
        out = corpus.save(tmp_path / "corpus.txt", fmt="text")
        loaded = JudgeCorpus.load(out, fmt="text")
        assert len(loaded) == len(corpus)

    def test_repr(self):
        corpus = JudgeCorpus.from_pairs(self._PAIRS, n=4, seed=0, good_ratio=0.5)
        r = repr(corpus)
        assert "JudgeCorpus" in r
        assert "n=4" in r

    def test_judge_example_format_text(self):
        ex = JudgeExample("q", "a", LABEL_GOOD)
        txt = ex.format_text()
        assert "JUDGE_PROMPT" in txt
        assert "JUDGE_RESPONSE" in txt
        assert "JUDGE_LABEL" in txt
        assert LABEL_GOOD in txt

    def test_judge_example_to_dict(self):
        ex = JudgeExample("q", "a", LABEL_BAD)
        d = ex.to_dict()
        assert d == {"prompt": "q", "response": "a", "label": LABEL_BAD}

    def test_mangle_response_produces_variant(self):
        rng = random.Random(42)
        good = "The quick brown fox"
        variants = {_mangle_response(good, rng) for _ in range(10)}
        # At least one variant should differ from the original
        assert any(v != good for v in variants)

    def test_synthesize_judge_corpus_shim(self, tmp_path):
        out = synthesize_judge_corpus(8, tmp_path / "judge.txt", seed=0)
        assert out.exists()
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 8


# ---------------------------------------------------------------------------
# TrainingMetrics
# ---------------------------------------------------------------------------

class TestTrainingMetrics:
    def test_on_step_records(self):
        m = TrainingMetrics()
        m.on_step(1, 2.5, lr=3e-4)
        m.on_step(2, 2.1, lr=2.9e-4)
        assert len(m._steps) == 2
        assert m._steps[0]["loss"] == 2.5
        assert m._steps[1]["step"] == 2

    def test_to_jsonl(self, tmp_path):
        m = TrainingMetrics()
        m.on_step(1, 1.5)
        m.on_step(2, 1.2)
        out = m.to_jsonl(tmp_path / "steps.jsonl")
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["step"] == 1
        assert first["loss"] == 1.5

    def test_plot_loss(self, tmp_path):
        try:
            m = TrainingMetrics()
            for i in range(5):
                m.on_step(i + 1, 2.0 - i * 0.3)
            out = m.plot_loss(tmp_path / "loss.png")
            assert out.exists()
        except ModuleNotFoundError:
            pytest.skip("matplotlib not installed")

    def test_plot_score_distribution(self, tmp_path):
        try:
            m = TrainingMetrics()
            out = m.plot_score_distribution([0.1, 0.5, 0.9, 0.7], tmp_path / "dist.png")
            assert out.exists()
        except ModuleNotFoundError:
            pytest.skip("matplotlib not installed")

    def test_close_no_error(self):
        m = TrainingMetrics()
        m.close()  # Should never raise


# ---------------------------------------------------------------------------
# parse_training_log
# ---------------------------------------------------------------------------

class TestParseTrainingLog:
    _LOG = (
        "[neo_oven.train] arch=hypernix step 10/100  loss=2.4321  ppl=11.39  lr=3.00e-04  ctx=512\n"
        "[neo_oven.train] arch=hypernix step 20/100  loss=2.1054  ppl=8.21  lr=2.95e-04  ctx=512\n"
        "Some other log line\n"
        "[neo_oven.train] arch=hypernix step 30/100  loss=1.9876  ppl=7.30  lr=2.90e-04  ctx=512\n"
    )

    def test_parses_all_steps(self):
        entries = parse_training_log(self._LOG)
        assert len(entries) == 3

    def test_fields_present(self):
        entries = parse_training_log(self._LOG)
        assert entries[0]["step"] == 10
        assert abs(entries[0]["loss"] - 2.4321) < 1e-4

    def test_empty_log(self):
        assert parse_training_log("") == []


# ---------------------------------------------------------------------------
# ARCH_PRESETS completeness
# ---------------------------------------------------------------------------

class TestArchPresets:
    def test_required_presets_exist(self):
        for name in ("hypernix", "llama3", "qwen2", "qwen3.5", "mistral", "gemma4", "nix"):
            assert name in ARCH_PRESETS, f"Missing preset: {name}"

    def test_each_preset_has_required_keys(self):
        required = {"attention_bias", "model_type", "rope_theta", "rms_norm_eps", "tie_word_embeddings"}
        for name, preset in ARCH_PRESETS.items():
            for key in required:
                assert key in preset, f"Preset {name!r} missing key {key!r}"


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------

def test_preheat_importable():
    from hypernix.neo_oven import preheat
    assert callable(preheat)


def test_new_oven_importable():
    from hypernix.neo_oven import new_oven
    assert callable(new_oven)


def test_neooven_class_importable():
    from hypernix.neo_oven import NeoOven
    assert NeoOven is not None


def test_plot_functions_importable():
    from hypernix.neo_oven import (
        plot_round_losses,
    )
    assert callable(plot_loss_curve)
    assert callable(plot_score_distribution)
    assert callable(plot_round_losses)
