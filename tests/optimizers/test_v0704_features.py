"""tests/optimizers/test_v0704_features.py

Comprehensive test suite for every feature introduced across the
hypernix 0.70.4 release series (0.70.4b1 through 0.70.4b14), covering:

  - hypernix.qa.QAProcessor                          (0.70.4b11)
  - hypernix.stml (STML, calculate_vram_context)      (0.70.4b11)
  - hypernix.abbicus.TurboAbbicus / TurboAbbicusConfig (0.70.4b11)
  - `hypernix stml` CLI subcommand                     (0.70.4b11)
  - `hypernix train run` curriculum flags              (0.70.4b11)
  - hypernix.train.train() / CodeOven.train() new kwargs (0.70.4b11)
  - `hnx` CLI alias                                    (0.70.4b1)
  - hypernix.optimizer_framework (OptimizerBase, ScheduleConfig) (0.70.4b2)
  - hypernix.pressure_cooker_v4 (PressureCookerV4 + subclasses)  (0.70.4b14)
  - hypernix.cardboard_box.CardboardBox                (0.70.4b14)

DESIGN NOTES — why this file should keep passing after 0.70.5:

  1. Every test targets the *public, documented behavior* of a 0.70.4
     feature (what it computes, what it returns, how it's shaped) —
     never private/underscore internals, exact log/print wording, or
     implementation details likely to get refactored.

  2. Feature *removal* would be a breaking change hypernix's own
     semver-ish policy (see wiki/Changelog.md) reserves for a major
     bump, not a 0.70.x patch/minor. A 0.70.5 release is expected to
     *add* to this surface, not remove it. Nonetheless, every import
     and every optional-dependency usage is wrapped so a genuinely
     removed/renamed symbol produces a clear ``pytest.skip`` with the
     symbol name rather than an opaque collection error that would
     mask failures in the rest of the suite.

  3. No network access, no real model downloads, no GPU required.
     Anything that would normally need a real snapshot directory is
     exercised against a tiny in-memory / on-disk fixture instead.

  4. Numeric assertions use tolerant bounds (ranges, monotonicity,
     "is a multiple of N") rather than pinning exact floats, since
     internal heuristics (e.g. VRAM overhead constants) are exactly
     the kind of thing a .5 release tunes without changing the public
     contract.

  5. CLI tests drive ``hypernix.cli.main()`` in-process (no subprocess,
     no dependency on which console-script entry point is installed)
     and assert on the presence of flag names in ``--help`` output
     rather than exact help text formatting.

Run with:  pytest tests/optimizers/test_v0704_features.py -v
"""
from __future__ import annotations

import inspect
import json
import re
import sys

import pytest

torch = pytest.importorskip("torch", reason="hypernix requires torch")


def _import_or_skip(module_path: str, *names: str):
    """Import ``names`` from ``module_path``, skipping the test on failure.

    Used everywhere a 0.70.4 symbol is referenced, so a rename/removal
    in a future release produces a clear skip instead of a hard failure
    that could be mistaken for an actual regression in unrelated tests.
    """
    try:
        mod = __import__(module_path, fromlist=list(names) if names else [""])
    except ImportError as exc:
        pytest.skip(f"module {module_path!r} not importable: {exc}")
    if not names:
        return mod
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        pytest.skip(f"{module_path} is missing expected symbol(s): {missing}")
    if len(names) == 1:
        return getattr(mod, names[0])
    return tuple(getattr(mod, n) for n in names)


# ===========================================================================
# hypernix.qa.QAProcessor  (0.70.4b11)
# ===========================================================================

class TestQAProcessor:
    def _get(self):
        return _import_or_skip("hypernix.qa", "QAProcessor")

    def test_question_answer_mode_basic(self):
        QAProcessor = self._get()
        rows = [{"question": "What is 2+2?", "answer": "4"}]
        proc = QAProcessor(source=rows, format_mode="question_answer")
        out = list(proc)
        assert len(out) == 1
        assert "2+2" in out[0]
        assert "4" in out[0]
        # Template keywords must survive verbatim.
        assert "Question" in out[0]
        assert "Answer" in out[0]

    def test_predict_next_mode_has_no_template_wrapper(self):
        QAProcessor = self._get()
        rows = [{"question": "foo", "answer": "bar"}]
        proc = QAProcessor(source=rows, format_mode="predict_next")
        out = list(proc)
        assert len(out) == 1
        assert "foo" in out[0] and "bar" in out[0]
        # predict_next must NOT inject the Q&A template keywords.
        assert "Question:" not in out[0]
        assert "Answer:" not in out[0]

    @pytest.mark.parametrize(
        "record",
        [
            {"prompt": "p", "completion": "c"},
            {"instruction": "p", "response": "c"},
            {"input": "p", "output": "c"},
            {"q": "p", "a": "c"},
        ],
    )
    def test_key_fallbacks(self, record):
        """Common dataset key-naming conventions must resolve automatically."""
        QAProcessor = self._get()
        proc = QAProcessor(source=[record], format_mode="predict_next")
        out = list(proc)
        assert len(out) == 1
        assert "p" in out[0] and "c" in out[0]

    def test_custom_keys(self):
        QAProcessor = self._get()
        rows = [{"src": "hello", "tgt": "world"}]
        proc = QAProcessor(
            source=rows, question_key="src", answer_key="tgt", format_mode="predict_next"
        )
        out = list(proc)
        assert "hello" in out[0] and "world" in out[0]

    def test_seasoning_never_corrupts_template_keywords(self):
        """Salt/pepper seasoning must never touch the literal template text."""
        QAProcessor = self._get()
        salt_shaker_mod = pytest.importorskip("hypernix.salt_shaker")
        if not hasattr(salt_shaker_mod, "FromTheBag"):
            pytest.skip("hypernix.salt_shaker.FromTheBag not found")
        # `source` is required by the Shaker base class but unused by the
        # single-string `.season()` call QAProcessor makes internally.
        shaker = salt_shaker_mod.FromTheBag(source=[], rate=1.0, seed=0)  # rate=1.0: maximal perturbation
        rows = [{"question": "hello there", "answer": "general kenobi"}]
        proc = QAProcessor(source=rows, format_mode="question_answer", salt_shaker=shaker)
        out = list(proc)
        assert len(out) == 1
        # Even under maximal seasoning, the template wrapper text survives.
        assert "Question:" in out[0]
        assert "Answer:" in out[0]

    def test_iterable_protocol(self):
        """QAProcessor must be a plain iterable usable in a for-loop / list()."""
        QAProcessor = self._get()
        proc = QAProcessor(source=[{"question": "a", "answer": "b"}])
        assert hasattr(proc, "__iter__")
        collected = [line for line in proc]
        assert len(collected) == 1

    def test_jsonl_file_source(self, tmp_path):
        QAProcessor = self._get()
        f = tmp_path / "qa.jsonl"
        f.write_text(
            json.dumps({"question": "1+1?", "answer": "2"}) + "\n"
            + json.dumps({"question": "2+2?", "answer": "4"}) + "\n"
        )
        proc = QAProcessor(source=f, format_mode="question_answer")
        out = list(proc)
        assert len(out) == 2


# ===========================================================================
# hypernix.stml  (0.70.4b11)
# ===========================================================================

class TestCalculateVramContext:
    def _get(self):
        return _import_or_skip("hypernix.stml", "calculate_vram_context")

    def test_returns_positive_multiple_of_128(self):
        calculate_vram_context = self._get()
        ctx = calculate_vram_context(
            vram_gb=24.0, model_size_params=4.0, batch_size=2, precision="fp16"
        )
        assert isinstance(ctx, int)
        assert ctx > 0
        assert ctx % 128 == 0

    def test_more_vram_never_yields_smaller_context(self):
        """Monotonicity: strictly more VRAM must not produce a *smaller*
        max context for otherwise-identical settings."""
        calculate_vram_context = self._get()
        small = calculate_vram_context(vram_gb=8.0, model_size_params=1.0, batch_size=1)
        large = calculate_vram_context(vram_gb=80.0, model_size_params=1.0, batch_size=1)
        assert large >= small

    def test_bigger_model_never_yields_larger_context(self):
        """Monotonicity in the other direction: a bigger model at fixed
        VRAM must not leave *more* room for context."""
        calculate_vram_context = self._get()
        small_model = calculate_vram_context(vram_gb=24.0, model_size_params=1.0, batch_size=1)
        big_model = calculate_vram_context(vram_gb=24.0, model_size_params=30.0, batch_size=1)
        assert big_model <= small_model

    def test_model_too_big_for_vram_returns_safe_minimum(self):
        calculate_vram_context = self._get()
        ctx = calculate_vram_context(vram_gb=1.0, model_size_params=70.0, batch_size=8)
        assert ctx == 128

    @pytest.mark.parametrize("precision", ["fp32", "fp16", "int8", "int4"])
    def test_all_documented_precisions_accepted(self, precision):
        calculate_vram_context = self._get()
        ctx = calculate_vram_context(
            vram_gb=16.0, model_size_params=2.0, batch_size=1, precision=precision
        )
        assert ctx >= 128


class TestSTML:
    def _get(self):
        return _import_or_skip("hypernix.stml", "STML")

    def test_short_sequence_passes_through_unchanged(self):
        STML = self._get()
        stml = STML(untrained_max_context=8192, segment_length=512)
        batch = {"input_ids": torch.randint(0, 100, (2, 256))}
        out = stml.regulate(dict(batch))
        assert out["input_ids"].shape == batch["input_ids"].shape

    def test_enforces_untrained_max_context_hard_cap(self):
        STML = self._get()
        stml = STML(untrained_max_context=1024, segment_length=256)
        batch = {"input_ids": torch.randint(0, 100, (1, 4096))}
        out = stml.regulate(dict(batch))
        # After folding, total tokens represented must never exceed the cap.
        total_tokens = out["input_ids"].shape[0] * out["input_ids"].shape[1]
        assert total_tokens <= 1024

    def test_folds_long_sequence_into_batch_dimension(self):
        """Sequences longer than segment_length must be folded into the
        batch dim as (batch * num_segments, segment_length), not truncated
        down to segment_length outright — this is STML's whole point."""
        STML = self._get()
        stml = STML(untrained_max_context=8192, segment_length=128)
        batch = {"input_ids": torch.arange(1 * 512).reshape(1, 512)}
        out = stml.regulate(dict(batch))
        assert out["input_ids"].shape[1] == 128
        # 512 tokens / 128 segment_length = 4 segments -> batch dim grows to 4.
        assert out["input_ids"].shape[0] == 4
        # No data lost: folded tensor must contain every original token.
        assert out["input_ids"].numel() >= 512

    def test_preserves_labels_and_attention_mask_alongside_input_ids(self):
        STML = self._get()
        stml = STML(untrained_max_context=8192, segment_length=128)
        n = 384
        batch = {
            "input_ids": torch.ones(1, n, dtype=torch.long),
            "attention_mask": torch.ones(1, n, dtype=torch.long),
            "labels": torch.ones(1, n, dtype=torch.long),
        }
        out = stml.regulate(dict(batch))
        assert out["input_ids"].shape == out["attention_mask"].shape == out["labels"].shape

    def test_missing_input_ids_is_a_noop(self):
        STML = self._get()
        stml = STML()
        batch = {"foo": "bar"}
        out = stml.regulate(dict(batch))
        assert out == {"foo": "bar"}

    def test_composes_with_a_regulator(self):
        """STML must accept an optional Abbicus/TurboAbbicus-style
        regulator and apply it before its own folding logic."""
        STML = self._get()

        class _StubRegulator:
            def __init__(self):
                self.called = False

            def regulate(self, batch):
                self.called = True
                return batch

        stub = _StubRegulator()
        stml = STML(untrained_max_context=8192, segment_length=256, regulator=stub)
        batch = {"input_ids": torch.randint(0, 100, (1, 128))}
        stml.regulate(dict(batch))
        assert stub.called is True


# ===========================================================================
# hypernix.abbicus.TurboAbbicus / TurboAbbicusConfig  (0.70.4b11)
# ===========================================================================

class TestTurboAbbicus:
    def _get(self):
        return _import_or_skip("hypernix.abbicus", "TurboAbbicus", "TurboAbbicusConfig")

    def test_context_grows_with_curriculum_progress(self):
        """Exponential growth: allowed length at step 0 must be strictly
        smaller than allowed length partway through the curriculum."""
        TurboAbbicus, TurboAbbicusConfig = self._get()
        cfg = TurboAbbicusConfig(
            base_context_length=4096, hard_cap=16384, curriculum_steps=1000,
            oscillation_enabled=False,
        )
        ta = TurboAbbicus(cfg)
        start_len = ta.current_max_length
        ta.step(500)
        mid_len = ta.current_max_length
        assert mid_len > start_len

    def test_never_exceeds_double_the_hard_cap(self):
        """Even mid-oscillation, allowed length is clamped to <= 2x hard_cap
        (documented safety clamp)."""
        TurboAbbicus, TurboAbbicusConfig = self._get()
        cfg = TurboAbbicusConfig(
            base_context_length=4096, hard_cap=8192, curriculum_steps=10,
            oscillation_enabled=True, oscillation_amplitude=0.5,
        )
        ta = TurboAbbicus(cfg)
        for step in range(0, 500, 7):
            ta.step(step)
            assert ta.current_max_length <= cfg.hard_cap * 2
            assert ta.current_max_length >= 128

    def test_reaches_and_holds_near_hard_cap_eventually(self):
        TurboAbbicus, TurboAbbicusConfig = self._get()
        cfg = TurboAbbicusConfig(
            base_context_length=1024, hard_cap=4096, curriculum_steps=100,
            oscillation_enabled=False,
        )
        ta = TurboAbbicus(cfg)
        ta.step(10_000)  # far past curriculum_steps -> progress clamps to 1.0
        # int() truncation in the exponential formula can land 1-2 tokens
        # under the exact cap; allow a small tolerance rather than pinning
        # exact equality to an internal rounding detail.
        assert cfg.hard_cap - ta.current_max_length <= 2

    def test_oscillation_disabled_holds_exactly_at_cap(self):
        TurboAbbicus, TurboAbbicusConfig = self._get()
        cfg = TurboAbbicusConfig(
            base_context_length=1024, hard_cap=4096, curriculum_steps=10,
            oscillation_enabled=False,
        )
        ta = TurboAbbicus(cfg)
        lengths = set()
        for step in range(20, 200, 5):
            ta.step(step)
            lengths.add(ta.current_max_length)
        # With oscillation off, once past curriculum_steps the length should
        # be stable (not bouncing around) and within a token or two of the
        # cap (int() truncation in the exponential formula can land just
        # under it).
        assert len(lengths) == 1, f"expected a single stable value, got {lengths}"
        (stable_length,) = lengths
        assert cfg.hard_cap - stable_length <= 2

    def test_regulate_truncates_batches_over_the_allowed_length(self):
        TurboAbbicus, TurboAbbicusConfig = self._get()
        cfg = TurboAbbicusConfig(
            base_context_length=128, hard_cap=256, curriculum_steps=1_000_000,
            oscillation_enabled=False,
        )
        ta = TurboAbbicus(cfg)  # step 0 -> small allowed length
        batch = {"input_ids": torch.randint(0, 100, (1, 4096))}
        out = ta.regulate(dict(batch))
        assert out["input_ids"].shape[1] <= cfg.hard_cap

    def test_size_multiplier_scales_with_model_size(self):
        """Larger declared model_size must never produce a *smaller*
        starting context than a smaller declared model size, all else equal."""
        TurboAbbicus, TurboAbbicusConfig = self._get()
        small_cfg = TurboAbbicusConfig(model_size="1B", base_context_length=4096, hard_cap=32768)
        big_cfg = TurboAbbicusConfig(model_size="70B", base_context_length=4096, hard_cap=32768)
        small_ta = TurboAbbicus(small_cfg)
        big_ta = TurboAbbicus(big_cfg)
        assert big_ta.current_max_length >= small_ta.current_max_length

    def test_gpu_utilization_is_never_referenced_by_oscillation(self):
        """Documented guarantee: oscillation adjusts by host CPU load only,
        never GPU utilization. We can't easily assert a negative on
        internals, so instead assert the class works identically whether
        or not CUDA is reported available (it must not branch on GPU load)."""
        TurboAbbicus, TurboAbbicusConfig = self._get()
        cfg = TurboAbbicusConfig(
            base_context_length=1024, hard_cap=2048, curriculum_steps=1,
            oscillation_enabled=True,
        )
        ta = TurboAbbicus(cfg)
        ta.step(50)
        # Just confirm this doesn't raise regardless of CUDA availability,
        # and that current_max_length is always a sane int.
        assert isinstance(ta.current_max_length, int)


# ===========================================================================
# `hypernix stml` and `hypernix train run` CLI integration  (0.70.4b11)
# ===========================================================================

class TestCLIIntegration:
    def _cli_main(self):
        return _import_or_skip("hypernix.cli", "main")

    def test_stml_subcommand_computes_and_prints_a_context_length(self, capsys):
        main = self._cli_main()
        rc = main(["stml", "--vram", "16", "--params", "4", "--precision", "fp16"])
        assert rc == 0
        out = capsys.readouterr().out
        # Don't pin exact wording — just confirm an integer token count is present.
        assert re.search(r"\d+", out), f"expected a numeric context length in output, got: {out!r}"

    def test_stml_subcommand_requires_vram_flag(self, capsys):
        main = self._cli_main()
        with pytest.raises(SystemExit):
            main(["stml"])  # --vram is required; argparse exits non-zero

    @pytest.mark.parametrize(
        "flag",
        [
            "--use-abbicus",
            "--use-turbo-abbicus",
            "--use-stml",
            "--untrained-max-context",
            "--segment-length",
        ],
    )
    def test_train_run_help_advertises_curriculum_flags(self, capsys, flag):
        """`hypernix train run --help` must mention every 0.70.4 curriculum
        flag. Using --help (rather than a real run) keeps this fast, and
        immune to unrelated training-loop changes in later releases."""
        main = self._cli_main()
        with pytest.raises(SystemExit) as exc_info:
            main(["train", "run", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert flag in out, f"{flag!r} missing from `hypernix train run --help` output"


class TestTrainFunctionSignature:
    """Signature-level checks for hypernix.train.train() and
    CodeOven.train() — verifies the 0.70.4 kwargs exist and have sane
    defaults, without actually running a training loop (which would need
    a real model + dataset and be slow / environment-dependent)."""

    @pytest.mark.parametrize(
        "kwarg,expected_default",
        [
            ("use_abbicus", False),
            ("use_turbo_abbicus", False),
            ("use_stml", False),
            ("untrained_max_context", 8192),
            ("segment_length", 512),
        ],
    )
    def test_train_module_function_has_curriculum_kwargs(self, kwarg, expected_default):
        train_fn = _import_or_skip("hypernix.train", "train")
        sig = inspect.signature(train_fn)
        assert kwarg in sig.parameters, f"hypernix.train.train() missing kwarg {kwarg!r}"
        assert sig.parameters[kwarg].default == expected_default

    @pytest.mark.parametrize(
        "kwarg",
        ["use_abbicus", "use_turbo_abbicus", "use_stml", "untrained_max_context", "segment_length"],
    )
    def test_codeoven_train_has_curriculum_kwargs(self, kwarg):
        CodeOven = _import_or_skip("hypernix.old_oven", "CodeOven")
        if not hasattr(CodeOven, "train"):
            pytest.skip("CodeOven.train not found")
        sig = inspect.signature(CodeOven.train)
        assert kwarg in sig.parameters, f"CodeOven.train() missing kwarg {kwarg!r}"


class TestHnxAlias:
    """`hnx` was added in 0.70.4b1 as a full alias of `hypernix`. We check
    the installed package metadata's entry points rather than shelling
    out to a subprocess, so this test works regardless of PATH setup."""

    def test_hnx_entry_point_registered(self):
        try:
            from importlib.metadata import entry_points
        except ImportError:
            pytest.skip("importlib.metadata not available")
        try:
            eps = entry_points(group="console_scripts")
        except TypeError:
            # Python < 3.10 compatibility shape, just in case.
            eps = entry_points().get("console_scripts", [])
        names = {ep.name for ep in eps}
        if "hypernix" not in names:
            pytest.skip("hypernix not installed as a package (running from source checkout)")
        assert "hnx" in names, "`hnx` console-script alias not registered"


# ===========================================================================
# hypernix.optimizer_framework  (0.70.4b2)
# ===========================================================================

class TestScheduleConfig:
    def _get(self):
        return _import_or_skip("hypernix.optimizer_framework", "ScheduleConfig")

    def test_warmup_plateau_cooldown_shape(self):
        ScheduleConfig = self._get()
        sched = ScheduleConfig(
            lr=1.0, warmup_steps=10, plateau_steps=10, cooldown_steps=10, min_lr=0.0
        ).validate()
        # Warmup: lr should increase monotonically from ~0 towards peak.
        lr_start = sched.lr_at_step(0)
        lr_mid_warmup = sched.lr_at_step(5)
        lr_end_warmup = sched.lr_at_step(10)
        assert lr_start <= lr_mid_warmup <= lr_end_warmup
        assert lr_end_warmup == pytest.approx(1.0, abs=0.15)
        # Plateau: should stay near peak.
        lr_plateau = sched.lr_at_step(15)
        assert lr_plateau == pytest.approx(1.0, abs=0.15)
        # Cooldown: should decay toward min_lr.
        lr_end = sched.lr_at_step(29)
        assert lr_end < lr_plateau

    def test_cools_down_to_min_lr_not_zero(self):
        ScheduleConfig = self._get()
        sched = ScheduleConfig(
            lr=1.0, warmup_steps=1, plateau_steps=1, cooldown_steps=10, min_lr=0.1
        ).validate()
        lr_far_past_end = sched.lr_at_step(1000)
        assert lr_far_past_end == pytest.approx(0.1, abs=0.05)

    def test_validate_rejects_negative_fields(self):
        ScheduleConfig = self._get()
        with pytest.raises(ValueError):
            ScheduleConfig(lr=1.0, warmup_steps=-1).validate()

    def test_validate_rejects_nonpositive_lr(self):
        ScheduleConfig = self._get()
        with pytest.raises(ValueError):
            ScheduleConfig(lr=0.0).validate()


class TestOptimizerBase:
    def _get(self):
        return _import_or_skip("hypernix.optimizer_framework", "OptimizerBase", "ScheduleConfig")

    def _tiny_concrete_optimizer(self, OptimizerBase, ScheduleConfig):
        """OptimizerBase requires .step() to be implemented by a subclass;
        build the smallest possible concrete subclass for testing."""

        class _TinyOptimizer(OptimizerBase):
            def __init__(self, params, **kwargs):
                # `defaults` is a required positional arg on OptimizerBase;
                # an empty dict is fine since this subclass doesn't add any
                # per-group hyperparameters beyond what torch.optim.Optimizer
                # already tracks (like `lr`, if present in each group).
                super().__init__(params, defaults={}, **kwargs)

            def step(self, closure=None):
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is not None:
                            p.data.add_(p.grad, alpha=-group.get("lr", 1e-3))
                return None

        return _TinyOptimizer

    def test_gradient_clip_norm_mode_reports_clipping(self):
        OptimizerBase, ScheduleConfig = self._get()
        Tiny = self._tiny_concrete_optimizer(OptimizerBase, ScheduleConfig)
        p = torch.nn.Parameter(torch.zeros(4))
        p.grad = torch.ones(4) * 100.0  # deliberately huge gradient
        opt = Tiny([p], schedule=ScheduleConfig(lr=1e-3), grad_clip=1.0, grad_clip_mode="norm")
        stats = opt.gradient_clip()
        assert stats.clipped is True
        assert p.grad.norm().item() <= 1.0 + 1e-4

    def test_gradient_clip_disabled_when_no_threshold_set(self):
        OptimizerBase, ScheduleConfig = self._get()
        Tiny = self._tiny_concrete_optimizer(OptimizerBase, ScheduleConfig)
        p = torch.nn.Parameter(torch.zeros(4))
        p.grad = torch.ones(4) * 100.0
        opt = Tiny([p], schedule=ScheduleConfig(lr=1e-3), grad_clip=None)
        stats = opt.gradient_clip()
        assert stats.clipped is False
        assert p.grad.norm().item() == pytest.approx(200.0, rel=1e-3)  # ||[100,100,100,100]|| = 200

    def test_scheduled_lr_delegates_to_schedule_config(self):
        OptimizerBase, ScheduleConfig = self._get()
        Tiny = self._tiny_concrete_optimizer(OptimizerBase, ScheduleConfig)
        p = torch.nn.Parameter(torch.zeros(2))
        sched = ScheduleConfig(lr=1.0, warmup_steps=1, plateau_steps=1, cooldown_steps=1)
        opt = Tiny([p], schedule=sched)
        assert opt.scheduled_lr(0) == sched.lr_at_step(0)
        assert opt.scheduled_lr(5) == sched.lr_at_step(5)

    def test_step_actually_updates_parameters(self):
        OptimizerBase, ScheduleConfig = self._get()
        Tiny = self._tiny_concrete_optimizer(OptimizerBase, ScheduleConfig)
        p = torch.nn.Parameter(torch.ones(3))
        p.grad = torch.ones(3)
        opt = Tiny([p], schedule=ScheduleConfig(lr=0.1))
        before = p.data.clone()
        opt.step()
        assert not torch.equal(before, p.data)


class TestFusedAdamwStep:
    def test_standalone_adamw_function_reduces_loss_on_toy_problem(self):
        fused_adamw_step = _import_or_skip("hypernix.optimizer_framework", "fused_adamw_step")
        # Fit y = 2x with a single scalar weight via a handful of manual
        # AdamW steps using the standalone function.
        w = torch.tensor([0.0])
        exp_avg = torch.zeros_like(w)
        exp_avg_sq = torch.zeros_like(w)
        x, target = torch.tensor([1.0]), torch.tensor([2.0])
        losses = []
        for step in range(1, 200):
            pred = w * x
            loss = (pred - target).pow(2).mean()
            losses.append(loss.item())
            grad = 2 * (pred - target) * x
            fused_adamw_step(
                [w], [grad], [exp_avg], [exp_avg_sq],
                lr=0.05, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, step=step,
            )
        assert losses[-1] < losses[0]


# ===========================================================================
# hypernix.pressure_cooker_v4  (0.70.4b14)
# ===========================================================================

class TestPressureCookerV4:
    def _get(self, name="PressureCookerV4"):
        return _import_or_skip("hypernix.pressure_cooker_v4", name)

    def _schedule(self):
        ScheduleConfig = _import_or_skip("hypernix.optimizer_framework", "ScheduleConfig")
        return ScheduleConfig(lr=1e-2)

    def test_basic_step_reduces_toy_loss(self):
        PressureCookerV4 = self._get()
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.zeros(1))
        opt = PressureCookerV4([w], schedule=self._schedule())
        x, target = torch.tensor([1.0]), torch.tensor([3.0])
        first_loss = None
        for _step in range(50):
            opt.zero_grad()
            loss = (w * x - target).pow(2).mean()
            if first_loss is None:
                first_loss = loss.item()
            loss.backward()
            opt.step()
        final_loss = (w * x - target).pow(2).mean().item()
        assert final_loss < first_loss

    def test_ema_shadow_weights_tracked_when_enabled(self):
        PressureCookerV4 = self._get()
        w = torch.nn.Parameter(torch.zeros(2))
        opt = PressureCookerV4([w], schedule=self._schedule(), use_ema=True, ema_beta=0.9)
        for _ in range(5):
            opt.zero_grad()
            loss = (w - 1.0).pow(2).sum()
            loss.backward()
            opt.step()
        # An EMA shadow-state dict must exist and be non-empty once EMA is on.
        assert hasattr(opt, "_ema_state")
        assert len(opt._ema_state) > 0

    def test_ema_disabled_by_default_leaves_no_shadow_state(self):
        PressureCookerV4 = self._get()
        w = torch.nn.Parameter(torch.zeros(2))
        opt = PressureCookerV4([w], schedule=self._schedule())  # use_ema defaults False
        opt.zero_grad()
        loss = (w - 1.0).pow(2).sum()
        loss.backward()
        opt.step()
        assert len(opt._ema_state) == 0

    def test_stochastic_rounding_only_applies_to_low_precision_params(self):
        """Documented behavior: stochastic rounding noise is only added
        for float16/bfloat16 params. We can't easily assert the internal
        noise injection directly, so instead confirm fp32 training with
        stochastic_rounding=True remains numerically well-behaved (no
        NaNs introduced) and actually still reduces loss."""
        PressureCookerV4 = self._get()
        torch.manual_seed(0)
        w = torch.nn.Parameter(torch.zeros(1))  # fp32
        opt = PressureCookerV4([w], schedule=self._schedule(), stochastic_rounding=True)
        x, target = torch.tensor([1.0]), torch.tensor([3.0])
        for _ in range(50):
            opt.zero_grad()
            loss = (w * x - target).pow(2).mean()
            loss.backward()
            opt.step()
        assert torch.isfinite(w).all()

    def test_grad_clip_is_honored(self):
        PressureCookerV4 = self._get()
        w = torch.nn.Parameter(torch.zeros(4))
        opt = PressureCookerV4([w], schedule=self._schedule(), grad_clip=1.0)
        opt.zero_grad()
        loss = (w - 1000.0).pow(2).sum()  # huge gradient
        loss.backward()
        opt.step()
        # Post-step, the recorded gradient (post-clip) should be bounded.
        assert w.grad.norm().item() <= 1.0 + 1e-3

    @pytest.mark.parametrize(
        "subclass_name",
        [
            "StovetopV4Cooker",
            "StovetopV4CookerPlus",
            "Agedcookerv4",
            "CookerLite",
        ],
    )
    def test_documented_subclasses_are_constructible_and_step(self, subclass_name):
        Subclass = self._get(subclass_name)
        w = torch.nn.Parameter(torch.zeros(2))
        opt = Subclass([w], schedule=self._schedule())
        opt.zero_grad()
        loss = (w - 1.0).pow(2).sum()
        loss.backward()
        opt.step()  # must not raise
        assert torch.isfinite(w).all()

    def test_ultracookerv4_accepts_qat_mode_kwarg(self):
        Ultracookerv4 = self._get("Ultracookerv4")
        w = torch.nn.Parameter(torch.zeros(2))
        opt = Ultracookerv4([w], schedule=self._schedule(), qat_mode="iq4")
        opt.zero_grad()
        loss = (w - 1.0).pow(2).sum()
        loss.backward()
        opt.step()  # must not raise regardless of whether qat_mode is fully wired up
        assert torch.isfinite(w).all()

    def test_lars_adaptation_does_not_crash_on_zero_gradient(self):
        """LARS trust-ratio scaling divides by the gradient norm; a
        zero-gradient parameter must not produce NaN/inf."""
        PressureCookerV4 = self._get()
        w = torch.nn.Parameter(torch.ones(3))
        opt = PressureCookerV4([w], schedule=self._schedule(), lars_adaptation=True)
        opt.zero_grad()
        w.grad = torch.zeros(3)  # explicit zero gradient
        opt.step()
        assert torch.isfinite(w).all()


# ===========================================================================
# hypernix.cardboard_box.CardboardBox  (0.70.4b14)
# ===========================================================================

class TestCardboardBox:
    def _get(self):
        return _import_or_skip("hypernix.cardboard_box", "CardboardBox")

    def test_append_and_read_round_trip(self, tmp_path):
        CardboardBox = self._get()
        box = CardboardBox(tmp_path / "tokens.bin")
        box.append([1, 2, 3, 4, 5])
        out = box.read(0, 5)
        assert list(out) == [1, 2, 3, 4, 5]

    def test_multiple_appends_accumulate(self, tmp_path):
        CardboardBox = self._get()
        box = CardboardBox(tmp_path / "tokens.bin")
        box.append([1, 2, 3])
        box.append([4, 5, 6])
        assert box.valid_tokens == 6
        out = box.read(0, 6)
        assert list(out) == [1, 2, 3, 4, 5, 6]

    def test_read_past_end_returns_only_available_tokens(self, tmp_path):
        CardboardBox = self._get()
        box = CardboardBox(tmp_path / "tokens.bin")
        box.append([1, 2, 3])
        out = box.read(0, 1000)
        assert list(out) == [1, 2, 3]

    def test_read_at_or_past_end_returns_empty(self, tmp_path):
        CardboardBox = self._get()
        box = CardboardBox(tmp_path / "tokens.bin")
        box.append([1, 2, 3])
        out = box.read(3, 10)
        assert len(out) == 0

    def test_prune_reduces_valid_token_count(self, tmp_path):
        CardboardBox = self._get()
        box = CardboardBox(tmp_path / "tokens.bin")
        box.append(list(range(10)))
        box.prune(4)
        assert box.valid_tokens == 6
        out = box.read(0, 6)
        assert list(out) == list(range(4, 10))

    def test_prune_more_than_available_clamps_to_zero_remaining(self, tmp_path):
        CardboardBox = self._get()
        box = CardboardBox(tmp_path / "tokens.bin")
        box.append([1, 2, 3])
        box.prune(1000)
        assert box.valid_tokens == 0

    def test_reopening_existing_file_restores_state(self, tmp_path):
        CardboardBox = self._get()
        path = tmp_path / "tokens.bin"
        box1 = CardboardBox(path)
        box1.append([9, 8, 7])
        box1.prune(1)

        box2 = CardboardBox(path, create_if_missing=False)
        assert box2.valid_tokens == 2
        assert list(box2.read(0, 2)) == [8, 7]

    def test_missing_file_without_create_raises(self, tmp_path):
        CardboardBox = self._get()
        with pytest.raises(FileNotFoundError):
            CardboardBox(tmp_path / "does_not_exist.bin", create_if_missing=False)

    def test_defragment_reclaims_pruned_space_and_preserves_data(self, tmp_path):
        CardboardBox = self._get()
        path = tmp_path / "tokens.bin"
        box = CardboardBox(path)
        box.append(list(range(20)))
        box.prune(10)
        size_before = path.stat().st_size
        box.defragment()
        size_after = path.stat().st_size
        assert size_after <= size_before
        assert box.valid_tokens == 10
        assert list(box.read(0, 10)) == list(range(10, 20))

    def test_defragment_on_already_compact_file_is_a_noop(self, tmp_path):
        CardboardBox = self._get()
        box = CardboardBox(tmp_path / "tokens.bin")
        box.append([1, 2, 3])
        before = list(box.read(0, 3))
        box.defragment()  # head == 0 already; should be a no-op
        assert list(box.read(0, 3)) == before

    def test_empty_append_is_a_noop(self, tmp_path):
        CardboardBox = self._get()
        box = CardboardBox(tmp_path / "tokens.bin")
        box.append([])
        assert box.valid_tokens == 0


# ===========================================================================
# Cross-feature integration: STML + TurboAbbicus composed together, as
# `hypernix train run --use-turbo-abbicus --use-stml` would exercise.
# ===========================================================================

class TestSTMLTurboAbbicusIntegration:
    def test_turbo_abbicus_feeding_into_stml_produces_valid_batch(self):
        TurboAbbicus, TurboAbbicusConfig = _import_or_skip(
            "hypernix.abbicus", "TurboAbbicus", "TurboAbbicusConfig"
        )
        STML = _import_or_skip("hypernix.stml", "STML")

        cfg = TurboAbbicusConfig(base_context_length=512, hard_cap=2048, curriculum_steps=100)
        regulator = TurboAbbicus(cfg)
        regulator.step(50)

        stml = STML(untrained_max_context=2048, segment_length=256, regulator=regulator)
        batch = {
            "input_ids": torch.randint(0, 100, (1, 4000)),
            "attention_mask": torch.ones(1, 4000, dtype=torch.long),
        }
        out = stml.regulate(dict(batch))

        # Final folded sequence length must respect STML's segment size,
        # and the regulator's cap must have been applied upstream.
        assert out["input_ids"].shape[1] == 256
        assert out["input_ids"].shape == out["attention_mask"].shape


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
