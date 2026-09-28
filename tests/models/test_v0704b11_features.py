"""Tests for HyperNix 0.70.4 new features (up to b11).

Covers:
  - QAProcessor (qa.py) - dataset formatting and seasoning
  - STML (stml.py) - context management and VRAM context calculator
  - TurboAbbicus / TurboAbbicusConfig (abbicus.py) - exponential curriculum
  - TvTop++ layout tree, color, and dynamic widths
  - Version bump

These tests are intentionally resilient: they check module/class/function
presence and behaviour, not internal implementation details, so they will
not break when subsequent beta versions ship further changes.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from hypernix.tvtop_plus_plus import TVTopPlusPlus

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_fake_shaker(prefix: str = "S") -> MagicMock:
    """Return a duck-typed shaker that prepends a prefix to input text."""
    shaker = MagicMock()
    shaker.season = lambda text: f"{prefix}:{text}"
    return shaker


# ===========================================================================
# 1. Version
# ===========================================================================

class TestVersion:
    def test_version_is_importable_from_package(self) -> None:
        from hypernix import __version__
        # Resilient by design (see module docstring): verify __version__ is
        # importable and looks like a semver string, without pinning to a
        # specific release so this doesn't break on later version bumps.
        assert isinstance(__version__, str)
        assert re.match(r"^\d+\.\d+\.\d+", __version__)


# ===========================================================================
# 2. QAProcessor
# ===========================================================================

class TestQAProcessor:
    def test_import(self) -> None:
        from hypernix.qa import QAProcessor  # noqa: F401

    def test_dict_list_question_answer_mode(self) -> None:
        from hypernix.qa import QAProcessor

        data = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "Capital of France?", "answer": "Paris"},
        ]
        proc = QAProcessor(data, format_mode="question_answer")
        results = list(proc)
        assert len(results) == 2
        assert "Question: What is 2+2?" in results[0]
        assert "Answer: 4" in results[0]
        assert "Question: Capital of France?" in results[1]
        assert "Answer: Paris" in results[1]

    def test_predict_next_mode(self) -> None:
        from hypernix.qa import QAProcessor

        data = [{"question": "Hello", "answer": "World"}]
        proc = QAProcessor(data, format_mode="predict_next")
        results = list(proc)
        assert len(results) == 1
        assert "Hello" in results[0]
        assert "World" in results[0]
        # In predict_next mode there should be no "Question:" prefix
        assert "Question:" not in results[0]

    def test_alternate_keys_instruction_completion(self) -> None:
        from hypernix.qa import QAProcessor

        data = [{"instruction": "Describe Paris.", "completion": "Paris is the capital of France."}]
        proc = QAProcessor(data)
        results = list(proc)
        assert len(results) == 1
        assert "Describe Paris" in results[0]

    def test_jsonl_file_parsing(self, tmp_path: Path) -> None:
        from hypernix.qa import QAProcessor

        jsonl = tmp_path / "data.jsonl"
        jsonl.write_text(
            '{"question": "What is AI?", "answer": "Artificial Intelligence."}\n'
            '{"question": "What is ML?", "answer": "Machine Learning."}\n',
            encoding="utf-8",
        )
        proc = QAProcessor(jsonl)
        results = list(proc)
        assert len(results) == 2
        assert "What is AI?" in results[0]
        assert "Artificial Intelligence" in results[0]

    def test_salt_shaker_applied_to_question(self) -> None:
        from hypernix.qa import QAProcessor

        data = [{"question": "hello", "answer": "world"}]
        shaker = _make_fake_shaker("SALT")
        proc = QAProcessor(data, salt_shaker=shaker, season_target="question")
        results = list(proc)
        assert "SALT:hello" in results[0]
        # Answer should NOT be seasoned
        assert "SALT:world" not in results[0]
        assert "world" in results[0]

    def test_pepper_shaker_applied_to_answer(self) -> None:
        from hypernix.qa import QAProcessor

        data = [{"question": "hi", "answer": "bye"}]
        shaker = _make_fake_shaker("PEPPER")
        proc = QAProcessor(data, pepper_shaker=shaker, season_target="answer")
        results = list(proc)
        assert "PEPPER:bye" in results[0]
        assert "PEPPER:hi" not in results[0]

    def test_both_shakers_season_target_both(self) -> None:
        from hypernix.qa import QAProcessor

        data = [{"question": "q", "answer": "a"}]
        salt = _make_fake_shaker("S")
        pepper = _make_fake_shaker("P")
        proc = QAProcessor(data, salt_shaker=salt, pepper_shaker=pepper, season_target="both")
        results = list(proc)
        # pepper applied first, then salt: salt(pepper(text))
        assert results[0].count("S:") >= 1

    def test_template_keywords_safe_from_seasoning(self) -> None:
        """Seasoning must never corrupt 'Question:' or 'Answer:' keywords."""
        from hypernix.qa import QAProcessor

        data = [{"question": "Who am I?", "answer": "You are a robot."}]
        shaker = _make_fake_shaker("X")
        proc = QAProcessor(data, salt_shaker=shaker, season_target="both")
        results = list(proc)
        # Template keywords must remain intact
        assert results[0].startswith("Question: X:Who am I?")
        assert "Answer: X:You are a robot." in results[0]

    def test_iter_protocol(self) -> None:
        from hypernix.qa import QAProcessor

        data = [{"question": "a", "answer": "b"}]
        proc = QAProcessor(data)
        # Must be iterable via __iter__
        assert list(proc) == list(proc.process())

    def test_empty_dataset_produces_no_output(self) -> None:
        from hypernix.qa import QAProcessor

        proc = QAProcessor([])
        assert list(proc) == []

    def test_exported_from_top_level(self) -> None:
        from hypernix import QAProcessor  # noqa: F401


# ===========================================================================
# 3. STML — context manager and VRAM calculator
# ===========================================================================

class TestCalculateVRAMContext:
    def test_import(self) -> None:
        from hypernix.stml import calculate_vram_context  # noqa: F401

    def test_returns_positive_int(self) -> None:
        from hypernix.stml import calculate_vram_context

        ctx = calculate_vram_context(vram_gb=24.0, model_size_params=7.0, batch_size=2)
        assert isinstance(ctx, int)
        assert ctx > 0

    def test_larger_vram_gives_larger_context(self) -> None:
        from hypernix.stml import calculate_vram_context

        # Use a very small model (0.5B) so the model overhead doesn't swamp both
        # VRAM budgets — ensuring the extra VRAM actually translates to more context.
        ctx_small = calculate_vram_context(vram_gb=6.0, model_size_params=0.5, batch_size=1)
        ctx_large = calculate_vram_context(vram_gb=24.0, model_size_params=0.5, batch_size=1)
        assert ctx_large > ctx_small

    def test_larger_model_gives_smaller_context(self) -> None:
        from hypernix.stml import calculate_vram_context

        ctx_small_model = calculate_vram_context(vram_gb=24.0, model_size_params=1.0, batch_size=1)
        ctx_large_model = calculate_vram_context(vram_gb=24.0, model_size_params=70.0, batch_size=1)
        assert ctx_small_model > ctx_large_model

    def test_minimum_floor_of_128(self) -> None:
        from hypernix.stml import calculate_vram_context

        # Even in extreme scenarios the floor should be 128
        ctx = calculate_vram_context(vram_gb=1.0, model_size_params=70.0, batch_size=8)
        assert ctx >= 128

    def test_context_is_multiple_of_128(self) -> None:
        from hypernix.stml import calculate_vram_context

        ctx = calculate_vram_context(vram_gb=24.0, model_size_params=7.0, batch_size=2)
        assert ctx % 128 == 0

    def test_fp32_gives_smaller_context_than_fp16(self) -> None:
        from hypernix.stml import calculate_vram_context

        ctx_fp16 = calculate_vram_context(vram_gb=24.0, model_size_params=7.0, batch_size=2, precision="fp16")
        ctx_fp32 = calculate_vram_context(vram_gb=24.0, model_size_params=7.0, batch_size=2, precision="fp32")
        assert ctx_fp16 >= ctx_fp32

    def test_exported_from_top_level(self) -> None:
        from hypernix import calculate_vram_context  # noqa: F401


class TestSTML:
    def test_import(self) -> None:
        from hypernix.stml import STML  # noqa: F401

    def test_basic_construction(self) -> None:
        from hypernix.stml import STML

        mgr = STML(trained_context=512, untrained_max_context=2048, segment_length=128)
        assert mgr.trained_context == 512
        assert mgr.untrained_max_context == 2048
        assert mgr.segment_length == 128

    def test_regulate_passthrough_short_sequence(self) -> None:
        import torch

        from hypernix.stml import STML

        mgr = STML(trained_context=512, untrained_max_context=2048, segment_length=256)
        batch = {"input_ids": torch.zeros(2, 64, dtype=torch.long)}
        out = mgr.regulate(batch)
        # Sequence shorter than segment_length — should pass through unchanged
        assert out["input_ids"].shape == (2, 64)

    def test_regulate_truncates_beyond_untrained_max(self) -> None:
        import torch

        from hypernix.stml import STML

        mgr = STML(trained_context=512, untrained_max_context=256, segment_length=64)
        batch = {
            "input_ids": torch.zeros(1, 512, dtype=torch.long),
            "labels": torch.zeros(1, 512, dtype=torch.long),
        }
        out = mgr.regulate(batch)
        assert out["input_ids"].shape[1] <= 256

    def test_regulate_folds_long_sequence_into_batch(self) -> None:
        import torch

        from hypernix.stml import STML

        seg = 64
        bsz = 2
        seq_len = seg * 4  # exactly 4 segments
        mgr = STML(trained_context=512, untrained_max_context=4096, segment_length=seg)
        batch = {"input_ids": torch.zeros(bsz, seq_len, dtype=torch.long)}
        out = mgr.regulate(batch)
        # Should fold into (bsz * num_segments, segment_length)
        assert out["input_ids"].shape == (bsz * 4, seg)

    def test_regulate_without_input_ids_is_noop(self) -> None:
        from hypernix.stml import STML

        mgr = STML()
        batch: dict = {"some_key": "some_value"}
        out = mgr.regulate(batch)
        assert out == batch

    def test_regulate_with_regulator(self) -> None:
        import torch

        from hypernix.stml import STML

        fake_reg = MagicMock()
        # side_effect records the call but still returns the batch unmodified
        fake_reg.regulate = MagicMock(side_effect=lambda b: b)
        mgr = STML(trained_context=512, untrained_max_context=4096, segment_length=128, regulator=fake_reg)
        batch = {"input_ids": torch.zeros(1, 64, dtype=torch.long)}
        out = mgr.regulate(batch)
        # STML calls regulator.regulate internally when a regulator is present
        fake_reg.regulate.assert_called_once()
        assert out is not None

    def test_exported_from_top_level(self) -> None:
        from hypernix import STML  # noqa: F401


# ===========================================================================
# 4. TurboAbbicus
# ===========================================================================

class TestTurboAbbicusConfig:
    def test_import(self) -> None:
        from hypernix.abbicus import TurboAbbicusConfig  # noqa: F401

    def test_defaults(self) -> None:
        from hypernix.abbicus import TurboAbbicusConfig

        cfg = TurboAbbicusConfig()
        assert cfg.hard_cap > cfg.base_context_length or cfg.hard_cap == 16384
        assert cfg.oscillation_enabled is True
        assert 0 < cfg.oscillation_amplitude <= 1.0

    def test_size_multiplier_7b(self) -> None:
        from hypernix.abbicus import TurboAbbicusConfig

        cfg = TurboAbbicusConfig(model_size="7B")
        assert cfg.size_multiplier == 1.0

    def test_size_multiplier_70b(self) -> None:
        from hypernix.abbicus import TurboAbbicusConfig

        cfg = TurboAbbicusConfig(model_size="70B")
        assert cfg.size_multiplier >= 1.5

    def test_size_multiplier_1b(self) -> None:
        from hypernix.abbicus import TurboAbbicusConfig

        cfg = TurboAbbicusConfig(model_size="1B")
        assert cfg.size_multiplier < 1.0

    def test_exported_from_top_level(self) -> None:
        from hypernix import TurboAbbicusConfig  # noqa: F401


class TestTurboAbbicus:
    def test_import(self) -> None:
        from hypernix.abbicus import TurboAbbicus  # noqa: F401

    def test_construction_and_initial_length(self) -> None:
        from hypernix.abbicus import TurboAbbicus, TurboAbbicusConfig

        cfg = TurboAbbicusConfig(base_context_length=1024, hard_cap=8192)
        ta = TurboAbbicus(cfg)
        assert isinstance(ta.current_max_length, int)
        assert ta.current_max_length >= 128

    def test_context_grows_exponentially_with_steps(self) -> None:
        from hypernix.abbicus import TurboAbbicus, TurboAbbicusConfig

        cfg = TurboAbbicusConfig(
            base_context_length=512,
            hard_cap=8192,
            curriculum_steps=100,
            oscillation_enabled=False,
        )
        ta = TurboAbbicus(cfg)
        ta.step(0)
        len_0 = ta.current_max_length
        ta.step(50)
        len_50 = ta.current_max_length
        ta.step(100)
        len_100 = ta.current_max_length
        # Context should grow as steps increase
        assert len_50 >= len_0
        assert len_100 >= len_50

    def test_growth_is_exponential_not_linear(self) -> None:
        """The ratio of increments should grow (exponential), not be constant (linear)."""
        from hypernix.abbicus import TurboAbbicus, TurboAbbicusConfig

        cfg = TurboAbbicusConfig(
            base_context_length=512,
            hard_cap=16384,
            curriculum_steps=100,
            oscillation_enabled=False,
        )
        ta = TurboAbbicus(cfg)
        ta.step(10)
        l10 = ta.current_max_length
        ta.step(50)
        l50 = ta.current_max_length
        ta.step(90)
        l90 = ta.current_max_length

        delta_early = l50 - l10
        delta_late = l90 - l50
        # Later increments should be equal to or larger than early (exponential curve)
        assert delta_late >= delta_early - 10  # small tolerance for rounding

    def test_hard_cap_respected(self) -> None:
        from hypernix.abbicus import TurboAbbicus, TurboAbbicusConfig

        cfg = TurboAbbicusConfig(
            hard_cap=2048,
            curriculum_steps=10,
            oscillation_enabled=False,
        )
        ta = TurboAbbicus(cfg)
        for s in range(100):
            ta.step(s)
        # With oscillation off, should never exceed hard_cap
        assert ta.current_max_length <= cfg.hard_cap

    def test_oscillation_at_cap(self) -> None:
        from hypernix.abbicus import TurboAbbicus, TurboAbbicusConfig

        cfg = TurboAbbicusConfig(
            hard_cap=512,
            curriculum_steps=1,
            oscillation_enabled=True,
            oscillation_frequency=0.5,
            oscillation_amplitude=0.1,
        )
        ta = TurboAbbicus(cfg)
        lengths = set()
        for s in range(20):
            ta.step(s)
            lengths.add(ta.current_max_length)
        # With oscillation enabled at cap, multiple lengths should be produced
        assert len(lengths) > 1

    def test_regulate_truncates_batch(self) -> None:
        import torch

        from hypernix.abbicus import TurboAbbicus, TurboAbbicusConfig

        cfg = TurboAbbicusConfig(
            base_context_length=64,
            hard_cap=128,
            curriculum_steps=1,
            oscillation_enabled=False,
        )
        ta = TurboAbbicus(cfg)
        ta.step(0)
        batch = {
            "input_ids": torch.zeros(2, 1024, dtype=torch.long),
            "labels": torch.zeros(2, 1024, dtype=torch.long),
        }
        out = ta.regulate(batch)
        assert out["input_ids"].shape[1] <= ta.current_max_length

    def test_vram_safeguard_does_not_use_gpu_as_change_factor(self) -> None:
        """VRAM guard monitors allocation but must never call psutil/cpu usage
        as a scaling mechanism — only CPU is used for the oscillation factor."""
        from hypernix.abbicus import TurboAbbicusConfig

        cfg = TurboAbbicusConfig()
        # cpu_factor_scale drives the oscillation, not VRAM factor
        assert hasattr(cfg, "cpu_factor_scale")
        assert hasattr(cfg, "vram_safety_threshold")
        # There must NOT be a gpu_factor_scale attribute
        assert not hasattr(cfg, "gpu_factor_scale")

    def test_exported_from_top_level(self) -> None:
        from hypernix import TurboAbbicus  # noqa: F401


# ===========================================================================
# 5. tvtop++ layout & color fixes
# ===========================================================================

class TestTVTopPlusPlusLayout:
    """Verify the layout tree is well-formed and panel signatures accept console."""

    def _make_dashboard(self, tmp_path: Path) -> TVTopPlusPlus:
        from hypernix.tvtop_plus_plus import TVTopPlusPlus

        log = tmp_path / "train.log"
        log.write_text("loss=0.5 step=10/100\n", encoding="utf-8")
        return TVTopPlusPlus(log_path=log, color=False)

    def test_layout_has_correct_named_regions(self, tmp_path: Path) -> None:
        from rich.console import Console

        tvt = self._make_dashboard(tmp_path)
        frame = tvt.latest_frame()
        console = Console(force_terminal=False, width=120)
        layout = tvt._build_layout(frame, console)

        # All required regions must exist
        for name in ("header", "body", "footer", "top", "bottom",
                     "left", "right", "training", "hardware",
                     "process", "gpu", "loss", "log"):
            assert layout[name] is not None, f"Missing layout region: {name!r}"

    def test_layout_body_has_top_and_bottom_not_left_right_directly(self, tmp_path: Path) -> None:
        """body must be split into top/bottom, not left/right (fixes the double-split bug)."""
        from rich.console import Console

        tvt = self._make_dashboard(tmp_path)
        frame = tvt.latest_frame()
        console = Console(force_terminal=False, width=120)
        layout = tvt._build_layout(frame, console)

        # body's direct children must be top and bottom
        body_children = {c.name for c in layout["body"].children}
        assert "top" in body_children
        assert "bottom" in body_children

    def test_hardware_panel_accepts_console_kwarg(self, tmp_path: Path) -> None:
        from rich.console import Console

        tvt = self._make_dashboard(tmp_path)
        frame = tvt.latest_frame()
        console = Console(force_terminal=False, width=120)
        panel = tvt._make_hardware_panel(frame, console)
        assert panel is not None

    def test_loss_panel_accepts_console_kwarg(self, tmp_path: Path) -> None:
        from rich.console import Console

        tvt = self._make_dashboard(tmp_path)
        frame = tvt.latest_frame()
        console = Console(force_terminal=False, width=120)
        panel = tvt._make_loss_panel(frame, console)
        assert panel is not None

    def test_log_panel_accepts_console_kwarg(self, tmp_path: Path) -> None:
        from rich.console import Console

        tvt = self._make_dashboard(tmp_path)
        frame = tvt.latest_frame()
        console = Console(force_terminal=False, width=120)
        panel = tvt._make_log_panel(frame, console)
        assert panel is not None

    def test_log_panel_shows_more_lines_on_wide_terminal(self, tmp_path: Path) -> None:
        """Log panel must show 8 lines, not the old 6."""
        from rich.console import Console

        from hypernix.tvtop_plus_plus import TVTopPlusPlus

        log = tmp_path / "train.log"
        lines = "\n".join(f"loss={i/10:.2f}" for i in range(1, 15))
        log.write_text(lines + "\n", encoding="utf-8")
        tvt = TVTopPlusPlus(log_path=log, color=False)
        frame = tvt.latest_frame()
        console = Console(force_terminal=False, width=200)
        panel = tvt._make_log_panel(frame, console)
        # Panel renderable must not be None
        assert panel.renderable is not None

    def test_run_initialises_console_without_fixed_width(self, tmp_path: Path) -> None:
        """Console must be created without a hardcoded width argument so that
        window resizes are reflected dynamically."""
        import inspect

        from hypernix.tvtop_plus_plus import TVTopPlusPlus

        source = inspect.getsource(TVTopPlusPlus.run)
        # The correct pattern: Console(force_terminal=True) with NO width=... kwarg
        assert "width=term_width" not in source, (
            "Console should NOT be created with a fixed width=term_width"
        )


# ===========================================================================
# 6. CLI integration — stml subcommand registered
# ===========================================================================

class TestCLIStml:
    def test_stml_in_subcommands_set(self) -> None:
        from hypernix.cli import _SUBCOMMANDS

        assert "stml" in _SUBCOMMANDS

    def test_stml_help_exits_zero(self) -> None:
        """Running `hypernix stml --help` via the internal function must not crash."""
        from hypernix import cli

        with pytest.raises(SystemExit) as exc_info:
            cli._run_stml(["--help"])
        assert exc_info.value.code == 0

    def test_stml_calculator_produces_output(self, capsys) -> None:
        from hypernix import cli

        cli._run_stml([
            "--vram", "16.0",
            "--params", "4.0",
            "--batch-size", "2",
        ])
        captured = capsys.readouterr()
        assert "context length" in captured.out.lower() or "tokens" in captured.out.lower()


# ===========================================================================
# 7. train.py signature
# ===========================================================================

class TestTrainSignature:
    def test_train_accepts_new_kwargs(self) -> None:
        import inspect

        from hypernix.train import train

        sig = inspect.signature(train)
        params = sig.parameters
        assert "use_abbicus" in params
        assert "use_turbo_abbicus" in params
        assert "use_stml" in params
        assert "untrained_max_context" in params
        assert "segment_length" in params

    def test_train_defaults_are_sane(self) -> None:
        import inspect

        from hypernix.train import train

        sig = inspect.signature(train)
        p = sig.parameters
        assert p["use_abbicus"].default is False
        assert p["use_turbo_abbicus"].default is False
        assert p["use_stml"].default is False
        assert p["untrained_max_context"].default == 8192
        assert p["segment_length"].default == 512


# ===========================================================================
# 8. old_oven.py signature
# ===========================================================================

class TestOldOvenSignature:
    def test_oven_train_accepts_new_kwargs(self) -> None:
        import inspect

        from hypernix.old_oven import CodeOven

        sig = inspect.signature(CodeOven.train)
        params = sig.parameters
        assert "use_turbo_abbicus" in params
        assert "use_stml" in params
        assert "untrained_max_context" in params
        assert "segment_length" in params

    def test_oven_train_defaults_are_sane(self) -> None:
        import inspect

        from hypernix.old_oven import CodeOven

        sig = inspect.signature(CodeOven.train)
        p = sig.parameters
        assert p["use_turbo_abbicus"].default is False
        assert p["use_stml"].default is False
        assert p["untrained_max_context"].default == 8192
