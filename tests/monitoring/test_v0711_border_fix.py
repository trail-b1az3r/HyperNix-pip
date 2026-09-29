"""Regression tests for the v0.71.1 tvtop++ / cctvtop border-corruption fix.

Bug: ``tv._bar_str`` / ``_gauge_line`` / ``_block_history_bar`` embed raw
ANSI SGR escape sequences (e.g. ``\\x1b[32m#\\x1b[0m``) directly into the
strings they return when called with ``color_enabled=True``. That's fine
for the plain-ANSI ``tvtop`` renderer, which measures visible width with
its own ``_strip_ansi``/``_visible_len`` helpers.

``tvtop_plus_plus.py`` (Rich-based ``tvtop++``, and ``cctvtop.py`` which
subclasses it) used to splice that same ANSI-laden output straight into a
``rich.text.Text`` via a plain f-string, e.g.::

    content.append(f"CPU History [{cpu_hist}]\\n", style="green")

Rich's ``Text`` has no idea those bytes are escape codes -- it treats them
as literal characters, so ``Text.cell_len`` (which drives where Rich draws
a Panel's right-hand border) comes out far larger than what's actually
visible on screen once the terminal interprets the embedded codes. The
observable symptom (confirmed against a live screenshot) is stray "│"
box-drawing characters and even literal escape-sequence text floating
outside the Hardware Vitals / GPU Details panels, with the right border
drawn in the wrong column.

The fix routes any string that may contain embedded ANSI through
``Text.from_ansi`` (via the new ``_append_ansi`` helper) so Rich parses
the escape codes into proper zero-width style spans instead of literal
characters.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from rich.text import Text

from hypernix.cctvtop import CCTVTop
from hypernix.tv import _bar_str, _block_history_bar, _gauge_line
from hypernix.tvtop_plus_plus import Frame, TVTopPlusPlus, _append_ansi


def _make_frame(**overrides) -> Frame:
    defaults = dict(
        has_training_data=True,
        step=50,
        total_steps=100,
        loss=1.2345,
        lr=3e-4,
        throughput=12.5,
        elapsed_seconds=30.0,
        cpu_percent=42.0,
        ram_percent=55.0,
        gpu_util_percent=88.0,
        gpu_mem_used_mib=2286,
        gpu_mem_total_mib=8192,
        gpu_temp_c=72.0,
        gpu_power_w=140.0,
        gpu_power_limit_w=180.0,
        gpu_name="NVIDIA GeForce GTX 1080",
        cpu_history=[10.0, 20.0, 90.0, 55.0],
        ram_history=[30.0, 40.0, 95.0, 60.0],
        gpu_util_history=[5.0, 88.0, 91.0, 60.0],
    )
    defaults.update(overrides)
    return Frame(**defaults)


# ---------------------------------------------------------------------------
# The raw ANSI-embedding helpers themselves are untouched (they're shared
# with the plain-ANSI `tvtop` renderer, which handles them correctly) --
# just confirm they really do still embed raw escapes when color_enabled,
# so the tests below are exercising the actual reported failure mode.
# ---------------------------------------------------------------------------

def test_ansi_bar_helpers_still_embed_raw_escapes_when_colored() -> None:
    assert "\x1b[" in _bar_str(0.5, 10, ascii_only=False, color_enabled=True)
    assert "\x1b[" in _gauge_line("CPU", 50.0, 10, ascii_only=False, color_enabled=True)
    assert "\x1b[" in _block_history_bar([10.0, 90.0], 4, True)


# ---------------------------------------------------------------------------
# _append_ansi — the fix itself
# ---------------------------------------------------------------------------

class TestAppendAnsiHelper:
    def test_strips_raw_escape_bytes_from_the_rendered_text(self) -> None:
        content = Text()
        bar = _bar_str(0.5, 10, ascii_only=False, color_enabled=True)
        _append_ansi(content, f"VRAM  {bar} 2286/8192 MiB\n", style="yellow")
        # No literal ESC byte should survive into the Text's plain content --
        # that's exactly what corrupted Rich's cell-width measurement.
        assert "\x1b" not in content.plain

    def test_cell_len_matches_visible_width_not_the_raw_byte_count(self) -> None:
        content = Text()
        bar = _bar_str(1.0, 20, ascii_only=False, color_enabled=True)
        raw = f"CPU History [{bar}]"
        _append_ansi(content, raw)
        # The naive f-string-embed approach would have made Text see this
        # as far longer than 20 block characters + "CPU History []".
        assert content.cell_len == len("CPU History [" + ("█" * 20) + "]")

    def test_color_spans_are_still_applied(self) -> None:
        """The whole point of embedding ANSI was per-cell color ramps
        (green/yellow/red) -- from_ansi must preserve those as real spans,
        not just discard the color information."""
        content = Text()
        hist = _block_history_bar([10.0, 90.0], 2, True)  # one green, one red cell
        _append_ansi(content, hist)
        assert len(content.spans) >= 2

    def test_plain_text_with_no_ansi_is_unaffected(self) -> None:
        content = Text()
        _append_ansi(content, "Loss       1.2345\n", style="yellow")
        assert content.plain == "Loss       1.2345\n"


# ---------------------------------------------------------------------------
# Panel-level regression tests -- build the real panels tvtop++ renders
# and confirm none of them leak raw ANSI bytes into their Text content.
# These would have failed before the fix (asserting equality with the
# corrupted, much-longer cell_len that caused the border to shift).
# ---------------------------------------------------------------------------

class TestPanelsDoNotLeakAnsiBytes:
    def _tvt(self, tmp_path: Path) -> TVTopPlusPlus:
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5 step=50/100\n", encoding="utf-8")
        return TVTopPlusPlus(log_path=log_file, color=True)

    def test_training_panel_progress_bar_is_clean(self, tmp_path: Path) -> None:
        tvt = self._tvt(tmp_path)
        frame = _make_frame()
        panel = tvt._make_training_panel(frame)
        assert "\x1b" not in panel.renderable.plain

    def test_hardware_panel_gauges_and_history_are_clean(self, tmp_path: Path) -> None:
        tvt = self._tvt(tmp_path)
        frame = _make_frame()
        panel = tvt._make_hardware_panel(frame, console=None)
        assert "\x1b" not in panel.renderable.plain
        # The history bars should still be present and readable.
        assert "CPU History [" in panel.renderable.plain
        assert "RAM History [" in panel.renderable.plain
        assert "GPU History [" in panel.renderable.plain

    def test_gpu_panel_vram_temp_power_bars_are_clean(self, tmp_path: Path) -> None:
        tvt = self._tvt(tmp_path)
        frame = _make_frame()
        panel = tvt._make_gpu_panel(frame)
        assert "\x1b" not in panel.renderable.plain
        assert "VRAM" in panel.renderable.plain
        assert "Temp" in panel.renderable.plain
        assert "Power" in panel.renderable.plain

    @pytest.mark.parametrize("color", [True, False])
    def test_no_panel_content_cell_len_exceeds_its_plain_length(
        self, tmp_path: Path, color: bool,
    ) -> None:
        """A corrupted Text (raw ESC bytes counted as cells) always reports
        cell_len >= len(plain) with room to spare for the escape bytes;
        a clean Text reports cell_len == number of actually-rendered cells,
        which for our ASCII/box-drawing content is <= len(plain)."""
        tvt = TVTopPlusPlus(log_path=tmp_path / "does-not-exist.log", color=color)
        frame = _make_frame()
        for panel in (
            tvt._make_training_panel(frame),
            tvt._make_hardware_panel(frame, console=None),
            tvt._make_gpu_panel(frame),
        ):
            text = panel.renderable
            assert "\x1b" not in text.plain
            assert text.cell_len <= len(text.plain)


# ---------------------------------------------------------------------------
# cctvtop inherits _make_training_panel / _make_hardware_panel / _make_gpu_panel
# unchanged, so the same corruption applied there too -- confirm the fix
# carries through the subclass.
# ---------------------------------------------------------------------------

class TestCCTVTopInheritsTheFix:
    def test_cctvtop_hardware_panel_is_clean(self, tmp_path: Path) -> None:
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5 step=1/10\n", encoding="utf-8")
        app = CCTVTop(log_path=log_file, color=True)
        frame = _make_frame()
        panel = app._make_hardware_panel(frame, console=None)
        assert "\x1b" not in panel.renderable.plain
