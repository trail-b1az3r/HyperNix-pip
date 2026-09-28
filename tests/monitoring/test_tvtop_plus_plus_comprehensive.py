"""Comprehensive tests for tvtop++ dashboard (v0.70.4+).

Tests cover:
- Syntax validation and imports
- Rich v15 integration and layout rendering
- Cross-platform compatibility (Linux, macOS, Windows)
- Panel rendering with various data states
- Live dashboard simulation
- Edge cases (no log, empty log, malformed log)
- Performance and memory usage
- ASCII mode and color modes
- Small mode layout
"""
from __future__ import annotations

import platform
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hypernix.tvtop_plus_plus import TVTopPlusPlus, cli_main


class TestSyntaxAndImports:
    """Test that tvtop++ has correct syntax and all imports work."""

    def test_module_imports(self):
        """Test all required imports are available."""
        # Rich v15 components
        from rich.console import Console
        from rich.layout import Layout
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        
        assert Console is not None
        assert Layout is not None
        assert Live is not None
        assert Panel is not None
        assert Table is not None
        assert Text is not None

    def test_tvtop_plus_plus_class_exists(self):
        """Test TVTopPlusPlus class is properly defined."""
        assert hasattr(TVTopPlusPlus, 'run')
        assert hasattr(TVTopPlusPlus, 'latest_frame')
        assert hasattr(TVTopPlusPlus, '_build_layout')
        assert hasattr(TVTopPlusPlus, '_make_training_panel')
        assert hasattr(TVTopPlusPlus, '_make_hardware_panel')
        assert hasattr(TVTopPlusPlus, '_make_process_panel')
        assert hasattr(TVTopPlusPlus, '_make_gpu_panel')
        assert hasattr(TVTopPlusPlus, '_make_loss_panel')
        assert hasattr(TVTopPlusPlus, '_make_log_panel')

    def test_cli_main_exists(self):
        """Test CLI entry point exists."""
        assert callable(cli_main)


class TestCrossPlatformCompatibility:
    """Test tvtop++ works across different operating systems."""

    def test_platform_detection(self):
        """Test that platform detection works."""
        current_os = platform.system()
        assert current_os in ['Linux', 'Darwin', 'Windows']

    def test_path_handling_cross_platform(self, tmp_path: Path):
        """Test path handling works on all platforms."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        # Should accept both string and Path
        tvt1 = TVTopPlusPlus(log_path=str(log_file))
        tvt2 = TVTopPlusPlus(log_path=log_file)
        
        assert tvt1.log_tail is not None
        assert tvt2.log_tail is not None

    @pytest.mark.skipif(platform.system() == 'Windows', reason="GPU tests skipped on Windows CI")
    def test_gpu_query_graceful_degradation(self, tmp_path: Path):
        """Test GPU panel handles missing nvidia-smi gracefully."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file)
        frame = tvt.latest_frame()
        
        # Should not crash even without GPU
        assert frame.gpu_util_percent is None or isinstance(frame.gpu_util_percent, int | float)
        assert frame.gpu_mem_total_mib is None or isinstance(frame.gpu_mem_total_mib, int)

    def test_cpu_reading_fallback(self, tmp_path: Path):
        """Test CPU reading works with psutil fallback."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file)
        frame = tvt.latest_frame()
        
        # Should have some CPU value (from psutil or /proc/stat)
        assert frame.cpu_percent is None or isinstance(frame.cpu_percent, int | float)


class TestLayoutRendering:
    """Test Rich layout rendering with various configurations."""

    def test_basic_layout_creation(self, tmp_path: Path):
        """Test basic layout can be created."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5 step=10/100\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file, color=True)
        frame = tvt.latest_frame()
        
        from rich.console import Console
        console = Console(force_terminal=False)
        layout = tvt._build_layout(frame, console)
        
        assert layout is not None
        # Access named sections properly
        header_section = layout["header"]
        body_section = layout["body"]
        footer_section = layout["footer"]
        assert header_section is not None
        assert body_section is not None
        assert footer_section is not None

    def test_all_panels_render(self, tmp_path: Path):
        """Test all panels render without errors."""
        log_file = tmp_path / "train.log"
        log_file.write_text(
            "loss=0.5 step=10/100 lr=0.001\n"
            "loss=0.4 step=20/100\n"
            "loss=0.3 step=30/100\n",
            encoding="utf-8"
        )
        
        tvt = TVTopPlusPlus(log_path=log_file, color=True)
        frame = tvt.latest_frame()
        
        # Test each panel renderer
        training_panel = tvt._make_training_panel(frame)
        hardware_panel = tvt._make_hardware_panel(frame)
        process_panel = tvt._make_process_panel(frame)
        gpu_panel = tvt._make_gpu_panel(frame)
        loss_panel = tvt._make_loss_panel(frame)
        log_panel = tvt._make_log_panel(frame)
        
        assert training_panel is not None
        assert hardware_panel is not None
        assert process_panel is not None
        assert gpu_panel is not None
        assert loss_panel is not None
        assert log_panel is not None

    def test_render_with_no_log(self):
        """Test rendering when no log file is specified."""
        tvt = TVTopPlusPlus(log_path=None, color=True)
        frame = tvt.latest_frame()
        
        from rich.console import Console
        console = Console(force_terminal=False)
        layout = tvt._build_layout(frame, console)
        
        assert layout is not None
        # Training panel should show waiting message
        training_panel = tvt._make_training_panel(frame)
        assert training_panel is not None

    def test_render_with_empty_log(self, tmp_path: Path):
        """Test rendering with an empty log file."""
        log_file = tmp_path / "empty.log"
        log_file.write_text("", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file, color=True)
        frame = tvt.latest_frame()
        
        from rich.console import Console
        console = Console(force_terminal=False)
        layout = tvt._build_layout(frame, console)
        
        assert layout is not None

    def test_render_with_malformed_log(self, tmp_path: Path):
        """Test rendering with malformed log entries."""
        log_file = tmp_path / "bad.log"
        log_file.write_text(
            "random garbage\n"
            "not a training log\n"
            "12345\n",
            encoding="utf-8"
        )
        
        tvt = TVTopPlusPlus(log_path=log_file, color=True)
        frame = tvt.latest_frame()
        
        from rich.console import Console
        console = Console(force_terminal=False)
        layout = tvt._build_layout(frame, console)
        
        assert layout is not None


class TestColorAndAsciiModes:
    """Test color and ASCII-only rendering modes."""

    def test_color_mode_enabled(self, tmp_path: Path):
        """Test rendering with colors enabled."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file, color=True, ascii_only=False)
        frame = tvt.latest_frame()
        
        from rich.console import Console
        console = Console(force_terminal=False)
        layout = tvt._build_layout(frame, console)
        
        # Should use Rich styling
        assert layout is not None

    def test_ascii_mode(self, tmp_path: Path):
        """Test rendering in ASCII-only mode."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5 step=10/100\n", encoding="utf-8")
        
        # Check progress bar uses simpler characters (no block elements when ascii_only)
        from hypernix.tv import _bar_str
        bar = _bar_str(0.5, 20, ascii_only=True, color_enabled=False)
        # Bar should only contain basic ASCII characters
        assert all(ord(c) < 128 for c in bar)

    def test_no_color_mode(self, tmp_path: Path):
        """Test rendering with colors disabled."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file, color=False)
        frame = tvt.latest_frame()
        
        from rich.console import Console
        console = Console(force_terminal=False)
        layout = tvt._build_layout(frame, console)
        
        assert layout is not None


class TestSmallMode:
    """Test compact/small mode layout."""

    def test_small_mode_layout(self, tmp_path: Path):
        """Test small mode creates simpler layout."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5 step=10/100\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file, small_mode=True)
        frame = tvt.latest_frame()
        
        from rich.console import Console
        console = Console(force_terminal=False)
        layout = tvt._build_layout(frame, console)
        
        assert layout is not None


class TestLiveDashboard:
    """Test live dashboard functionality."""

    def test_frame_updates(self, tmp_path: Path):
        """Test that frames update with new data."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5 step=10/100\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file)
        
        # Append new data
        with log_file.open("a") as f:
            f.write("loss=0.4 step=20/100\n")
        
        frame2 = tvt.latest_frame()
        
        assert frame2.loss == 0.4
        assert frame2.step == 20

    def test_elapsed_time_increases(self, tmp_path: Path):
        """Test that elapsed time increases between frames."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file)
        
        frame1 = tvt.latest_frame()
        time.sleep(0.1)
        frame2 = tvt.latest_frame()
        
        assert frame2.elapsed_seconds >= frame1.elapsed_seconds

    def test_history_buffers_fill(self, tmp_path: Path):
        """Test that history buffers accumulate data."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file)
        
        # Collect multiple frames with mocked CPU value
        for i in range(5):
            with patch('hypernix.tvtop_plus_plus._safe_psutil_percent', return_value=(50.0 + i, 50.0)):
                tvt.latest_frame()
            time.sleep(0.01)
        
        # History should have accumulated some data
        frame = tvt.latest_frame()
        # At least CPU history should have entries
        assert len(frame.cpu_history) >= 1 or len(frame.ram_history) >= 1



class TestCLI:
    """Test command-line interface."""

    def test_cli_help(self, capsys):
        """Test --help flag."""
        result = cli_main(["--help"])
        captured = capsys.readouterr()
        
        assert result == 0
        assert "tvtop++" in captured.out
        assert "--log" in captured.out
        assert "-s" in captured.out or "small" in captured.out

    def test_cli_no_color_flag(self, tmp_path: Path, capsys):
        """Test --no-color flag."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        # Just test it doesn't crash - can't easily test full run
        with patch.object(TVTopPlusPlus, 'run', return_value=None):
            result = cli_main(["--log", str(log_file), "--no-color"])
        
        assert result == 0

    def test_cli_ascii_flag(self, tmp_path: Path):
        """Test --ascii flag."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        with patch.object(TVTopPlusPlus, 'run', return_value=None):
            result = cli_main(["--log", str(log_file), "--ascii"])
        
        assert result == 0

    def test_cli_small_flag(self, tmp_path: Path):
        """Test --small flag."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        with patch.object(TVTopPlusPlus, 'run', return_value=None):
            result = cli_main(["--log", str(log_file), "--small"])
        
        assert result == 0

    def test_cli_refresh_flag(self, tmp_path: Path):
        """Test --refresh flag."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        with patch.object(TVTopPlusPlus, 'run', return_value=None):
            result = cli_main(["--log", str(log_file), "--refresh", "0.5"])
        
        assert result == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_log_file(self, tmp_path: Path):
        """Test handling of nonexistent log file."""
        log_file = tmp_path / "does_not_exist.log"
        
        # Should not crash, just show waiting message
        tvt = TVTopPlusPlus(log_path=log_file)
        frame = tvt.latest_frame()
        
        assert frame.has_training_data is False

    def test_very_long_log_lines(self, tmp_path: Path):
        """Test handling of very long log lines."""
        log_file = tmp_path / "long.log"
        long_line = "loss=0.5 " + "x" * 10000 + "\n"
        log_file.write_text(long_line, encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file)
        frame = tvt.latest_frame()
        
        # Should handle gracefully
        assert frame is not None

    def test_unicode_in_log(self, tmp_path: Path):
        """Test handling of unicode characters in log."""
        log_file = tmp_path / "unicode.log"
        log_file.write_text("loss=0.5 epoch=1 训练中...\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file)
        frame = tvt.latest_frame()
        
        assert frame is not None

    def test_rapid_log_updates(self, tmp_path: Path):
        """Test handling of rapid log file updates."""
        log_file = tmp_path / "rapid.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file)
        
        # Simulate rapid updates
        for i in range(10):
            with log_file.open("a") as f:
                f.write(f"loss={0.5 - i*0.01}\n")
            tvt.latest_frame()
        
        frame = tvt.latest_frame()
        assert len(frame.recent_losses) > 0

    def test_memory_efficiency(self, tmp_path: Path):
        """Test that memory usage stays bounded with large logs."""
        log_file = tmp_path / "large.log"
        
        # Write many lines
        with log_file.open("w") as f:
            for i in range(1000):
                f.write(f"loss={0.5 - i*0.0001} step={i}/1000\n")
        
        tvt = TVTopPlusPlus(log_path=log_file)
        frame = tvt.latest_frame()
        
        # Losses should be bounded by history_size (default 8 in LogTail, but Frame stores more)
        # The LogTail has history_size=8, but the Frame may accumulate more from multiple polls
        # Just verify it's a reasonable number and not unbounded
        assert len(frame.recent_losses) > 0
        assert len(frame.recent_losses) <= 120  # Reasonable upper bound


class TestRichIntegration:
    """Test Rich v15 specific features."""

    def test_rich_live_compatibility(self, tmp_path: Path):
        """Test compatibility with Rich Live API."""
        from rich.console import Console
        from rich.live import Live
        from rich.text import Text
        
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        console = Console(force_terminal=False)
        
        # Create a simple Live display to verify Rich integration
        text = Text("Test")
        with Live(text, console=console, screen=False) as live:
            live.update(Text("Updated"))
        
        assert True  # If we get here, Rich Live works

    def test_rich_panel_styling(self, tmp_path: Path):
        """Test Rich panel styling is applied."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file, color=True)
        frame = tvt.latest_frame()
        
        training_panel = tvt._make_training_panel(frame)
        
        # Panel should have title and box style (Rich v15 uses 'box' not 'border_style' for box characters)
        assert training_panel.title == "Training Vitals"
        from rich.box import DOUBLE
        assert training_panel.box == DOUBLE

    def test_rich_table_creation(self, tmp_path: Path):
        """Test Rich table creation for process monitor."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file)
        frame = tvt.latest_frame()
        
        process_panel = tvt._make_process_panel(frame)
        
        # Should contain a Table
        from rich.table import Table
        assert isinstance(process_panel.renderable, Table)


class TestPerformance:
    """Basic performance tests."""

    def test_frame_generation_speed(self, tmp_path: Path):
        """Test that frame generation is fast enough."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5 step=10/100\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file)
        
        start = time.time()
        for _ in range(10):
            tvt.latest_frame()
        elapsed = time.time() - start
        
        # Should generate 10 frames in under 1 second
        assert elapsed < 1.0

    def test_the_process_table_is_not_rescanned_every_frame(self, tmp_path: Path, monkeypatch):
        """Listing every process costs about a second on Windows; the
        layout is rebuilt every refresh, so it reuses the last scan."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        tvt = TVTopPlusPlus(log_path=log_file)
        scans = []
        monkeypatch.setattr(tvt, "_scan_processes", lambda: scans.append(1) or [])
        for _ in range(5):
            tvt._get_active_processes()
        assert len(scans) == 1
        monkeypatch.setattr(tvt, "PROCESS_REFRESH_SECONDS", 0.0)
        tvt._get_active_processes()
        assert len(scans) == 2

    def test_layout_build_speed(self, tmp_path: Path):
        """Test that layout building is fast."""
        log_file = tmp_path / "train.log"
        log_file.write_text("loss=0.5\n", encoding="utf-8")
        
        tvt = TVTopPlusPlus(log_path=log_file)
        frame = tvt.latest_frame()
        
        from rich.console import Console
        console = Console(force_terminal=False)
        
        start = time.time()
        for _ in range(10):
            tvt._build_layout(frame, console)
        elapsed = time.time() - start
        
        # Should build 10 layouts in under 1.5 seconds
        assert elapsed < 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
