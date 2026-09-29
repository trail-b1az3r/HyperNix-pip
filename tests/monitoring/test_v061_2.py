"""Regression tests for v0.61.2 — tvtop btop-style multi-panel
rewrite.

Adds rich CPU panel (per-core grid + history graph), Memory panel
(USED / CACHE / FREE / SWAP breakdown bars + history), GPU panel
(util / VRAM / temp / power gauges + history).  Replaces the old
single ``hardware`` panel.
"""
from __future__ import annotations

from pathlib import Path

from hypernix import tv


class TestNewProbes:
    def test_memory_breakdown_returns_dict_or_none(self) -> None:
        out = tv._read_memory_breakdown()
        # Either a dict with at least total_mib, or None on hosts
        # without psutil and without /proc/meminfo.
        if out is not None:
            assert "total_mib" in out
            assert out["total_mib"] >= 0

    def test_per_core_returns_list_or_none(self) -> None:
        out = tv._safe_psutil_per_core()
        if out is not None:
            assert isinstance(out, list)
            assert all(isinstance(v, float) for v in out)

    def test_proc_stat_per_core_needs_two_calls(self) -> None:
        # First call returns None (no delta), second returns a list
        # whose length matches /proc/stat's per-core entries — at
        # least 1 on every Linux host.  Skip the assert on non-Linux.
        # Reset the function-attribute prev-sample memo so we get a
        # genuine "first call" regardless of prior test order.
        if hasattr(tv._read_proc_stat_per_core, "_prev"):
            del tv._read_proc_stat_per_core._prev  # type: ignore[attr-defined]
        first = tv._read_proc_stat_per_core()
        second = tv._read_proc_stat_per_core()
        if second is not None:
            assert isinstance(second, list)
            assert len(second) >= 1
            assert first is None  # delta math requires a prior sample


class TestRichRender:
    def test_render_includes_per_core_grid_when_available(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text("step 1/10 loss=2.0\n", encoding="utf-8")
        tvt = tv.TVTop(log_path=log, color=False, width=120)
        # Two frames so the /proc/stat per-core sampler has a delta.
        tvt.latest_frame()
        out = tvt.render(tvt.latest_frame())
        # On any host with at least 1 CPU we expect the per-core
        # label "c 0" to appear in the CPU panel.
        if any(line.startswith("c") for line in out.splitlines()):
            assert "c 0" in out or "c 1" in out

    def test_render_memory_panel_shows_breakdown_or_fallback(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text("step 1/10 loss=2.0\n", encoding="utf-8")
        tvt = tv.TVTop(log_path=log, color=False, width=120)
        out = tvt.render(tvt.latest_frame())
        # Either USED/FREE bars from the breakdown, or the legacy
        # `RAM N%` fallback, or the no-data placeholder.
        assert ("USED" in out) or ("RAM " in out) or ("no memory data" in out)

    def test_render_gpu_panel_handles_no_gpu(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text("step 1/10 loss=2.0\n", encoding="utf-8")
        tvt = tv.TVTop(log_path=log, color=False, width=120)
        out = tvt.render(tvt.latest_frame())
        # On a host without nvidia-smi the GPU panel renders the
        # "(no GPU detected ...)" placeholder.  When nvidia-smi is
        # available, "UTIL" is in the panel.
        assert ("no GPU detected" in out) or ("UTIL" in out)

    def test_render_footer_shows_core_count_and_gpu_label(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text("step 1/10 loss=2.0\n", encoding="utf-8")
        tvt = tv.TVTop(log_path=log, color=False, width=120)
        out = tvt.render(tvt.latest_frame())
        assert "cores" in out
        assert "gpu=" in out


class TestHistoryDeques:
    def test_cpu_history_grows_per_frame(self, tmp_path: Path, monkeypatch) -> None:
        # Force a non-None CPU sample so the deque grows.
        monkeypatch.setattr(tv, "_safe_psutil_percent", lambda: (42.0, 50.0))
        log = tmp_path / "train.log"
        log.write_text("step 1/10 loss=2.0\n", encoding="utf-8")
        tvt = tv.TVTop(log_path=log, color=False, width=120)
        tvt.latest_frame()
        tvt.latest_frame()
        tvt.latest_frame()
        assert len(tvt._cpu_history) >= 3
        assert all(v == 42.0 for v in tvt._cpu_history)

    def test_history_capped_at_120(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(tv, "_safe_psutil_percent", lambda: (10.0, 20.0))
        log = tmp_path / "train.log"
        log.write_text("step 1/10 loss=2.0\n", encoding="utf-8")
        tvt = tv.TVTop(log_path=log, color=False, width=120)
        for _ in range(150):
            tvt.latest_frame()
        assert len(tvt._cpu_history) == 120
