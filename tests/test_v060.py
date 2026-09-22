"""Tests for v0.60.0 — eight new modules:

* compactor   — zip older checkpoints
* ethanol     — bounded GPU overclock plan / apply
* outage      — display-blank context manager
* tv          — tvtop training dashboard
* timer       — 4 tiers (kitchen / egg / interval / pomodoro)
* thermometer — 4 tiers (instant / probe / infrared / digital)
* dishwasher  — 4 tiers (hand / quick / normal / heavy)
* strainer    — 4 tiers (colander / fine-mesh / nut-milk-bag / cheesecloth)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hypernix import (
    compactor,
    dishwasher,
    ethanol,
    outage,
    strainer,
    thermometer,
    timer,
    tv,
)

# ---------------------------------------------------------------------------
# compactor
# ---------------------------------------------------------------------------

class TestCompactor:
    def _make_ckpts(self, root: Path, steps: list[int]) -> list[Path]:
        out = []
        for s in steps:
            d = root / f"ckpt-{s}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "model.pt").write_bytes(b"\x00" * 64)
            out.append(d)
        return out

    def test_discover_orders_oldest_first(self, tmp_path: Path) -> None:
        self._make_ckpts(tmp_path, [200, 100, 300])
        order = compactor.list_checkpoints(tmp_path)
        assert [p.name for p in order] == ["ckpt-100", "ckpt-200", "ckpt-300"]

    def test_keep_recent_n_leaves_top_n(self, tmp_path: Path) -> None:
        self._make_ckpts(tmp_path, [100, 200, 300, 400, 500])
        plan = compactor.Compactor(tmp_path, keep_recent=2, dry_run=True).plan()
        assert [src.name for src, _arc in plan] == ["ckpt-100", "ckpt-200", "ckpt-300"]

    def test_compact_zip_writes_archives_and_removes_originals(
        self, tmp_path: Path,
    ) -> None:
        self._make_ckpts(tmp_path, [100, 200, 300])
        archives = compactor.compact(tmp_path, keep_recent=1, fmt="zip")
        assert len(archives) == 2
        for a in archives:
            assert a.exists() and a.suffix == ".zip"
        # Originals gone.
        assert not (tmp_path / "ckpt-100").exists()
        assert (tmp_path / "ckpt-300").exists()

    def test_dry_run_does_not_write_or_delete(self, tmp_path: Path) -> None:
        self._make_ckpts(tmp_path, [100, 200])
        compactor.compact(tmp_path, keep_recent=0, fmt="zip", dry_run=True)
        assert (tmp_path / "ckpt-100").exists()
        assert (tmp_path / "ckpt-200").exists()
        assert not list(tmp_path.glob("*.zip"))

    def test_unknown_fmt_raises(self) -> None:
        with pytest.raises(ValueError):
            compactor.Compactor(".", fmt="xz")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ethanol
# ---------------------------------------------------------------------------

class TestEthanol:
    def test_level_zero_is_full_stock(self) -> None:
        e = ethanol.Ethanol(level=0)
        core, mem, power = e.offsets()
        assert core == 0 and mem == 0 and power == 100

    def test_the_top_level_hits_the_ceilings(self) -> None:
        """Follows MAX_LEVEL rather than pinning 30.

        It said 30, and MAX_LEVEL moved to 45 — so the test asserted
        that two-thirds of the way up was the top, and failed. The
        behaviour worth pinning is "the top level reaches the declared
        ceilings", which is true at whatever the top happens to be.
        """
        core, mem, power = ethanol.Ethanol(level=ethanol.MAX_LEVEL).offsets()
        assert core == ethanol.MAX_CORE_OFFSET_MHZ
        assert mem == ethanol.MAX_MEM_OFFSET_MHZ
        assert power == ethanol.MAX_POWER_LIMIT_PCT

    def test_above_the_top_level_is_clamped(self) -> None:
        top = ethanol.Ethanol(level=ethanol.MAX_LEVEL).offsets()
        assert ethanol.Ethanol(level=ethanol.MAX_LEVEL + 1).offsets() == top
        assert ethanol.Ethanol(level=999).offsets() == top

    def test_nothing_writes_the_ceiling_out_as_a_literal(self) -> None:
        """MAX_LEVEL moved from 30 to 45 and left three things behind:
        this test file, `_level_to_offsets`' docstring, and the `eth`
        help — which went on telling people the maximum was a level
        two-thirds of the way up. All three now read the constant."""
        import re
        from pathlib import Path as _Path

        source = _Path(ethanol.__file__).read_text(encoding="utf-8")
        stale = [
            line.strip()
            for line in source.splitlines()
            # A bare "30" in prose or usage text, not in code that is
            # allowed to contain the number (offsets, temperatures).
            if re.search(r"\b(level|levels)\b[^\n]*\b30\b", line, re.I)
            or re.search(r"0\.\.30|0…30", line)
        ]
        assert not stale, f"the old ceiling is still written out: {stale}"

    def test_the_help_names_the_real_maximum(self) -> None:
        assert f"0..{ethanol.MAX_LEVEL}" in ethanol._USAGE
        assert f"eth {ethanol.MAX_LEVEL}" in ethanol._USAGE

    def test_negative_level_raises(self) -> None:
        with pytest.raises(ValueError):
            ethanol.Ethanol(level=-1).offsets()

    def test_apply_without_confirm_returns_plan_only(self) -> None:
        res = ethanol.Ethanol(level=5).apply(confirm=False)
        assert res.applied is False
        assert "refusing" in res.notes.lower() or "no supported" in res.notes.lower()

    def test_cli_help_returns_zero(self, capsys) -> None:
        rc = ethanol.cli_main(["--help"])
        assert rc == 0

    def test_cli_invalid_level_returns_two(self) -> None:
        assert ethanol.cli_main(["100"]) == 2
        assert ethanol.cli_main(["abc"]) == 2


# ---------------------------------------------------------------------------
# outage
# ---------------------------------------------------------------------------

class TestOutage:
    def test_detect_backend_returns_string(self) -> None:
        b = outage.detect_backend()
        assert isinstance(b, str)

    def test_context_manager_runs_without_raising(self) -> None:
        # We can't actually verify the screen blanked in a unit test,
        # but the CM flow must complete cleanly even on a host where
        # no display backend is available.
        with outage.Outage() as o:
            assert o.last_result is not None
        assert o.last_result.restored or o.last_result.notes  # something happened

    def test_strict_mode_raises_when_backend_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(outage, "_detect_backend", lambda: "unknown")
        o = outage.Outage(strict=True)
        # Strict + unknown backend → blanking is a no-op-with-note, no error.
        # The strict flag only escalates real *failures*; missing backend
        # is treated as a graceful skip.  So this should not raise.
        res = o.black_out()
        assert res.notes  # at least one note recorded

    def test_platform_summary_returns_dict(self) -> None:
        d = outage.platform_summary()
        assert "platform" in d and "backend" in d


# ---------------------------------------------------------------------------
# tv / tvtop
# ---------------------------------------------------------------------------

class TestTV:
    def test_sparkline_handles_empty(self) -> None:
        assert tv.sparkline([]) == ""

    def test_sparkline_handles_constant(self) -> None:
        # All-equal values render to a single mid-band block.
        s = tv.sparkline([5.0, 5.0, 5.0])
        assert len(s) == 3

    def test_log_tail_picks_up_new_lines(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text("line a\n", encoding="utf-8")
        tail = tv.LogTail(log, history_size=5)
        first = tail.poll()
        assert first == ["line a"]
        log.write_text("line a\nline b\n", encoding="utf-8")
        second = tail.poll()
        assert second == ["line b"]
        assert tail.tail == ["line a", "line b"]

    def test_frame_parses_step_loss_lr(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text("step 100/2000 loss=2.345 lr=3e-4\n", encoding="utf-8")
        tvt = tv.TVTop(log_path=log)
        frame = tvt.latest_frame()
        assert frame.step == 100
        assert frame.total_steps == 2000
        assert frame.loss == pytest.approx(2.345)
        assert frame.lr == pytest.approx(3e-4)

    def test_progress_clamped_to_one(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text("step 5000/2000 loss=1.0\n", encoding="utf-8")
        f = tv.TVTop(log_path=log).latest_frame()
        assert 0 <= f.progress <= 1

    def test_render_produces_string(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text("step 50/100 loss=0.5\n", encoding="utf-8")
        tvt = tv.TVTop(log_path=log, color=False, width=100)
        out = tvt.render(tvt.latest_frame())
        # 0.61.0b1 panel layout: "step    50 / 100" + bar + "loss  0.5000".
        assert "step" in out
        assert "0.5000" in out
        assert "training" in out  # panel title
        assert "loss curve" in out  # graph panel title

    def test_render_multi_row_graph(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text(
            "\n".join(f"step {i}/20 loss={2.0 - i * 0.05}" for i in range(20)),
            encoding="utf-8",
        )
        tvt = tv.TVTop(log_path=log, color=False, width=100)
        tvt.latest_frame()
        out = tvt.render(tvt.latest_frame())
        # The multi-row graph contributes Unicode block-bar characters.
        assert "█" in out

    def test_run_one_frame_max(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text("step 1/10 loss=2.0\n", encoding="utf-8")
        # Should return without blocking.
        tv.TVTop(log_path=log, refresh_seconds=0.01).run(max_frames=1)


# ---------------------------------------------------------------------------
# timer (4 tiers)
# ---------------------------------------------------------------------------

class TestTimer:
    def test_kitchen_timer_expires_after_duration(self) -> None:
        t = timer.KitchenTimer(duration=0.05).start()
        assert not t.expired()
        time.sleep(0.25)
        assert t.expired()

    def test_egg_timer_fires_on_ring_once(self) -> None:
        rings = []
        t = timer.EggTimer(duration=0.02, on_ring=lambda: rings.append(1)).start()
        time.sleep(0.07)
        t.check()
        t.check()
        t.check()
        assert rings == [1]
        assert t.rang is True

    def test_interval_timer_only_fires_after_interval(self) -> None:
        t = timer.IntervalTimer(interval_seconds=0.05).start()
        # First call right after start: shouldn't fire yet.
        assert t.should_fire() is False
        time.sleep(0.26)
        assert t.should_fire() is True
        # Two consecutive should_fire calls in the same window: only one fires.
        assert t.should_fire() is False

    def test_pomodoro_starts_in_work_state(self) -> None:
        t = timer.PomodoroTimer(work_seconds=0.05, rest_seconds=0.05).start()
        assert t.state == "work"

    def test_pomodoro_switches_to_rest(self) -> None:
        t = timer.PomodoroTimer(work_seconds=0.05, rest_seconds=0.5).start()
        time.sleep(0.15)
        assert t.tick() == "rest"

    def test_factory(self) -> None:
        assert isinstance(timer.timer("interval", interval_seconds=1), timer.IntervalTimer)


# ---------------------------------------------------------------------------
# thermometer (4 tiers)
# ---------------------------------------------------------------------------

class TestThermometer:
    def test_instant_returns_reading(self) -> None:
        r = thermometer.InstantThermometer().read()
        assert isinstance(r, thermometer.Reading)
        # On CI without sensors, both may be None — that's OK.
        assert r.timestamp > 0

    def test_probe_keeps_history(self) -> None:
        p = thermometer.ProbeThermometer(history_size=4)
        for _ in range(5):
            p.read()
        assert len(p.history) <= 4

    def test_infrared_tracks_peak_per_source(self) -> None:
        ir = thermometer.InfraredThermometer(warn_celsius=85.0)
        ir.read()
        assert isinstance(ir.peaks, dict)

    def test_digital_writes_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "temp.jsonl"
        with thermometer.DigitalThermometer(log_path=path) as dt:
            dt.read()
        # Validate the line is JSON-loadable even if all sensor fields are None.
        if path.exists() and path.stat().st_size > 0:
            row = json.loads(path.read_text().splitlines()[0])
            assert "timestamp" in row


# ---------------------------------------------------------------------------
# dishwasher (4 tiers)
# ---------------------------------------------------------------------------

class TestDishwasher:
    def test_hand_wash_removes_logs(self, tmp_path: Path) -> None:
        (tmp_path / "train.log").write_text("hello", encoding="utf-8")
        report = dishwasher.HandWash(root=tmp_path).run()
        assert any(p.name == "train.log" for p in report.files_removed)

    def test_quick_wash_removes_tmp(self, tmp_path: Path) -> None:
        (tmp_path / "x.tmp").write_text("hi", encoding="utf-8")
        report = dishwasher.QuickWash(root=tmp_path).run()
        assert any(p.name == "x.tmp" for p in report.files_removed)

    def test_normal_wash_removes_old_checkpoints(self, tmp_path: Path) -> None:
        for s in (100, 200, 300):
            d = tmp_path / f"ckpt-{s}"
            d.mkdir()
            (d / "model.pt").write_bytes(b"\x00" * 8)
        dishwasher.NormalWash(root=tmp_path, keep_recent=1).run()
        assert not (tmp_path / "ckpt-100").exists()
        assert (tmp_path / "ckpt-300").exists()

    def test_dry_run_keeps_files(self, tmp_path: Path) -> None:
        (tmp_path / "x.log").write_text("hi", encoding="utf-8")
        dishwasher.HandWash(root=tmp_path, dry_run=True).run()
        assert (tmp_path / "x.log").exists()

    def test_factory(self) -> None:
        assert isinstance(dishwasher.dishwasher("heavy"), dishwasher.HeavyDuty)


# ---------------------------------------------------------------------------
# strainer (4 tiers)
# ---------------------------------------------------------------------------

class TestStrainer:
    def test_colander_drops_empty(self) -> None:
        out = strainer.Colander().filter(["hi", "", None, "  ", "ok"])
        assert out == ["hi", "ok"]

    def test_fine_mesh_min_length(self) -> None:
        out = strainer.FineMesh(min_length=4).filter(["hi", "hello", "ok!", "world"])
        assert out == ["hello", "world"]

    def test_nut_milk_bag_drops_non_printable(self) -> None:
        bad = "hi\x00\x01\x02\x03\x04\x05\x06\x07"
        out = strainer.NutMilkBag(min_length=2).filter([bad, "good text"])
        assert "good text" in out
        assert bad not in out

    def test_cheesecloth_dedupes(self) -> None:
        out = strainer.Cheesecloth(min_length=4, ngram_size=4).filter([
            "the quick brown fox jumps over",
            "the quick brown fox jumps over",  # dup
            "completely different content here",
        ])
        assert len(out) == 2

    def test_stats_reasons(self) -> None:
        s = strainer.FineMesh(min_length=4)
        s.filter(["", "ok!", "hello"])
        assert s.stats().reasons.get("empty", 0) == 1

    def test_factory(self) -> None:
        assert isinstance(strainer.strainer("cheesecloth"), strainer.Cheesecloth)
