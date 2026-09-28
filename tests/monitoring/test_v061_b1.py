"""Tests for v0.61.0b1 — three new modules + tvtop visual rewrite.

* ups       — weather + scheduled-outage panic mode
* injection — thinking / testing / system-override / custom token splicers
* plasma    — quick GPU benchmark for ETA calibration
* tv        — multi-row loss graph + log sanitisation + auto-detect
              filtering (these supplement the existing TestTV cases
              in tests/system/test_v060.py).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hypernix import injection, plasma, tv, ups

# ---------------------------------------------------------------------------
# UPS
# ---------------------------------------------------------------------------

class TestUPS:
    def test_offline_check_returns_inactive_status(self) -> None:
        u = ups.UPS(offline=True)
        s = u.check()
        assert s.active is False
        assert s.panic is False

    def test_panic_runs_snapshot_fn_exactly_once(self) -> None:
        calls: list[int] = []

        def fake_outage(_addr: str | None) -> bool:
            return True

        u = ups.UPS(
            offline=True,
            outage_check_fn=fake_outage,
            snapshot_fn=lambda: calls.append(1),
        )
        u.check(force=True)
        u.check(force=True)
        u.check(force=True)
        # Snapshot fires once on the panic *transition*, not every poll.
        assert calls == [1]

    def test_panic_triples_save_cadence(self) -> None:
        u = ups.UPS(
            offline=True,
            outage_check_fn=lambda _a: True,
            cadence_multiplier=3,
        )
        u.check(force=True)
        assert u.adjusted_save_every(900) == 300

    def test_no_panic_keeps_save_cadence(self) -> None:
        u = ups.UPS(offline=True)
        u.check(force=True)
        assert u.adjusted_save_every(900) == 900

    def test_severe_weather_codes_set_is_correct(self) -> None:
        # Heavy rain (65), thunderstorm (95), thunderstorm + heavy hail (99).
        for c in (65, 95, 99):
            assert c in ups.SEVERE_WEATHER_CODES

    def test_summary_describes_state(self) -> None:
        u = ups.UPS(offline=True, outage_check_fn=lambda _a: True)
        u.check(force=True)
        assert "SCHEDULED OUTAGE" in u.last_status.summary

    def test_history_records_active_threats(self) -> None:
        u = ups.UPS(offline=True, outage_check_fn=lambda _a: True)
        u.check(force=True)
        assert len(u.history) == 1
        assert u.history[0].scheduled_outage is True

    def test_cadence_multiplier_must_be_at_least_1(self) -> None:
        with pytest.raises(ValueError):
            ups.UPS(offline=True, cadence_multiplier=0)


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

class TestInjection:
    def test_thinking_wraps_text(self) -> None:
        out = injection.ThinkingInjector().inject_text("2+2?")
        assert out == "<think>2+2?</think>"

    def test_thinking_injects_into_messages(self) -> None:
        msgs = injection.ThinkingInjector().inject_messages(
            [{"role": "user", "content": "hi"}],
        )
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "system"]
        assert "<think>" in msgs[0]["content"]
        assert "</think>" in msgs[-1]["content"]

    def test_testing_prefix_only(self) -> None:
        out = injection.TestingInjector().inject_text("question?")
        assert out.startswith("<|test|>")
        assert "<|/test|>" not in out  # mode='prefix' ignores close

    def test_system_override_appends_at_end(self) -> None:
        msgs = injection.SystemOverrideInjector().inject_messages(
            [{"role": "user", "content": "hi"}],
        )
        assert msgs[-1]["role"] == "system"
        assert "<|system_override|>" in msgs[-1]["content"]

    def test_custom_injector(self) -> None:
        inj = injection.CustomInjector(open="<|tool|>", close="<|/tool|>", mode="wrap")
        assert inj.inject_text("x") == "<|tool|>x<|/tool|>"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            injection.Injector(mode="weird")

    def test_factory_dispatch(self) -> None:
        assert isinstance(injection.injector("thinking"), injection.ThinkingInjector)
        assert isinstance(injection.injector("testing"), injection.TestingInjector)
        with pytest.raises(ValueError):
            injection.injector("does-not-exist")

    def test_one_shot_inject_helper(self) -> None:
        out = injection.inject(
            [{"role": "user", "content": "hi"}],
            kind="thinking",
        )
        assert isinstance(out, list)
        assert len(out) == 3

    def test_one_shot_inject_string_form(self) -> None:
        assert injection.inject("hi", kind="thinking") == "<think>hi</think>"

    def test_history_records_each_event(self) -> None:
        inj = injection.ThinkingInjector()
        inj.inject_text("a")
        inj.inject_text("b")
        assert len(inj.history) == 2


# ---------------------------------------------------------------------------
# Plasma
# ---------------------------------------------------------------------------

class TestPlasma:
    @pytest.fixture
    def tiny_cfg(self) -> plasma.PlasmaConfig:
        return plasma.PlasmaConfig(
            hidden_size=32, num_layers=1, seq_length=8,
            batch_size=2, vocab_size=64,
            warmup_steps=1, measure_steps=2,
        )

    def test_quick_benchmark_returns_result(self, tiny_cfg) -> None:
        r = plasma.quick_benchmark(tiny_cfg, device="cpu")
        assert isinstance(r, plasma.PlasmaResult)
        assert r.step_ms > 0
        assert r.tokens_per_sec > 0
        assert len(r.samples_ms) == tiny_cfg.measure_steps

    def test_calibration_factor_is_positive(self, tiny_cfg) -> None:
        r = plasma.quick_benchmark(tiny_cfg, device="cpu")
        assert r.calibration_factor > 0

    def test_summary_mentions_step_and_throughput(self, tiny_cfg) -> None:
        r = plasma.quick_benchmark(tiny_cfg, device="cpu")
        s = r.summary()
        assert "step=" in s and "throughput=" in s

    def test_calibrate_alarm_scales_estimate(self, tiny_cfg) -> None:
        from hypernix.smoke_alarm import RadsAlarm

        alarm = RadsAlarm()
        original = alarm.estimate_step_seconds()
        result = plasma.PlasmaResult(
            config=tiny_cfg, device="cpu", dtype="float32",
            step_ms=100.0, tokens_per_sec=1.0,
            calibration_factor=2.5,
        )
        plasma.calibrate_alarm(alarm, result)
        scaled = alarm.estimate_step_seconds()
        assert scaled == pytest.approx(original * 2.5)

    def test_calibrate_alarm_rejects_object_without_method(self) -> None:
        with pytest.raises(TypeError):
            plasma.calibrate_alarm(object(), plasma.PlasmaResult(
                config=plasma.PlasmaConfig(), device="cpu", dtype="float32",
                step_ms=1.0, tokens_per_sec=1.0,
            ))

    def test_plasma_alias_matches_quick_benchmark(self, tiny_cfg) -> None:
        # Just verify the alias is wired; results vary run to run so
        # we only check the returned type.
        assert isinstance(plasma.plasma(tiny_cfg, device="cpu"), plasma.PlasmaResult)


# ---------------------------------------------------------------------------
# tv visual upgrade
# ---------------------------------------------------------------------------

class TestTVVisualUpgrade:
    def test_multi_row_graph_returns_height_rows(self) -> None:
        rows = tv.multi_row_graph([1, 2, 3, 4, 5], width=10, height=4)
        assert len(rows) == 4
        assert all(len(r) == 10 for r in rows)

    def test_multi_row_graph_handles_empty(self) -> None:
        rows = tv.multi_row_graph([], width=10, height=3)
        assert len(rows) == 3

    def test_multi_row_graph_constant_values(self) -> None:
        rows = tv.multi_row_graph([5.0] * 8, width=8, height=3)
        assert len(rows) == 3
        assert all(len(r) == 8 for r in rows)

    def test_log_tail_sanitises_binary_garbage(self, tmp_path: Path) -> None:
        log = tmp_path / "junk.log"
        # Mix valid log lines with binary noise.
        log.write_bytes(b"step 1/10 loss=2.0\n\x00\x01\x02\x03 binary\nstep 2/10 loss=1.9\n")
        tail = tv.LogTail(log)
        tail.poll()
        joined = "\n".join(tail.tail)
        # Non-printable chars get replaced with '?'.
        assert "\x00" not in joined
        assert "\x01" not in joined
        assert "step 1/10" in joined

    def test_autodetect_skips_non_training_log(self, tmp_path: Path) -> None:
        # Non-training log (browser-style JSON garbage).
        (tmp_path / "konsole.log").write_text(
            '{"id":"abc","type":"library-playlists"}\n' * 5,
            encoding="utf-8",
        )
        # Real training log written more recently.
        (tmp_path / "train-real.log").write_text(
            "step 100/2000 loss=2.345 lr=3e-4\n", encoding="utf-8",
        )
        chosen = tv._autodetect_log(tmp_path)
        assert chosen is not None
        assert chosen.name == "train-real.log"

    def test_looks_like_training_log_detects_pattern(self, tmp_path: Path) -> None:
        good = tmp_path / "good.log"
        good.write_text("step 5/10 loss=0.5\n", encoding="utf-8")
        bad = tmp_path / "bad.log"
        bad.write_text("hello world\nrandom text\n", encoding="utf-8")
        assert tv._looks_like_training_log(good) is True
        assert tv._looks_like_training_log(bad) is False

    def test_render_uses_panel_frames(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text("step 50/100 loss=0.5\n", encoding="utf-8")
        tvt = tv.TVTop(log_path=log, color=False, width=120)
        out = tvt.render(tvt.latest_frame())
        # Rounded panel corners.
        assert "╭" in out and "╯" in out
        # 0.61.2 split the old "hardware" panel into cpu / memory / gpu.
        for title in ("cpu", "memory", "gpu", "training", "loss curve", "recent log"):
            assert title in out, f"missing panel title: {title}"

    def test_render_empty_state(self, tmp_path: Path) -> None:
        log = tmp_path / "train.log"
        log.write_text("not a training log\n", encoding="utf-8")
        tvt = tv.TVTop(log_path=log, color=False, width=100)
        out = tvt.render(tvt.latest_frame())
        # Empty-state header instead of fake "step 0".
        assert "waiting" in out.lower()
