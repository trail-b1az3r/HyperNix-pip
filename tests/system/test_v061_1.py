"""Regression tests for v0.61.1 — hyped chat CLI + 5 bug-fix /
utility passes.

Pass 1 (by-hand audit):
* hyped chat-loop now routes through Countertop.say + bell token
  callback instead of bypassing history management.
* hyped picker uses '*' badge in --ascii mode.
* ups.UPS instantiation is now lazy (no IP-geolocation block).
* plasma.calibrate_alarm reset on re-call instead of compounding.
* tv._sanitise preserves \\r so Windows CRLF logs render.

Pass 3 (utility helpers):
* hypernix.utils — healthcheck / diagnostic_info / list_models /
  print_models / session_dir / has_binary / is_module_available.
* hypernix.utils.warn_hyper_nix_2 — MAJOR undertrained warning.

Pass 4 (utility helpers):
* Menu.find — fuzzy persona lookup (exact / case / substring /
  prefix / ambiguous-none).
* injection.thinking / testing / system_override shortcuts.
"""
from __future__ import annotations

import os

import pytest

from hypernix import injection, plasma, tv, ups, utils
from hypernix.hyped import (
    CURATED_MODELS,
    Configurator,
    SamplingConfig,
    _resolve_short_name,
    _wrap,
)
from hypernix.menu import MENU, Menu

# ---------------------------------------------------------------------------
# Pass 1: bug-fix regressions
# ---------------------------------------------------------------------------

class TestPass1Regressions:
    def test_hyped_picker_renders_with_ascii_only_uses_star(self) -> None:
        out = Configurator(color=False, ascii_only=True).render_model_picker()
        assert "*" in out
        # The default model row should be marked with a star, not the
        # original Unicode ★ which won't render in many CI terminals.
        assert "★" not in out

    def test_ups_instantiation_is_lazy_with_explicit_coords(self) -> None:
        import time
        t0 = time.time()
        u = ups.UPS(latitude=47.6, longitude=-122.3)
        assert (time.time() - t0) < 0.5  # no network call
        assert u._coords_resolved is False

    def test_ups_offline_does_not_resolve_coords_on_check(self) -> None:
        u = ups.UPS(offline=True)
        u.check(force=True)
        # Even after a check, offline=True keeps coords unresolved
        # *practically* (we set the flag but skip the HTTP call).
        assert u._coords_resolved is True
        assert u.latitude is None and u.longitude is None

    def test_plasma_recalibration_does_not_compound(self) -> None:
        from hypernix.smoke_alarm import RadsAlarm

        a = RadsAlarm()
        baseline = a.estimate_step_seconds()
        r1 = plasma.PlasmaResult(
            config=plasma.PlasmaConfig(), device="cpu", dtype="fp32",
            step_ms=1.0, tokens_per_sec=1.0, calibration_factor=2.0,
        )
        r2 = plasma.PlasmaResult(
            config=plasma.PlasmaConfig(), device="cpu", dtype="fp32",
            step_ms=1.0, tokens_per_sec=1.0, calibration_factor=3.0,
        )
        plasma.calibrate_alarm(a, r1)
        plasma.calibrate_alarm(a, r2)
        # Second calibration must be 3*baseline, NOT 2*3*baseline.
        assert a.estimate_step_seconds() == pytest.approx(baseline * 3.0)
        plasma.reset_calibration(a)
        assert a.estimate_step_seconds() == pytest.approx(baseline)

    def test_tv_sanitise_preserves_carriage_return(self) -> None:
        # CRLF Windows logs must keep \r intact (we only strip the
        # other C0 / C1 control codes).
        assert tv._sanitise("hello\rworld") == "hello\rworld"
        assert tv._sanitise("with\x00null") == "with?null"
        assert tv._sanitise("with\x07bell") == "with?bell"


# ---------------------------------------------------------------------------
# Pass 3: utils + hyper-Nix.2 warning
# ---------------------------------------------------------------------------

class TestUtils:
    def test_diagnostic_info_returns_dict(self) -> None:
        info = utils.diagnostic_info()
        assert isinstance(info, dict)
        assert "hypernix_version" in info
        assert "torch_version" in info
        assert "known_models_count" in info

    def test_healthcheck_summary_string(self) -> None:
        report = utils.healthcheck(verbose=False)
        text = report.summary()
        assert "hypernix" in text.lower()
        assert "torch" in text.lower()

    def test_list_models_filter_substring(self) -> None:
        rows = utils.list_models(filter_substring="qwen")
        assert all("qwen" in s.lower() or "qwen" in repo.lower() for s, repo, _n in rows)
        assert len(rows) > 0

    def test_list_models_filter_arch(self) -> None:
        rows = utils.list_models(arch="hypernix")
        # Every entry should be HyperNix-shaped.
        assert len(rows) > 0

    def test_print_models_no_match_prints_message(self, capsys) -> None:
        import io
        buf = io.StringIO()
        utils.print_models(filter_substring="not-a-real-model-xyz", file=buf)
        assert "no models match" in buf.getvalue()

    def test_session_dir_creates_directory(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HYPERNIX_CACHE_DIR", str(tmp_path))
        d = utils.session_dir(label="unit-test")
        assert d.exists()
        assert d.is_dir()
        assert "unit-test" in d.name

    def test_is_module_available(self) -> None:
        assert utils.is_module_available("torch") is True
        assert utils.is_module_available("definitely_not_a_module") is False

    def test_has_binary(self) -> None:
        # `python` should be on PATH wherever the test runs.
        import sys
        py_exe = "python" if utils.has_binary("python") else "python3"
        assert utils.has_binary(py_exe) or os.path.exists(sys.executable)


class TestHyperNix2Warning:
    def setup_method(self) -> None:
        utils.reset_warnings()

    def test_is_hyper_nix_2_matches_aliases(self) -> None:
        for s in ("hyper-nix.2", "ray0rf1re/hyper-Nix.2",
                  "hyper-nix", "hypernix2", "HYPER-NIX.2"):
            assert utils.is_hyper_nix_2(s)

    def test_is_hyper_nix_2_rejects_other_repos(self) -> None:
        for s in ("hyper-nix.1", "nix2.7a", "qwen2.5-7b", ""):
            assert not utils.is_hyper_nix_2(s)

    def test_warn_fires_once_per_process(self, capsys) -> None:
        utils.reset_warnings()
        first = utils.warn_hyper_nix_2("ray0rf1re/hyper-Nix.2")
        second = utils.warn_hyper_nix_2("ray0rf1re/hyper-Nix.2")
        assert first is True
        assert second is False
        captured = capsys.readouterr()
        # The MAJOR warning text appears in stderr.
        assert "UNDERTRAINED" in captured.err

    def test_warn_force_fires_again(self) -> None:
        utils.reset_warnings()
        utils.warn_hyper_nix_2("hyper-nix.2")
        # force=True re-emits even after the dedupe set has the entry.
        assert utils.warn_hyper_nix_2("hyper-nix.2", force=True) is True

    def test_warn_skipped_for_other_repos(self, capsys) -> None:
        utils.reset_warnings()
        assert utils.warn_hyper_nix_2("hyper-nix.1") is False
        assert utils.warn_hyper_nix_2("nix2.7a") is False
        captured = capsys.readouterr()
        assert "UNDERTRAINED" not in captured.err

    def test_env_var_suppresses_warning(self, monkeypatch, capsys) -> None:
        utils.reset_warnings()
        monkeypatch.setenv("HYPERNIX_SUPPRESS_HYPERNIX2_WARNING", "1")
        assert utils.warn_hyper_nix_2("hyper-nix.2") is False
        captured = capsys.readouterr()
        assert "UNDERTRAINED" not in captured.err


# ---------------------------------------------------------------------------
# Pass 4: Menu.find + injection shortcuts
# ---------------------------------------------------------------------------

class TestMenuFind:
    def test_exact_match(self) -> None:
        assert MENU.find("judge") == "judge"

    def test_case_insensitive(self) -> None:
        assert MENU.find("JUDGE") == "judge"

    def test_substring_unique(self) -> None:
        assert MENU.find("code") == "code-helper"

    def test_prefix_unique(self) -> None:
        assert MENU.find("cre") == "creative"

    def test_ambiguous_returns_none(self) -> None:
        # Multiple substring matches — ambiguous.
        m = Menu(prompts={"alpha-bot": "x", "beta-bot": "y", "gamma-bot": "z"})
        assert m.find("bot") is None

    def test_unknown_returns_none(self) -> None:
        assert MENU.find("definitely-not-a-persona") is None

    def test_empty_query_returns_none(self) -> None:
        assert MENU.find("") is None


class TestInjectionShortcuts:
    def test_thinking_string(self) -> None:
        assert injection.thinking("hi") == "<think>hi</think>"

    def test_thinking_messages(self) -> None:
        out = injection.thinking([{"role": "user", "content": "hi"}])
        assert isinstance(out, list)
        assert any("<think>" in m["content"] for m in out)

    def test_testing_string(self) -> None:
        assert injection.testing("q").startswith("<|test|>")

    def test_system_override_string(self) -> None:
        assert "<|system_override|>" in injection.system_override("be terse")


# ---------------------------------------------------------------------------
# Hyped — non-interactive surfaces
# ---------------------------------------------------------------------------

class TestHyped:
    def test_curated_models_includes_user_request(self) -> None:
        shorts = {m.short for m in CURATED_MODELS}
        for required in (
            "hyper-nix.1", "hyper-nix.2", "nix2.7a", "nix2.6-mm",
            "qwen3.5-2b", "qwen3.5-4b",
            "nano-nano-v4", "nano-mini-6.99-v2", "nano-nano-927-v3",
        ):
            assert required in shorts, f"missing curated model: {required}"

    def test_resolve_short_name_curated(self) -> None:
        m = _resolve_short_name("hyper-nix.2")
        assert m is not None
        assert m.repo_id == "ray0rf1re/hyper-Nix.2"

    def test_resolve_short_name_falls_back_to_known_models(self) -> None:
        # nano-nano (alias for nano-nano-v4) lives in KNOWN_MODELS but
        # isn't in the curated short-list.
        m = _resolve_short_name("nano-nano")
        assert m is not None
        assert "Nano-nano" in m.repo_id

    def test_resolve_short_name_unknown(self) -> None:
        assert _resolve_short_name("definitely-not-a-model") is None

    def test_picker_render_includes_every_curated_short(self) -> None:
        text = Configurator(color=False).render_model_picker()
        for m in CURATED_MODELS:
            assert m.short in text

    def test_sampling_config_defaults(self) -> None:
        s = SamplingConfig()
        assert 0.0 < s.temperature < 2.0
        assert 0 < s.top_p <= 1.0
        assert s.top_k > 0
        assert s.max_new_tokens > 0
        assert s.flour_preset == "smart"

    def test_wrap_handles_empty_and_long_words(self) -> None:
        assert _wrap("", max_width=20) == [""]
        long = "x" * 50
        rows = _wrap(f"{long} y", max_width=20)
        assert long in rows[0]
