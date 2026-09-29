"""ethanol — GPU overclocking helper.

The module shipped with no tests at all, which is how `eth auto` came to
advertise "auto-pick safe level based on temperature" while being the
literal constant 15, and how level 0 came to set the *highest* clock
index on ROCm while being documented as "reset to stock".

Nothing here touches a real GPU: the vendor tools are stubbed at the
`_run`/`shutil.which` boundary, which is exactly where the module's own
seam is.
"""
from __future__ import annotations

import subprocess

import pytest

from hypernix.system import ethanol as eth


def _completed(stdout: str = "", returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def commands(monkeypatch) -> list[list[str]]:
    """Record every vendor command instead of running it."""
    recorded: list[list[str]] = []

    def fake_run(cmd):
        recorded.append(list(cmd))
        joined = " ".join(cmd)
        if "temperature.gpu" in joined:
            return _completed("60")
        if "power.default_limit" in joined:
            return _completed("320")
        return _completed("ok")

    monkeypatch.setattr(eth, "_run", fake_run)
    return recorded


class TestLevelMapping:
    def test_level_zero_is_stock(self):
        assert eth._level_to_offsets(0) == (0, 0, 100)

    def test_max_level_hits_the_ceilings(self):
        core, mem, power = eth._level_to_offsets(eth.MAX_LEVEL)
        assert core == eth.MAX_CORE_OFFSET_MHZ
        assert mem == eth.MAX_MEM_OFFSET_MHZ
        assert power == eth.MAX_POWER_LIMIT_PCT

    def test_levels_above_max_clamp_rather_than_raise(self):
        assert eth._level_to_offsets(999) == eth._level_to_offsets(eth.MAX_LEVEL)

    def test_negative_is_rejected(self):
        with pytest.raises(ValueError):
            eth._level_to_offsets(-1)

    def test_offsets_increase_monotonically(self):
        previous = (-1, -1, -1)
        for level in range(eth.MAX_LEVEL + 1):
            current = eth._level_to_offsets(level)
            assert current >= previous
            previous = current


class TestAutoLevel:
    """`eth auto` has to read a temperature. It used to return 15 and
    claim it had."""

    @pytest.mark.parametrize(
        ("fraction", "expected"),
        [(0.40, 20), (0.65, 15), (0.78, 10), (0.95, 5), (1.10, 0)],
    )
    def test_level_follows_temperature(self, monkeypatch, fraction, expected):
        """Stated as a fraction of the abort temperature, not in degrees.

        The absolute version of this test passed for a year and then
        started asserting the wrong thing the moment somebody lowered
        `THERMAL_ABORT_C` — because the bands were three fixed degrees
        and one derived one, and lowering the threshold put the derived
        band below two of the fixed ones. A test written in degrees
        cannot notice that; one written in fractions fails the moment
        the ladder stops being a ladder.
        """
        temp = eth.THERMAL_ABORT_C * fraction
        monkeypatch.setattr(eth.Ethanol, "read_temperature", lambda self: temp)
        level, reason = eth.Ethanol(level=0, backend="nvidia").auto_level()
        assert level == expected
        assert f"{temp:.0f}" in reason

    def test_the_bands_ascend(self):
        """The bug itself. A table that does not ascend has a dead band
        in it, and every temperature past the out-of-order entry falls
        off the end to stock instead of matching."""
        for abort in (40.0, 60.0, 75.0, 91.5, 105.0):
            ceilings = [c for c, _ in eth.auto_level_bands(abort)]
            assert ceilings == sorted(ceilings), f"out of order at {abort}"
            assert len(set(ceilings)) == len(ceilings)

    def test_the_top_band_is_the_abort_temperature(self):
        """Above it, `apply` refuses anyway — a band that reached past
        it would be picking a level the next call would not apply."""
        assert eth.auto_level_bands(60.0)[-1][0] == 60.0

    def test_it_never_picks_a_level_apply_would_refuse(self, monkeypatch):
        """`_check_temperature_safe` refuses levels above 10 once the GPU
        is past the abort temperature. `auto_level` choosing 15 or 20
        there would be the two halves of the module disagreeing."""
        for fraction in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2):
            temp = eth.THERMAL_ABORT_C * fraction
            monkeypatch.setattr(eth.Ethanol, "read_temperature",
                                lambda self, t=temp: t)
            level, _ = eth.Ethanol(level=0, backend="nvidia").auto_level()
            machine = eth.Ethanol(level=level, backend="nvidia")
            safe, _, why = machine._check_temperature_safe()
            assert safe, f"auto picked {level} at {temp:.0f}C but apply says: {why}"

    def test_a_hot_gpu_gets_no_overclock(self, monkeypatch):
        monkeypatch.setattr(eth.Ethanol, "read_temperature", lambda self: 99.0)
        level, reason = eth.Ethanol(level=0, backend="nvidia").auto_level()
        assert level == 0
        assert "too hot" in reason

    def test_an_unreadable_temperature_stays_at_stock(self, monkeypatch):
        """"I can't tell how hot this is" must not mean "overclock it
        anyway" — the old constant did exactly that."""
        monkeypatch.setattr(eth.Ethanol, "read_temperature", lambda self: None)
        level, reason = eth.Ethanol(level=0, backend="nvidia").auto_level()
        assert level == 0
        assert "could not read" in reason

    def test_auto_is_not_a_constant(self, monkeypatch):
        """A regression guard with teeth: two different temperatures must
        not produce the same level."""
        levels = set()
        for temp in (35.0, 70.0, 95.0):
            monkeypatch.setattr(eth.Ethanol, "read_temperature", lambda self, t=temp: t)
            levels.add(eth.Ethanol(level=0, backend="nvidia").auto_level()[0])
        assert len(levels) > 1


class TestReadTemperature:
    def test_reads_nvidia_smi(self, monkeypatch):
        monkeypatch.setattr(eth, "_run", lambda cmd: _completed("61\n"))
        assert eth.Ethanol(level=0, backend="nvidia").read_temperature() == 61.0

    def test_returns_none_when_the_tool_fails(self, monkeypatch):
        monkeypatch.setattr(eth, "_run", lambda cmd: _completed("", returncode=9))
        assert eth.Ethanol(level=0, backend="nvidia").read_temperature() is None

    def test_returns_none_on_unparseable_output(self, monkeypatch):
        """None, not 0.0 — an unreadable GPU must not look like the
        coldest possible one."""
        monkeypatch.setattr(eth, "_run", lambda cmd: _completed("N/A"))
        assert eth.Ethanol(level=0, backend="nvidia").read_temperature() is None

    def test_no_backend_reads_nothing(self):
        assert eth.Ethanol(level=0, backend="none").read_temperature() is None


class TestResetIsActuallyAReset:
    """Level 0 is documented as "reset to stock" on every backend."""

    def test_rocm_level_zero_resets_instead_of_maxing_out(self, commands):
        eth.Ethanol(level=0, backend="rocm").apply(confirm=True)
        joined = " ".join(" ".join(c) for c in commands)
        assert "--resetclocks" in joined
        assert "--setperflevel auto" in joined
        # The bug: `--setsclk 7` pinned the card to its highest clock
        # index on the one command a user reaches for to undo an
        # overclock.
        assert "--setsclk" not in joined

    def test_rocm_nonzero_level_still_raises_clocks(self, commands):
        eth.Ethanol(level=20, backend="rocm").apply(confirm=True)
        joined = " ".join(" ".join(c) for c in commands)
        assert "--setperflevel manual" in joined
        assert "--setsclk" in joined

    def test_intel_level_zero_resets_instead_of_a_no_op(self, commands):
        eth.Ethanol(level=0, backend="intel").apply(confirm=True)
        joined = " ".join(" ".join(c) for c in commands)
        assert "-d" in joined
        assert "+0" not in joined

    def test_nvidia_level_zero_zeroes_the_offsets(self, commands):
        eth.Ethanol(level=0, backend="nvidia").apply(confirm=True)
        joined = " ".join(" ".join(c) for c in commands)
        assert "GPUGraphicsClockOffsetAllPerformanceLevels=0" in joined
        assert "GPUMemoryTransferRateOffsetAllPerformanceLevels=0" in joined


class TestConfirmGate:
    def test_nothing_runs_without_confirmation(self, commands):
        result = eth.Ethanol(level=10, backend="nvidia").apply()
        assert result.applied is False
        assert commands == []

    def test_env_var_confirms(self, commands, monkeypatch):
        monkeypatch.setenv("HYPERNIX_ETHANOL_CONFIRM", "1")
        result = eth.Ethanol(level=10, backend="nvidia").apply()
        assert result.applied is True
        assert commands

    def test_no_backend_says_so_instead_of_asking_for_confirmation(self, commands):
        """Telling a user with no GPU tools to set a confirm flag sends
        them to the same dead end one step later."""
        result = eth.Ethanol(level=10, backend="none").apply()
        assert result.applied is False
        assert "no supported overclocking tool" in result.notes
        assert "CONFIRM" not in result.notes
        assert commands == []


class TestThermalAbort:
    def test_a_hot_gpu_aborts_a_big_overclock(self, monkeypatch, commands):
        monkeypatch.setattr(eth.Ethanol, "read_temperature", lambda self: 92.0)
        result = eth.Ethanol(level=20, backend="nvidia").apply(confirm=True)
        assert result.applied is False
        assert "too hot" in result.notes
        assert commands == []

    def test_a_hot_gpu_still_allows_a_reset(self, monkeypatch, commands):
        """The abort must never block level 0 — refusing to let someone
        undo an overclock because the card is hot is backwards."""
        monkeypatch.setattr(eth.Ethanol, "read_temperature", lambda self: 92.0)
        result = eth.Ethanol(level=0, backend="nvidia").apply(confirm=True)
        assert result.applied is True

    def test_auto_throttle_can_be_turned_off(self, monkeypatch, commands):
        monkeypatch.setattr(eth.Ethanol, "read_temperature", lambda self: 92.0)
        result = eth.Ethanol(level=20, backend="nvidia").apply(
            confirm=True, auto_throttle=False,
        )
        assert result.applied is True


class TestStatus:
    def test_parses_an_nvidia_row(self, monkeypatch):
        monkeypatch.setattr(
            eth, "_run",
            lambda cmd: _completed("NVIDIA RTX 4080, 58, 143.2, 320.0, 2505, 11201, 47"),
        )
        info = eth.Ethanol(level=0, backend="nvidia").status()
        assert info["name"] == "NVIDIA RTX 4080"
        assert info["temperature_c"] == 58.0
        assert info["power_draw_w"] == 143.2
        assert info["clock_graphics_mhz"] == 2505.0
        assert info["utilization_pct"] == 47.0

    def test_missing_readings_are_none_not_absent(self, monkeypatch):
        """A caller has to be able to tell "not reported" from "no such
        key"."""
        monkeypatch.setattr(eth, "_run", lambda cmd: _completed("GPU, [N/A], [N/A], 320, 2505, 11201, 47"))
        info = eth.Ethanol(level=0, backend="nvidia").status()
        assert "temperature_c" in info
        assert info["temperature_c"] is None

    def test_no_backend_returns_a_shaped_dict(self):
        info = eth.Ethanol(level=0, backend="none").status()
        assert info["backend"] == "none"
        assert info["temperature_c"] is None

    def test_status_changes_nothing(self, commands):
        eth.Ethanol(level=0, backend="nvidia").status()
        for cmd in commands:
            joined = " ".join(cmd)
            assert "--set" not in joined
            assert "-pl" not in joined
            assert "-a " not in joined


class TestCLI:
    def test_help_exits_zero(self, capsys):
        assert eth.cli_main(["--help"]) == 0
        assert "eth status" in capsys.readouterr().out

    def test_no_args_shows_help(self, capsys):
        assert eth.cli_main([]) == 0
        assert "usage:" in capsys.readouterr().out

    def test_rejects_a_non_numeric_level(self, capsys):
        assert eth.cli_main(["banana"]) == 2
        assert "expected a level" in capsys.readouterr().err

    def test_rejects_an_out_of_range_level(self, capsys):
        assert eth.cli_main([str(eth.MAX_LEVEL + 1)]) == 2
        assert "must be in" in capsys.readouterr().err

    def test_rejects_a_bad_gpu_index(self, capsys):
        assert eth.cli_main(["5", "--gpu", "x"]) == 2

    def test_gpu_index_without_a_value(self, capsys):
        assert eth.cli_main(["5", "--gpu"]) == 2

    def test_reset_is_level_zero(self, monkeypatch, commands):
        monkeypatch.setattr(eth, "_detect_backend", lambda: "nvidia")
        eth.cli_main(["reset", "--confirm"])
        assert "GPUGraphicsClockOffsetAllPerformanceLevels=0" in " ".join(
            " ".join(c) for c in commands
        )

    def test_status_subcommand(self, monkeypatch, capsys):
        monkeypatch.setattr(eth, "_detect_backend", lambda: "nvidia")
        monkeypatch.setattr(
            eth, "_run",
            lambda cmd: _completed("Fake GPU, 58, 143.2, 320.0, 2505, 11201, 47"),
        )
        assert eth.cli_main(["status"]) == 0
        out = capsys.readouterr().out
        assert "Fake GPU" in out
        assert "58" in out

    def test_status_without_a_backend_fails(self, monkeypatch, capsys):
        monkeypatch.setattr(eth, "_detect_backend", lambda: "none")
        assert eth.cli_main(["status"]) == 1
        assert "no supported GPU tool" in capsys.readouterr().err

    def test_no_backend_exits_nonzero(self, monkeypatch):
        """It used to exit 0 on a machine with no GPU tools, which looks
        exactly like a successful overclock in a script."""
        monkeypatch.setattr(eth, "_detect_backend", lambda: "none")
        assert eth.cli_main(["5"]) == 1

    def test_plan_only_exits_zero(self, monkeypatch):
        monkeypatch.setattr(eth, "_detect_backend", lambda: "nvidia")
        assert eth.cli_main(["5"]) == 0


class TestModuleEntryPoint:
    def test_has_a_main_guard(self):
        from pathlib import Path

        source = Path(eth.__file__).read_text(encoding="utf-8")
        assert '__name__ == "__main__"' in source
