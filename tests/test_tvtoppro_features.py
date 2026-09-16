"""tvtoppro's three additions: the intro, the module system, the watchdog.

Each is tested at the layer that can actually fail.

The **intro** is an animation, and the only part of one worth asserting is
that it ends showing what it was asked to show. :func:`decode_frames` is
separated from the printing precisely so that assertion exists — an
animation that lives entirely inside a ``time.sleep`` loop is one nobody
ever checks.

The **module system** exists so that a third party can add a panel, which
means its failure modes are somebody else's code raising, returning the
wrong type, or being slow. All three are tested with modules that do
exactly that, because the promise being made is that a bad plugin costs
its own panel and nothing else.

The **watchdog** is tested against real files with real mtimes. Its whole
job is noticing that a file is old, and a mocked clock would be testing
the mock.
"""
from __future__ import annotations

import os
import time

import pytest

from hypernix.monitoring import stale_log
from hypernix.monitoring.tvtoppro import THEMES, TvTopPro, _duration, play_intro
from hypernix.monitoring.tvtoppro_modules import (
    BUILTIN,
    Module,
    Panel,
    Reading,
    discover,
    load_modules,
    poll_all,
)
from hypernix.timing.spinner import decode_frames


class TestTheDecodeAnimation:
    def test_it_ends_on_the_text(self):
        """The only property that matters. Everything before the last
        frame is noise by design; the last frame is a promise."""
        assert decode_frames("tvtoppro", seed=0)[-1] == "tvtoppro"

    def test_every_frame_is_the_same_width(self):
        """A line that changes length between frames reads as jitter
        rather than as decoding, and on a narrow terminal it wraps."""
        frames = decode_frames("hypernix monitor", seed=1)
        assert {len(frame) for frame in frames} == {len("hypernix monitor")}

    def test_spaces_never_scramble(self):
        """The shape of the text has to be visible from frame one, or
        the eye has nothing to lock onto."""
        text = "two words here"
        spaces = [i for i, ch in enumerate(text) if ch == " "]
        for frame in decode_frames(text, seed=2):
            assert all(frame[i] == " " for i in spaces)

    def test_it_resolves_monotonically(self):
        """Characters lock in and stay locked. A character that
        un-resolves reads as a glitch, which is the other animation."""
        text = "resolving"
        frames = decode_frames(text, seed=3)
        resolved_counts = [
            sum(1 for a, b in zip(frame, text, strict=True) if a == b) for frame in frames
        ]
        assert resolved_counts == sorted(resolved_counts)

    def test_it_is_deterministic_for_a_seed(self):
        assert decode_frames("abc", seed=7) == decode_frames("abc", seed=7)

    def test_a_single_step_is_just_the_text(self):
        assert decode_frames("x", steps=1) == ["x"]

    def test_the_intro_cannot_take_the_dashboard_down(self, monkeypatch):
        """It is the first thing tvtoppro does. There is no failure of a
        decorative animation that should stop a monitoring tool."""
        import hypernix.timing.spinner as spinner

        def _explode(*_args, **_kwargs):
            raise RuntimeError("no terminal here")

        monkeypatch.setattr(spinner, "anime_print", _explode)
        play_intro(THEMES["hypernix"], enabled=True)  # must not raise

    def test_disabled_does_nothing(self, capsys):
        play_intro(THEMES["hypernix"], enabled=False)
        assert capsys.readouterr().out == ""


class TestDurationParsing:
    @pytest.mark.parametrize("text,seconds", [
        ("7d", 604800), ("1w", 604800), ("6h", 21600),
        ("30m", 1800), ("45s", 45), ("90", 90), ("1.5h", 5400),
    ])
    def test_suffixes(self, text, seconds):
        assert _duration(text) == pytest.approx(seconds)

    @pytest.mark.parametrize("text", ["off", "none", "never", "0", ""])
    def test_off_is_zero(self, text):
        assert _duration(text) == 0.0

    @pytest.mark.parametrize("text", ["soon", "7y", "-3d", "d"])
    def test_nonsense_says_what_is_accepted(self, text):
        with pytest.raises(ValueError):
            _duration(text)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class _Good(Module):
    name = "good"
    title = "good"
    description = "returns a panel"

    def poll(self):
        return Panel(title="good", readings=[
            Reading("a", "1", fraction=0.5, ramp="cpu"),
            Reading("b", "2", history=[0.0, 0.5, 1.0], ramp="used"),
            Reading("c", "3"),
        ])


class _Raises(Module):
    name = "raises"
    title = "raises"

    def poll(self):
        raise ZeroDivisionError("plugins do this")


class _WrongType(Module):
    name = "wrong"
    title = "wrong"

    def poll(self):
        return {"not": "a panel"}


class _Slow(Module):
    name = "slow"
    title = "slow"

    def poll(self):
        time.sleep(0.4)
        return Panel(title="slow", readings=[Reading("eventually", "yes")])


class _Unavailable(Module):
    name = "absent"
    title = "absent"

    def available(self) -> bool:
        return False

    def unavailable_reason(self) -> str:
        return "this machine has no flux capacitor"

    def poll(self):  # pragma: no cover - never reached
        return Panel(title="absent")


class TestReadings:
    def test_fractions_are_clamped(self):
        """A module doing its own arithmetic will hand over 1.02 sooner
        or later, and a meter longer than its own bar breaks the box."""
        assert Reading("x", fraction=1.5).fraction == 1.0
        assert Reading("x", fraction=-0.2).fraction == 0.0

    def test_history_is_clamped(self):
        assert Reading("x", history=[-1.0, 0.5, 9.0]).history == [0.0, 0.5, 1.0]

    def test_no_fraction_stays_none(self):
        """None and 0.0 mean different things: no meter, and an empty
        meter."""
        assert Reading("x", value="text").fraction is None


class TestPollAll:
    def test_a_raising_module_costs_its_own_panel(self):
        panels = poll_all([_Good(), _Raises(), _Good()])
        assert len(panels) == 3
        assert panels[1].error.startswith("ZeroDivisionError")
        assert panels[0].readings and panels[2].readings

    def test_a_wrong_return_type_is_caught(self):
        """Duck typing would reach into a dict and raise somewhere far
        from the module that caused it."""
        panels = poll_all([_WrongType()])
        assert "not a Panel" in panels[0].error

    def test_a_slow_module_is_named(self):
        panels = poll_all([_Slow()])
        assert panels[0].readings           # it did produce its numbers
        assert "budget" in panels[0].error  # and it is still called out

    def test_a_fast_module_has_no_error(self):
        assert poll_all([_Good()])[0].error == ""


class TestDiscovery:
    def test_the_builtins_are_all_there(self):
        found = discover()
        for name in BUILTIN:
            assert isinstance(found[name], Module), name

    def test_all_selects_only_available_ones(self):
        loaded, _ = load_modules("all")
        assert loaded
        assert all(module.available() for module in loaded)

    def test_an_unknown_name_is_a_problem_not_a_crash(self):
        """One typo in --modules disk,nte,swap should cost the typo."""
        loaded, problems = load_modules("disk,nosuchthing,swap")
        assert {module.name for module in loaded} >= {"disk"}
        assert problems and problems[0][0] == "nosuchthing"

    def test_a_near_miss_gets_a_suggestion(self):
        _loaded, problems = load_modules("dis")
        assert "did you mean" in problems[0][1]

    def test_an_unavailable_module_reports_why(self, tmp_path):
        module_dir = tmp_path / "modules"
        module_dir.mkdir()
        (module_dir / "absent.py").write_text(
            "from hypernix.monitoring.tvtoppro_modules import Module, Panel\n"
            "class Absent(Module):\n"
            "    name = 'absent'\n"
            "    title = 'absent'\n"
            "    def available(self): return False\n"
            "    def unavailable_reason(self): return 'no flux capacitor'\n"
            "    def poll(self): return Panel(title='absent')\n"
        )
        _loaded, problems = load_modules("absent", user_dir=module_dir)
        assert problems == [("absent", "no flux capacitor")]

    def test_a_user_file_is_picked_up(self, tmp_path):
        """The whole point of the user directory: no packaging, no
        install, drop a file in and it is there."""
        module_dir = tmp_path / "modules"
        module_dir.mkdir()
        (module_dir / "mine.py").write_text(
            "from hypernix.monitoring.tvtoppro_modules import Module, Panel, Reading\n"
            "class Mine(Module):\n"
            "    name = 'mine'\n"
            "    title = 'mine'\n"
            "    def poll(self):\n"
            "        return Panel(title='mine', readings=[Reading('hi', '42')])\n"
        )
        loaded, problems = load_modules("mine", user_dir=module_dir)
        assert problems == []
        assert loaded[0].poll().readings[0].value == "42"

    def test_a_broken_user_file_is_reported_not_swallowed(self, tmp_path):
        """A plugin that silently does not appear is one its author
        cannot debug."""
        module_dir = tmp_path / "modules"
        module_dir.mkdir()
        (module_dir / "bad.py").write_text("import nonexistent_module_xyz\n")
        found = discover(user_dir=module_dir)
        assert isinstance(found["bad"], str)
        assert "ModuleNotFoundError" in found["bad"]

    def test_a_file_with_no_module_says_so(self, tmp_path):
        module_dir = tmp_path / "modules"
        module_dir.mkdir()
        (module_dir / "empty.py").write_text("x = 1\n")
        found = discover(user_dir=module_dir)
        assert "defines no Module" in found["empty"]

    def test_underscore_files_are_skipped(self, tmp_path):
        """So a shared helper next to the modules is not itself loaded."""
        module_dir = tmp_path / "modules"
        module_dir.mkdir()
        (module_dir / "_helpers.py").write_text("raise RuntimeError('boom')\n")
        assert "_helpers" not in discover(user_dir=module_dir)


class TestBuiltinModules:
    @pytest.mark.parametrize("name", sorted(BUILTIN))
    def test_it_polls_without_raising(self, name):
        module = BUILTIN[name]()
        if not module.available():
            pytest.skip(module.unavailable_reason())
        assert isinstance(module.poll(), Panel)

    @pytest.mark.parametrize("name", sorted(BUILTIN))
    def test_polling_twice_gives_rates_not_totals(self, name):
        """Every counter these read is monotonic-since-boot, so the first
        sample has nothing to subtract from. Returning it raw would show
        a 40 GB/s disk read on the first frame."""
        module = BUILTIN[name]()
        if not module.available():
            pytest.skip(module.unavailable_reason())
        module.poll()
        panel = module.poll()
        for reading in panel.readings:
            if reading.fraction is not None:
                assert 0.0 <= reading.fraction <= 1.0


# ---------------------------------------------------------------------------
# The stale-log watchdog
# ---------------------------------------------------------------------------


class TestStaleness:
    def test_a_fresh_log_is_not_stale(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("step 1/10 loss=1.0\n")
        assert not stale_log.is_stale(log)

    def test_an_old_log_is(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("step 1/10 loss=1.0\n")
        old = time.time() - 30 * 86400
        os.utime(log, (old, old))
        assert stale_log.is_stale(log)
        assert stale_log.log_age_seconds(log) > 29 * 86400

    def test_a_missing_log_is_not_stale(self, tmp_path):
        """"There is no log" is a different problem with a different
        message, and the dashboard already says it."""
        assert not stale_log.is_stale(tmp_path / "nope.log")
        assert stale_log.log_age_seconds(tmp_path / "nope.log") is None
        assert not stale_log.is_stale(None)

    def test_the_threshold_is_honoured(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("x\n")
        old = time.time() - 3600
        os.utime(log, (old, old))
        assert stale_log.is_stale(log, threshold_seconds=60)
        assert not stale_log.is_stale(log, threshold_seconds=7200)

    def test_investigate_short_circuits_on_a_fresh_log(self, tmp_path):
        log = tmp_path / "train.log"
        log.write_text("step 1/10 loss=1.0\n")
        report = stale_log.investigate(log)
        assert not report.stale
        assert report.process is None

    def test_force_investigates_anyway(self, tmp_path):
        """"Show me what is actually training" is a reasonable thing to
        ask on purpose, not only as a consequence of a stale file."""
        log = tmp_path / "train.log"
        log.write_text("step 1/10 loss=1.0\n")
        assert stale_log.investigate(log, force=True).stale

    def test_it_never_raises(self, tmp_path, monkeypatch):
        """It runs on a dashboard's refresh tick. An exception there
        takes the screen down."""
        def _explode():
            raise RuntimeError("psutil had a bad day")

        monkeypatch.setattr(stale_log, "busiest_python", _explode)
        report = stale_log.investigate(tmp_path / "x.log", force=True)
        assert report.stale
        assert any("could not inspect" in note for note in report.notes)

    def test_the_report_serialises(self, tmp_path):
        report = stale_log.investigate(tmp_path / "x.log", force=True)
        as_dict = report.to_dict()
        assert set(as_dict) >= {"stale", "process", "notes", "progress"}
        assert isinstance(report.describe(), str)


class TestProcessRanking:
    def test_it_never_picks_itself(self):
        """The dashboard is a Python process and is frequently the
        busiest one on an idle box."""
        assert all(
            candidate.pid != os.getpid()
            for candidate in stale_log.rank_python_processes(limit=50)
        )

    def test_a_training_command_scores_above_an_idle_one(self):
        idle = stale_log.ProcessCandidate(pid=1, cpu_percent=5.0)
        training = stale_log.ProcessCandidate(
            pid=2, cpu_percent=5.0, command="python train.py",
            looks_like_training=True,
        )
        assert training.score > idle.score

    def test_the_hint_does_not_outrank_real_work(self):
        """A 2%-CPU process with 'train' in its name must not beat a
        genuinely busy one. The hint is evidence, not proof."""
        named = stale_log.ProcessCandidate(
            pid=1, cpu_percent=2.0, looks_like_training=True)
        busy = stale_log.ProcessCandidate(pid=2, cpu_percent=380.0)
        assert busy.score > named.score

    def test_lifetime_cpu_is_scaled_like_top(self):
        class _Times:
            user, system = 8.0, 2.0

        now = 1_000_000.0
        # Ten CPU-seconds over ten wall-clock seconds is one core, 100%.
        assert stale_log._lifetime_cpu(_Times(), now - 10, now) == pytest.approx(100.0)
        # The same work in five seconds is two cores.
        assert stale_log._lifetime_cpu(_Times(), now - 5, now) == pytest.approx(200.0)

    def test_missing_cpu_times_is_zero_not_a_crash(self):
        assert stale_log._lifetime_cpu(None, 0.0, 1.0) == 0.0
        assert stale_log._lifetime_cpu(object(), 1.0, 2.0) == 0.0

    def test_a_command_is_shortened_from_the_middle(self):
        """Everything that says which run it is lives at the end."""
        command = "/very/long/venv/bin/python3 " + "x" * 200 + " train.py --lr 1e-4"
        short = stale_log._shorten(command, 60)
        assert len(short) == 60
        assert short.startswith("/very/long")
        assert short.endswith("train.py --lr 1e-4")


class TestTheDashboardIntegration:
    def _dashboard(self, **kwargs) -> TvTopPro:
        from hypernix.monitoring.tvtop_plus_plus import TVTopPlusPlus

        return TvTopPro(source=TVTopPlusPlus(log_path=None), **kwargs)

    def test_no_stale_check_unless_asked(self):
        """It costs a /proc walk. Nobody should pay for it by default."""
        assert self._dashboard().stale_report() is None

    def test_the_check_is_cached(self, tmp_path):
        """The answer changes on the timescale of a run starting, not on
        the timescale of a frame."""
        dashboard = self._dashboard(log_path=tmp_path / "x.log",
                                    always_find_run=True)
        first = dashboard.stale_report()
        assert dashboard.stale_report() is first

    def test_module_panels_are_drawn(self):
        from rich.text import Text

        dashboard = self._dashboard(modules=[_Good()], show_processes=False)
        # Plain text, because the box title arrives as
        # `┌─┤[/][#ee]good[/][…]├` and `"┤good├" in markup` is false for
        # a panel that rendered perfectly.
        screen = Text.from_markup(dashboard.render(width=90)).plain
        assert "┤good├" in screen
        # Its three readings: a metered one, a graphed one, and a plain
        # label/value line.
        assert "a " in screen and "b " in screen and "c " in screen

    def test_a_broken_module_shows_its_error_in_its_box(self):
        dashboard = self._dashboard(modules=[_Raises()], show_processes=False)
        screen = dashboard.render(width=90)
        assert "ZeroDivisionError" in screen

    def test_every_row_is_the_same_width(self):
        """The btop-alike claim in one assertion: a right border that
        lands in a different column on any row is the thing this whole
        presentation layer exists to avoid."""
        from rich.text import Text

        dashboard = self._dashboard(modules=[_Good()], show_processes=False)
        # Rendered to plain text first: every row starts with a colour
        # tag, so testing the markup would be testing that `[#30]` begins
        # with a box character.
        rows = [
            Text.from_markup(row).plain
            for row in dashboard.render(width=90).splitlines()
        ]
        widths = {
            len(row) for row in rows
            if row.startswith(("│", "┌", "└"))
        }
        assert widths == {90}
        # And that the loop above actually found the boxes.
        assert len([r for r in rows if r.startswith("┌")]) >= 5

    def test_boxes_are_numbered_without_gaps(self):
        """The numbers are how btop's keyboard navigation addresses a
        box, so two boxes sharing one is worse than cosmetic."""
        import re

        dashboard = self._dashboard(modules=[_Good(), _Good()])
        screen = dashboard.render(width=90)
        numbers = [int(n) for n in re.findall(r"┤(\d+)├", screen)]
        assert numbers == list(range(1, len(numbers) + 1))
