"""``hnx prot`` and the screen that stayed on.

The report was "prot doesn't make the monitors black/sleep". The code
was one line::

    subprocess.run(["xset", "dpms", "force", state], check=False,
                   stdout=DEVNULL, stderr=DEVNULL)

and that line fails, silently and identically, in four different
situations: DPMS disabled, a Wayland session, no graphical session at
all, and macOS. ``check=False`` meant a non-zero exit was fine,
``DEVNULL`` meant the error message was discarded, and
``except Exception: pass`` covered whatever was left — so the screen
stayed on, the terminal went into raw mode anyway, and the user was told
"Monitor will sleep".

These tests are about that shape rather than about ``xset``: that a
failure is reported, that the session decides the method, and that raw
mode is not entered on the strength of a blank that did not happen.
"""
from __future__ import annotations

import subprocess

import pytest

from hypernix.system import blanking, protect


@pytest.fixture(autouse=True)
def _no_real_commands(monkeypatch):
    """Nothing in this file runs a real xset, swaymsg or pmset, and
    nothing in it depends on the platform it happens to run on.

    The platform pin is not tidiness. ``detect_session`` answers
    ``"darwin"`` from ``sys.platform`` alone, before it looks at a single
    environment variable -- which is correct, because a Mac has no
    ``DISPLAY`` to consult -- so on a macOS runner every test here that
    sets up an X11 or Wayland session was describing a session the code
    never saw, and got pmset or nothing. Windows fell through to the
    Linux detector and answered ``"windows"``.

    These tests are about which method a given *session* selects, so the
    session is something each one states rather than inherits. The one
    test that is about macOS sets ``darwin`` itself.
    """
    def refuse(*args, **kwargs):  # pragma: no cover - only on a mistake
        raise AssertionError(f"a test tried to run {args!r}")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(blanking.sys, "platform", "linux")
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE", "SWAYSOCK",
                 "HYPRLAND_INSTANCE_SIGNATURE"):
        monkeypatch.delenv(name, raising=False)


class Recorder:
    """Stands in for ``subprocess.run``, remembering every argv."""

    def __init__(self, returncode: int = 0, stderr: str = "", stdout: str = ""):
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(
            argv, self.returncode, stdout=self.stdout, stderr=self.stderr
        )

    @property
    def flat(self) -> list[str]:
        return [" ".join(call) for call in self.calls]


def use_x11(monkeypatch, recorder: Recorder) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setattr(blanking.shutil, "which",
                        lambda name: f"/usr/bin/{name}" if name == "xset" else None)
    monkeypatch.setattr(subprocess, "run", recorder)


class TestTheSessionDecidesTheMethod:
    """The Wayland half of the bug. ``xset`` under Wayland is not a
    degraded answer, it is no answer: there is no X server to ask."""

    def test_x11_uses_xset(self, monkeypatch):
        use_x11(monkeypatch, Recorder())
        chosen = blanking.choose_blanker()
        assert chosen is not None and chosen.name == "xset"

    def test_wayland_does_not_use_xset(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setattr(blanking.shutil, "which", lambda name: f"/usr/bin/{name}")
        chosen = blanking.choose_blanker()
        assert chosen is not None
        assert chosen.name != "xset"
        assert chosen.session == "wayland"

    def test_hyprland_beats_a_generic_wlroots_tool(self, monkeypatch):
        """wlopm is often installed next to hyprctl and hyprctl is the
        one that works, so "installed" cannot be the whole test."""
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "abc123")
        monkeypatch.setattr(blanking.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert blanking.choose_blanker().name == "hyprctl"

    def test_a_compositor_tool_without_its_compositor_is_not_chosen(self, monkeypatch):
        """swaymsg exists on plenty of machines that are not running
        sway, and it cannot blank anything on them."""
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setattr(
            blanking.shutil, "which",
            lambda name: "/usr/bin/swaymsg" if name == "swaymsg" else None,
        )
        assert blanking.choose_blanker() is None

    def test_macos_has_a_method(self, monkeypatch):
        """The old code checked ``sys.platform.startswith("linux")`` and
        did nothing otherwise. A Mac has ``pmset displaysleepnow``."""
        monkeypatch.setattr(blanking.sys, "platform", "darwin")
        monkeypatch.setattr(blanking.shutil, "which", lambda name: f"/usr/bin/{name}")
        chosen = blanking.choose_blanker()
        assert chosen is not None and chosen.name == "pmset"


class TestDpmsIsEnabledBeforeItIsForced:
    """The single most common cause. ``xset dpms force off`` against a
    disabled DPMS extension is accepted and ignored: the X server takes
    the request, does nothing, and exits 0. Nothing anywhere says so."""

    def test_plus_dpms_runs_before_force_off(self, monkeypatch):
        recorder = Recorder()
        use_x11(monkeypatch, recorder)
        assert blanking.set_monitor_state("off").ok
        assert recorder.flat == ["xset +dpms", "xset dpms force off"]

    def test_a_session_that_had_dpms_off_gets_it_back(self, monkeypatch):
        """Somebody who disabled DPMS did it on purpose — usually to
        stop a machine blanking mid-presentation. Leaving it enabled
        breaks something on the way out."""
        recorder = Recorder()
        use_x11(monkeypatch, recorder)
        blanking.set_monitor_state("on", restore_dpms=False)
        assert recorder.flat == ["xset dpms force on", "xset -dpms"]

    def test_a_session_that_had_dpms_on_is_left_alone(self, monkeypatch):
        recorder = Recorder()
        use_x11(monkeypatch, recorder)
        blanking.set_monitor_state("on", restore_dpms=True)
        assert recorder.flat == ["xset dpms force on"]

    def test_the_prior_setting_is_read_from_xset_q(self, monkeypatch):
        recorder = Recorder(stdout="  DPMS is Disabled\n")
        use_x11(monkeypatch, recorder)
        assert blanking.dpms_enabled() is False
        recorder.stdout = "  DPMS is Enabled\n"
        assert blanking.dpms_enabled() is True


class TestFailuresAreReported:
    """All of them used to look like success."""

    def test_a_nonzero_exit_is_not_success(self, monkeypatch):
        recorder = Recorder(returncode=1, stderr="xset:  unable to open display\n")
        use_x11(monkeypatch, recorder)
        result = blanking.set_monitor_state("off")
        assert not result.ok
        assert "unable to open display" in result.reason

    def test_a_missing_binary_is_not_success(self, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        monkeypatch.setattr(blanking.shutil, "which", lambda name: None)
        result = blanking.set_monitor_state("off")
        assert not result.ok
        assert "x11-xserver-utils" in result.hint

    def test_a_hung_compositor_does_not_hang_prot(self, monkeypatch):
        """Raw mode is entered right after this call. A blanking command
        that never returns means a terminal that never comes back."""
        def hang(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, blanking.COMMAND_TIMEOUT)

        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        monkeypatch.setattr(blanking.shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setattr(subprocess, "run", hang)
        result = blanking.set_monitor_state("off")
        assert not result.ok
        assert "did not answer" in result.reason

    def test_no_graphical_session_says_which_problem_it_is(self, monkeypatch):
        monkeypatch.setattr(blanking.shutil, "which", lambda name: f"/usr/bin/{name}")
        result = blanking.set_monitor_state("off")
        assert not result.ok
        assert "DISPLAY" in result.reason

    def test_success_names_the_method(self, monkeypatch):
        use_x11(monkeypatch, Recorder())
        assert blanking.set_monitor_state("off").method == "xset"


class TestItDoesNotLockAScreenItDidNotBlank:
    """The compounding failure: raw mode on a lit screen. The machine
    looks unlocked, the keyboard looks dead, and the only message on
    screen promises the opposite of what happened."""

    def test_it_refuses_rather_than_entering_raw_mode(self, monkeypatch, capsys):
        monkeypatch.setattr(blanking.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            protect.termios, "tcgetattr",
            lambda fd: pytest.fail("entered raw mode without blanking the screen"),
        )
        assert protect.start_protection() == 1
        out = capsys.readouterr().out
        assert "--force" in out

    def test_force_is_the_way_past_it(self, monkeypatch):
        """Refusing is the default, not a wall."""
        import inspect

        assert "force" in inspect.signature(protect.start_protection).parameters

    def test_the_cli_passes_force_through(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            protect, "start_protection", lambda *, force=False: seen.update(force=force) or 0
        )
        protect.cli_main(["start", "--force"])
        assert seen == {"force": True}

    def test_the_promise_is_not_made_before_the_attempt(self):
        """"Monitor will sleep" was printed before anything was tried,
        which is how a failure became a lie rather than a message."""
        import inspect

        source = inspect.getsource(protect.start_protection)
        assert "Monitor will sleep" not in source


class TestOutageAndProtAgree:
    """They did not, and that is how both of them ended up missing the
    DPMS enable: two copies of "turn the screen off", each slightly
    different, each fixed separately or not at all."""

    def test_outage_uses_the_same_table(self, monkeypatch):
        from hypernix.system import outage

        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "abc123")
        monkeypatch.setattr(blanking.shutil, "which", lambda name: f"/usr/bin/{name}")
        assert outage.detect_backend() == blanking.choose_blanker().name

    def test_outage_no_longer_reaches_for_xset_under_wayland(self, monkeypatch):
        """Its old detector fell through to ``xset`` on any Wayland
        session without ``wlopm`` installed. There is no X server there
        for xset to talk to."""
        from hypernix.system import outage

        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        monkeypatch.setattr(
            blanking.shutil, "which",
            lambda name: "/usr/bin/xset" if name == "xset" else None,
        )
        assert outage.detect_backend() != "xset"

    def test_outage_enables_dpms_too(self, monkeypatch):
        from hypernix.system import outage

        recorder = Recorder()
        use_x11(monkeypatch, recorder)
        outage.Outage(backend="xset").black_out()
        # `xset q` first: the prior DPMS setting has to be read before
        # it is changed, so restoring can put it back.
        assert recorder.flat == ["xset q", "xset +dpms", "xset dpms force off"]

    def test_there_is_only_one_copy_of_the_commands(self):
        """A literal ``xset dpms force`` outside the shared table means
        somebody has started a third copy."""
        import ast
        import inspect

        from hypernix.system import outage

        def code_only(module) -> str:
            """The module's source with every docstring removed.

            Both of these talk about DPMS at length in prose, and
            ``protect``'s quotes the original broken one-liner verbatim
            so the next person can see what was wrong with it. Matching
            on that would make the documentation fail the test.
            """
            tree = ast.parse(inspect.getsource(module))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef,
                                         ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                body = getattr(node, "body", [])
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    node.body = body[1:] or [ast.Pass()]
            return ast.unparse(ast.fix_missing_locations(tree))

        for module in (outage, protect):
            source = code_only(module)
            for argv in ("'dpms', 'force'", "'displaysleepnow'", "'--off', '*'"):
                assert argv not in source, f"{module.__name__} grew its own {argv}"

    def test_the_check_above_would_catch_one(self):
        """Guarding the guard: `ast.unparse` normalises quoting, so a
        pattern written with the wrong quotes would match nothing and
        pass forever."""
        import ast

        normalised = ast.unparse(ast.parse('run(["xset", "dpms", "force", "off"])'))
        assert "'dpms', 'force'" in normalised
