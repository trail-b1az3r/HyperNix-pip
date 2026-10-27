"""`hnx prot` on Hyprland, with hyprlock or Quickshell doing the locking.

Hyprland blanked already (``hyprctl dispatch dpms off``), but nothing
kept the session from locking while it was dark: the lock guard knew
GNOME, KDE, Xfce and the X screensaver, and on Hyprland the locker is
hypridle -- which runs hyprlock, ``loginctl lock-session`` or a Quickshell
lock after its timeouts -- or Quickshell's own IdleMonitor. And
``misc:key_press_enables_dpms`` lit the screen at the wake word's first
letter. Each of those is checked here against a faked session.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from hypernix.system import blanking, protect


class FakeSession:
    """hyprctl, pgrep, systemctl and pkill as subprocess.run sees them, and
    the inhibitor processes as subprocess.Popen sees them."""

    def __init__(self, monkeypatch, tmp_path, *, running=("hypridle",), options=None,
                 unit_active=False, tools=None):
        self.running = set(running)
        self.options = dict(options or {"misc:key_press_enables_dpms": 1, "misc:mouse_move_enables_dpms": 0})
        self.unit_active = unit_active
        self.calls: list[list[str]] = []
        self.spawned: list[FakeProcess] = []
        self.tools = set(tools if tools is not None else
                         ("hyprctl", "pgrep", "systemctl", "pkill", "systemd-inhibit"))
        monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "abc")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setattr(blanking.shutil, "which",
                            lambda name: f"/usr/bin/{name}" if name in self.tools else None)
        monkeypatch.setattr(subprocess, "run", self.run)
        monkeypatch.setattr(blanking.subprocess, "Popen", self.popen)

    def run(self, argv, **kwargs):
        argv = [argv[0].rsplit("/", 1)[-1], *argv[1:]]
        self.calls.append(argv)
        tool, rest = argv[0], argv[1:]
        out, code = "", 0
        if tool == "pgrep":
            code = 0 if rest[-1] in self.running else 1
            out = "4242" if code == 0 else ""
        elif tool == "hyprctl" and rest[:1] == ["getoption"]:
            out = json.dumps({"option": rest[1], "int": self.options.get(rest[1], 0), "set": True})
        elif tool == "hyprctl" and rest[:1] == ["keyword"]:
            self.options[rest[1]] = 1 if rest[2] == "true" else 0
        elif tool == "hyprctl" and rest[:2] == ["dispatch", "exec"]:
            self.running.add(rest[2])
        elif tool == "systemctl":
            if rest[1] == "is-active":
                out, code = ("active", 0) if self.unit_active else ("inactive", 3)
            elif rest[1] == "stop":
                self.unit_active = False
                self.running.discard("hypridle")
            elif rest[1] == "start":
                self.unit_active = True
                self.running.add("hypridle")
        elif tool == "pkill":
            code = 0 if rest[-1] in self.running else 1
            self.running.discard(rest[-1])
        else:
            code = 1
        return subprocess.CompletedProcess(argv, code, stdout=out, stderr="")

    def popen(self, argv, **kwargs):
        process = FakeProcess([argv[0].rsplit("/", 1)[-1], *argv[1:]])
        self.spawned.append(process)
        return process

    @property
    def holding(self) -> list[list[str]]:
        return [p.argv for p in self.spawned if p.alive]


class FakeProcess:
    def __init__(self, argv):
        self.argv = argv
        self.alive = True

        class _Stdin:
            def close(inner):  # noqa: N805 - a tiny stand-in
                self.alive = False

        self.stdin = _Stdin()

    def wait(self, timeout=None):
        if self.alive:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return 0

    def terminate(self):
        self.alive = False

    def kill(self):
        self.alive = False


def _write(tmp_path, relative, text):
    path = tmp_path / "config" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_hypridle_is_held_off_with_a_logind_inhibitor(monkeypatch, tmp_path):
    session = FakeSession(monkeypatch, tmp_path)
    guard = blanking.LockGuard(state_file=str(tmp_path / "lock.json"))
    held = guard.disable()
    assert ["systemd-inhibit", "--what=idle", "--who=hnx prot",
            "--why=the screen is off for protection, not idle", "--mode=block", "cat"] in session.holding
    assert "idle inhibited (logind)" in held
    guard.restore()
    assert session.holding == []


def test_the_wake_word_is_not_its_own_first_letter(monkeypatch, tmp_path):
    """key_press_enables_dpms lit the screen at the first key typed."""
    session = FakeSession(monkeypatch, tmp_path)
    guard = blanking.LockGuard(state_file=str(tmp_path / "lock.json"))
    held = guard.disable()
    assert session.options["misc:key_press_enables_dpms"] == 0
    assert "Hyprland misc:key_press_enables_dpms" in held
    # One that was off already is left alone, so it is not turned on after.
    assert not any("mouse_move" in h for h in held)
    guard.restore()
    assert session.options == {"misc:key_press_enables_dpms": 1, "misc:mouse_move_enables_dpms": 0}


@pytest.mark.parametrize("unit_active", [True, False])
def test_a_hypridle_that_ignores_logind_is_stopped_and_started_again(monkeypatch, tmp_path, unit_active):
    _write(tmp_path, "hypr/hypridle.conf",
           "general {\n    lock_cmd = pidof hyprlock || hyprlock\n    ignore_systemd_inhibit = true\n}\n")
    session = FakeSession(monkeypatch, tmp_path, unit_active=unit_active)
    guard = blanking.LockGuard(state_file=str(tmp_path / "lock.json"))
    held = guard.disable()
    assert "hypridle (stopped)" in held
    assert "hypridle" not in session.running
    assert not any(p[0] == "systemd-inhibit" for p in session.holding)  # it would be ignored
    guard.restore()
    assert "hypridle" in session.running
    restart = ["systemctl", "--user", "start", "hypridle.service"] if unit_active else \
        ["hyprctl", "dispatch", "exec", "hypridle"]
    assert restart in session.calls


def test_a_crash_with_hypridle_stopped_is_undone_by_restore(monkeypatch, tmp_path, capsys):
    _write(tmp_path, "hypr/hypridle.conf", "general {\n  ignore_systemd_inhibit = yes  # mine\n}\n")
    session = FakeSession(monkeypatch, tmp_path, unit_active=True)
    state = tmp_path / "lock.json"
    blanking.LockGuard(state_file=str(state)).disable()     # ...and the process dies
    assert "hypridle" not in session.running
    monkeypatch.setattr(protect, "LOCK_STATE_FILE", state)
    assert protect.cli_main(["restore"]) == 0
    assert "hypridle" in session.running
    assert session.options["misc:key_press_enables_dpms"] == 1


def test_quickshell_watching_idle_gets_a_wayland_inhibitor(monkeypatch, tmp_path):
    _write(tmp_path, "quickshell/shell.qml",
           "import Quickshell.Wayland\nIdleMonitor { timeout: 300; onIsIdleChanged: lock.locked = true }\n")
    session = FakeSession(monkeypatch, tmp_path, running=("quickshell",),
                          tools=("hyprctl", "pgrep", "systemd-inhibit", "wlinhibit"))
    guard = blanking.LockGuard()
    held = guard.disable()
    assert ["wlinhibit"] in session.holding
    assert "idle inhibited (Wayland, for Quickshell)" in held
    guard.restore()
    assert session.holding == []


def test_quickshell_without_wlinhibit_is_said_not_hidden(monkeypatch, tmp_path):
    _write(tmp_path, "quickshell/lock/Idle.qml", "IdleMonitor { timeout: 120 }\n")
    FakeSession(monkeypatch, tmp_path, running=("qs",))
    guard = blanking.LockGuard()
    guard.disable()
    assert any("wlinhibit" in note for note in guard.notes)


def test_a_quickshell_that_does_not_watch_idle_needs_nothing(monkeypatch, tmp_path):
    _write(tmp_path, "quickshell/shell.qml", "PanelWindow { }\n")
    session = FakeSession(monkeypatch, tmp_path, running=("quickshell",),
                          tools=("hyprctl", "pgrep", "systemd-inhibit", "wlinhibit"))
    guard = blanking.LockGuard()
    guard.disable()
    assert ["wlinhibit"] not in session.holding
    assert guard.notes == []


def test_hypridle_without_systemd_inhibit_is_said(monkeypatch, tmp_path):
    FakeSession(monkeypatch, tmp_path, tools=("hyprctl", "pgrep"))
    guard = blanking.LockGuard()
    guard.disable()
    assert any("may lock" in note for note in guard.notes)


def test_nothing_happens_off_hyprland(monkeypatch, tmp_path):
    session = FakeSession(monkeypatch, tmp_path, running=())
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE")
    guard = blanking.LockGuard()
    assert guard.disable() == []
    assert session.spawned == [] and not any(c[0] == "hyprctl" for c in session.calls)
