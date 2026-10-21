"""blanking — turn the screen off, on every kind of session there is.

One module because there were two, and they disagreed.

:mod:`hypernix.system.protect` ran ``xset dpms force off``;
:mod:`hypernix.system.outage` ran ``xset dpms force off`` or ``wlopm``.
Both were missing the same thing, and it is the reason the report came
in as "prot doesn't make the monitors black":

    ``xset dpms force off`` is a *request to the DPMS extension*. If
    DPMS is disabled — which it is on a lot of desktops, because the
    desktop environment handles power management itself — the X server
    accepts the request, does nothing, and exits 0.

So the screen stayed on and the exit status said it had not. Add
``check=False``, two ``DEVNULL`` streams and an ``except Exception: pass`` on
top, as ``protect`` had, and there was no way to find out.

Enabling DPMS first fixes it, and the fix has to be in one place,
because a second copy is a second thing to not fix. What is here:

* the DPMS enable, and putting it back the way it was found;
* a method per session type rather than per platform — a Wayland
  session has no X server for ``xset`` to ask, and "Linux" is not the
  question;
* Hyprland, sway, wlroots, GNOME and KDE, because "Wayland" is not one
  thing either;
* never locking the screen (0.72.8): the freedesktop screensaver path
  blanked GNOME and KDE by *activating the screensaver*, and an active
  screensaver locks the session, so the wake word woke a lock screen.
  It is gone, and :class:`LockGuard` turns lock-on-idle off for as long
  as the screen is dark and puts it back afterwards;
* macOS, which the ``protect`` path checked for Linux and silently
  skipped;
* failures that come back as a reason and a remedy instead of a
  discarded stderr.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

#: How long any single blanking command gets. A compositor that has not
#: answered in this long is wedged, and waiting longer for it only means
#: the caller finds out later -- for ``prot``, with the terminal already
#: in raw mode.
COMMAND_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# What kind of session is this
# ---------------------------------------------------------------------------


def detect_session() -> str:
    """``"x11"``, ``"wayland"``, ``"darwin"`` or ``"none"``.

    The Linux half is :func:`hypernix.monitoring.remote_desktop.detect_session`
    — imported rather than re-derived, because "is this X11 or Wayland"
    having two answers in one program is how the cctvtop bug happened and
    there is no version of that which is worth a second copy.

    macOS is added here because it has a working answer to this feature
    and the old code did not check for it, so ``hnx prot`` on a Mac
    blanked nothing and said it had.
    """
    if sys.platform == "darwin":
        return "darwin"
    from ..monitoring.remote_desktop import detect_session as _linux_session

    return _linux_session()


@dataclass(frozen=True)
class Blanker:
    """One way of turning the screen off, and back on again.

    *before* runs first and its failures are not fatal: for ``xset`` it
    is ``+dpms``, and enabling an extension that is already enabled is
    not an error worth reporting.

    *env* names an environment variable the compositor sets. Several
    Wayland compositors ship tools that are installed but only work
    under their own compositor, so "the binary exists" is not the same
    question as "this will work here".
    """

    name: str
    session: str
    binary: str
    off: tuple[str, ...]
    on: tuple[str, ...]
    before: tuple[tuple[str, ...], ...] = ()
    env: str = ""
    install: str = ""
    #: A desktop it only works on, matched against XDG_CURRENT_DESKTOP.
    desktop: str = ""

    @property
    def available(self) -> bool:
        if not shutil.which(self.binary):
            return False
        if self.env and not os.environ.get(self.env):
            return False
        if self.desktop and self.desktop not in current_desktop():
            return False
        return True


#: Every method, most specific first.
#:
#: Order matters within a session type: a Hyprland session may well have
#: ``wlopm`` installed, and ``hyprctl`` is the one that will work.
BLANKERS: tuple[Blanker, ...] = (
    # X11. `+dpms` first, because `force off` against a disabled DPMS
    # extension is accepted and ignored -- the whole original bug.
    Blanker(
        name="xset", session="x11", binary="xset",
        before=(("+dpms",),),
        off=("dpms", "force", "off"), on=("dpms", "force", "on"),
        env="DISPLAY",
        install="Install x11-xserver-utils (Debian/Ubuntu) or xorg-xset (Arch).",
    ),
    # Wayland, compositor by compositor. There is no shared protocol for
    # this: wlr-output-power-management is a wlroots extension, and
    # GNOME and KDE each do their own thing over D-Bus.
    Blanker(
        name="hyprctl", session="wayland", binary="hyprctl",
        off=("dispatch", "dpms", "off"), on=("dispatch", "dpms", "on"),
        env="HYPRLAND_INSTANCE_SIGNATURE",
    ),
    Blanker(
        name="swaymsg", session="wayland", binary="swaymsg",
        off=("output", "*", "power", "off"),
        on=("output", "*", "power", "on"),
        env="SWAYSOCK",
    ),
    # GNOME: Mutter's own display power-save mode (3 = off, 0 = on). Not
    # the freedesktop screensaver this used to activate: an active
    # screensaver *locks the session*, which is not what blanking means.
    Blanker(
        name="gnome", session="wayland", binary="gdbus", desktop="gnome",
        off=("call", "--session", "--dest", "org.gnome.Mutter.DisplayConfig",
             "--object-path", "/org/gnome/Mutter/DisplayConfig",
             "--method", "org.freedesktop.DBus.Properties.Set",
             "org.gnome.Mutter.DisplayConfig", "PowerSaveMode", "<int32 3>"),
        on=("call", "--session", "--dest", "org.gnome.Mutter.DisplayConfig",
            "--object-path", "/org/gnome/Mutter/DisplayConfig",
            "--method", "org.freedesktop.DBus.Properties.Set",
            "org.gnome.Mutter.DisplayConfig", "PowerSaveMode", "<int32 0>"),
        install="GNOME: gdbus comes with glib (libglib2.0-bin).",
    ),
    # KDE Plasma 5.27+/6: DPMS through KScreen, which does not lock.
    Blanker(
        name="kscreen-doctor", session="wayland", binary="kscreen-doctor", desktop="kde",
        off=("--dpms", "off"), on=("--dpms", "on"),
        install="KDE: install kscreen (kscreen-doctor).",
    ),
    Blanker(
        name="wlopm", session="wayland", binary="wlopm",
        off=("--off", "*"), on=("--on", "*"),
        env="WAYLAND_DISPLAY",
        install="Install wlopm (wlroots compositors: sway, river, Wayfire).",
    ),
    # macOS. `displaysleepnow` is the system's own "sleep the display
    # now", and any keypress wakes it -- including the wake word, which
    # is why nothing needs to undo it. `on` is still wired up so the
    # wake path is the same shape everywhere.
    Blanker(
        name="pmset", session="darwin", binary="pmset",
        off=("displaysleepnow",), on=(),
    ),
)


def current_desktop() -> str:
    """The desktop, lower-cased: ``"gnome"``, ``"kde"``, ``"x-cinnamon"``..."""
    return ":".join(os.environ.get(name, "") for name in
                    ("XDG_CURRENT_DESKTOP", "XDG_SESSION_DESKTOP", "DESKTOP_SESSION")).lower()


def choose_blanker(session: str | None = None) -> Blanker | None:
    """The method that will work on this session, or ``None``."""
    session = session or detect_session()
    for blanker in BLANKERS:
        if blanker.session == session and blanker.available:
            return blanker
    return None


@dataclass(frozen=True)
class BlankResult:
    """What happened, in enough detail to act on.

    The old function returned ``None`` whatever happened, which is why a
    screen that stayed on produced no message: there was nothing to
    report and nowhere to report it.
    """

    ok: bool
    method: str = ""
    reason: str = ""
    hint: str = ""

    def describe(self) -> str:
        if self.ok:
            return f"via {self.method}"
        return f"{self.reason}{(' ' + self.hint) if self.hint else ''}"


def _run(blanker: Blanker, args: tuple[str, ...]) -> tuple[bool, str]:
    """Run one command. Returns ``(ok, stderr)`` — neither is discarded."""
    try:
        done = subprocess.run(
            [blanker.binary, *args],
            capture_output=True, text=True, timeout=COMMAND_TIMEOUT,
        )
    except FileNotFoundError:
        return False, f"{blanker.binary} is not installed"
    except subprocess.TimeoutExpired:
        return False, f"{blanker.binary} did not answer within {COMMAND_TIMEOUT:.0f}s"
    except OSError as exc:
        return False, f"{blanker.binary}: {exc}"
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip().splitlines()
        return False, (
            f"{blanker.binary} exited {done.returncode}"
            + (f": {detail[0]}" if detail else "")
        )
    return True, ""


def dpms_enabled() -> bool | None:
    """Whether X11's DPMS extension is on, or ``None`` if unknown.

    Read before enabling it so :func:`set_monitor_state` can put it back
    the way it found it. Someone who disabled DPMS did so deliberately —
    usually to stop a machine blanking during a presentation — and a
    tool that leaves it enabled has broken something on its way out.
    """
    if not shutil.which("xset") or not os.environ.get("DISPLAY"):
        return None
    try:
        done = subprocess.run(
            ["xset", "q"], capture_output=True, text=True, timeout=COMMAND_TIMEOUT  # nosec B607 - found by shutil.which above
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    text = done.stdout
    if "DPMS is Enabled" in text:
        return True
    if "DPMS is Disabled" in text:
        return False
    return None


def _no_method(session: str) -> BlankResult:
    """The refusal, naming which of the four cases this is."""
    if session == "none":
        return BlankResult(
            False,
            reason="No graphical session here — no DISPLAY and no WAYLAND_DISPLAY.",
            hint="There is no screen to blank from a TTY or over SSH.",
        )
    wanted = [b for b in BLANKERS if b.session == session]
    installs = [b.install for b in wanted if b.install]
    return BlankResult(
        False,
        reason=f"No way to blank the screen on this {session} session.",
        hint=" ".join(dict.fromkeys(installs)),
    )


def set_monitor_state(
    state: str,
    *,
    blanker: Blanker | None = None,
    restore_dpms: bool | None = None,
) -> BlankResult:
    """Turn the monitor ``"off"`` or ``"on"``. Says whether it worked.

    *restore_dpms* is what :func:`dpms_enabled` returned before the
    screen was blanked; passing ``False`` to the ``"on"`` call disables
    DPMS again on the way out, leaving the session as it was found.
    """
    if state not in ("on", "off"):
        raise ValueError(f"state is 'on' or 'off', not {state!r}")

    session = detect_session()
    blanker = blanker or choose_blanker(session)
    if blanker is None:
        return _no_method(session)

    if state == "off":
        for preparation in blanker.before:
            # Best effort by design: `xset +dpms` on a server where DPMS
            # is already enabled is a no-op that returns 0, and on one
            # built without the extension it fails and `force off` will
            # fail too, with a better message.
            _run(blanker, preparation)

    args = blanker.off if state == "off" else blanker.on
    if args:
        ok, detail = _run(blanker, args)
        if not ok:
            return BlankResult(
                False, method=blanker.name,
                reason=f"Could not turn the monitor {state}: {detail}.",
                hint=blanker.install,
            )

    if state == "on" and restore_dpms is False and blanker.name == "xset":
        _run(blanker, ("-dpms",))

    return BlankResult(True, method=blanker.name)



# ---------------------------------------------------------------------------
# Keeping the session unlocked while the screen is dark
# ---------------------------------------------------------------------------

#: gsettings keys that lock the session when the screensaver comes on.
_GSETTINGS_LOCKS = (
    ("org.gnome.desktop.screensaver", "lock-enabled"),
    ("org.cinnamon.desktop.screensaver", "lock-enabled"),
    ("org.mate.screensaver", "lock-enabled"),
)


def _out(argv: list[str]) -> str | None:
    """stdout of a command found on PATH, or None if it is missing or failed."""
    exe = shutil.which(argv[0])
    if not exe:
        return None
    try:
        done = subprocess.run([exe, *argv[1:]], capture_output=True, text=True, timeout=COMMAND_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


@dataclass
class LockGuard:
    """Turn lock-on-idle off while ``prot`` holds the screen dark, then put it back.

    A dark screen is an idle session, and an idle session locks: after
    GNOME's idle delay, KDE's Autolock, or the X screensaver's timeout.
    So the screen-off itself was not the only thing that locked; waiting
    long enough did too. Each setting changed is recorded with the value
    it had, in memory and in a state file, so a crash can be undone with
    ``hnx prot restore``. Settings that were already off are not touched.
    """

    state_file: str = ""
    changed: list[tuple[str, list[str]]] = field(default_factory=list)

    def _record(self, what: str, restore: list[str]) -> None:
        self.changed.append((what, restore))
        self._save()

    def _save(self) -> None:
        if not self.state_file:
            return
        import json
        from pathlib import Path

        path = Path(self.state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([{"what": w, "restore": r} for w, r in self.changed]), encoding="utf-8")

    def disable(self) -> list[str]:
        """Switch off what would lock the session. Returns what was changed."""
        for schema, key in _GSETTINGS_LOCKS:
            if _out(["gsettings", "get", schema, key]) == "true" and \
                    _out(["gsettings", "set", schema, key, "false"]) is not None:
                self._record(f"{schema} {key}", ["gsettings", "set", schema, key, "true"])
        for tool in ("kreadconfig6", "kreadconfig5"):
            value = _out([tool, "--file", "kscreenlockerrc", "--group", "Daemon", "--key", "Autolock"])
            if value is None:
                continue
            if value.lower() != "false":
                writer = tool.replace("read", "write")
                if _out([writer, "--file", "kscreenlockerrc", "--group", "Daemon",
                         "--key", "Autolock", "false"]) is not None:
                    self._record("KDE Autolock", [writer, "--file", "kscreenlockerrc", "--group", "Daemon",
                                                  "--key", "Autolock", "--delete"])
                    _reload_kde_locker()
            break
        xfce = _out(["xfconf-query", "-c", "xfce4-screensaver", "-p", "/lock/enabled"])
        if xfce == "true" and _out(["xfconf-query", "-c", "xfce4-screensaver", "-p", "/lock/enabled",
                                    "-s", "false"]) is not None:
            self._record("xfce4-screensaver lock", ["xfconf-query", "-c", "xfce4-screensaver", "-p",
                                                    "/lock/enabled", "-s", "true"])
        if os.environ.get("DISPLAY"):
            timeout = _xset_screensaver_timeout()
            if timeout and _out(["xset", "s", "off"]) is not None:
                self._record("X screensaver", ["xset", "s", str(timeout[0]), str(timeout[1])])
        return [what for what, _ in self.changed]

    def restore(self) -> list[str]:
        """Put back everything :meth:`disable` changed, newest first."""
        restored = []
        for what, argv in reversed(self.changed):
            if _out(argv) is not None:
                restored.append(what)
            if what == "KDE Autolock":
                _reload_kde_locker()
        self.changed = []
        if self.state_file:
            from pathlib import Path

            Path(self.state_file).unlink(missing_ok=True)
        return restored

    @classmethod
    def from_state(cls, state_file: str) -> LockGuard:
        """The guard a crashed ``prot`` left behind, to restore."""
        import json
        from pathlib import Path

        guard = cls(state_file=state_file)
        path = Path(state_file)
        if path.is_file():
            try:
                guard.changed = [(row["what"], list(row["restore"]))
                                 for row in json.loads(path.read_text(encoding="utf-8"))]
            except (ValueError, KeyError, TypeError):
                guard.changed = []
        return guard


def _reload_kde_locker() -> None:
    _out(["dbus-send", "--session", "--dest=org.freedesktop.ScreenSaver", "--type=method_call",
          "/ScreenSaver", "org.kde.screensaver.configure"])


def _xset_screensaver_timeout() -> tuple[int, int] | None:
    """X's screensaver (timeout, cycle) when it is on, else None."""
    import re

    text = _out(["xset", "q"]) or ""
    match = re.search(r"timeout:\s*(\d+)\s+cycle:\s*(\d+)", text)
    if not match or match.group(1) == "0":
        return None
    return int(match.group(1)), int(match.group(2))


def macos_locks_on_sleep() -> bool | None:
    """Whether a display sleep on this Mac locks it straight away (None: unknown).

    ``sysadminctl -screenLock status`` prints to stderr, so it is read
    from both streams.
    """
    exe = shutil.which("sysadminctl")
    if not exe:
        return None
    try:
        done = subprocess.run([exe, "-screenLock", "status"], capture_output=True, text=True,
                              timeout=COMMAND_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (done.stdout + done.stderr).lower()
    if "screenlock is off" in text:
        return False
    if "immediate" in text or "delay is 0" in text:
        return True
    return None
