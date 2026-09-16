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
``check=False``, two ``DEVNULL``\s and an ``except Exception: pass`` on
top, as ``protect`` had, and there was no way to find out.

Enabling DPMS first fixes it, and the fix has to be in one place,
because a second copy is a second thing to not fix. What is here:

* the DPMS enable, and putting it back the way it was found;
* a method per session type rather than per platform — a Wayland
  session has no X server for ``xset`` to ask, and "Linux" is not the
  question;
* Hyprland, sway, wlroots and the freedesktop screensaver, because
  "Wayland" is not one thing either;
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
from dataclasses import dataclass

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

    @property
    def available(self) -> bool:
        if not shutil.which(self.binary):
            return False
        if self.env and not os.environ.get(self.env):
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
    Blanker(
        name="wlopm", session="wayland", binary="wlopm",
        off=("--off", "*"), on=("--on", "*"),
        env="WAYLAND_DISPLAY",
        install="Install wlopm (wlroots compositors: sway, river, Wayfire).",
    ),
    # GNOME and anything else implementing the freedesktop screensaver
    # interface. Activating the screensaver is not a DPMS off, but it
    # does blank the screen, which is what was asked for, and it is the
    # only thing that works on a stock GNOME Wayland session.
    Blanker(
        name="screensaver", session="wayland", binary="dbus-send",
        off=("--session", "--dest=org.freedesktop.ScreenSaver",
             "--type=method_call", "/org/freedesktop/ScreenSaver",
             "org.freedesktop.ScreenSaver.SetActive", "boolean:true"),
        on=("--session", "--dest=org.freedesktop.ScreenSaver",
            "--type=method_call", "/org/freedesktop/ScreenSaver",
            "org.freedesktop.ScreenSaver.SetActive", "boolean:false"),
        env="WAYLAND_DISPLAY",
        install="Install dbus (dbus-send) for the GNOME/KDE screensaver path.",
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
            ["xset", "q"], capture_output=True, text=True, timeout=COMMAND_TIMEOUT
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

