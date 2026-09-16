"""outage — turn the display off during training.

Long training runs don't need the panel lit up.  An ``outage``
blanks the display when the run starts, then *guarantees* the
display comes back on when training:

* finishes successfully,
* raises (KeyboardInterrupt / RuntimeError / OOM / anything),
* explicitly calls :meth:`Outage.restore`.

The restore-on-anything semantic comes from the context-manager
``__exit__`` so a crash mid-training still leaves you with a
working screen.

Quick use::

    from hypernix.outage import Outage

    with Outage():
        train_for_six_hours()       # screen off; comes back when done

Manual control::

    o = Outage().black_out()
    try:
        train_for_six_hours()
    finally:
        o.restore()

Backends come from :mod:`hypernix.system.blanking`, shared with
``hnx prot`` so the two cannot drift apart again:

* Linux X11: ``xset +dpms`` then ``xset dpms force off``, and
  ``... force on`` to wake. The ``+dpms`` matters: ``force off``
  against a disabled DPMS extension is accepted and ignored, and the
  screen stays on.
* Linux Wayland: ``hyprctl``, ``swaymsg``, ``wlopm`` or the
  freedesktop screensaver, whichever the session actually has
* macOS: ``pmset displaysleepnow`` (auto-wake on input)
* Windows: ``SendMessageW(HWND_BROADCAST, WM_SYSCOMMAND,
  SC_MONITORPOWER, 2)`` via ``ctypes.windll``

Each call is wrapped so a missing tool / unsupported environment
just records the failure on the returned :class:`OutageResult`
without raising.  Set ``strict=True`` to escalate to an
exception instead.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from . import blanking


@dataclass
class OutageResult:
    backend: str
    blanked: bool = False
    restored: bool = False
    notes: list[str] = field(default_factory=list)
    error: str | None = None


def _has(name: str) -> bool:
    return shutil.which(name) is not None


def _detect_backend() -> str:
    """The name of the method that will work here.

    Delegated to :mod:`hypernix.system.blanking`, which is also what
    ``hnx prot`` uses. The version that lived here asked
    ``sys.platform`` and then looked for two binaries, which got three
    things wrong that the shared table gets right: it chose ``xset`` on
    a Wayland session that had no ``wlopm`` (there is no X server for
    it to talk to), it never enabled DPMS before forcing it off (the
    request is accepted and ignored, and the screen stays on), and it
    knew nothing about Hyprland, sway or the freedesktop screensaver.

    The strings it returns are unchanged for ``xset``, ``wlopm``,
    ``pmset`` and ``windows``; the new compositors add their own, and
    :meth:`Outage._do_blank` dispatches on the object rather than the
    name so they need no branch here.
    """
    if sys.platform == "win32":
        return "windows"
    chosen = blanking.choose_blanker()
    if chosen is not None:
        return chosen.name
    return "linux-none" if sys.platform.startswith("linux") else "unknown"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _blanker_named(name: str | None) -> blanking.Blanker | None:
    """The shared table's entry for a backend name, or ``None``.

    ``None`` covers both "no backend here" and a name a caller forced
    that this build does not have, and both mean the same thing to
    :class:`Outage`: record a note and skip rather than raise, which is
    the documented behaviour and what ``strict`` deliberately does not
    escalate.
    """
    return next((b for b in blanking.BLANKERS if b.name == name), None)


def _windows_set_monitor(state: int) -> int:
    """state: 2 = off, -1 = on.  Returns rc-like int (0 == ok)."""
    try:
        import ctypes
        # SC_MONITORPOWER = 0xF170, WM_SYSCOMMAND = 0x0112
        # HWND_BROADCAST = 0xFFFF
        return ctypes.windll.user32.SendMessageW(
            0xFFFF, 0x0112, 0xF170, state,
        )
    except Exception:  # noqa: BLE001
        return -1


@dataclass
class Outage:
    """Display blanker / restorer.

    Args:
        backend: Force a specific backend; ``None`` (default) auto-
            detects.  Recognised: ``"xset"``, ``"wlopm"``,
            ``"pmset"``, ``"windows"``, ``"none"`` (no-op).
        strict: Raise ``RuntimeError`` if blanking or restoring
            fails.  Default is to record the failure on
            :attr:`last_result` and keep going.
        on_restore: Optional callable run after the display is
            woken up.
    """

    backend: str | None = None
    strict: bool = False
    on_restore: Callable[[], Any] | None = None
    last_result: OutageResult | None = field(default=None, init=False, repr=False)
    #: What ``xset q`` said about DPMS before the screen was blanked, so
    #: restoring can leave the session as it was found. ``None`` until
    #: something has been blanked, and on every non-X11 backend.
    _dpms_was_enabled: bool | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = _detect_backend()

    # ------------------------------------------------------------------
    # Core ops
    # ------------------------------------------------------------------

    def black_out(self) -> OutageResult:
        """Turn the display off."""
        result = OutageResult(backend=self.backend or "unknown")
        try:
            self._do_blank(result)
            result.blanked = True
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            if self.strict:
                self.last_result = result
                raise
        self.last_result = result
        return result

    def restore(self) -> OutageResult:
        """Turn the display back on.  Always callable, even if
        :meth:`black_out` failed — restore is a best-effort wake."""
        result = OutageResult(backend=self.backend or "unknown")
        try:
            self._do_restore(result)
            result.restored = True
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            if self.strict:
                self.last_result = result
                raise
        if self.on_restore is not None:
            try:
                self.on_restore()
            except Exception as exc:  # noqa: BLE001
                result.notes.append(f"on_restore raised: {exc}")
        self.last_result = result
        return result

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Outage:
        self.black_out()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        # Always restore, even if the body raised.
        self.restore()
        return False  # never suppress the original exception

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------

    def _do_blank(self, result: OutageResult) -> None:
        b = self.backend
        if b == "windows":
            rc = _windows_set_monitor(2)
            result.notes.append(f"windows monitor off rc={rc}")
            return
        blanker = _blanker_named(b)
        if blanker is None:
            result.notes.append(
                f"no display-off backend available (backend={b!r}); skipping",
            )
            return
        # Read before blanking so :meth:`_do_restore` can put a session
        # that had DPMS deliberately disabled back the way it was.
        self._dpms_was_enabled = blanking.dpms_enabled()
        outcome = blanking.set_monitor_state("off", blanker=blanker)
        result.notes.append(f"{blanker.name}: {outcome.describe()}")
        if not outcome.ok:
            raise RuntimeError(outcome.reason)

    def _do_restore(self, result: OutageResult) -> None:
        b = self.backend
        if b == "windows":
            rc = _windows_set_monitor(-1)
            result.notes.append(f"windows monitor on rc={rc}")
            return
        blanker = _blanker_named(b)
        if blanker is None:
            result.notes.append(
                f"no display-on backend available (backend={b!r}); skipping",
            )
            return
        if blanker.name == "pmset":
            # pmset has no explicit wake -- input wakes it. Best effort:
            # ``caffeinate -u -t 1`` nudges the display.
            r = _run(["caffeinate", "-u", "-t", "1"])
            result.notes.append(f"caffeinate -u rc={r.returncode}")
            return
        outcome = blanking.set_monitor_state(
            "on", blanker=blanker, restore_dpms=self._dpms_was_enabled,
        )
        result.notes.append(f"{blanker.name}: {outcome.describe()}")


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def outage(*, backend: str | None = None, strict: bool = False) -> Outage:
    return Outage(backend=backend, strict=strict)


def black_out(*, backend: str | None = None) -> OutageResult:
    """One-shot blank.  Caller is responsible for restoring."""
    return Outage(backend=backend).black_out()


def restore_display(*, backend: str | None = None) -> OutageResult:
    """One-shot wake."""
    return Outage(backend=backend).restore()


def detect_backend() -> str:
    """Return the backend that would be used on this host."""
    return _detect_backend()


def platform_summary() -> dict[str, Any]:
    """Diagnostic helper for bug reports."""
    return {
        "platform": sys.platform,
        "system": platform.system(),
        "backend": _detect_backend(),
        "session": blanking.detect_session(),
        "xset": _has("xset"),
        "wlopm": _has("wlopm"),
        "pmset": _has("pmset"),
        "available": [b.name for b in blanking.BLANKERS if b.available],
    }


__all__ = [
    "Outage",
    "OutageResult",
    "black_out",
    "detect_backend",
    "outage",
    "platform_summary",
    "restore_display",
]
