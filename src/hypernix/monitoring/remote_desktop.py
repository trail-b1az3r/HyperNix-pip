"""hypernix.monitoring.remote_desktop — cctvtop's VNC panel, working.

What was wrong
--------------
cctvtop's ``ensure_vnc()`` had five bugs, and the interesting thing about
them is that four are the same bug: it reported what it *intended* rather
than what happened.

1. **It reported "Running" for a server that had failed to start.**
   ``x11vnc`` was launched with ``-bg``, so it forks and the parent exits
   immediately — a zero exit status from ``Popen`` means the fork
   happened, not that a port was bound. On a machine with no X display
   (every Wayland desktop, every headless box, every container) x11vnc
   printed ``cannot open display`` and died, and the panel said
   ``RUNNING`` with a connect address that refused every connection. That
   is the whole of "cctvtop's Remote Desktop is broken": it was not
   broken quietly, it was broken while claiming to work.

2. **A missing ``tailscale`` took out the whole panel.** The
   ``tailscale ip -4`` call was inside the same ``try`` as everything
   else, so ``FileNotFoundError`` on a machine without Tailscale — which
   is most machines — replaced a perfectly good VNC status with
   ``Error: [Errno 2] No such file or directory: 'tailscale'``.

3. **The status was read once, in the constructor, and never again.**
   x11vnc could die thirty seconds in and the panel would say RUNNING
   until the dashboard was restarted.

4. **The port was hardcoded to 5900.** x11vnc takes the next free port
   when 5900 is busy, so the address shown was wrong exactly when a
   second display was the reason you wanted it.

5. **``-nopw``, on every interface.** Not a display bug: starting
   ``cctvtop`` to watch a training log also opened an unauthenticated
   remote-control socket for the whole desktop, on 0.0.0.0, without
   saying so.

What this does instead
----------------------
:func:`probe` **connects to the port**. Not ``pgrep``, not an exit
status — a TCP connection and a read of the RFB version string the
protocol opens with. A server that says ``RFB 003.008`` is a server that
will accept the next client; anything else is reported as what it is.

Nothing starts unless :func:`start` is called, and cctvtop only calls it
when asked. A monitoring dashboard should not have side effects on the
machine it is monitoring.

:func:`start` generates a password, writes it 0600, and binds to
localhost. Reaching it from another machine is then an SSH tunnel or
Tailscale, both of which authenticate. ``--insecure`` still exists,
because somebody on an air-gapped lab network has a real reason for it,
and it now says out loud what it is doing rather than being the default.

Wayland
-------
x11vnc cannot serve a Wayland session; it is an X11 screen scraper and
there is no X11 screen. Under Wayland this uses ``wayvnc`` when it is
installed and says which compositors ship it when it is not, instead of
running x11vnc and producing bug 1 again.
"""
from __future__ import annotations

import logging
import os
import secrets
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "RemoteDesktopError",
    "Status",
    "Backend",
    "DEFAULT_PORT",
    "detect_session",
    "detect_backend",
    "probe",
    "start",
    "stop",
    "status",
    "addresses",
    "password_file",
]

#: Where VNC starts looking. x11vnc calls this display 0.
DEFAULT_PORT = 5900

#: How many ports past :data:`DEFAULT_PORT` to probe when looking for a
#: server somebody else started. x11vnc walks upwards from 5900 when the
#: port is busy, so a second display lands on 5901 and the old code's
#: hardcoded 5900 pointed at the wrong one.
PORT_SCAN = 8

#: Seconds to wait for a server to come up before calling it failed.
#: x11vnc binds its port well inside a second; the margin is for a loaded
#: machine, not for a server that is going to work eventually.
START_TIMEOUT = 4.0

#: Seconds a cached :class:`Status` stays fresh. The panel redraws at
#: 1 Hz and a TCP connect per redraw is wasteful, but a status that is
#: minutes stale is the bug this module exists to fix.
CACHE_SECONDS = 3.0


class RemoteDesktopError(Exception):
    """A remote desktop could not be started, stopped or inspected."""


@dataclass(frozen=True)
class Backend:
    """One VNC server program, and what it needs to run."""

    name: str
    #: "x11" or "wayland" — the session type it can serve.
    session: str
    #: Install hint, shown when the binary is missing.
    install: str

    @property
    def available(self) -> bool:
        return bool(shutil.which(self.name))


BACKENDS: tuple[Backend, ...] = (
    Backend("x11vnc", "x11", "apt install x11vnc / dnf install x11vnc"),
    # GNOME and KDE ship their own; wayvnc is the one that works with
    # wlroots compositors (Sway, Hyprland, river) and is the only one
    # that can be driven from a command line like this.
    Backend("wayvnc", "wayland", "apt install wayvnc, or use your desktop's own screen sharing"),
)


@dataclass
class Status:
    """What is actually true about the remote desktop right now."""

    #: True only when a TCP connection was made and the server spoke RFB.
    running: bool = False
    port: int = 0
    #: "x11vnc", "wayvnc", or "" when nothing answered.
    backend: str = ""
    #: The RFB protocol version the server announced, e.g. "003.008".
    protocol: str = ""
    #: "x11", "wayland" or "none".
    session: str = "none"
    #: Whether this process started it, so :func:`stop` knows what it owns.
    ours: bool = False
    #: False when the server was started with -nopw.
    password_protected: bool = True
    #: The interface it is bound to, as passed to :func:`start`.
    listen: str = "localhost"
    #: Where to connect from, best first. Never empty when running.
    addresses: list[str] = field(default_factory=list)
    #: Why it is not running, when it is not. Actionable, not a code.
    reason: str = ""
    #: What to do about *reason*.
    hint: str = ""
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "port": self.port,
            "backend": self.backend,
            "protocol": self.protocol,
            "session": self.session,
            "ours": self.ours,
            "password_protected": self.password_protected,
            "listen": self.listen,
            "addresses": list(self.addresses),
            "reason": self.reason,
            "hint": self.hint,
        }

    def describe(self) -> str:
        if not self.running:
            lines = [f"Remote desktop: {self.reason or 'not running'}"]
            if self.hint:
                lines.append(f"  {self.hint}")
            return "\n".join(lines)
        lines = [
            f"Remote desktop: running on port {self.port}"
            + (f" ({self.backend})" if self.backend else "")
        ]
        for address in self.addresses:
            lines.append(f"  vnc://{address}")
        if not self.password_protected:
            lines.append(
                "  ! No password. Anyone who can reach this port has the "
                "keyboard and mouse."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# What kind of session is this
# ---------------------------------------------------------------------------


def detect_session() -> str:
    """``"x11"``, ``"wayland"`` or ``"none"``.

    ``XDG_SESSION_TYPE`` first because it is the answer when it is set;
    the ``WAYLAND_DISPLAY``/``DISPLAY`` fallback covers the case where it
    is not, which is most login managers older than about 2020.

    Getting this right is what stops bug 1 from coming back: under
    Wayland there is no X11 screen for x11vnc to scrape, and running it
    anyway produces a process that exits immediately and a panel that
    says RUNNING.
    """
    declared = os.environ.get("XDG_SESSION_TYPE", "").strip().lower()
    if declared in ("x11", "wayland"):
        return declared
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "none"


def detect_backend(session: str | None = None) -> Backend | None:
    """The installed server that can serve this session, or ``None``."""
    session = session or detect_session()
    for backend in BACKENDS:
        if backend.session == session and backend.available:
            return backend
    return None


# ---------------------------------------------------------------------------
# Is anything actually listening
# ---------------------------------------------------------------------------


def _rfb_handshake(port: int, host: str = "127.0.0.1", timeout: float = 0.4) -> str:
    """The RFB version a server on *port* announces, or ``""``.

    A VNC server's first act is to send twelve bytes, ``RFB xxx.yyy\\n``,
    before the client says anything. So one connect and one read is a
    complete, unambiguous answer to "will this accept a viewer" — which
    is what the panel is claiming, and what neither ``pgrep`` nor a
    ``Popen`` exit status can tell you.

    The connection is closed without replying. An RFB server treats that
    as a client that changed its mind, which costs it a log line and
    nothing else.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout) as stream:
            stream.settimeout(timeout)
            greeting = stream.recv(12)
    except (TimeoutError, OSError):
        return ""
    if greeting.startswith(b"RFB "):
        return greeting[4:11].decode("ascii", errors="replace")
    return ""


def _owner_of(port: int) -> str:
    """The program listening on *port*, when it can be determined.

    Best effort and never required: it fills in "x11vnc" beside the port
    number, and an empty string is a missing label rather than a failure.
    """
    for command in (
        ["ss", "-lptnH", f"sport = :{port}"],
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
    ):
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=1.0, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        for backend in BACKENDS:
            if backend.name in result.stdout:
                return backend.name
    return ""


def probe(port: int | None = None, *, scan: bool = True) -> tuple[int, str]:
    """Find a live VNC server. Returns ``(port, protocol)``, or ``(0, "")``.

    Scans upward from :data:`DEFAULT_PORT` because x11vnc takes the next
    free port when 5900 is busy — which is exactly the situation where
    the hardcoded 5900 the old code printed was the wrong number.
    """
    if port is not None:
        return ((port, protocol) if (protocol := _rfb_handshake(port)) else (0, ""))
    if not scan:
        return ((DEFAULT_PORT, protocol)
                if (protocol := _rfb_handshake(DEFAULT_PORT)) else (0, ""))
    for candidate in range(DEFAULT_PORT, DEFAULT_PORT + PORT_SCAN):
        protocol = _rfb_handshake(candidate)
        if protocol:
            return candidate, protocol
    return 0, ""


# ---------------------------------------------------------------------------
# Where to connect from
# ---------------------------------------------------------------------------


def _tailscale_ip() -> str:
    """This machine's Tailscale IPv4, or ``""``.

    Its own function with its own error handling, which is the entire
    point: the old code ran this inside the same ``try`` as the VNC
    status, so not having Tailscale installed — the normal case —
    replaced a working status with a FileNotFoundError string.
    """
    if not shutil.which("tailscale"):
        return ""
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=2.0, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    # One address per line, and `-4` does not always mean one line: a
    # machine in two tailnets gets two. The first is the one to show.
    for line in result.stdout.splitlines():
        candidate = line.strip()
        if candidate:
            return candidate
    return ""


def _lan_ip() -> str:
    """This machine's LAN address, without sending anything.

    A UDP socket ``connect``ed to an outside address performs no I/O — it
    only asks the routing table which local interface would be used — so
    this works with no network and tells us the address a machine on the
    same LAN would use.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe_socket:
            probe_socket.settimeout(0.2)
            probe_socket.connect(("192.0.2.1", 9))   # TEST-NET-1, routed nowhere
            return probe_socket.getsockname()[0]
    except OSError:
        return ""


def addresses(port: int, *, listen: str = "localhost") -> list[str]:
    """Where a viewer should point, best first.

    A localhost-bound server is only reachable through a tunnel, so
    saying "localhost:5900" without the tunnel command is half an
    instruction. The SSH line is the other half.
    """
    found = [f"localhost:{port}"]
    if listen == "localhost":
        host = socket.gethostname()
        found.append(f"(tunnel: ssh -L {port}:localhost:{port} {host})")
        return found
    tailscale = _tailscale_ip()
    if tailscale:
        # First when present: it is authenticated and encrypted, which
        # the LAN address is not.
        found.insert(0, f"{tailscale}:{port}")
    if listen in ("lan", "all"):
        lan = _lan_ip()
        if lan and lan != tailscale:
            found.append(f"{lan}:{port}")
    return found


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def password_file() -> Path:
    """Where the generated VNC password lives."""
    root = os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state")
    return Path(root) / "hypernix" / "vncpasswd"


def _ensure_password(backend: str) -> tuple[Path | None, str]:
    """Create a password file if there is not one. Returns ``(path, plain)``.

    ``plain`` is empty when the file already existed — the stored form is
    obfuscated rather than hashed, and recovering it to print would mean
    implementing VNC's DES scramble to hand somebody a password they set
    themselves. They know it; a new one can be had by deleting the file.
    """
    target = password_file()
    if target.exists():
        return target, ""
    plain = secrets.token_urlsafe(9)[:8]     # VNC truncates past 8 characters
    target.parent.mkdir(parents=True, exist_ok=True)
    if backend == "x11vnc":
        try:
            result = subprocess.run(
                ["x11vnc", "-storepasswd", plain, str(target)],
                capture_output=True, text=True, timeout=10.0, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RemoteDesktopError(
                f"Could not store a VNC password: {exc}"
            ) from exc
        if result.returncode != 0 or not target.exists():
            raise RemoteDesktopError(
                "x11vnc -storepasswd failed: "
                + (result.stderr.strip() or "no output")
            )
    else:
        # wayvnc reads a plain-text file. 0600 below is doing the work.
        target.write_text(plain)
    # Before returning, not after: the file is written by a subprocess
    # with the process umask, and a world-readable VNC password is the
    # same problem as no password with extra steps.
    target.chmod(0o600)
    return target, plain


# ---------------------------------------------------------------------------
# Starting and stopping
# ---------------------------------------------------------------------------

#: The process this module started, so :func:`stop` only stops its own.
_OURS: subprocess.Popen | None = None
_OUR_PORT: int = 0
_CACHE: tuple[float, Status] | None = None


def _listen_args(backend: str, listen: str) -> list[str]:
    if backend != "x11vnc":
        return []
    if listen == "localhost":
        return ["-localhost"]
    if listen == "tailscale":
        address = _tailscale_ip()
        if not address:
            raise RemoteDesktopError(
                "listen='tailscale' but this machine has no Tailscale address. "
                "Run `tailscale up`, or use listen='localhost' and an SSH tunnel."
            )
        return ["-listen", address]
    if listen in ("lan", "all"):
        return []
    raise RemoteDesktopError(
        f"Unknown listen mode {listen!r}. One of: localhost, tailscale, lan, all."
    )


def start(
    *,
    port: int | None = None,
    listen: str = "localhost",
    insecure: bool = False,
    view_only: bool = False,
    timeout: float = START_TIMEOUT,
) -> Status:
    """Start a VNC server and **verify** that it came up.

    Returns a :class:`Status` whose ``running`` is true only if a TCP
    connection was made and the server spoke RFB. It does not return
    "started it, probably fine" — that was the bug.

    *listen* defaults to localhost, so reaching it from elsewhere means
    an SSH tunnel or Tailscale, both of which authenticate. *insecure*
    starts it with no password at all, which is a real requirement on
    some isolated networks and is never a default.
    """
    global _OURS, _OUR_PORT, _CACHE
    _CACHE = None

    session = detect_session()
    if session == "none":
        raise RemoteDesktopError(
            "No graphical session here — no DISPLAY and no WAYLAND_DISPLAY. "
            "There is no desktop to serve. On a headless box, start one with "
            "Xvfb and set DISPLAY before running this."
        )
    backend = detect_backend(session)
    if backend is None:
        wanted = next((b for b in BACKENDS if b.session == session), None)
        raise RemoteDesktopError(
            f"This is a {session} session and "
            + (f"{wanted.name} is not installed. {wanted.install}"
               if wanted else "no supported VNC server exists for it.")
        )

    existing_port, protocol = probe(port)
    if existing_port:
        # Somebody else's, or a previous run's. Adopting it beats
        # starting a second server that will land on a different port
        # and confuse the panel.
        return status(port=existing_port, force=True)

    chosen = port or DEFAULT_PORT
    password: Path | None = None
    plain = ""
    if not insecure:
        password, plain = _ensure_password(backend.name)

    if backend.name == "x11vnc":
        command = [
            "x11vnc",
            "-display", os.environ.get("DISPLAY", ":0"),
            "-rfbport", str(chosen),
            "-forever", "-shared", "-q",
            # Not -bg. Backgrounding is why the old code could not tell a
            # server that started from one that died: the parent exits 0
            # either way. Held as a child here, so its exit status and
            # its stderr are both available when the probe fails.
            "-noxdamage",
        ]
        command += _listen_args("x11vnc", listen)
        command += ["-rfbauth", str(password)] if password else ["-nopw"]
        if view_only:
            command.append("-viewonly")
    else:
        command = ["wayvnc", "--render-cursor"]
        if password:
            command += ["--config", str(password)]
        command += ["0.0.0.0" if listen in ("lan", "all") else "127.0.0.1", str(chosen)]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise RemoteDesktopError(f"Could not run {backend.name}: {exc}") from exc

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _rfb_handshake(chosen):
            _OURS, _OUR_PORT = process, chosen
            result = status(port=chosen, force=True)
            result.ours = True
            result.listen = listen
            result.password_protected = password is not None
            if plain:
                # Only on the run that generated it. Printed by the
                # caller, not here -- a library that writes a password to
                # stdout writes it into whatever log is capturing stdout.
                result.hint = f"VNC password: {plain}  (stored in {password})"
            return result
        if process.poll() is not None:
            # It died. This is the case the old code called "Running".
            stderr = ""
            try:
                stderr = (process.stderr.read() or b"").decode(
                    "utf-8", errors="replace"
                ).strip()
            except (OSError, ValueError):
                pass
            detail = stderr.splitlines()[-1] if stderr else f"exit code {process.returncode}"
            extra = ""
            if "cannot open display" in stderr.lower():
                extra = (
                    " x11vnc scrapes an X11 screen and there is not one here. "
                    "Under Wayland, install wayvnc or use the desktop's own "
                    "screen sharing."
                )
            raise RemoteDesktopError(f"{backend.name} exited: {detail}.{extra}")
        time.sleep(0.1)

    process.terminate()
    raise RemoteDesktopError(
        f"{backend.name} did not accept a connection on port {chosen} within "
        f"{timeout:.0f}s. It may still be starting; check with "
        f"`hnx cctvtop --remote-desktop-status`."
    )


def stop() -> bool:
    """Stop the server this process started. Returns whether there was one.

    Deliberately does not kill a server it did not start. Somebody else's
    x11vnc may be carrying somebody else's session, and a monitoring
    dashboard exiting is not a reason to drop it.
    """
    global _OURS, _OUR_PORT, _CACHE
    _CACHE = None
    if _OURS is None:
        return False
    process, _OURS, _OUR_PORT = _OURS, None, 0
    if process.poll() is not None:
        return False
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
    return True


def status(*, port: int | None = None, force: bool = False) -> Status:
    """What is true right now, cached for :data:`CACHE_SECONDS`.

    Called on every dashboard redraw, which is the fix for the third bug:
    the old code read the status once in a constructor, so a server that
    died thirty seconds in showed RUNNING until the dashboard restarted.
    """
    global _CACHE
    now = time.time()
    if not force and _CACHE is not None and now - _CACHE[0] < CACHE_SECONDS:
        return _CACHE[1]

    session = detect_session()
    live_port, protocol = probe(port)
    if live_port:
        result = Status(
            running=True,
            port=live_port,
            protocol=protocol,
            backend=_owner_of(live_port),
            session=session,
            ours=(live_port == _OUR_PORT and _OURS is not None and _OURS.poll() is None),
            addresses=addresses(live_port, listen="all"),
        )
    else:
        backend = detect_backend(session)
        if session == "none":
            reason = "no graphical session (no DISPLAY, no WAYLAND_DISPLAY)"
            hint = "Nothing to serve. On a headless box, start Xvfb first."
        elif backend is None:
            wanted = next((b for b in BACKENDS if b.session == session), None)
            reason = f"no VNC server installed for this {session} session"
            hint = wanted.install if wanted else ""
        else:
            reason = "not running"
            hint = "Start it with `hnx cctvtop --remote-desktop`."
        result = Status(running=False, session=session, reason=reason, hint=hint)

    _CACHE = (now, result)
    return result
