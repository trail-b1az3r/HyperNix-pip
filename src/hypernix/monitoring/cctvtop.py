"""Python wrapper for the C++ cctvtop dashboard (upgraded).

The Remote Desktop panel is :mod:`hypernix.monitoring.remote_desktop` now.
See that module for what was wrong with the version that lived here —
briefly, it reported "RUNNING" for a server that had failed to start, and
it started one without being asked, with no password, on every interface.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from hypernix.monitoring.tv import LogTail, _autodetect_log
from hypernix.monitoring.tvtop_plus_plus import SPINNERS, Frame, TVTopPlusPlus


def _resolve_cctvtop_log(explicit: Path | str | None = None) -> Path | None:
    """Resolve the training log cctvtop should tail.

    No longer hardcoded to a single path -- this checks, in order:

    1. An explicit path (``--log`` flag / constructor argument).
    2. ``~/checkpoints/train.log`` -- cctvtop's preferred, prioritized
       convention (matches tvtop's autodetect priority).
    3. ``./checkpoints/train.log`` -- the legacy cwd-relative location
       cctvtop used to hardcode unconditionally.
    4. Whatever :func:`hypernix.tv._autodetect_log` finds by scanning
       the current directory for a training-shaped log.

    Returns ``None`` if nothing is found anywhere.
    """
    if explicit is not None:
        return Path(explicit)

    home_default = Path.home() / "checkpoints" / "train.log"
    if home_default.exists():
        return home_default

    cwd_default = Path.cwd() / "checkpoints" / "train.log"
    if cwd_default.exists():
        return cwd_default

    return _autodetect_log()


def ensure_vnc() -> dict[str, str]:
    """The old shape of this function, over the new implementation.

    Kept because it was importable and somebody may have imported it.
    Two behaviours changed and both were bugs:

    * It no longer *starts* anything. Running a dashboard to watch a
      training log should not open a remote-control socket for the whole
      desktop as a side effect. ``cctvtop --remote-desktop`` starts one.
    * ``"Running"`` now means a TCP connection was made and the server
      spoke RFB, not that a fork succeeded. The old version said
      ``"Running"`` for an x11vnc that had printed ``cannot open
      display`` and exited, which is the whole of "cctvtop's Remote
      Desktop is broken".

    New code should use :func:`hypernix.monitoring.remote_desktop.status`,
    which returns all of it rather than two strings.
    """
    from hypernix.monitoring import remote_desktop

    state = remote_desktop.status()
    if state.running:
        return {"status": "Running", "ip": state.addresses[0] if state.addresses else ""}
    return {"status": state.reason or "Not running", "ip": "Unknown"}


class CCTVTop(TVTopPlusPlus):
    def __init__(self, *args, remote_desktop: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # cctvtop no longer forces a hardcoded log path: if the caller
        # already supplied one (e.g. via cli_main's --log handling) it is
        # respected as-is; otherwise it's resolved through the same
        # priority chain as tvtop (~/checkpoints/train.log first).
        if self.log_path is None:
            resolved = _resolve_cctvtop_log(None)
            if resolved is not None:
                self.log_path = resolved
                self.log_tail = LogTail(Path(self.log_path), history_size=8)
        # Whether *this* run was asked to start a server. The status
        # itself is read per frame in _update_layout, not cached here:
        # reading it once in a constructor is why a server that died
        # thirty seconds in showed RUNNING until the dashboard was
        # restarted.
        self.remote_desktop = remote_desktop

    def _init_layout(self) -> Layout:
        layout = super()._init_layout()
        # Replace the right column to include VNC
        layout["right"].split_column(
            Layout(name="hardware", ratio=4),
            Layout(name="gpu", ratio=2),
            Layout(name="vnc", ratio=1)
        )
        return layout

    def _update_layout(self, f: Frame, console: Console, layout: Layout) -> None:
        # Let the parent update training, process, hardware, gpu, loss, log
        super()._update_layout(f, console, layout)
        
        # Override header text for CCTVTop
        spinner_char = SPINNERS[self.tick % len(SPINNERS)] if not self.ascii_only else "*"
        title_text = Text.assemble(
            (" ", "default"),
            (spinner_char, "bright_cyan"),
            ("  ✦ HYPERNIX CCTVTop (VNC+PRO) ✦  ", "bold bright_red"),
            (time.strftime(" %Y-%m-%d %H:%M:%S "), "dim"),
        )
        layout["header"].update(Panel(title_text, style="bold red", padding=(0, 2)))
        
        layout["vnc"].update(Panel(
            self._remote_desktop_text(),
            title="Remote Desktop",
            border_style="blue",
        ))

    def _remote_desktop_text(self) -> Text:
        """The panel, from a status read this frame.

        Every branch says something specific. The old panel ended every
        failure with "(Please install x11vnc)" -- including the failures
        where x11vnc was installed and the problem was a Wayland session,
        a missing DISPLAY, or a port already in use.
        """
        from hypernix.monitoring import remote_desktop as rd

        state = rd.status()
        text = Text()
        text.append("VNC Server: ", style="bold white")
        if state.running:
            text.append("RUNNING", style="bold green")
            if state.backend:
                text.append(f"  {state.backend}", style="dim")
            text.append("\n")
            for address in state.addresses[:2]:
                text.append(f"{address}\n", style="cyan")
            if not state.password_protected:
                # Loud, because it means anyone who can reach the port
                # has the keyboard and mouse.
                text.append("no password set", style="bold red")
            return text

        text.append(f"{state.reason}\n", style="bold red")
        if state.hint:
            text.append(state.hint, style="dim")
        return text


def cli_main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])

    if "--help" in args or "-h" in args:
        print(
            "usage: hnx cctvtop [--log path] [--remote-desktop] [--help]\n\n"
            "A pure-Python training dashboard with hardware metrics and a remote\n"
            "desktop panel. Prioritizes ~/checkpoints/train.log, then falls back to\n"
            "./checkpoints/train.log, then auto-detects a training-shaped log under\n"
            "the current directory. Pass --log <path> to point at something else.\n\n"
            "Remote desktop:\n"
            "  --remote-desktop          start a VNC server, with a password, bound\n"
            "                            to localhost. Reach it over an SSH tunnel\n"
            "                            or Tailscale.\n"
            "  --remote-desktop-listen   localhost (default) | tailscale | lan | all\n"
            "  --remote-desktop-view-only  serve the screen without keyboard/mouse\n"
            "  --remote-desktop-insecure   no password at all. Says so on screen.\n"
            "  --remote-desktop-status   print what is running and exit\n\n"
            "Without --remote-desktop nothing is started: the panel reports what is\n"
            "already there. A dashboard should not open a remote-control socket for\n"
            "your whole desktop as a side effect of watching a log."
        )
        return 0

    from hypernix.monitoring import remote_desktop as rd

    def _take_value(flag: str, default: str) -> str:
        if flag in args:
            index = args.index(flag)
            if index + 1 < len(args):
                value = args[index + 1]
                del args[index:index + 2]
                return value
            del args[index]
        return default

    listen = _take_value("--remote-desktop-listen", "localhost")
    view_only = "--remote-desktop-view-only" in args
    insecure = "--remote-desktop-insecure" in args
    want_rd = "--remote-desktop" in args
    for flag in ("--remote-desktop-view-only", "--remote-desktop-insecure",
                 "--remote-desktop"):
        while flag in args:
            args.remove(flag)

    if "--remote-desktop-status" in args:
        print(rd.status(force=True).describe())
        # Non-zero when nothing is running, so a script can gate on it.
        return 0 if rd.status().running else 1

    explicit_log: Path | None = None
    if "--log" in args:
        i = args.index("--log")
        if i + 1 < len(args):
            explicit_log = Path(args[i + 1])
            del args[i : i + 2]

    if want_rd:
        try:
            state = rd.start(listen=listen, insecure=insecure, view_only=view_only)
        except rd.RemoteDesktopError as exc:
            # Fatal rather than a warning. Somebody who passed
            # --remote-desktop wants a remote desktop, and starting the
            # dashboard anyway with a panel saying "not running" is how
            # the previous version hid this exact failure.
            print(f"cctvtop: could not start the remote desktop: {exc}",
                  file=sys.stderr)
            return 2
        print(state.describe())
        if state.hint:
            print("cctvtop: remote desktop credentials were generated; not displaying sensitive details.")
        if insecure:
            print(
                "cctvtop: no VNC password. Anyone who can reach "
                f"port {state.port} has this desktop's keyboard and mouse.",
                file=sys.stderr,
            )

    log_file = _resolve_cctvtop_log(explicit_log)
    if log_file is None or not log_file.exists():
        home_default = Path.home() / "checkpoints" / "train.log"
        cwd_default = Path.cwd() / "checkpoints" / "train.log"
        print(
            f"Error: no training log found.\n"
            f"Looked for: {explicit_log or '(no --log given)'}, {home_default}, "
            f"{cwd_default}, and an auto-detected *.log under {Path.cwd()}.\n"
            "Pass --log <path> for an explicit location.",
            file=sys.stderr,
        )
        return 1

    app = CCTVTop(log_path=log_file, refresh_seconds=1.0,
                  remote_desktop=want_rd)
    try:
        app.run()
    finally:
        # Only stops a server this process started -- rd.stop() checks.
        # Somebody else's x11vnc may be carrying somebody else's session,
        # and a dashboard exiting is not a reason to drop it.
        if want_rd and rd.stop():
            print("cctvtop: stopped the remote desktop it started.")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
