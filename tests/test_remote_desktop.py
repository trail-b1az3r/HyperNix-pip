"""cctvtop's Remote Desktop panel, and the five bugs it had.

The old ``ensure_vnc()`` reported what it *intended* rather than what
happened, and four of its five bugs were that same mistake in different
places. Each one has a test here, named for the failure rather than for
the function, because "cctvtop's Remote Desktop is broken" was never a
crash — it was a panel confidently displaying a connect address that
refused every connection.

The fixtures serve a real socket. A VNC server's first act is to send
twelve bytes of ``RFB xxx.yyy\\n`` before the client says anything, so a
fake that sends exactly that is indistinguishable from the real thing
for everything this module does — and testing against a real listener is
the only way to check a function whose entire job is "is something
actually listening".
"""
from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import pytest

from hypernix.monitoring import remote_desktop as rd


@pytest.fixture(autouse=True)
def _no_cache():
    """status() caches for a few seconds, which two tests in a row would
    otherwise share."""
    rd._CACHE = None
    yield
    rd._CACHE = None


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _Listener:
    """A socket that answers with *greeting* and closes."""

    def __init__(self, port: int, greeting: bytes) -> None:
        self.port = port
        self.greeting = greeting
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> _Listener:
        self._thread.start()
        # Wait for the bind rather than sleeping a guessed interval.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return self
            except OSError:
                time.sleep(0.01)
        raise RuntimeError("the fixture listener never came up")

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _serve(self) -> None:
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", self.port))
        server.listen(8)
        server.settimeout(0.1)
        while not self._stop.is_set():
            try:
                client, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                if self.greeting:
                    client.sendall(self.greeting)
            except OSError:
                pass
            client.close()
        server.close()


def _vnc(port: int) -> _Listener:
    return _Listener(port, b"RFB 003.008\n")


class TestBugOneFalseRunning:
    """The headline bug: "RUNNING" for a server that had failed to start.

    x11vnc was launched with ``-bg``, so it forks and the parent exits 0
    whether or not a port was ever bound. On any machine without an X
    display — every Wayland desktop, every headless box, every container
    — it printed ``cannot open display``, died, and the panel said
    RUNNING with an address that refused every connection.
    """

    def test_a_server_that_died_is_not_running(self, tmp_path, monkeypatch):
        binary = tmp_path / "x11vnc"
        binary.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = '-storepasswd' ]; then : > \"$3\"; exit 0; fi\n"
            "echo 'x11vnc: cannot open display \":0\"' >&2\n"
            "exit 1\n"
        )
        binary.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
        monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

        with pytest.raises(rd.RemoteDesktopError) as caught:
            rd.start(port=_free_port(), timeout=3.0)
        message = str(caught.value)
        assert "cannot open display" in message
        # And it explains the fix rather than just quoting the error.
        assert "wayvnc" in message

    def test_running_means_a_completed_handshake(self):
        """Not pgrep, not an exit status. A connection, and the RFB
        version string the protocol opens with."""
        port = _free_port()
        with _vnc(port):
            found, protocol = rd.probe(port)
            assert found == port
            assert protocol == "003.008"

    def test_a_listener_that_is_not_vnc_is_not_a_vnc_server(self):
        """The old check was `pgrep -x x11vnc`, which cannot tell the
        difference between a VNC server and anything else on the port."""
        port = _free_port()
        with _Listener(port, b"HTTP/1.1 200 OK\r\n"):
            assert rd.probe(port) == (0, "")

    def test_a_silent_listener_is_not_one_either(self):
        """A socket that accepts and says nothing is a half-open
        connection, not a VNC server."""
        port = _free_port()
        with _Listener(port, b""):
            assert rd.probe(port) == (0, "")

    def test_nothing_listening_is_not_running(self):
        assert rd.probe(_free_port()) == (0, "")


class TestBugTwoMissingTailscale:
    """A missing ``tailscale`` took out the whole panel.

    The ``tailscale ip -4`` call lived in the same ``try`` as everything
    else, so FileNotFoundError on a machine without Tailscale — which is
    most machines — replaced a perfectly good VNC status with
    ``Error: [Errno 2] No such file or directory: 'tailscale'``.
    """

    def test_no_tailscale_still_gives_a_status(self, monkeypatch):
        monkeypatch.setattr(rd.shutil, "which",
                            lambda name: None if name == "tailscale" else "/usr/bin/" + name)
        port = _free_port()
        with _vnc(port):
            state = rd.status(port=port, force=True)
        assert state.running
        assert state.addresses

    def test_a_tailscale_that_errors_is_just_no_address(self, monkeypatch):
        monkeypatch.setattr(rd.shutil, "which", lambda _name: "/usr/bin/tailscale")

        def _fail(*_args, **_kwargs):
            raise OSError("tailscaled is not running")

        monkeypatch.setattr(rd.subprocess, "run", _fail)
        assert rd._tailscale_ip() == ""

    def test_a_tailscale_with_two_addresses_gives_one(self, monkeypatch):
        """`-4` does not always mean one line: a machine in two tailnets
        gets two, and the old `.strip()` kept both plus the newline
        between them, producing `100.x\\nfd7a:…:5900`."""
        monkeypatch.setattr(rd.shutil, "which", lambda _name: "/usr/bin/tailscale")

        class _Result:
            returncode = 0
            stdout = "100.64.0.1\n100.64.0.2\n"

        monkeypatch.setattr(rd.subprocess, "run", lambda *a, **k: _Result())
        assert rd._tailscale_ip() == "100.64.0.1"


class TestBugThreeStaleStatus:
    """The status was read once, in a constructor, and never again."""

    def test_the_status_follows_reality(self):
        port = _free_port()
        with _vnc(port):
            assert rd.status(port=port, force=True).running
        assert not rd.status(port=port, force=True).running

    def test_it_is_cached_but_not_forever(self, monkeypatch):
        """A TCP connect on every one of a 1 Hz dashboard's redraws is
        wasteful; a status that is minutes stale is the bug."""
        calls = []

        def _counted(port=None, **kwargs):
            calls.append(port)
            return (0, "")

        monkeypatch.setattr(rd, "probe", _counted)
        rd.status(force=True)
        rd.status()
        rd.status()
        assert len(calls) == 1
        assert rd.CACHE_SECONDS < 60


class TestBugFourTheHardcodedPort:
    """5900 was hardcoded, and x11vnc takes the next free port when 5900
    is busy — which is exactly the case where you wanted a second
    display."""

    def test_the_scan_finds_a_server_above_5900(self, monkeypatch):
        monkeypatch.setattr(rd, "DEFAULT_PORT", _free_port() - 2)
        port = rd.DEFAULT_PORT + 2
        with _vnc(port):
            found, _protocol = rd.probe()
        assert found == port

    def test_the_reported_port_is_the_real_one(self, monkeypatch):
        monkeypatch.setattr(rd, "DEFAULT_PORT", _free_port() - 1)
        port = rd.DEFAULT_PORT + 1
        with _vnc(port):
            state = rd.status(force=True)
        assert state.port == port
        assert all(str(port) in address for address in state.addresses[:1])


class TestBugFiveNoPasswordOnEveryInterface:
    """Starting cctvtop to watch a log also opened an unauthenticated
    remote-control socket for the whole desktop, on 0.0.0.0."""

    def test_nothing_starts_unless_asked(self, monkeypatch):
        """ensure_vnc() used to start a server as a side effect of
        reading the status."""
        started = []
        monkeypatch.setattr(rd.subprocess, "Popen",
                            lambda *a, **k: started.append(a) or (_ for _ in ()).throw(
                                AssertionError("should not have started anything")))
        from hypernix.monitoring.cctvtop import ensure_vnc

        ensure_vnc()
        assert started == []

    def test_localhost_is_the_default_bind(self):
        assert rd._listen_args("x11vnc", "localhost") == ["-localhost"]

    def test_widening_it_is_explicit(self, monkeypatch):
        monkeypatch.setattr(rd, "_tailscale_ip", lambda: "100.64.0.1")
        assert rd._listen_args("x11vnc", "tailscale") == ["-listen", "100.64.0.1"]
        assert rd._listen_args("x11vnc", "all") == []

    def test_tailscale_without_tailscale_is_refused(self, monkeypatch):
        """Rather than silently falling back to every interface, which is
        the opposite of what was asked for."""
        monkeypatch.setattr(rd, "_tailscale_ip", lambda: "")
        with pytest.raises(rd.RemoteDesktopError, match="no Tailscale address"):
            rd._listen_args("x11vnc", "tailscale")

    def test_an_unknown_listen_mode_is_refused(self):
        with pytest.raises(rd.RemoteDesktopError, match="Unknown listen mode"):
            rd._listen_args("x11vnc", "everywhere")

    def test_a_generated_password_is_not_world_readable(self, tmp_path, monkeypatch):
        """Written by a subprocess, so it gets the process umask. A
        world-readable VNC password is the same problem as no password
        with extra steps."""
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        target, plain = rd._ensure_password("wayvnc")
        assert plain and len(plain) <= 8
        assert target.stat().st_mode & 0o077 == 0

    def test_an_existing_password_is_not_replaced(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        first, plain = rd._ensure_password("wayvnc")
        first.write_text("set-by-hand")
        again, second_plain = rd._ensure_password("wayvnc")
        assert again == first
        assert again.read_text() == "set-by-hand"
        # Empty, because the stored form cannot be read back.
        assert second_plain == ""

    def test_the_status_says_when_there_is_no_password(self):
        state = rd.Status(running=True, port=5900, password_protected=False,
                          addresses=["localhost:5900"])
        assert "no password" in state.describe().lower()


class TestSessionDetection:
    """Getting this wrong is what lets bug 1 back in: under Wayland there
    is no X11 screen for x11vnc to scrape."""

    @pytest.mark.parametrize("declared", ["x11", "wayland"])
    def test_the_declared_type_wins(self, monkeypatch, declared):
        monkeypatch.setenv("XDG_SESSION_TYPE", declared)
        assert rd.detect_session() == declared

    def test_wayland_display_without_a_declaration(self, monkeypatch):
        monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
        assert rd.detect_session() == "wayland"

    def test_x_display_without_a_declaration(self, monkeypatch):
        monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        assert rd.detect_session() == "x11"

    def test_headless_is_none(self, monkeypatch):
        for name in ("XDG_SESSION_TYPE", "WAYLAND_DISPLAY", "DISPLAY"):
            monkeypatch.delenv(name, raising=False)
        assert rd.detect_session() == "none"

    def test_no_session_refuses_rather_than_trying(self, monkeypatch):
        for name in ("XDG_SESSION_TYPE", "WAYLAND_DISPLAY", "DISPLAY"):
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(rd.RemoteDesktopError, match="No graphical session"):
            rd.start()

    def test_wayland_does_not_reach_for_x11vnc(self, monkeypatch):
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.setattr(
            rd.shutil, "which",
            lambda name: "/usr/bin/x11vnc" if name == "x11vnc" else None,
        )
        assert rd.detect_backend() is None
        with pytest.raises(rd.RemoteDesktopError, match="wayvnc is not installed"):
            rd.start()


class TestTheStatusReport:
    def test_every_failure_says_something_specific(self, monkeypatch):
        """The old panel ended every failure with "(Please install
        x11vnc)" — including the ones where x11vnc was installed and the
        problem was a Wayland session or a missing DISPLAY."""
        for name in ("XDG_SESSION_TYPE", "WAYLAND_DISPLAY", "DISPLAY"):
            monkeypatch.delenv(name, raising=False)
        state = rd.status(force=True)
        assert not state.running
        assert "graphical session" in state.reason
        assert "x11vnc" not in state.hint

    def test_a_localhost_bind_comes_with_the_tunnel_command(self):
        """"localhost:5900" without the tunnel is half an instruction."""
        found = rd.addresses(5900, listen="localhost")
        assert any("ssh -L 5900:localhost:5900" in entry for entry in found)

    def test_tailscale_is_offered_before_the_lan(self, monkeypatch):
        """It is authenticated and encrypted; the LAN address is not."""
        monkeypatch.setattr(rd, "_tailscale_ip", lambda: "100.64.0.1")
        monkeypatch.setattr(rd, "_lan_ip", lambda: "192.168.1.5")
        found = rd.addresses(5900, listen="lan")
        assert found[0] == "100.64.0.1:5900"

    def test_it_serialises(self):
        assert set(rd.Status().to_dict()) >= {"running", "port", "reason", "hint"}

    def test_stop_without_a_server_is_false_not_an_error(self):
        rd._OURS = None
        assert rd.stop() is False


class TestTheLegacyShim:
    """``ensure_vnc`` was importable, so it still is."""

    def test_it_still_returns_two_strings(self):
        from hypernix.monitoring.cctvtop import ensure_vnc

        result = ensure_vnc()
        assert set(result) == {"status", "ip"}

    def test_running_now_means_running(self, monkeypatch):
        from hypernix.monitoring.cctvtop import ensure_vnc

        monkeypatch.setattr(
            rd, "status",
            lambda **_kwargs: rd.Status(running=True, port=5900,
                                        addresses=["localhost:5900"]),
        )
        assert ensure_vnc() == {"status": "Running", "ip": "localhost:5900"}


class TestTheCli:
    def test_status_exits_non_zero_when_nothing_runs(self, monkeypatch, capsys):
        from hypernix.monitoring.cctvtop import cli_main

        monkeypatch.setattr(
            rd, "status",
            lambda **_kwargs: rd.Status(running=False, reason="not running"),
        )
        assert cli_main(["--remote-desktop-status"]) == 1
        assert "not running" in capsys.readouterr().out

    def test_a_failed_start_is_fatal(self, monkeypatch, capsys):
        """Somebody who passed --remote-desktop wants a remote desktop.
        Starting the dashboard anyway with a panel saying "not running"
        is how the previous version hid this exact failure."""
        from hypernix.monitoring.cctvtop import cli_main

        def _refuse(**_kwargs):
            raise rd.RemoteDesktopError("no display here")

        monkeypatch.setattr(rd, "start", _refuse)
        assert cli_main(["--remote-desktop"]) == 2
        assert "no display here" in capsys.readouterr().err

    def test_the_help_documents_the_flags(self, capsys):
        from hypernix.monitoring.cctvtop import cli_main

        assert cli_main(["--help"]) == 0
        out = capsys.readouterr().out
        for flag in ("--remote-desktop", "--remote-desktop-listen",
                     "--remote-desktop-insecure", "--remote-desktop-status"):
            assert flag in out
        # And says what happens without it.
        assert "nothing is started" in out
