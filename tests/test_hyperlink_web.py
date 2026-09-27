"""HyperLink on the web, on port 37965 (0.72.6.post1).

The T1 server hosts a site that works like the HyperLink app, bound to
127.0.0.1 and this machine's Tailscale addresses and nothing else. These
hold the listener to those addresses, the page to its security headers
and its escaping, and the site to sharing the API's origin and its
authentication.
"""
from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import urllib.request
from pathlib import Path

import pytest
from conftest import clear_t1_config

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient  # noqa: E402

from hypernix.t1api import hyperlink_web  # noqa: E402

WEB = Path(hyperlink_web.__file__).resolve().parent.parent / "hyperlink" / "web"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    clear_t1_config(monkeypatch)
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1.sqlite3"))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def api_app(monkeypatch):
    from hypernix.t1api.app import create_app

    monkeypatch.setenv("T1_TRUSTED_NETWORK", "1")
    return create_app()


class TestThePageAndTheApiShareAnOrigin:
    def test_the_page_is_served_with_its_headers(self, api_app):
        client = TestClient(hyperlink_web.build_web_app(api_app), client=("127.0.0.1", 5000))
        page = client.get("/")
        assert page.status_code == 200 and "<title>HyperLink</title>" in page.text
        policy = page.headers["content-security-policy"]
        assert "script-src 'self'" in policy and "frame-ancestors 'none'" in policy
        assert page.headers["x-frame-options"] == "DENY"
        for path, kind in (("/web/app.js", "javascript"), ("/web/app.css", "css"),
                           ("/web/icon.svg", "svg")):
            response = client.get(path)
            assert response.status_code == 200 and kind in response.headers["content-type"]

    def test_everything_else_is_the_t1_api(self, api_app):
        client = TestClient(hyperlink_web.build_web_app(api_app), client=("127.0.0.1", 5000))
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert client.get("/hyperlink/sessions").status_code == 200  # keyless on loopback here

    def test_no_request_path_becomes_a_file_path(self, api_app):
        client = TestClient(hyperlink_web.build_web_app(api_app), client=("127.0.0.1", 5000))
        for path in ("/web/../../pyproject.toml", "/web/app.js/../index.html", "/web/secret.txt"):
            response = client.get(path)
            assert response.status_code == 404, path
            assert "[project]" not in response.text


class TestWhereItListens:
    def test_only_tailnet_addresses_are_taken(self, monkeypatch):
        monkeypatch.setattr(hyperlink_web.shutil, "which", lambda name: "/usr/bin/tailscale")
        monkeypatch.setattr(hyperlink_web.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
            a[0], 0, stdout="100.101.102.103\n192.168.1.5\nfd7a:115c:a1e0::1\n8.8.8.8\n", stderr=""))
        assert hyperlink_web.tailnet_addresses() == ["100.101.102.103", "fd7a:115c:a1e0::1"]

    def test_no_tailscale_is_no_tailnet(self, monkeypatch):
        monkeypatch.setattr(hyperlink_web.shutil, "which", lambda name: None)

        class Psutil:
            @staticmethod
            def net_if_addrs():
                class Entry:
                    def __init__(self, address):
                        self.address = address
                return {"eth0": [Entry("192.168.1.5")], "lo": [Entry("127.0.0.1")]}

        monkeypatch.setitem(__import__("sys").modules, "psutil", Psutil)
        monkeypatch.setattr(hyperlink_web, "_tailscale_binary", lambda: "")
        # With no Tailscale the route to 100.100.100.100 is the LAN's.
        monkeypatch.setattr(hyperlink_web, "_routed_addresses", lambda: ["192.168.1.5"])
        assert hyperlink_web.tailnet_addresses() == []

    def test_the_route_finds_it_without_the_command_or_psutil(self, monkeypatch):
        """0.72.6.post2: the t1api extra has no psutil, and a systemd unit's
        PATH, or a Mac with Tailscale from the App Store, has no `tailscale`,
        so the site listened on 127.0.0.1 alone and a phone was refused."""
        monkeypatch.setattr(hyperlink_web, "_tailscale_binary", lambda: "")
        monkeypatch.setitem(__import__("sys").modules, "psutil", None)
        asked = []

        class Probe:
            def __init__(self, family, kind):
                self.family = family

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def connect(self, address):
                asked.append(address)
                if self.family == socket.AF_INET6:
                    raise OSError("no IPv6 route")

            def getsockname(self):
                return ("100.101.102.103", 50000)

        monkeypatch.setattr(hyperlink_web.socket, "socket", Probe)
        assert hyperlink_web.tailnet_addresses() == ["100.101.102.103"]
        assert ("100.100.100.100", 53) in asked  # Tailscale's own address, asked of the kernel

    def test_a_minimal_path_still_finds_the_command(self, monkeypatch):
        import os

        monkeypatch.setattr(hyperlink_web.shutil, "which", lambda name: None)
        monkeypatch.setattr(os.path, "exists", lambda path: path == "/usr/bin/tailscale")
        ran = []

        def run(argv, **kwargs):
            ran.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout="100.64.1.2\n", stderr="")

        monkeypatch.setattr(hyperlink_web.subprocess, "run", run)
        assert hyperlink_web.tailnet_addresses() == ["100.64.1.2"]
        assert ran == [["/usr/bin/tailscale", "ip"]]

    def test_it_says_where_it_is_in_the_log(self, api_app, monkeypatch, capsys):
        monkeypatch.setattr(hyperlink_web, "tailnet_addresses", lambda: [])
        port = _free_port()
        listener = hyperlink_web.WebListener(api_app, port=port)
        listener.start()
        try:
            err = capsys.readouterr().err
            assert f"HyperLink on the web: http://127.0.0.1:{port}" in err
            assert "No Tailscale address on this machine" in err
        finally:
            listener.stop()

    def test_status_says_whether_it_answers(self, api_app, monkeypatch):
        monkeypatch.setattr(hyperlink_web, "tailnet_addresses", lambda: [])
        assert hyperlink_web.status_lines(False, 37965) == ["web       off (T1_WEB_ENABLED=0)"]
        port = _free_port()
        assert "NOT answering" in hyperlink_web.status_lines(True, port)[0]
        listener = hyperlink_web.WebListener(api_app, port=port)
        listener.start()
        try:
            import time

            for _ in range(50):
                lines = hyperlink_web.status_lines(True, port)
                if lines[0].endswith(" answering"):
                    break
                time.sleep(0.1)
            assert lines[0] == f"web       http://127.0.0.1:{port}/  answering"
            assert "only this machine" in lines[1]
        finally:
            listener.stop()

    def test_it_serves_on_loopback_and_never_everywhere(self, api_app, monkeypatch):
        monkeypatch.setattr(hyperlink_web, "tailnet_addresses", lambda: [])
        port = _free_port()
        listener = hyperlink_web.WebListener(api_app, port=port)
        listener.start()
        try:
            assert listener.addresses == ["127.0.0.1"]
            assert "0.0.0.0" not in listener.urls()[0]
            import time

            for _ in range(50):
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as page:
                        assert b"HyperLink" in page.read()
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                pytest.fail("the web site never answered")
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as health:
                assert json.loads(health.read())["status"] == "ok"
        finally:
            listener.stop()

    def test_it_is_on_by_default_on_37965(self, monkeypatch):
        from hypernix.t1api.config import T1APIConfig

        monkeypatch.delenv("T1_WEB_ENABLED", raising=False)
        monkeypatch.delenv("T1_WEB_PORT", raising=False)
        config = T1APIConfig(token_secret="x" * 40)
        assert config.web_enabled is True and config.web_port == 37965

    def test_the_server_starts_and_stops_it(self, monkeypatch):
        from hypernix.t1api.app import create_app

        monkeypatch.setattr(hyperlink_web, "tailnet_addresses", lambda: [])
        monkeypatch.setenv("T1_WEB_ENABLED", "1")
        monkeypatch.setenv("T1_WEB_PORT", str(_free_port()))
        app = create_app()
        with TestClient(app):
            assert app.state.t1_web is not None and app.state.t1_web.addresses == ["127.0.0.1"]
        assert app.state.t1_web._server is None

    def test_off_means_off(self, monkeypatch):
        from hypernix.t1api.app import create_app

        monkeypatch.setenv("T1_WEB_ENABLED", "0")
        assert create_app().state.t1_web is None


class TestThePageItself:
    def test_it_is_dark(self):
        """Dark whatever the system is set to (asked for in 0.72.6.post1)."""
        css = (WEB / "app.css").read_text()
        root = css[css.index(":root {"):css.index("}", css.index(":root {"))]
        assert "color-scheme: dark" in root and "--bg: #0e0e13" in root
        assert "prefers-color-scheme: light" not in css
        assert '<meta name="color-scheme" content="dark">' in (WEB / "index.html").read_text()

    def test_a_failed_reply_keeps_its_error_on_screen(self):
        """0.72.6.post2: the page redrew the chat from the server after a
        reply, which wiped the error, so a message with nothing to answer
        it (no model loaded, LM Studio not running) showed nothing at all."""
        js = (WEB / "app.js").read_text()
        send = js[js.index("async function send"):js.index("async function stop")]
        redraw = send.index("await openSession(sessionId)")
        assert "if (failed)" in send[redraw:], "the error is re-shown after the redraw"
        assert "metadata.error" in js  # and drawn from the server's copy
        assert "Runner tab" in js
        css = (WEB / "app.css").read_text()
        assert ".meta.error { color: var(--danger)" in css

    def test_no_inline_script_so_the_policy_can_forbid_it(self):
        html = (WEB / "index.html").read_text()
        assert re.findall(r"<script(?![^>]*\bsrc=)", html) == []
        assert "onclick=" not in html and "javascript:" not in html

    def test_model_text_only_reaches_innerhtml_through_render(self):
        js = (WEB / "app.js").read_text()
        for match in re.finditer(r"\.innerHTML\s*=\s*([^;]+);", js):
            assert match.group(1).strip().startswith("render("), match.group(0)

    @pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
    def test_render_escapes_what_a_model_writes(self):
        js = (WEB / "app.js").read_text()
        functions = js[js.index("function escapeHtml"):js.index("function shortModel")]
        script = functions + """
const cases = [
  render('<script>alert(1)</script>'),
  render('<img src=x onerror=alert(1)>'),
  render('see https://example.com/a?b=1&c="x" now'),
  render('```\\n<b>code</b>\\n```'),
  render('[x](javascript:alert(1))'),
];
console.log(JSON.stringify(cases));
"""
        done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=30)
        assert done.returncode == 0, done.stderr
        out = json.loads(done.stdout)
        assert "<script" not in out[0] and "&lt;script&gt;" in out[0]
        assert "<img" not in out[1]
        # The link stops at the quote, and no entity is cut in half.
        assert '<a href="https://example.com/a?b=1&amp;c=" target' in out[2]
        assert "&quot;x&quot; now" in out[2]
        assert "<pre><code>&lt;b&gt;code&lt;/b&gt;</code></pre>" in out[3]
        assert 'href="javascript' not in out[4]


class TestAKeylessCallerGetsTheReadOnlyTools:
    """The page on a trusted network signs in without a key. Its model was
    offered no T1 tools at all, so it could not answer "which version is
    this server"; it now gets the ones that only read."""

    def _request(self, headers=()):
        from starlette.requests import Request

        return Request({"type": "http", "method": "POST", "path": "/", "root_path": "",
                        "scheme": "http", "query_string": b"", "headers": list(headers),
                        "server": ("127.0.0.1", 37965)})

    def test_keyless_is_read_only(self):
        from hypernix.t1api.routers.hyperlink import _t1_access

        class Principal:
            scopes = ("read", "write")
            is_admin = False

        access = _t1_access(self._request(), Principal())
        assert access is not None and access.token == "" and access.allow_mutating is False
        assert access.base_url == "http://127.0.0.1:37965"

    def test_with_a_key_it_is_the_callers_own(self):
        from hypernix.t1api.routers.hyperlink import _t1_access

        class Principal:
            scopes = ("read", "write")
            is_admin = False

        access = _t1_access(self._request([(b"authorization", b"Bearer T1_abc")]), Principal())
        assert access.token == "T1_abc" and access.allow_mutating is True
