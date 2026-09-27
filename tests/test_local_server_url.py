"""The command-line clients find the T1 API where `hypernix-t1 start` put it.

Reported against 0.72.6: the server was on 8001 and

    hypernix-t1 runner plan qwen3-8b
    Could not reach the T1 API at http://127.0.0.1:8000/runner/plan:
    [Errno 111] Connection refused

`runner_cli` knew --url, T1_URL and 8000. The shell wrapper had exported
T1_PORT from the server's .env for it, and it never looked.
"""
from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import clear_t1_config

from hypernix.t1api import chat_cli, localserver, runner_cli


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    clear_t1_config(monkeypatch)
    for name in ("T1_URL", "T1_PORT", "T1_HOST", "T1_ADMIN_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("T1_CONFIG_DIR", str(tmp_path / "t1api"))
    (tmp_path / "t1api").mkdir()
    return tmp_path / "t1api"


class TestWhereTheServerIs:
    def test_nothing_configured_is_8000(self):
        assert localserver.configured_url() == ("http://127.0.0.1:8000", "the default port")

    def test_t1_port_from_the_environment(self, monkeypatch):
        """What the shell wrapper exports from the .env before running us."""
        monkeypatch.setenv("T1_PORT", "8001")
        assert localserver.configured_url()[0] == "http://127.0.0.1:8001"

    def test_t1_port_from_the_servers_env_file(self, _isolated):
        (_isolated / ".env").write_text("T1_HOST=0.0.0.0\nT1_PORT='8001'\n")
        url, source = localserver.configured_url()
        assert url == "http://127.0.0.1:8001", "0.0.0.0 is reached on loopback"
        assert ".env" in source

    def test_t1_url_and_an_explicit_url_win(self, monkeypatch):
        monkeypatch.setenv("T1_PORT", "8001")
        monkeypatch.setenv("T1_URL", "http://box:9000/")
        assert localserver.configured_url()[0] == "http://box:9000"
        assert localserver.configured_url("http://other:1")[0] == "http://other:1"

    def test_an_ipv6_host_is_bracketed(self, monkeypatch):
        monkeypatch.setenv("T1_HOST", "::1")
        monkeypatch.setenv("T1_PORT", "8002")
        assert localserver.configured_url()[0] == "http://[::1]:8002"

    def test_the_runner_uses_it(self, monkeypatch):
        monkeypatch.setenv("T1_PORT", "8001")
        assert runner_cli._base_url("") == "http://127.0.0.1:8001"


@pytest.fixture
def t1_like():
    """Something answering /health the way the T1 API does, and chat."""
    seen: list[tuple[str, str, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):  # noqa: N802
            seen.append(("GET", self.path, {}))
            if self.path == "/health":
                return self._json(200, {"status": "ok", "request_id": "r"})
            if self.path == "/runner/status":
                return self._json(200, {"loaded": False, "backends": ["auto"]})
            if self.path.startswith("/hyperlink/sessions/"):
                return self._json(404, {"error": {"message": "no such chat"}})
            return self._json(404, {})

        def do_PATCH(self):  # noqa: N802
            seen.append(("PATCH", self.path, self._body()))
            return self._json(200, {"session": {"session_id": "sess_1"}})

        def do_POST(self):  # noqa: N802
            body = self._body()
            seen.append(("POST", self.path, body))
            if self.path == "/hyperlink/sessions":
                return self._json(200, {"session": {"session_id": "sess_1"}})
            if self.path.endswith("/chat/stream"):
                frames = [
                    {"type": "start", "session_id": "sess_1"},
                    {"type": "tool", "tool": "server_version", "ok": True},
                    {"type": "delta", "text": "Hello "},
                    {"type": "delta", "text": "there."},
                    {"type": "done", "model_id": "hypernix.3-mini", "finish_reason": "stop"},
                ]
                data = "".join(f"data: {json.dumps(f)}\n\n" for f in frames).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return None
            return self._json(404, {})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", seen
    server.shutdown()


class TestFindingItWhenTheConfigIsWrong:
    def test_is_t1_api_checks_the_shape(self, t1_like):
        url, _ = t1_like
        assert localserver.is_t1_api(url)
        assert not localserver.is_t1_api("http://127.0.0.1:9")  # discard port: nothing there

    def test_the_runner_finds_it_and_says_where(self, t1_like, monkeypatch, capsys):
        url, seen = t1_like
        port = int(url.rsplit(":", 1)[1])
        monkeypatch.setattr(localserver, "SCAN_PORTS", (9, port))
        monkeypatch.setenv("T1_PORT", "9")  # configured wrong
        assert runner_cli.main(["status"]) == 0
        err = capsys.readouterr().err
        assert f"using the T1 API found at {url}" in err
        assert ("GET", "/runner/status", {}) in seen

    def test_a_named_url_is_not_second_guessed(self, t1_like, monkeypatch):
        url, _ = t1_like
        port = int(url.rsplit(":", 1)[1])
        monkeypatch.setattr(localserver, "SCAN_PORTS", (port,))
        with pytest.raises(SystemExit) as exc:
            runner_cli.main(["--url", "http://127.0.0.1:9", "status"])
        assert "Could not reach" in str(exc.value)

    def test_nothing_anywhere_says_so_and_how(self, monkeypatch):
        monkeypatch.setattr(localserver, "SCAN_PORTS", (9,))
        monkeypatch.setenv("T1_PORT", "9")
        with pytest.raises(SystemExit) as exc:
            runner_cli.main(["status"])
        message = str(exc.value)
        assert "from T1_PORT" in message and "hypernix-t1 start" in message


class TestChat:
    def test_a_new_chat_with_a_system_prompt(self, t1_like, monkeypatch, capsys):
        url, seen = t1_like
        code = chat_cli.main(["--url", url, "-s", "Be terse.", "Which", "model?"])
        assert code == 0
        out = capsys.readouterr()
        assert out.out == "Hello there.\n"
        assert "[server_version ✓]" in out.err
        assert "hypernix.3-mini" in out.err
        created = next(b for m, p, b in seen if p == "/hyperlink/sessions")
        assert created["system_prompt"] == "Be terse."
        sent = next(b for m, p, b in seen if p.endswith("/chat/stream"))
        assert sent["content"] == "Which model?"

    def test_the_chat_is_carried_on(self, t1_like, _isolated, capsys):
        url, seen = t1_like
        chat_cli.main(["--url", url, "first"])
        assert (_isolated / "cli_session").read_text().strip() == "sess_1"

    def test_quiet_prints_only_the_reply(self, t1_like, capsys):
        url, _ = t1_like
        chat_cli.main(["--url", url, "-q", "hi"])
        out = capsys.readouterr()
        assert out.out == "Hello there.\n" and "hypernix.3-mini" not in out.err

    def test_standard_input(self, t1_like, monkeypatch, capsys):
        url, seen = t1_like
        monkeypatch.setattr("sys.stdin", io.StringIO("from a pipe\n"))
        assert chat_cli.main(["--url", url]) == 0
        sent = next(b for m, p, b in seen if p.endswith("/chat/stream"))
        assert sent["content"] == "from a pipe"

    def test_the_shell_wrapper_dispatches_it(self):
        from pathlib import Path

        script = (Path(__file__).resolve().parent.parent / "bin" / "hypernix-t1").read_text()
        assert "chat)              cmd_chat" in script
        assert "hypernix.t1api.chat_cli" in script
