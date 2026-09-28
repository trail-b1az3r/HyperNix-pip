"""``waiter``'s connection diagnostics, version report, and help topics.

The failure this exists for::

    Could not reach http://127.0.0.1:1234/hyperlink/pair: [Errno 111]
    Connection refused

accurate, and it answers none of the three questions the reader has: is
anything listening, is this the address I meant, and what do I type next.
Port 1234 is LM Studio's — a bridge target the *server* talks to, never
the address waiter should point at — and nothing in that message says so.
"""
from __future__ import annotations

import contextlib
import io
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hypernix.waiter.diagnose import (
    T1_DEFAULT_PORT,
    WELL_KNOWN_PORTS,
    diagnose,
    format_diagnosis,
    get_json,
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def serve():
    """Run a handler on a free port for the duration of one test."""
    servers = []

    def start(handler_cls) -> str:
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _json_handler(routes: dict[str, dict]):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path in routes:
                body = json.dumps(routes[self.path]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def log_message(self, *args):  # keep the test output clean
            pass

    return Handler


class TestNothingListening:
    def test_it_says_so_rather_than_repeating_the_errno(self):
        url = f"http://127.0.0.1:{free_port()}/hyperlink/pair"
        result = diagnose(url, "[Errno 111] Connection refused")
        assert result.port_open is False
        assert result.nothing_listening
        text = format_diagnosis(result)
        assert "Nothing is listening" in text

    def test_it_names_a_command_that_starts_the_server(self):
        url = f"http://127.0.0.1:{free_port()}/status"
        text = format_diagnosis(diagnose(url, "refused"))
        assert "run_local.sh" in text or "start-t1.sh" in text

    def test_it_says_where_the_address_came_from(self):
        """Frequently the whole answer: an -I from weeks ago is still saved."""
        url = f"http://127.0.0.1:{free_port()}/status"
        text = format_diagnosis(
            diagnose(url, "refused", source="the saved config (~/.hypernix/waiter.jsonl)")
        )
        assert "Address from: the saved config" in text


class TestWellKnownPorts:
    """The bug in the report: 1234 is LM Studio, not the T1 API."""

    def test_1234_is_recognised_as_lm_studio(self):
        result = diagnose("http://127.0.0.1:1234/hyperlink/pair", "refused", probe=False)
        assert result.known_use is not None
        assert result.known_use.software == "LM Studio"

    def test_the_advice_points_at_the_t1_port(self):
        text = format_diagnosis(
            diagnose("http://127.0.0.1:1234/hyperlink/pair", "refused", probe=False)
        )
        assert "LM Studio" in text
        assert f":{T1_DEFAULT_PORT}" in text
        assert "waiter serv" in text

    def test_it_explains_that_lm_studio_is_reached_through_the_server(self):
        """The misconception behind the mistake, not just the wrong number."""
        text = format_diagnosis(diagnose("http://h:1234/x", "refused", probe=False))
        assert "through the T1 server" in text

    def test_ollama_too(self):
        result = diagnose("http://127.0.0.1:11434/x", "refused", probe=False)
        assert result.known_use.software == "Ollama"

    def test_the_t1_port_is_not_flagged_as_someone_elses(self):
        result = diagnose(f"http://127.0.0.1:{T1_DEFAULT_PORT}/x", "refused", probe=False)
        assert result.known_use is None

    def test_every_entry_is_a_distinct_port(self):
        ports = [use.port for use in WELL_KNOWN_PORTS]
        assert len(ports) == len(set(ports))
        assert T1_DEFAULT_PORT not in ports


class TestIdentifyingWhatAnswered:
    def test_a_t1_server_is_recognised(self, serve):
        base = serve(_json_handler({
            "/status": {"server_name": "test", "t1_api_version": "1.0.26.8.1.0"}
        }))
        assert diagnose(f"{base}/hyperlink/pair", "404", timeout=3.0).responder == (
            "a T1 API server"
        )

    def test_an_openai_compatible_server_is_recognised(self, serve):
        """LM Studio on a port that is not 1234 still gets diagnosed."""
        base = serve(_json_handler({"/v1/models": {"data": [{"id": "qwen"}]}}))
        result = diagnose(f"{base}/hyperlink/pair", "404", timeout=3.0)
        assert "OpenAI-compatible" in result.responder
        assert "rather than a T1 API" in format_diagnosis(result)

    def test_a_404_does_not_end_the_probe_early(self, serve):
        """The bug this catches.

        ``/status`` is tried before ``/v1/models``. Returning on the first
        HTTP error meant a 404 there reported "an HTTP server" and never
        reached the route that identifies LM Studio.
        """
        base = serve(_json_handler({"/v1/models": {"data": []}}))
        assert "OpenAI-compatible" in diagnose(f"{base}/x", timeout=3.0).responder

    def test_port_open_is_true_when_something_answers(self, serve):
        base = serve(_json_handler({"/": {}}))
        assert diagnose(f"{base}/x", timeout=3.0).port_open is True


class TestProbesIgnoreProxies:
    """A proxy answers a different question than "is this host up".

    An environment with HTTP_PROXY set and no no_proxy for localhost —
    containers do this routinely — would otherwise have every local probe
    fail through the proxy and be reported as the server being down.
    """

    def test_get_json_works_with_a_bogus_proxy_configured(self, serve, monkeypatch):
        base = serve(_json_handler({"/status": {"server_name": "x"}}))
        for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
            monkeypatch.setenv(var, "http://127.0.0.1:9")  # discard port
        assert get_json(f"{base}/status", timeout=3.0) == {"server_name": "x"}

    def test_identification_works_with_a_bogus_proxy(self, serve, monkeypatch):
        base = serve(_json_handler({"/status": {"t1_api_version": "1.0.26.8.1.0"}}))
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        assert diagnose(f"{base}/x", timeout=3.0).responder == "a T1 API server"


class TestNeverRaises:
    """A diagnostic that throws replaces a real error with a worse one."""

    @pytest.mark.parametrize(
        "url",
        ["", "not-a-url", "http://", "http://[::1]:99999/x", "ftp://host/x"],
    )
    def test_malformed_urls_are_survivable(self, url):
        result = diagnose(url, "refused", probe=False)
        assert isinstance(format_diagnosis(result), str)

    def test_get_json_returns_none_for_junk(self, serve):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                body = b"not json at all"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        base = serve(Handler)
        assert get_json(f"{base}/status", timeout=3.0) is None


# ---------------------------------------------------------------------------
# The CLI surface
# ---------------------------------------------------------------------------


def run_cli(*argv: str) -> tuple[int, str]:
    from hypernix.waiter.cli import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue() + err.getvalue()


@pytest.fixture
def waiter_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "hypernix.waiter.local_config._DEFAULT_CONFIG_PATH",
        tmp_path / "waiter.config.jsonl",
    )
    return tmp_path


class TestVersionCommand:
    def test_it_reports_the_versions_that_move_independently(self, waiter_home):
        import hypernix
        from hypernix.t1api.version import T1_VERSION_SHORT
        from hypernix.waiter import __waiter_version__

        code, text = run_cli("version")
        assert code == 0
        assert hypernix.__version__ in text
        assert __waiter_version__ in text
        assert T1_VERSION_SHORT in text
        assert "v1, v2, v2short" in text

    def test_json(self, waiter_home):
        code, text = run_cli("version", "--json")
        assert code == 0
        data = json.loads(text)
        assert data["key_versions"]["available"] == ["v1", "v2", "v2short", "v2.1"]
        assert data["key_versions"]["reserved"] == []
        assert data["t1_api"]["min_client"]

    def test_it_omits_the_server_line_rather_than_guessing(self, waiter_home):
        """No server configured is not the same as a server of unknown version."""
        code, text = run_cli("version")
        assert code == 0
        assert "Server:" not in text

    def test_it_reads_the_flat_version_field(self, serve, waiter_home):
        """`t1_version` is an object; `t1_api_version` is the string.

        Reading the wrong one printed a dict into the version report.
        """
        from hypernix.waiter.local_config import WaiterConfigStore, WaiterLocalConfig

        base = serve(_json_handler({
            "/status": {
                "server_name": "blaze-test",
                "t1_api_version": "1.0.26.8.1.0",
                "t1_version": {"short": "1.0.26.8.1.0", "long": "1.0.2026.8.1.0"},
            }
        }))
        WaiterConfigStore().save(WaiterLocalConfig(server=base, local_only=True))
        code, text = run_cli("version")
        assert code == 0
        assert "Server:      t1 v1.0.26.8.1.0  (blaze-test)" in text
        assert "{" not in text, "a dict leaked into the version report"


class TestUsageText:
    def test_the_t1_version_is_not_hard_coded(self, waiter_home):
        """It said 1.0.26.8.0.1 long after the API moved to 1.0.26.8.1.0.

        A version that has to be updated by hand in a second place is a
        version that will be wrong.
        """
        from hypernix.t1api.version import T1_VERSION_SHORT

        code, text = run_cli("--help")
        assert code == 0
        assert f"T1 v{T1_VERSION_SHORT}" in text

    def test_the_new_subcommands_are_listed(self, waiter_home):
        _, text = run_cli("--help")
        assert "waiter version" in text
        assert "waiter help" in text

    def test_an_unknown_subcommand_shows_the_usage_and_fails(self, waiter_home):
        code, text = run_cli("nonsense")
        assert code == 1
        assert "Unknown subcommand" in text


class TestHelpTopics:
    def test_it_lists_topics(self, waiter_home):
        code, text = run_cli("help")
        assert code == 0
        for topic in ("connect", "keys", "hyperlink", "find"):
            assert topic in text

    @pytest.mark.parametrize("topic", ["connect", "keys", "hyperlink", "find"])
    def test_each_topic_renders(self, waiter_home, topic):
        code, text = run_cli("help", topic)
        assert code == 0
        assert len(text.splitlines()) > 5

    def test_connect_warns_about_the_bridge_ports(self, waiter_home):
        """The mistake that produced the report."""
        _, text = run_cli("help", "connect")
        assert "1234" in text and "11434" in text

    def test_an_unknown_topic_lists_the_real_ones(self, waiter_home):
        code, text = run_cli("help", "nope")
        assert code == 1
        assert "connect" in text
