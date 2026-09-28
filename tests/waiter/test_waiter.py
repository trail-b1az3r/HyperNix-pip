"""Tests for hypernix.waiter — the client, local config, and CLI dispatch.

Uses a real (stdlib ``http.server``) HTTP server rather than mocking
``urllib`` internals, so these tests exercise the actual request/response
path the ``waiter`` console script uses in production, just against a fake
backend instead of a live T1 API. See ``tests/test_t1api_*.py`` for the
server-side equivalents.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hypernix.waiter.cli import main
from hypernix.waiter.client import T1Client, T1ClientError
from hypernix.waiter.local_config import WaiterConfigStore, WaiterLocalConfig


class _FakeT1Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D401 - silence default request logging
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - stdlib naming convention
        routes = {
            "/health": (200, {"status": "ok", "request_id": "r1"}),
            "/status": (
                200,
                {
                    "status": "ok",
                    "environment": "development",
                    "t1_api_version": "0.71.5b1",
                    "hypernix_version": "0.71.5b1",
                    "beta": "beta1",
                    "model_count": 1,
                    "storage_backend": "sqlite",
                    "request_id": "r2",
                },
            ),
            "/models": (
                200,
                {
                    "models": [
                        {
                            "model_id": "nanonix-nano",
                            "display_name": "nanoNix-nano",
                            "version": "1.0",
                            "architecture": "dense",
                            "status": "example",
                            "availability": "public",
                            "minimum_plan": "free",
                            "free_tier_available": True,
                            "routing_priority": 40,
                        }
                    ],
                    "count": 1,
                    "request_id": "r3",
                },
            ),
        }
        if self.path in routes:
            code, payload = routes[self.path]
            self._send(code, payload)
        else:
            self._send(404, {"error": {"code": "NOT_FOUND", "message": "no route"}, "request_id": "r4"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw or b"{}")
        if self.path == "/auth/t1/validate":
            key = body.get("key", "")
            if key == "T1_bad":
                self._send(
                    401,
                    {"error": {"code": "AUTH_INVALID_KEY", "message": "bad key"}, "request_id": "r5"},
                )
                return
            self._send(
                200,
                {
                    "key_id": "abc12345",
                    "key_type": "admin" if "admin" in key else "user",
                    "scopes": ["read", "write"] + (["admin"] if "admin" in key else []),
                    "active": True,
                    "is_admin": "admin" in key,
                    "expires_at": None,
                    "request_id": "r6",
                },
            )
        else:
            self._send(404, {"error": {"code": "NOT_FOUND", "message": "no route"}, "request_id": "r7"})


@pytest.fixture
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _FakeT1Handler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()


class _FakeT1Beta2Handler(BaseHTTPRequestHandler):
    """Minimal fake covering just the Beta 2 endpoints the CLI subcommand
    tests exercise — separate from _FakeT1Handler (Beta 1) to keep each
    fake focused on what it's actually testing."""

    def log_message(self, *a):
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        from urllib.parse import urlparse

        path = urlparse(self.path).path
        if path == "/servers":
            self._send(200, {"servers": [{"server_id": "abcd1234ef", "name": "prod-1", "address": "https://p.example.com", "trust_level": "trusted", "status": "unknown"}], "count": 1, "request_id": "r"})
        elif path == "/modules":
            self._send(200, {"modules": [{"module_id": "mod123456", "name": "recommender", "version": "1.0.0", "status": "active", "size_bytes": 1024}], "count": 1, "request_id": "r"})
        elif path.startswith("/jobs/"):
            job_id = path.split("/")[2]
            self._send(200, {"job": {"job_id": job_id, "kind": "module_sync", "status": "succeeded", "result": {"ok": True}, "error": None, "created_by": "u1", "created_at": 1.0, "started_at": 1.0, "finished_at": 1.1, "payload": {}}, "request_id": "r"})
        elif path == "/events":
            self._send(200, {"events": [{"event_id": "e1", "type": "server.registered", "data": {"name": "prod-1"}, "source": "servers", "ts": 1.0}], "count": 1, "request_id": "r"})
        elif path == "/billing/balance":
            self._send(200, {"account_type": "user", "account_id": "u1", "balance": 42.5, "request_id": "r"})
        else:
            self._send(404, {"error": {"code": "NOT_FOUND", "message": "no route"}, "request_id": "r"})

    def do_POST(self):  # noqa: N802
        from urllib.parse import urlparse

        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        if path == "/models/route":
            self._send(200, {"model_id": "nanonix-mini-plus", "reason": "primary", "cascade_position": 0, "policy_name": "free_tier_default", "considered": [], "request_id": "r"})
        elif path == "/servers/register":
            self._send(200, {"server": {"server_id": "newserver01", "name": body["name"], "address": body["address"], "trust_level": "untrusted", "status": "unknown"}, "request_id": "r"})
        else:
            self._send(404, {"error": {"code": "NOT_FOUND", "message": "no route"}, "request_id": "r"})


@pytest.fixture
def beta2_server():
    srv = HTTPServer(("127.0.0.1", 0), _FakeT1Beta2Handler)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()


# ===========================================================================
# T1Client
# ===========================================================================


class TestT1Client:
    def test_health(self, fake_server):
        client = T1Client(base_url=fake_server)
        assert client.health()["status"] == "ok"

    def test_validate_success(self, fake_server):
        client = T1Client(base_url=fake_server)
        result = client.validate("T1_admin_key")
        assert result["key_id"] == "abc12345"
        assert result["is_admin"] is True

    def test_validate_failure_raises_with_code(self, fake_server):
        client = T1Client(base_url=fake_server)
        with pytest.raises(T1ClientError) as exc_info:
            client.validate("T1_bad")
        assert exc_info.value.code == "AUTH_INVALID_KEY"
        assert exc_info.value.status == 401

    def test_auth_required_call_without_credential_raises_locally(self, fake_server):
        client = T1Client(base_url=fake_server, credential=None)
        with pytest.raises(T1ClientError):
            client.usage_current()

    def test_unreachable_server_raises_clean_error(self):
        client = T1Client(base_url="http://127.0.0.1:1", timeout=1.0)
        with pytest.raises(T1ClientError):
            client.health()


# ===========================================================================
# WaiterLocalConfig / WaiterConfigStore
# ===========================================================================


class TestLocalConfig:
    def test_plain_roundtrip(self, tmp_path):
        store = WaiterConfigStore(tmp_path / "cfg.jsonl", encrypt=False)
        cfg = WaiterLocalConfig(server="s", key="k", port=1234)
        store.save(cfg)
        assert store.load() == cfg

    def test_encrypted_roundtrip(self, tmp_path):
        store = WaiterConfigStore(tmp_path / "cfg.jsonl", encrypt=True)
        cfg = WaiterLocalConfig(server="s", key="k")
        store.save(cfg)
        raw = (tmp_path / "cfg.jsonl").read_text()
        assert "server" not in raw  # not plaintext on disk
        assert store.load() == cfg

    def test_encrypted_file_readable_without_encrypt_flag_on_load(self, tmp_path):
        """The bug this test guards: reading back a file saved with -E
        must work even when the reader didn't pass encrypt=True, since
        real CLI usage often reads via a different subcommand than the
        one that wrote it (see hypernix.waiter.cli's _cmd_config)."""
        write_store = WaiterConfigStore(tmp_path / "cfg.jsonl", encrypt=True)
        cfg = WaiterLocalConfig(server="s", key="k")
        write_store.save(cfg)
        read_store = WaiterConfigStore(tmp_path / "cfg.jsonl", encrypt=False)
        assert read_store.load() == cfg

    def test_missing_file_returns_none(self, tmp_path):
        store = WaiterConfigStore(tmp_path / "nope.jsonl")
        assert store.load() is None


# ===========================================================================
# CLI dispatch (via main())
# ===========================================================================


class TestCliDispatch:
    def test_health_command(self, fake_server, capsys):
        rc = main(["health", "-I", fake_server])
        assert rc == 0

    def test_models_command(self, fake_server, capsys):
        rc = main(["models", "-I", fake_server, "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "nanonix-nano" in out

    def test_whoami_requires_key(self, fake_server):
        rc = main(["whoami", "-I", fake_server])
        assert rc == 1

    def test_whoami_with_key(self, fake_server):
        rc = main(["whoami", "-I", fake_server, "-K", "T1_user_key"])
        assert rc == 0

    def test_unknown_subcommand_returns_1(self):
        rc = main(["not-a-real-command"])
        assert rc == 1

    def test_serv_auto_end_to_end(self, fake_server, tmp_path):
        cfgfile = tmp_path / "cfg.jsonl"
        rc = main(["serv", "-A", "-I", fake_server, "-K", "T1_user_key", "-F", str(cfgfile)])
        assert rc == 0
        assert cfgfile.exists()
        store = WaiterConfigStore(cfgfile)
        saved = store.load()
        assert saved.server == fake_server
        assert saved.key == "T1_user_key"

    def test_serv_auto_encrypted_round_trips_through_config_command(self, fake_server, tmp_path, capsys):
        cfgfile = tmp_path / "cfg.jsonl"
        rc = main(["serv", "-A", "-I", fake_server, "-K", "T1_admin_key", "-E", "-F", str(cfgfile)])
        assert rc == 0
        capsys.readouterr()  # discard 'serv' output before capturing 'config' output below
        rc = main(["config", "-F", str(cfgfile), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["server"] == fake_server
        assert payload["key"].endswith("…")  # masked, not the raw secret

    def test_serv_no_action_flags_is_an_error(self, fake_server, tmp_path):
        cfgfile = tmp_path / "cfg.jsonl"
        rc = main(["serv", "-I", fake_server, "-K", "T1_user_key", "-F", str(cfgfile)])
        assert rc == 1

    def test_serv_admin_surface_flags_dont_crash_and_dont_pretend_to_work(self, fake_server, tmp_path, capsys):
        """-B/-W/-r/-a have no server endpoint yet — they should be
        accepted, stored locally, and clearly flagged as not enforced,
        never silently dropped or falsely reported as applied."""
        cfgfile = tmp_path / "cfg.jsonl"
        rc = main(
            [
                "serv",
                "-A",
                "-I",
                fake_server,
                "-K",
                "T1_user_key",
                "-F",
                str(cfgfile),
                "-B",
                "1.2.3.4",
                "-W",
                "5.6.7.8",
            ]
        )
        assert rc == 0
        err = capsys.readouterr().err
        assert "Beta 2/3" in err or "not" in err.lower()


class TestBeta2Subcommands:
    """Beta 2 subcommands (route/servers/modules/jobs/events/billing)
    against a fake server that implements just enough of the real Beta 2
    HTTP contract to exercise the client + CLI dispatch path."""

    def test_route_automatic(self, beta2_server):
        rc = main(["route", "-I", beta2_server, "-K", "T1_x", "--plan", "free"])
        assert rc == 0

    def test_servers_list(self, beta2_server):
        rc = main(["servers", "-I", beta2_server, "-K", "T1_x"])
        assert rc == 0

    def test_servers_register(self, beta2_server, capsys):
        rc = main(
            ["servers", "-I", beta2_server, "-K", "T1_x", "--register", "prod-2", "--address", "https://p2.example.com"]
        )
        assert rc == 0
        assert "prod-2" in capsys.readouterr().out

    def test_servers_register_without_address_errors_locally(self, beta2_server):
        """--register without --address should fail fast client-side,
        never even making a request."""
        rc = main(["servers", "-I", beta2_server, "-K", "T1_x", "--register", "prod-2"])
        assert rc == 1

    def test_modules_list(self, beta2_server):
        rc = main(["modules", "-I", beta2_server, "-K", "T1_x"])
        assert rc == 0

    def test_modules_upload_requires_file(self, beta2_server):
        rc = main(["modules", "-I", beta2_server, "-K", "T1_x", "--upload", "mod123"])
        assert rc == 1

    def test_jobs_get(self, beta2_server):
        rc = main(["jobs", "-I", beta2_server, "-K", "T1_x", "get", "job123"])
        assert rc == 0

    def test_events(self, beta2_server):
        rc = main(["events", "-I", beta2_server, "-K", "T1_x"])
        assert rc == 0

    def test_billing_balance(self, beta2_server, capsys):
        rc = main(["billing", "-I", beta2_server, "-K", "T1_x"])
        assert rc == 0
        assert "42.5" in capsys.readouterr().out


class TestDoctorTypedStatus:
    """`waiter doctor` reads the typed status view.

    waiter's T1Client overrides status() to return the raw envelope,
    because that is what the CLI's table renderers consume. Doctor wants
    the typed object and used to assume it got one, which crashed with
    "'dict' object has no attribute 'environment'" the first time it ran
    against a real server — a TestClient-only test suite never caught it
    because nothing exercised that path.
    """

    def test_status_override_returns_a_dict(self, fake_server):
        from hypernix.waiter.client import T1Client as WaiterClient

        payload = WaiterClient(base_url=fake_server).status()
        assert isinstance(payload, dict)

    def test_the_typed_view_is_built_from_it(self, fake_server):
        from hypernix.t1sdk.models import ServerStatus
        from hypernix.waiter.client import T1Client as WaiterClient

        status = ServerStatus.from_dict(WaiterClient(base_url=fake_server).status())
        # Attribute access is what doctor does; this is the assertion that
        # would have failed before the fix.
        assert isinstance(status.environment, str)
        assert isinstance(status.production_warnings, list)
