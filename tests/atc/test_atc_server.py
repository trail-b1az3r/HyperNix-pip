"""atc (0.73.0 pre-alpha 1) in front of a real T1 app.

The jail, probe refusals, the body limit, compression, the admin routes
and the end-to-end tunnel, driven by :class:`hypernix.atc.client.ATCTunnel`.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
from urllib.parse import urlsplit

import pytest

pytest.importorskip("fastapi")

from hypernix.security.gkey_cli import main as gkey  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")
OUTSIDER = ("203.0.113.50", 4000)


def _admin_key() -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert gkey(["create", "--type", "admin", "--scopes", "admin,read,write"]) == 0
    for line in ANSI.sub("", buf.getvalue()).splitlines():
        match = re.search(r"\bKey:\s*(\S+)", line)
        if match:
            return match.group(1)
    raise AssertionError("no key printed")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("hypernix.security.keymaster._DEFAULT_STORE", tmp_path / "keymaster")
    monkeypatch.setattr("hypernix.security.gatekeeper._DEFAULT_DATA", tmp_path / "gatekeeper")
    monkeypatch.setenv("T1_TOKEN_SECRET", "x" * 64)
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1api.sqlite3"))
    monkeypatch.setenv("T1_BACKUP_DIR", str(tmp_path / "backups"))
    for name in list(__import__("os").environ):
        if name.startswith("T1_ATC_"):
            monkeypatch.delenv(name)
    return monkeypatch


def _client(client_addr=OUTSIDER):
    from fastapi.testclient import TestClient

    from hypernix.t1api.app import create_app

    return TestClient(create_app(), client=client_addr)


class TestTheShield:
    def test_probes_are_404_and_three_jail(self, env):
        with _client() as client:
            for _ in range(3):
                assert client.get("/.env").status_code == 404
            reply = client.get("/health")
            assert reply.status_code == 429
            assert reply.json()["error"]["details"]["reason"] == "atc_jailed"
            assert int(reply.headers["retry-after"]) >= 60

    def test_this_machine_is_not_jailed(self, env):
        with _client(("127.0.0.1", 5000)) as client:
            for _ in range(5):
                assert client.get("/wp-admin/").status_code == 404
            assert client.get("/health").status_code == 200

    def test_an_admin_sees_the_jail_and_can_release(self, env):
        admin = {"Authorization": "Bearer " + _admin_key()}
        with _client(("127.0.0.1", 5000)) as client:
            reply = client.post("/atc/jail", json={"client": "198.51.100.7"}, headers=admin)
            assert reply.status_code == 200, reply.text
            status = client.get("/atc/status", headers=admin).json()
            assert [j["client"] for j in status["shield"]["jailed"]] == ["198.51.100.7"]
            assert client.post("/atc/release", json={"client": "198.51.100.7"},
                               headers=admin).status_code == 200
            assert client.post("/atc/release", json={"client": "198.51.100.7"},
                               headers=admin).status_code == 404

    def test_status_is_admin_only(self, env):
        with _client() as client:
            assert client.get("/atc/status").status_code in (401, 403)


class TestLimitsAndCompression:
    def test_a_body_over_the_limit_is_413(self, env):
        env.setenv("T1_ATC_MAX_BODY_MB", "0.001")
        with _client() as client:
            reply = client.post("/auth/t1/validate", content=b"x" * 5000,
                                headers={"content-type": "application/json"})
            assert reply.status_code == 413
            assert reply.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"

    def test_big_json_is_gzipped_when_asked(self, env):
        with _client(("127.0.0.1", 5000)) as client:
            reply = client.get("/openapi.json", headers={"accept-encoding": "gzip"})
            if reply.status_code == 404:
                pytest.skip("docs are not exposed in this configuration")
            assert reply.headers.get("content-encoding") == "gzip"
            assert "Accept-Encoding" in reply.headers.get("vary", "")
            assert reply.json()["openapi"]

    def test_off_means_off(self, env):
        env.setenv("T1_ATC_ENABLED", "0")
        with _client() as client:
            assert client.app.state.t1_atc is None
            for _ in range(5):
                client.get("/.env")
            assert client.get("/health").status_code == 200
            assert client.post("/atc/handshake", json={"public_key": "A" * 43}).status_code == 501


pytest.importorskip("cryptography")


class _Opener:
    """urllib's urlopen, answered by a TestClient."""

    def __init__(self, client):
        self.client = client
        self.raw: list[bytes] = []

    def __call__(self, req, timeout=None):
        path = urlsplit(req.full_url).path
        reply = self.client.post(path, content=req.data, headers=dict(req.header_items()))
        self.raw.append(reply.content)

        class _Resp(io.BytesIO):
            status = reply.status_code

        return _Resp(reply.content)


def _tunnel(client, **kw):
    from hypernix.atc.client import ATCTunnel

    opener = _Opener(client)
    return ATCTunnel("http://testserver", opener=opener, **kw), opener


class TestTheTunnel:
    def test_a_request_and_its_reply_travel_sealed(self, env):
        key = _admin_key()
        with _client() as client:
            tunnel, opener = _tunnel(client)
            status, headers, body = tunnel.request(
                "POST", "/auth/t1/validate", body={"key": key},
                headers={"Authorization": "Bearer " + key})
            assert status == 200, body
            assert json.loads(body)["is_admin"] is True
            assert headers["content-type"].startswith("application/json")
            assert all(key.encode() not in raw for raw in opener.raw)
            assert client.app.state.t1_atc.stats["tunnelled"] == 1

    def test_the_fingerprint_is_the_t2c_identity_and_pins(self, env):
        from hypernix.atc import e2e

        with _client() as client:
            fingerprint = client.get("/auth/t2c/public-key").json()["fingerprint"]
            tunnel, _ = _tunnel(client, pin=fingerprint)
            assert tunnel.request("GET", "/health")[0] == 200
            wrong, _ = _tunnel(client, pin="not-" + fingerprint)
            with pytest.raises(e2e.E2EError, match="pinned"):
                wrong.request("GET", "/health")

    def test_a_replayed_request_is_refused(self, env):
        from hypernix.atc import e2e

        with _client() as client:
            tunnel, _ = _tunnel(client)
            tunnel.request("GET", "/health")
            session = tunnel._session
            blob = e2e.seal_request(session["c2s"], session["id"], 1, method="GET", path="/health")
            reply = client.post("/atc/tunnel", content=blob,
                                headers={"X-ATC-Session": session["id"], "X-ATC-Seq": "1"})
            assert reply.status_code == 409
            assert reply.json()["error"]["details"]["reason"] == "atc_replay"

    def test_an_unknown_session_is_401(self, env):
        with _client() as client:
            reply = client.post("/atc/tunnel", content=b"x" * 40,
                                headers={"X-ATC-Session": "nope", "X-ATC-Seq": "1"})
            assert reply.status_code == 401

    def test_a_probe_inside_the_tunnel_is_still_a_probe(self, env):
        with _client() as client:
            tunnel, _ = _tunnel(client)
            assert tunnel.request("GET", "/.git/config")[0] == 404
            assert client.app.state.t1_atc.stats["probes"] == 1

    def test_required_e2e_refuses_the_clear_but_not_the_tunnel(self, env):
        env.setenv("T1_ATC_E2E_REQUIRED", "1")
        with _client() as client:
            reply = client.get("/models")
            assert reply.status_code == 403
            assert reply.json()["error"]["details"]["reason"] == "e2e_required"
            assert client.get("/health").status_code == 200
            tunnel, _ = _tunnel(client)
            status, _, _ = tunnel.request("GET", "/models")
            assert status != 403
        with _client(("127.0.0.1", 5000)) as local:
            assert local.get("/models").status_code != 403
