"""Conceal mode, 36-hour retention and /server/info (0.72.6).

Through the real app: a key turns conceal on; its address stops being
kept except where security needs it; what it made more than 36 hours
ago is deleted, and its memories, preferences and usage counts are not.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")

from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.privacy import RETENTION_HOURS, mask_address  # noqa: E402

OLD = time.time() - (RETENTION_HOURS + 1) * 3600


@pytest.fixture
def api(tmp_path, monkeypatch):
    from conftest import clear_t1_config
    from fastapi.testclient import TestClient

    from hypernix.security.gatekeeper import Gatekeeper
    from hypernix.t1api.app import create_app
    from hypernix.t1api.config import T1APIConfig

    clear_t1_config(monkeypatch)
    km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
    user = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE})
    app = create_app(
        config=T1APIConfig(
            token_secret="test-secret-value-that-is-long-enough",
            db_path=str(tmp_path / "t1.sqlite3"),
            module_storage_dir=str(tmp_path / "m"),
            hyperlink_files_dir=str(tmp_path / "f"),
            server_name="rack-one",
            server_description="The box under the stairs",
            server_owner="Sam",
            server_url="https://t1.example.net",
        ),
        keymaster=km,
        gatekeeper=Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False),
    )
    return app, TestClient(app, client=("198.51.100.23", 5000)), user


def _auth(key):
    return {"Authorization": f"Bearer {key}"}


class TestMaskAddress:
    @pytest.mark.parametrize("address,masked", [
        ("198.51.100.23", "198.51.100.0/24"),
        ("2001:db8:abcd:12::1", "2001:db8:abcd::/48"),
        ("not-an-ip", "concealed"),
        ("", ""),
    ])
    def test_it_keeps_the_network_not_the_address(self, address, masked):
        assert mask_address(address) == masked


class TestServerInfo:
    def test_it_is_public_and_says_who_runs_it(self, api):
        _app, client, _ = api
        info = client.get("/server/info").json()
        assert (info["name"], info["description"], info["owner"], info["url"]) == (
            "rack-one", "The box under the stairs", "Sam", "https://t1.example.net")
        assert info["public"] is True
        assert info["features"]["retention_hours"] == 36
        assert info["features"]["conceal_min_access_level"] == 3
        assert info["hypernix_version"]


class TestConceal:
    def test_off_by_default(self, api):
        _app, client, user = api
        assert client.get("/privacy/conceal", headers=_auth(user.key)).json()["concealed"] is False

    def test_on_and_off(self, api):
        _app, client, user = api
        on = client.post("/privacy/conceal", headers=_auth(user.key)).json()
        assert on["concealed"] is True and on["retention_hours"] == 36
        off = client.delete("/privacy/conceal", headers=_auth(user.key)).json()
        assert off["concealed"] is False

    def test_a_level_two_key_is_refused(self, api):
        from hypernix.security.t2keys import T2KeyGenerator

        _app, client, user = api
        low = T2KeyGenerator.from_t1(user.key, access_level=2).raw
        got = client.post("/privacy/conceal", headers=_auth(low))
        assert got.status_code == 403
        assert "level 3" in got.text

    def test_a_level_three_key_may(self, api):
        from hypernix.security.t2keys import T2KeyGenerator

        _app, client, user = api
        three = T2KeyGenerator.from_t1(user.key, access_level=3).raw
        assert client.post("/privacy/conceal", headers=_auth(three)).status_code == 200

    def test_the_address_is_masked_except_on_security_records(self, api):
        from hypernix.t1api.audit import AuditCategory

        app, client, user = api
        client.post("/privacy/conceal", headers=_auth(user.key))
        audit = app.state.t1_audit_log
        kept = audit.record("thing.done", actor_key_id=user.key_id, client_ip="198.51.100.23")
        security = audit.record("thing.refused", category=AuditCategory.SECURITY,
                                actor_key_id=user.key_id, client_ip="198.51.100.23")
        other = audit.record("thing.done", actor_key_id="someone-else", client_ip="198.51.100.23")
        assert kept.client_ip == "198.51.100.0/24"
        assert security.client_ip == "198.51.100.23"
        assert other.client_ip == "198.51.100.23"


class TestRetention:
    def _age(self, app, table, column, where, value):
        with app.state.t1_backend.connect() as conn:
            conn.execute(f"UPDATE {table} SET {column} = ? WHERE {where} = ?", (OLD, value))  # noqa: S608

    def test_old_data_goes_and_the_profile_stays(self, api):
        from hypernix.t1api.audit import AuditCategory

        app, client, user = api
        owner = user.key_id
        sessions = app.state.t1_session_store
        old = sessions.create(owner=owner, title="old")
        sessions.append(old.session_id, role="user", content="from last week", owner=owner)
        new = sessions.create(owner=owner, title="new")
        sessions.append(new.session_id, role="user", content="from today", owner=owner)
        self._age(app, "hyperlink_messages", "created_at", "session_id", old.session_id)
        self._age(app, "hyperlink_sessions", "updated_at", "session_id", old.session_id)

        files = app.state.t1_attachment_store
        stale = files.put(b"old bytes", filename="old.txt", owner=owner)
        fresh = files.put(b"new bytes", filename="new.txt", owner=owner)
        self._age(app, "hyperlink_files", "created_at", "file_id", stale.file_id)

        memory = app.state.t1_memory_store.create(owner=owner, content="Prefers tea")
        with app.state.t1_backend.connect() as conn:
            conn.execute("UPDATE hyperlink_memories SET created_at = ?", (OLD,))

        audit = app.state.t1_audit_log
        plain = audit.record("x", actor_key_id=owner)
        guard = audit.record("y", category=AuditCategory.SECURITY, actor_key_id=owner)
        with app.state.t1_backend.connect() as conn:
            conn.execute("UPDATE audit_events SET ts = ?", (OLD,))

        # Someone who did not ask for any of this keeps everything.
        bystander = sessions.create(owner="bystander", title="theirs")
        self._age(app, "hyperlink_sessions", "updated_at", "session_id", bystander.session_id)

        got = client.post("/privacy/conceal", headers=_auth(user.key)).json()
        assert got["swept"]["messages"] == 1
        assert got["swept"]["sessions"] == 1
        assert got["swept"]["files"] == 1
        assert got["swept"]["audit_records"] == 1

        remaining = {s.session_id for s in sessions.list_sessions(owner=owner)}
        assert remaining == {new.session_id}
        assert [f.file_id for f in files.list_files(owner=owner)] == [fresh.file_id]
        assert [m.memory_id for m in app.state.t1_memory_store.list(owner=owner)] == [memory.memory_id]
        with app.state.t1_backend.connect() as conn:
            ids = {r["audit_id"] for r in conn.execute("SELECT audit_id FROM audit_events").fetchall()}
        assert guard.audit_id in ids and plain.audit_id not in ids
        assert sessions.get(bystander.session_id, owner="bystander")

    def test_a_key_that_is_not_concealed_is_never_swept(self, api):
        app, _client, user = api
        sessions = app.state.t1_session_store
        old = sessions.create(owner=user.key_id, title="old")
        self._age(app, "hyperlink_sessions", "updated_at", "session_id", old.session_id)
        assert app.state.t1_retention.sweep() == []
        assert sessions.get(old.session_id, owner=user.key_id)

    def test_the_sweeper_thread_starts_and_stops_with_the_server(self, api):
        from fastapi.testclient import TestClient

        app, _client, _user = api
        with TestClient(app):
            thread = app.state.t1_retention._thread
            assert thread is not None and thread.is_alive()
        assert app.state.t1_retention._stop.is_set()
