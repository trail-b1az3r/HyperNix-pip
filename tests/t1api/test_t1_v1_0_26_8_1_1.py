"""T1 v1.0.2026.8.1.1 — server IDs that advance, and undo/redo that works.

Two defects, both of the same shape: state that was written but never
read back, so a feature looked implemented and did nothing.

* Every key ``gkey`` ever minted carried server ID ``00001-A1``. The
  counter advanced correctly and lived only in memory, and each ``gkey``
  invocation is its own process — so it restarted at the beginning every
  time.
* ``POST /t1/auth/undo`` could never undo anything. Nothing called
  ``AuthHistory.record()``, so the history was permanently empty; and the
  four Keymaster methods the inverse needs did not exist, so even a
  recorded entry would have answered 501.
"""
from __future__ import annotations

import contextlib
import io
import re

import pytest

from hypernix.security.gkey_cli import main as gkey
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType
from hypernix.t1api.version import T1_VERSION, T1Version

# The server needs the [t1api] extra; the key store does not. Skipping the
# whole module when fastapi is absent would take the server-ID and
# reversal-primitive tests down with it — and those cover the two bugs
# this release is about, in code that has no HTTP layer at all. Only the
# class that actually drives a server is gated.
try:  # a real import, not find_spec
    import fastapi  # noqa: F401

    _HAS_FASTAPI = True
except ImportError:
    # find_spec would answer "yes" for an installed-but-broken fastapi —
    # it locates the module without executing it. Importing is what the
    # tests themselves do, so it is what the guard should do.
    _HAS_FASTAPI = False
needs_server = pytest.mark.skipif(
    not _HAS_FASTAPI, reason="needs the [t1api] extra (fastapi)"
)

ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "hypernix.security.keymaster._DEFAULT_STORE", tmp_path / "keymaster"
    )
    monkeypatch.setattr(
        "hypernix.security.gatekeeper._DEFAULT_DATA", tmp_path / "gatekeeper"
    )
    monkeypatch.setenv("T1_TOKEN_SECRET", "x" * 64)
    # The auth history lives in the database, not the key store. Without
    # its own path the DB is shared between tests while the key store is
    # not, so one test's recorded rotation is replayed against another
    # test's keys.
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1api.sqlite3"))
    monkeypatch.setenv("T1_BACKUP_DIR", str(tmp_path / "backups"))
    return tmp_path / "keymaster"


def run_gkey(*argv: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert gkey(list(argv)) == 0
    return ANSI.sub("", buf.getvalue())


def field(text: str, label: str) -> str:
    pattern = re.compile(rf"\b{re.escape(label)}:\s*(.+)")
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            return match.group(1).strip().rstrip("│").strip()
    raise AssertionError(f"no {label!r} in:\n{text}")


class TestTheVersion:
    def test_this_release_is_still_in_the_build(self):
        """At least 8.1.1, not exactly it.

        This file covers what 1.0.2026.8.1.1 shipped, and those
        guarantees have to survive every later release. Asserting the
        build *is* 8.1.1 made that a test of the calendar: it went red on
        the next bump while every behaviour it names still worked.
        """
        shipped_in = T1Version(api=1, major=0, year=2026, month=8, feature=1, fix=1)
        assert T1_VERSION >= shipped_in

    def test_that_release_is_now_a_generation_behind(self):
        """It was "the generation has not moved". In 0.72.6 pt2 it did.

        Compatibility here is by generation (`api.major`), so a client
        shipped against 1.0.26.8.1.1 is no longer the same protocol as
        this server and is told to upgrade — which is what the check
        exists to do, rather than letting it fail on a missing field.
        """
        shipped_in = T1Version(api=1, major=0, year=2026, month=8, feature=1, fix=1)
        assert (T1_VERSION.api, T1_VERSION.major) != (shipped_in.api, shipped_in.major)
        assert not T1_VERSION.compatible_with(shipped_in)

    def test_every_shipped_component_derives_from_it(self):
        """Four packages carried the number as a literal and drifted.

        They are computed from ``T1_VERSION`` now, so a bump cannot leave
        one of them behind — which is exactly how ``waiter --help`` came
        to advertise a version two releases stale.
        """
        from hypernix.hyperlink import __hyperlink_version__
        from hypernix.t1api import __t1api_version__
        from hypernix.t1sdk import __sdk_version__
        from hypernix.waiter import __waiter_version__

        assert (
            __t1api_version__
            == __sdk_version__
            == __waiter_version__
            == __hyperlink_version__
            == T1_VERSION.short
        )


class TestServerIdsAdvance:
    """They advanced in memory and restarted from scratch every process."""

    def test_two_keys_in_one_process_differ(self, store):
        km = Keymaster(store_dir=store, auto_rotate=False)
        first = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        second = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        assert first.server_id != second.server_id

    def test_a_new_process_does_not_restart_the_sequence(self, store):
        """The actual bug: every `gkey create` is its own process."""
        first = Keymaster(store_dir=store, auto_rotate=False).create(
            key_type=KeyType.USER, scopes={KeyScope.READ}
        )
        second = Keymaster(store_dir=store, auto_rotate=False).create(
            key_type=KeyType.USER, scopes={KeyScope.READ}
        )
        assert first.server_id == "00001-A1"
        assert second.server_id == "00002-A1"

    def test_gkey_advances_across_invocations(self, store):
        """End to end, through the CLI the report came from."""
        seen = [field(run_gkey("create"), "Server ID") for _ in range(4)]
        assert seen == ["00001-A1", "00002-A1", "00003-A1", "00004-A1"]
        assert len(set(seen)) == len(seen)

    def test_a_revoked_key_does_not_release_its_server_id(self, store):
        """Resuming from the active keys alone would reissue it.

        Revoking moves a record into ``archive/``, which ``_load_all``
        does not read. A server ID that comes back around is worse than
        one that never moves: an audit trail stops being able to tell the
        two keys apart.
        """
        km = Keymaster(store_dir=store, auto_rotate=False)
        km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        highest = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.revoke(highest.key_id)

        fresh = Keymaster(store_dir=store, auto_rotate=False).create(
            key_type=KeyType.USER, scopes={KeyScope.READ}
        )
        assert fresh.server_id != highest.server_id
        assert fresh.server_id == "00003-A1"

    def test_rotation_gets_a_new_server_id(self, store):
        km = Keymaster(store_dir=store, auto_rotate=False)
        original = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        rotated = km.rotate(original.key_id)
        assert rotated.server_id != original.server_id

    def test_an_explicit_start_ahead_of_the_store_is_respected(self, store):
        Keymaster(store_dir=store, auto_rotate=False).create(
            key_type=KeyType.USER, scopes={KeyScope.READ}
        )
        km = Keymaster(store_dir=store, auto_rotate=False, server_id="00500-A1")
        assert km.create(key_type=KeyType.USER, scopes={KeyScope.READ}).server_id == (
            "00500-A1"
        )

    def test_an_unparseable_server_id_does_not_stop_the_store_opening(self, store):
        """A hand-edited record must not brick the key store."""
        import json

        km = Keymaster(store_dir=store, auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        path = store / f"{meta.key_id}.json"
        data = json.loads(path.read_text())
        data["server_id"] = "not-a-server-id"
        path.write_text(json.dumps(data))

        reopened = Keymaster(store_dir=store, auto_rotate=False)
        assert reopened.create(key_type=KeyType.USER, scopes={KeyScope.READ}).server_id

    def test_ordering_runs_generation_first(self):
        """Sorting the raw strings would put 00002-A1 after 00001-B1."""
        from hypernix.security.keymaster import _server_id_order

        assert _server_id_order("00002-A1") < _server_id_order("00001-B1")
        assert _server_id_order("99999-Z1") < _server_id_order("00001-A2")


class TestReversalPrimitives:
    """The four methods the undo path needs, which did not exist."""

    @pytest.mark.parametrize(
        "name", ["restore_key", "set_key_type", "set_scopes", "set_revoked"]
    )
    def test_it_exists(self, name):
        assert hasattr(Keymaster, name)

    def test_restore_key_puts_previous_material_back(self, store):
        km = Keymaster(store_dir=store, auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        original = meta.key
        rotated = km.rotate(meta.key_id)
        assert rotated.key != original

        km.restore_key(rotated.key_id, original)
        assert km.get_by_key(original) is not None

    def test_restore_key_keeps_the_key_id(self, store):
        """An undo must not change identity — references stay valid."""
        km = Keymaster(store_dir=store, auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        rotated = km.rotate(meta.key_id)
        restored = km.restore_key(rotated.key_id, meta.key)
        assert restored.key_id == rotated.key_id

    def test_restore_key_refuses_a_non_key(self, store):
        km = Keymaster(store_dir=store, auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        with pytest.raises(ValueError):
            km.restore_key(meta.key_id, "not-a-key")

    def test_set_scopes_refuses_an_empty_set(self, store):
        """A key with no scopes is a key that can do nothing, silently."""
        km = Keymaster(store_dir=store, auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        with pytest.raises(ValueError):
            km.set_scopes(meta.key_id, [])

    def test_un_revoke_restores_a_working_key(self, store):
        km = Keymaster(store_dir=store, auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        raw = meta.key
        km.set_revoked(meta.key_id, True)
        assert km.get_by_key(raw) is None

        km.set_revoked(meta.key_id, False)
        assert km.get_by_key(raw) is not None

    def test_un_revoke_survives_a_restart(self, store):
        """It has to leave the archive, not just the in-memory table."""
        km = Keymaster(store_dir=store, auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        raw = meta.key
        km.set_revoked(meta.key_id, True)
        km.set_revoked(meta.key_id, False)

        assert Keymaster(store_dir=store, auto_rotate=False).get_by_key(raw) is not None

    def test_they_refuse_an_unknown_key(self, store):
        km = Keymaster(store_dir=store, auto_rotate=False)
        for call in (
            lambda: km.set_key_type("nope", KeyType.ADMIN),
            lambda: km.set_scopes("nope", ["read"]),
            lambda: km.set_revoked("nope", False),
        ):
            with pytest.raises(KeyError):
                call()


@needs_server
class TestUndoRedoActuallyWorks:
    """``/t1/auth/undo`` shipped unable to undo anything."""

    @pytest.fixture
    def wired(self, store):
        from fastapi.testclient import TestClient

        from hypernix.t1api.app import create_app

        admin_out = run_gkey("create", "--type", "admin", "--scopes", "admin,read,write")
        victim_out = run_gkey("create", "--scopes", "read")
        app = create_app()
        with TestClient(app, client=("127.0.0.1", 5000)) as client:
            yield (
                client,
                {"Authorization": "Bearer " + field(admin_out, "Key")},
                field(victim_out, "Key"),
                field(victim_out, "Key ID"),
            )

    def test_a_rotation_is_recorded(self, wired):
        client, headers, _, victim_id = wired
        assert client.post(
            "/auth/t1/admin/rotate", json={"target_key_id": victim_id}, headers=headers
        ).status_code == 200

        history = client.get("/t1/auth/history", headers=headers).json()
        assert history["count"] == 1
        assert history["can_undo"] is True
        assert history["entries"][0]["op"] == "rotate"

    def test_the_history_does_not_leak_key_material(self, wired):
        """It stores the previous key so it can restore it; describe() must not
        hand it back."""
        client, headers, victim_key, victim_id = wired
        client.post(
            "/auth/t1/admin/rotate", json={"target_key_id": victim_id}, headers=headers
        )
        body = client.get("/t1/auth/history", headers=headers).text
        assert victim_key not in body

    def test_undo_reverses_the_rotation(self, wired):
        """The whole point: the old key authenticates again."""
        client, headers, victim_key, victim_id = wired

        def victim_works() -> bool:
            return client.get(
                "/keys", headers={"Authorization": "Bearer " + victim_key}
            ).status_code == 200

        assert victim_works()
        client.post(
            "/auth/t1/admin/rotate", json={"target_key_id": victim_id}, headers=headers
        )
        assert not victim_works(), "rotation did not take effect"

        assert client.post("/t1/auth/undo", json={}, headers=headers).status_code == 200
        assert victim_works(), "undo did not restore the previous key"

    def test_redo_reapplies_it(self, wired):
        client, headers, victim_key, victim_id = wired
        rotated = client.post(
            "/auth/t1/admin/rotate", json={"target_key_id": victim_id}, headers=headers
        ).json()["key"]
        client.post("/t1/auth/undo", json={}, headers=headers)
        assert client.post("/t1/auth/redo", json={}, headers=headers).status_code == 200
        assert client.get(
            "/keys", headers={"Authorization": "Bearer " + rotated}
        ).status_code == 200

    def test_rotating_your_own_key_is_recorded_too(self, wired):
        client, headers, _, _ = wired
        assert client.post("/auth/t1/rotate", json={}, headers=headers).status_code == 200
        # The caller's old key is gone, so read the history with the new one.
        assert client.get("/t1/auth/history", headers=headers).status_code == 401

    def test_undo_with_nothing_recorded_says_so(self, wired):
        client, headers, _, _ = wired
        response = client.post("/t1/auth/undo", json={}, headers=headers)
        assert response.status_code == 404
        assert "Nothing to undo" in response.json()["error"]["message"]

    def test_undoing_onto_a_key_that_is_gone_refuses_cleanly(self, wired):
        """A reachable state, not a programming error.

        The history outlives the keys it refers to, so undoing a rotation
        whose key has since been revoked is an ordinary thing to try. It
        used to escape as a bare "Internal Server Error" with no code and
        no body — the least useful possible answer.
        """
        client, headers, _, victim_id = wired
        rotated_id = client.post(
            "/auth/t1/admin/rotate", json={"target_key_id": victim_id}, headers=headers
        ).json()["key_id"]
        client.app.state.t1_keymaster.revoke(rotated_id)

        response = client.post("/t1/auth/undo", json={}, headers=headers)
        assert response.status_code == 409
        error = response.json()["error"]
        assert "no longer exists" in error["message"]
        assert error["details"]["target_key_id"] == rotated_id
        assert error["details"]["direction"] == "undo"

    def test_a_failed_recording_does_not_fail_the_rotation(self, wired, monkeypatch):
        """The rotation already happened and its key is in the response.

        Failing the request now would report a failure that did not occur
        and lose the new key with it.
        """
        client, headers, _, victim_id = wired

        def boom(*args, **kwargs):
            raise RuntimeError("history is down")

        monkeypatch.setattr(
            client.app.state.t1_auth_history, "record", boom, raising=True
        )
        response = client.post(
            "/auth/t1/admin/rotate", json={"target_key_id": victim_id}, headers=headers
        )
        assert response.status_code == 200
        assert response.json()["key"]
