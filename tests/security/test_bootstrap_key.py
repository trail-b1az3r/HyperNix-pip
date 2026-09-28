"""The loopback-only admin key a new server issues itself.

A fresh T1 API has an empty key store and every route that could fill it
is admin-only — a closed loop. The documented way out is running ``gkey``
on the server box, which works and is one more thing to discover at
exactly the moment someone is finding out whether any of this runs at
all. It is why ``waiter hyperlink pair`` could not be used on a new
server: pairing is admin-only and there was no admin.

Three limits make printing an admin key on a terminal reasonable, and
each is tested here: loopback only, three days, once.
"""
from __future__ import annotations

import time

import pytest

from hypernix.security.keymaster import Keymaster
from hypernix.t1api.bootstrap import (
    BOOTSTRAP_TTL_SECONDS,
    bootstrap_banner,
    ensure_bootstrap_key,
    is_bootstrap_key,
    is_loopback,
)

try:
    import fastapi  # noqa: F401

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

needs_server = pytest.mark.skipif(not _HAS_FASTAPI, reason="needs the [t1api] extra")


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
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1api.sqlite3"))
    monkeypatch.setenv("T1_HYPERLINK_ENABLED", "1")
    return tmp_path / "keymaster"


class TestIsLoopback:
    @pytest.mark.parametrize(
        "address", ["127.0.0.1", "127.0.0.53", "::1", "[::1]", "::1%lo0"]
    )
    def test_local_addresses(self, address):
        assert is_loopback(address)

    @pytest.mark.parametrize(
        "address", ["192.168.1.5", "10.0.0.1", "100.64.1.2", "8.8.8.8", "::"]
    )
    def test_remote_addresses(self, address):
        assert not is_loopback(address)

    @pytest.mark.parametrize("address", ["", "   ", "localhost", "not-an-ip", None])
    def test_anything_unparseable_is_not_loopback(self, address):
        """The one check between a printed admin key and the network.

        An address it cannot parse must not get the benefit of the doubt,
        and a hostname is not an address however local it looks.
        """
        assert not is_loopback(address)


class TestMinting:
    def test_a_new_store_gets_a_key(self, store):
        key = ensure_bootstrap_key(Keymaster(store_dir=store, auto_rotate=False))
        assert key is not None and key.created
        assert key.key.startswith("T2_")

    def test_it_is_an_admin_key(self, store):
        from hypernix.security.keymaster import KeyScope, KeyType

        km = Keymaster(store_dir=store, auto_rotate=False)
        key = ensure_bootstrap_key(km)
        meta = km.get(key.key_id)
        assert meta.key_type is KeyType.ADMIN
        assert KeyScope.ADMIN in meta.scopes

    def test_it_carries_a_password_component(self, store):
        key = ensure_bootstrap_key(Keymaster(store_dir=store, auto_rotate=False))
        assert key.password and key.password in key.key

    def test_it_lasts_three_days(self, store):
        key = ensure_bootstrap_key(Keymaster(store_dir=store, auto_rotate=False))
        assert BOOTSTRAP_TTL_SECONDS == 3 * 24 * 3600
        assert 71.0 < key.expires_in_hours <= 72.0

    def test_a_second_call_does_not_mint_another(self, store):
        """Restarting a server must not litter it with admin keys."""
        km = Keymaster(store_dir=store, auto_rotate=False)
        first = ensure_bootstrap_key(km)
        second = ensure_bootstrap_key(km)
        assert first.created and not second.created
        assert second.key == "", "the store keeps no plaintext to show twice"

    def test_it_survives_a_restart_without_being_reissued(self, store):
        ensure_bootstrap_key(Keymaster(store_dir=store, auto_rotate=False))
        again = ensure_bootstrap_key(Keymaster(store_dir=store, auto_rotate=False))
        assert not again.created

    def test_an_expired_key_is_replaced(self, store):
        """On day four it issues another rather than locking the operator out."""
        km = Keymaster(store_dir=store, auto_rotate=False)
        ensure_bootstrap_key(km, ttl_seconds=1)
        time.sleep(1.2)
        assert ensure_bootstrap_key(km, ttl_seconds=3600).created

    def test_the_key_is_tagged_so_it_can_be_recognised(self, store):
        km = Keymaster(store_dir=store, auto_rotate=False)
        key = ensure_bootstrap_key(km)
        assert is_bootstrap_key(km.get(key.key_id))

    def test_an_ordinary_key_is_not_a_bootstrap_key(self, store):
        from hypernix.security.keymaster import KeyScope, KeyType

        km = Keymaster(store_dir=store, auto_rotate=False)
        meta = km.create(key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN})
        assert not is_bootstrap_key(meta)


class TestBanner:
    def test_it_shows_the_key_and_both_limits(self, store):
        key = ensure_bootstrap_key(Keymaster(store_dir=store, auto_rotate=False))
        text = bootstrap_banner(key)
        assert key.key in text
        assert "only from this machine" in text
        assert "72 more hours" in text

    def test_it_does_not_invent_a_port(self, store):
        """uvicorn owns the bind address, so the config does not know it.

        A copy-pasteable command that connects to the wrong place is
        worse than one that obviously needs filling in.
        """
        key = ensure_bootstrap_key(Keymaster(store_dir=store, auto_rotate=False))
        text = bootstrap_banner(key)
        assert "<port>" in text
        assert ":8000" not in text

    def test_a_configured_url_is_used_when_there_is_one(self, store):
        key = ensure_bootstrap_key(Keymaster(store_dir=store, auto_rotate=False))
        text = bootstrap_banner(key, base_url="http://desktop.ts.net:8000")
        assert "http://desktop.ts.net:8000" in text
        assert "<port>" not in text

    def test_nothing_is_shown_for_a_key_that_already_existed(self, store):
        km = Keymaster(store_dir=store, auto_rotate=False)
        ensure_bootstrap_key(km)
        assert bootstrap_banner(ensure_bootstrap_key(km)) == ""


@needs_server
class TestItWorksAndOnlyLocally:
    @pytest.fixture
    def app(self, store):
        from hypernix.t1api.app import create_app

        return create_app()

    @staticmethod
    def _client(app, address):
        from fastapi.testclient import TestClient

        return TestClient(app, client=(address, 5000))

    def test_the_server_mints_one_on_first_start(self, app):
        assert app.state.t1_bootstrap_key is not None
        assert app.state.t1_bootstrap_key.created

    def test_it_reaches_the_admin_surface_from_loopback(self, app):
        key = app.state.t1_bootstrap_key.key
        with self._client(app, "127.0.0.1") as client:
            response = client.get("/keys", headers={"Authorization": "Bearer " + key})
        assert response.status_code == 200

    def test_it_can_mint_a_pairing_code(self, app):
        """The operation that was blocking HyperLink on a new server."""
        key = app.state.t1_bootstrap_key.key
        with self._client(app, "127.0.0.1") as client:
            response = client.post(
                "/hyperlink/pair", json={"label": "phone"},
                headers={"Authorization": "Bearer " + key},
            )
        assert response.status_code == 200

    @pytest.mark.parametrize("address", ["192.168.1.50", "100.64.1.2", "8.8.8.8"])
    def test_it_is_refused_from_anywhere_else(self, app, address):
        key = app.state.t1_bootstrap_key.key
        with self._client(app, address) as client:
            response = client.get("/keys", headers={"Authorization": "Bearer " + key})
        assert response.status_code == 403
        assert response.json()["error"]["details"]["reason"] == (
            "bootstrap_key_is_local_only"
        )

    def test_the_hyperlink_path_is_covered_too(self, app):
        """HyperLink resolves T2 keys without going through
        get_auth_context, so both routes need the restriction. Enforced
        on one of two is not enforced.
        """
        key = app.state.t1_bootstrap_key.key
        with self._client(app, "192.168.1.50") as client:
            response = client.get(
                "/hyperlink/endpoints", headers={"Authorization": "Bearer " + key}
            )
        assert response.status_code == 403

    def test_an_ordinary_admin_key_still_works_remotely(self, app):
        """The restriction must apply to this key and nothing else."""
        from hypernix.security.keymaster import KeyScope, KeyType

        meta = app.state.t1_keymaster.create(
            key_type=KeyType.ADMIN,
            scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE},
        )
        with self._client(app, "192.168.1.50") as client:
            response = client.get(
                "/keys", headers={"Authorization": "Bearer " + meta.key}
            )
        assert response.status_code == 200

    def test_it_can_be_turned_off(self, store, monkeypatch):
        from hypernix.t1api.app import create_app

        monkeypatch.setenv("T1_BOOTSTRAP_KEY", "0")
        assert create_app().state.t1_bootstrap_key is None
