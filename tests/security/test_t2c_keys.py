"""v2.1 (T2C) keys: minted, sealed, sent, opened, revoked.

The claims a T2C key makes, each checked here against the real key store
and the real server:

* the key that crosses the network changes every day, and one from
  outside the grace window is refused;
* only the server that issued it can open it;
* a device is bound to one key — its secret cannot vouch for another;
* the visible access level cannot be edited;
* a revoked device's keys stop working at once;
* the kit, which makes keys, is refused if it is ever sent as one.
"""
from __future__ import annotations

import base64
import contextlib
import datetime
import io
import os
import re

import pytest

pytest.importorskip("cryptography")

from hypernix.security import rotorvault as rv  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.security.t2c import (  # noqa: E402
    T2CAuthority,
    T2CError,
    T2CKit,
    build_inner,
    compose_key,
    today_utc,
    wrap_device_secret,
)
from hypernix.security.t2keys import T2KeyGenerator  # noqa: E402

_KEY_CACHE: dict[str, object] = {}


@pytest.fixture(autouse=True)
def fast_rsa(monkeypatch):
    """One 2048-bit key for the whole run: 3072-bit generation is seconds
    each, and what is under test is the protocol, not the key size."""
    def cached(bits: int = 3072):
        if "key" not in _KEY_CACHE:
            _KEY_CACHE["key"] = rv.generate_rsa_private_key.__wrapped__(2048)
        return _KEY_CACHE["key"]

    cached.__wrapped__ = getattr(rv.generate_rsa_private_key, "__wrapped__", rv.generate_rsa_private_key)
    monkeypatch.setattr(rv, "generate_rsa_private_key", cached)


def _t1_and_t2(level: int = 3) -> tuple[str, str]:
    from hypernix.security.keymaster import T1KeyGenerator

    t1 = T1KeyGenerator.generate()
    return t1, T2KeyGenerator.from_t1(t1, access_level=level).raw


# ---------------------------------------------------------------------------
# The authority, on its own
# ---------------------------------------------------------------------------


class TestTheAuthority:
    def test_a_kit_makes_a_key_the_server_opens(self, tmp_path):
        authority = T2CAuthority(tmp_path / "t2c")
        t1, t2 = _t1_and_t2(4)
        kit = authority.issue_kit(t2, bound_key=t1, access_level=4)
        opened, device = authority.open(kit.key_for())
        assert (opened, device) == (t2, kit.device_id)

    def test_the_key_changes_every_day(self, tmp_path):
        authority = T2CAuthority(tmp_path / "t2c")
        t1, t2 = _t1_and_t2()
        kit = authority.issue_kit(t2, bound_key=t1, access_level=3)
        day = datetime.date(2026, 9, 23)
        keys = {kit.key_for(day + datetime.timedelta(days=n)) for n in range(3)}
        assert len(keys) == 3
        assert all(t2 not in key for key in keys)

    def test_the_grace_window_is_one_day_either_side(self, tmp_path):
        authority = T2CAuthority(tmp_path / "t2c")
        t1, t2 = _t1_and_t2()
        kit = authority.issue_kit(t2, bound_key=t1, access_level=3)
        today = datetime.date(2026, 9, 23)
        for offset in (-1, 0, 1):
            authority.open(kit.key_for(today + datetime.timedelta(days=offset)), today=today)
        for offset in (-2, 2, 30):
            with pytest.raises(T2CError, match="not valid today"):
                authority.open(kit.key_for(today + datetime.timedelta(days=offset)), today=today)

    def test_another_server_cannot_open_it(self, tmp_path):
        """Same device registration copied across — the inner is sealed for
        the issuing server's RSA key and nobody else's."""
        issuing = T2CAuthority(tmp_path / "a")
        t1, t2 = _t1_and_t2()
        kit = issuing.issue_kit(t2, bound_key=t1, access_level=3)
        other = T2CAuthority(tmp_path / "b")
        other._private_key = rv.generate_rsa_private_key.__wrapped__(2048)
        other.register_device(bound_key=t1, secret=kit.secret)
        devices = other._load_devices()
        (new_id,) = devices
        devices[kit.device_id] = devices.pop(new_id)
        other._save_devices(devices)
        with pytest.raises(T2CError, match="different server"):
            other.open(kit.key_for())

    def test_a_device_cannot_vouch_for_a_different_key(self, tmp_path):
        authority = T2CAuthority(tmp_path / "t2c")
        mine, _ = _t1_and_t2()
        _, someone_elses = _t1_and_t2()
        device_id, secret = authority.register_device(bound_key=mine)
        inner = build_inner(authority.private_key().public_key(), someone_elses)
        key = compose_key(device_id, secret, inner, 3, today_utc())
        with pytest.raises(T2CError, match="different key"):
            authority.open(key)

    def test_the_visible_level_cannot_be_raised(self, tmp_path):
        authority = T2CAuthority(tmp_path / "t2c")
        t1, t2 = _t1_and_t2(level=2)
        kit = authority.issue_kit(t2, bound_key=t1, access_level=2)
        forged = re.sub(r"-2$", "-9", kit.key_for())
        with pytest.raises(T2CError, match="access level"):
            authority.open(forged)

    def test_revoking_stops_it(self, tmp_path):
        authority = T2CAuthority(tmp_path / "t2c")
        t1, t2 = _t1_and_t2()
        kit = authority.issue_kit(t2, bound_key=t1, access_level=3)
        assert authority.revoke(kit.device_id, bound_key=t1)
        with pytest.raises(T2CError, match="revoked"):
            authority.open(kit.key_for())

    def test_only_the_owner_revokes(self, tmp_path):
        authority = T2CAuthority(tmp_path / "t2c")
        t1, t2 = _t1_and_t2()
        kit = authority.issue_kit(t2, bound_key=t1, access_level=3)
        stranger, _ = _t1_and_t2()
        assert authority.revoke(kit.device_id, bound_key=stranger) is False
        authority.open(kit.key_for())

    def test_nothing_is_written_until_it_is_used(self, tmp_path):
        T2CAuthority(tmp_path / "t2c")
        assert not (tmp_path / "t2c").exists()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_the_private_key_is_owner_only(self, tmp_path):
        authority = T2CAuthority(tmp_path / "t2c")
        authority.fingerprint()
        assert (tmp_path / "t2c" / "server_rsa.pem").stat().st_mode & 0o077 == 0


class TestTheKit:
    def test_it_round_trips_as_text(self, tmp_path):
        authority = T2CAuthority(tmp_path / "t2c")
        t1, t2 = _t1_and_t2()
        kit = authority.issue_kit(t2, bound_key=t1, access_level=3, label="laptop")
        text = kit.to_text()
        assert text.startswith("T2CK_")
        assert T2CKit.from_text(text) == kit

    def test_a_damaged_kit_says_so(self):
        with pytest.raises(T2CError, match="damaged"):
            T2CKit.from_text("T2CK_" + base64.urlsafe_b64encode(b"{}").decode())
        with pytest.raises(T2CError, match="T2CK_"):
            T2CKit.from_text("T2_nope")

    def test_describe_never_includes_the_secret(self, tmp_path):
        authority = T2CAuthority(tmp_path / "t2c")
        t1, t2 = _t1_and_t2()
        kit = authority.issue_kit(t2, bound_key=t1, access_level=3)
        text = repr(kit.describe())
        assert kit.inner not in text
        assert base64.urlsafe_b64encode(kit.secret).decode() not in text


# ---------------------------------------------------------------------------
# gkey create -v v2.1
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("hypernix.security.keymaster._DEFAULT_STORE", tmp_path / "keymaster")
    monkeypatch.setattr("hypernix.security.gatekeeper._DEFAULT_DATA", tmp_path / "gatekeeper")
    return tmp_path


def _gkey(*argv: str) -> tuple[int, str]:
    from hypernix.security.gkey_cli import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, re.sub(r"\x1b\[[0-9;]*m", "", out.getvalue() + err.getvalue())


def _line(text: str, label: str) -> str:
    for line in text.splitlines():
        if line.startswith(f"{label}:") or line.strip().startswith(f"{label}:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError(f"no {label} in {text}")


class TestGkey:
    def test_it_mints_a_key_and_a_kit_that_authenticate(self, store):
        from hypernix.security.gatekeeper import Gatekeeper
        from hypernix.t1api.auth import T1AuthService

        code, text = _gkey("create", "-v", "v2.1", "--level", "5", "--scopes", "read,write")
        assert code == 0, text
        key, kit_text = _line(text, "Key"), _line(text, "Kit")
        assert key.startswith("T2C_") and key.endswith("-5")
        kit = T2CKit.from_text(kit_text)

        km = Keymaster(auto_rotate=False)
        service = T1AuthService(km, Gatekeeper(keymaster=km), token_secret="x" * 32)
        context = service.validate_key(key)
        assert (context.t2_family, context.t2_access_level) == ("T2C", 5)
        assert context.t2c_device_id == kit.device_id
        tomorrow = today_utc() + datetime.timedelta(days=1)
        assert service.validate_key(kit.key_for(tomorrow)).key_id == context.key_id

    def test_the_clear_key_is_not_printed(self, store):
        """The point of v2.1 is a key that is never written down in the clear."""
        code, text = _gkey("create", "-v", "v2.1")
        assert code == 0, text
        assert "T1_" not in text
        assert "T2_" not in text.replace("T2C_", "").replace("T2CK_", "")


# ---------------------------------------------------------------------------
# Through the server
# ---------------------------------------------------------------------------


@pytest.fixture
def api(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
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
        ),
        keymaster=km,
        gatekeeper=Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False),
    )
    return TestClient(app), user.key, tmp_path


def _client_side_kit(client, t1_key: str, level: int = 3) -> T2CKit:
    """What waiter does for `serv -E`: nothing secret leaves in the clear."""
    published = client.get("/auth/t2c/public-key").json()
    secret = os.urandom(32)
    registered = client.post("/auth/t2c/devices", json={
        "key": t1_key, "access_level": level, "label": "test",
        "wrapped_secret": wrap_device_secret(published["public_key_pem"], secret),
    })
    assert registered.status_code == 200, registered.text
    t2 = T2KeyGenerator.from_t1(t1_key, access_level=level).raw
    return T2CKit(device_id=registered.json()["device_id"], secret=secret,
                  inner=build_inner(published["public_key_pem"], t2), access_level=level,
                  server_fingerprint=published["fingerprint"])


class TestTheApi:
    def test_the_public_key_is_public(self, api):
        client, _, _ = api
        body = client.get("/auth/t2c/public-key").json()
        assert body["algorithm"] == "RSA-OAEP-SHA256"
        assert "BEGIN PUBLIC KEY" in body["public_key_pem"]
        assert body["fingerprint"].startswith("SHA256:")

    def test_register_then_authenticate_then_revoke(self, api):
        client, t1_key, _ = api
        kit = _client_side_kit(client, t1_key)
        headers = {"Authorization": f"Bearer {kit.key_for()}"}
        listed = client.get("/auth/t2c/devices", headers=headers)
        assert listed.status_code == 200, listed.text
        assert [d["device_id"] for d in listed.json()["devices"]] == [kit.device_id]

        validated = client.post("/auth/t1/validate", json={"key": kit.key_for()})
        assert validated.status_code == 200, validated.text

        gone = client.delete(f"/auth/t2c/devices/{kit.device_id}", headers=headers)
        assert gone.status_code == 200
        after = client.get("/auth/t2c/devices", headers=headers)
        assert after.status_code == 401
        assert "revoked" in after.text

    def test_the_kit_is_refused_as_a_credential(self, api):
        client, t1_key, _ = api
        kit = _client_side_kit(client, t1_key)
        got = client.get("/auth/t2c/devices", headers={"Authorization": f"Bearer {kit.to_text()}"})
        assert got.status_code == 401
        assert "kit, not a key" in got.text

    def test_a_device_is_registered_with_the_key_itself(self, api):
        client, t1_key, _ = api
        kit = _client_side_kit(client, t1_key)
        pem = client.get("/auth/t2c/public-key").json()["public_key_pem"]
        for key in (kit.key_for(), "T1S.x.y"):
            got = client.post("/auth/t2c/devices", json={
                "key": key, "wrapped_secret": wrap_device_secret(pem, os.urandom(32))})
            assert got.status_code == 400, got.text

    def test_a_secret_not_wrapped_for_this_server_is_refused(self, api):
        client, t1_key, _ = api
        got = client.post("/auth/t2c/devices", json={
            "key": t1_key, "wrapped_secret": base64.b64encode(os.urandom(256)).decode()})
        assert got.status_code == 400

    def test_an_unknown_key_cannot_register(self, api):
        client, _, _ = api
        from hypernix.security.keymaster import T1KeyGenerator

        pem = client.get("/auth/t2c/public-key").json()["public_key_pem"]
        got = client.post("/auth/t2c/devices", json={
            "key": T1KeyGenerator.generate(), "wrapped_secret": wrap_device_secret(pem, os.urandom(32))})
        assert got.status_code == 401

    def test_the_rsa_key_lives_beside_the_key_store(self, api):
        client, _, tmp_path = api
        assert not (tmp_path / "km" / "t2c").exists()
        client.get("/auth/t2c/public-key")
        assert (tmp_path / "km" / "t2c" / "server_rsa.pem").exists()
