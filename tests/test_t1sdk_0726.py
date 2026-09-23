"""The SDK's 0.72.6 surface, against the real app: server info, conceal,
and sealing a key into a v2.1 kit that is then the client's credential."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cryptography")

from sdk_bridge import sdk_client  # noqa: E402

from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402


@pytest.fixture
def server(tmp_path, monkeypatch):
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
            server_name="sdk-box", server_owner="Ari",
        ),
        keymaster=km,
        gatekeeper=Gatekeeper(keymaster=km, data_dir=tmp_path / "gk", log_to_file=False),
    )
    return TestClient(app), user


class TestServerInfo:
    def test_no_credential_needed(self, server):
        test_client, _ = server
        info = sdk_client(test_client).server_info()
        assert (info["name"], info["owner"]) == ("sdk-box", "Ari")


class TestConceal:
    def test_round_trip(self, server):
        test_client, user = server
        client = sdk_client(test_client, user.key)
        assert client.conceal_status()["concealed"] is False
        assert client.conceal()["concealed"] is True
        assert client.conceal(False)["concealed"] is False


class TestSealKey:
    def test_the_kit_is_a_working_credential(self, server):
        test_client, user = server
        kit = sdk_client(test_client, user.key).seal_key(label="laptop", access_level=4)
        assert kit.access_level == 4
        sealed = sdk_client(test_client, kit.to_text())
        who = sealed.validate()
        assert (who["key_family"], who["access_level"], who["key_id"]) == ("T2C", 4, user.key_id)
        devices = sealed.t2c_devices()
        assert [d["device_id"] for d in devices] == [kit.device_id]

    def test_the_kit_never_goes_on_the_wire(self, server):
        test_client, user = server
        kit = sdk_client(test_client, user.key).seal_key()
        sealed = sdk_client(test_client, kit.to_text())
        wire = sealed.transport.wire_credential()
        assert wire.startswith("T2C_") and "T2CK_" not in wire

    def test_revoking_the_device_ends_it(self, server):
        from hypernix.t1sdk import T1AuthError

        test_client, user = server
        kit = sdk_client(test_client, user.key).seal_key()
        sealed = sdk_client(test_client, kit.to_text())
        sealed.t2c_revoke_device(kit.device_id)
        with pytest.raises(T1AuthError):
            sealed.validate()

    def test_a_sealed_key_cannot_be_sealed_again(self, server):
        test_client, user = server
        kit = sdk_client(test_client, user.key).seal_key()
        with pytest.raises(ValueError):
            sdk_client(test_client, kit.to_text()).seal_key()
