"""Tests for hypernix.t1api.servers — the Server Registry."""
from __future__ import annotations

import pytest

from hypernix.t1api.db import SQLiteBackend
from hypernix.t1api.errors import T1APIError, T1ErrorCode
from hypernix.t1api.servers import ServerRegistry, ServerStatus, TrustLevel


@pytest.fixture
def registry(tmp_path) -> ServerRegistry:
    return ServerRegistry(SQLiteBackend(tmp_path / "t1api.sqlite3"))


class TestRegister:
    def test_register_returns_entry_with_untrusted_default(self, registry: ServerRegistry):
        s = registry.register(name="prod-1", address="https://prod1.example.com", registered_by="admin")
        assert s.trust_level == TrustLevel.UNTRUSTED
        assert s.server_id

    def test_private_address_rejected_by_default(self, registry: ServerRegistry):
        with pytest.raises(T1APIError) as exc_info:
            registry.register(name="x", address="http://192.168.1.5:9000", registered_by="admin")
        assert exc_info.value.code == T1ErrorCode.SSRF_BLOCKED

    def test_private_address_allowed_with_flag(self, registry: ServerRegistry):
        s = registry.register(
            name="local-1",
            address="http://100.64.1.2:9000",
            registered_by="admin",
            allow_private_address=True,
            trust_level=TrustLevel.LOCAL,
        )
        assert s.trust_level == TrustLevel.LOCAL

    def test_two_registrations_get_distinct_ids(self, registry: ServerRegistry):
        s1 = registry.register(name="a", address="https://a.example.com", registered_by="admin")
        s2 = registry.register(name="b", address="https://b.example.com", registered_by="admin")
        assert s1.server_id != s2.server_id


class TestTrust:
    def test_untrusted_server_fails_require_trusted(self, registry: ServerRegistry):
        s = registry.register(name="x", address="https://x.example.com", registered_by="admin")
        with pytest.raises(T1APIError) as exc_info:
            registry.require_trusted(s.server_id)
        assert exc_info.value.code == T1ErrorCode.SERVER_UNTRUSTED

    def test_promoted_server_passes_require_trusted(self, registry: ServerRegistry):
        s = registry.register(name="x", address="https://x.example.com", registered_by="admin")
        registry.update(s.server_id, trust_level=TrustLevel.TRUSTED)
        result = registry.require_trusted(s.server_id)
        assert result.trust_level == TrustLevel.TRUSTED

    def test_local_server_passes_require_trusted(self, registry: ServerRegistry):
        s = registry.register(
            name="local", address="http://127.0.0.1:9000", registered_by="admin",
            allow_private_address=True, trust_level=TrustLevel.LOCAL,
        )
        result = registry.require_trusted(s.server_id)
        assert result.trust_level == TrustLevel.LOCAL


class TestLookupAndList:
    def test_get_unknown_returns_none(self, registry: ServerRegistry):
        assert registry.get("nope") is None

    def test_require_unknown_raises_server_not_found(self, registry: ServerRegistry):
        with pytest.raises(T1APIError) as exc_info:
            registry.require("nope")
        assert exc_info.value.code == T1ErrorCode.SERVER_NOT_FOUND

    def test_list_filters_by_trust_level(self, registry: ServerRegistry):
        registry.register(name="a", address="https://a.example.com", registered_by="admin")
        registry.register(
            name="b", address="http://127.0.0.1", registered_by="admin",
            allow_private_address=True, trust_level=TrustLevel.LOCAL,
        )
        untrusted = registry.list(trust_level=TrustLevel.UNTRUSTED)
        assert len(untrusted) == 1
        assert untrusted[0].name == "a"

    def test_list_filters_by_status(self, registry: ServerRegistry):
        s = registry.register(name="a", address="https://a.example.com", registered_by="admin")
        registry.update(s.server_id, status=ServerStatus.ONLINE)
        registry.register(name="b", address="https://b.example.com", registered_by="admin")
        online = registry.list(status=ServerStatus.ONLINE)
        assert len(online) == 1
        assert online[0].name == "a"


class TestUpdate:
    def test_update_touches_updated_at(self, registry: ServerRegistry):
        s = registry.register(name="a", address="https://a.example.com", registered_by="admin")
        updated = registry.update(s.server_id, name="renamed")
        assert updated.name == "renamed"
        assert updated.updated_at >= s.updated_at

    def test_update_last_seen_only_when_requested(self, registry: ServerRegistry):
        s = registry.register(name="a", address="https://a.example.com", registered_by="admin")
        assert s.last_seen is None
        touched = registry.update(s.server_id, touch_last_seen=True)
        assert touched.last_seen is not None

    def test_update_unknown_raises(self, registry: ServerRegistry):
        with pytest.raises(T1APIError):
            registry.update("nope", name="x")


class TestDelete:
    def test_delete_removes_entry(self, registry: ServerRegistry):
        s = registry.register(name="a", address="https://a.example.com", registered_by="admin")
        registry.delete(s.server_id)
        assert registry.get(s.server_id) is None

    def test_delete_unknown_raises_server_not_found(self, registry: ServerRegistry):
        with pytest.raises(T1APIError) as exc_info:
            registry.delete("nope")
        assert exc_info.value.code == T1ErrorCode.SERVER_NOT_FOUND
