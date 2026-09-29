"""Tests for hypernix.t1api.modules — the Module Registry."""
from __future__ import annotations

import pytest

from hypernix.t1api.db import SQLiteBackend
from hypernix.t1api.errors import T1APIError, T1ErrorCode
from hypernix.t1api.modules import ModuleRegistry, ModuleStatus, SourceType


@pytest.fixture
def registry(tmp_path) -> ModuleRegistry:
    return ModuleRegistry(SQLiteBackend(tmp_path / "t1api.sqlite3"), storage_dir=tmp_path / "modules")


class TestCreate:
    def test_create_starts_as_draft(self, registry: ModuleRegistry):
        m = registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        assert m.status == ModuleStatus.DRAFT
        assert m.source_type == SourceType.NONE

    def test_duplicate_name_version_rejected(self, registry: ModuleRegistry):
        registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        with pytest.raises(T1APIError) as exc_info:
            registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        assert exc_info.value.code == T1ErrorCode.MODULE_ALREADY_EXISTS

    def test_same_name_different_version_allowed(self, registry: ModuleRegistry):
        registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        m2 = registry.create(name="mymodule", version="1.1.0", owner_key_id="user1")
        assert m2.version == "1.1.0"


class TestLocalUpload:
    def test_upload_sets_active_and_checksum(self, registry: ModuleRegistry):
        m = registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        content = b"module bytes"
        updated = registry.upload_local(m.module_id, content, filename="module.bin")
        assert updated.status == ModuleStatus.ACTIVE
        assert updated.size_bytes == len(content)
        assert updated.checksum is not None

    def test_content_roundtrips(self, registry: ModuleRegistry):
        m = registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        content = b"exact bytes to verify" * 50
        registry.upload_local(m.module_id, content, filename="module.bin")
        assert registry.read_content(m.module_id) == content

    def test_checksum_is_sha256_of_content(self, registry: ModuleRegistry):
        import hashlib

        m = registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        content = b"checksum me"
        updated = registry.upload_local(m.module_id, content, filename="module.bin")
        assert updated.checksum == hashlib.sha256(content).hexdigest()

    def test_path_traversal_in_filename_rejected(self, registry: ModuleRegistry):
        m = registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        with pytest.raises(T1APIError) as exc_info:
            registry.upload_local(m.module_id, b"evil", filename="../../../etc/passwd")
        assert exc_info.value.code == T1ErrorCode.PATH_TRAVERSAL_REJECTED

    def test_upload_to_unknown_module_raises_not_found(self, registry: ModuleRegistry):
        with pytest.raises(T1APIError) as exc_info:
            registry.upload_local("nope", b"x", filename="f.bin")
        assert exc_info.value.code == T1ErrorCode.MODULE_NOT_FOUND

    def test_read_content_before_upload_raises(self, registry: ModuleRegistry):
        m = registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        with pytest.raises(T1APIError):
            registry.read_content(m.module_id)


class TestRemoteSource:
    def test_register_remote_source_sets_pending_fetch(self, registry: ModuleRegistry):
        m = registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        updated = registry.register_remote_source(m.module_id, "https://example.com/module.tar.gz")
        assert updated.status == ModuleStatus.PENDING_FETCH
        assert updated.source_type == SourceType.REMOTE

    def test_remote_source_ssrf_blocked(self, registry: ModuleRegistry):
        m = registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        with pytest.raises(T1APIError) as exc_info:
            registry.register_remote_source(m.module_id, "http://169.254.169.254/secret")
        assert exc_info.value.code == T1ErrorCode.SSRF_BLOCKED

    def test_remote_source_private_ip_needs_explicit_allow(self, registry: ModuleRegistry):
        m = registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        with pytest.raises(T1APIError):
            registry.register_remote_source(m.module_id, "http://192.168.1.1/module.tar.gz")
        # explicit allow works
        updated = registry.register_remote_source(
            m.module_id, "http://192.168.1.1/module.tar.gz", allow_private=True
        )
        assert updated.status == ModuleStatus.PENDING_FETCH


class TestSync:
    def test_mark_synced_adds_server(self, registry: ModuleRegistry):
        m = registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        updated = registry.mark_synced(m.module_id, "server-1")
        assert updated.deployed_servers == ["server-1"]

    def test_mark_synced_is_idempotent(self, registry: ModuleRegistry):
        m = registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        registry.mark_synced(m.module_id, "server-1")
        updated = registry.mark_synced(m.module_id, "server-1")
        assert updated.deployed_servers == ["server-1"]

    def test_mark_synced_multiple_servers(self, registry: ModuleRegistry):
        m = registry.create(name="mymodule", version="1.0.0", owner_key_id="user1")
        registry.mark_synced(m.module_id, "server-1")
        updated = registry.mark_synced(m.module_id, "server-2")
        assert set(updated.deployed_servers) == {"server-1", "server-2"}


class TestListAndDelete:
    def test_list_filters_by_status(self, registry: ModuleRegistry):
        m1 = registry.create(name="a", version="1.0.0", owner_key_id="user1")
        registry.create(name="b", version="1.0.0", owner_key_id="user1")
        registry.upload_local(m1.module_id, b"x", filename="a.bin")
        active = registry.list(status=ModuleStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].name == "a"

    def test_list_filters_by_owner(self, registry: ModuleRegistry):
        registry.create(name="a", version="1.0.0", owner_key_id="user1")
        registry.create(name="b", version="1.0.0", owner_key_id="user2")
        mine = registry.list(owner_key_id="user1")
        assert len(mine) == 1
        assert mine[0].name == "a"

    def test_delete_removes_entry_and_file(self, registry: ModuleRegistry):
        m = registry.create(name="a", version="1.0.0", owner_key_id="user1")
        registry.upload_local(m.module_id, b"content", filename="a.bin")
        registry.delete(m.module_id)
        assert registry.get(m.module_id) is None

    def test_delete_unknown_raises_module_not_found(self, registry: ModuleRegistry):
        with pytest.raises(T1APIError) as exc_info:
            registry.delete("nope")
        assert exc_info.value.code == T1ErrorCode.MODULE_NOT_FOUND

    def test_require_unknown_raises_module_not_found(self, registry: ModuleRegistry):
        with pytest.raises(T1APIError) as exc_info:
            registry.require("nope")
        assert exc_info.value.code == T1ErrorCode.MODULE_NOT_FOUND
