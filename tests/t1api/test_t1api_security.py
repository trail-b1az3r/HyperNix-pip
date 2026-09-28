"""Tests for hypernix.t1api.security — the SSRF and path-traversal guards
shared by the server registry and module system."""
from __future__ import annotations

import pytest

from hypernix.t1api.errors import T1APIError, T1ErrorCode
from hypernix.t1api.security import sanitize_module_path, validate_remote_address


class TestValidateRemoteAddress:
    def test_public_https_url_passes(self):
        assert validate_remote_address("https://example.com/module.tar.gz")

    def test_public_http_url_passes(self):
        assert validate_remote_address("http://example.com/module.tar.gz")

    @pytest.mark.parametrize("scheme", ["file", "ftp", "gopher", "javascript", "data"])
    def test_non_http_schemes_blocked(self, scheme):
        with pytest.raises(T1APIError) as exc_info:
            validate_remote_address(f"{scheme}://example.com/x")
        assert exc_info.value.code == T1ErrorCode.SSRF_BLOCKED

    def test_cloud_metadata_ip_always_blocked(self):
        with pytest.raises(T1APIError) as exc_info:
            validate_remote_address("http://169.254.169.254/latest/meta-data/")
        assert exc_info.value.code == T1ErrorCode.SSRF_BLOCKED

    def test_cloud_metadata_ip_blocked_even_with_allow_private(self):
        with pytest.raises(T1APIError):
            validate_remote_address("http://169.254.169.254/", allow_private=True)

    def test_private_ip_blocked_by_default(self):
        with pytest.raises(T1APIError) as exc_info:
            validate_remote_address("http://192.168.1.5:8080/")
        assert exc_info.value.code == T1ErrorCode.SSRF_BLOCKED

    def test_private_ip_allowed_with_flag(self):
        assert validate_remote_address("http://192.168.1.5:8080/", allow_private=True)

    def test_localhost_blocked_by_default(self):
        with pytest.raises(T1APIError):
            validate_remote_address("http://localhost:8000/")

    def test_localhost_allowed_with_flag(self):
        assert validate_remote_address("http://localhost:8000/", allow_private=True)

    def test_empty_host_blocked(self):
        with pytest.raises(T1APIError):
            validate_remote_address("http:///path-only")


class TestSanitizeModulePath:
    def test_normal_relative_path_resolves_under_base(self, tmp_path):
        base = tmp_path / "modules"
        base.mkdir()
        result = sanitize_module_path("mymodule/v1.tar.gz", base)
        assert result.is_relative_to(base.resolve())

    def test_parent_traversal_rejected(self, tmp_path):
        base = tmp_path / "modules"
        base.mkdir()
        with pytest.raises(T1APIError) as exc_info:
            sanitize_module_path("../../../etc/passwd", base)
        assert exc_info.value.code == T1ErrorCode.PATH_TRAVERSAL_REJECTED

    def test_absolute_path_override_rejected(self, tmp_path):
        base = tmp_path / "modules"
        base.mkdir()
        with pytest.raises(T1APIError) as exc_info:
            sanitize_module_path("/etc/passwd", base)
        assert exc_info.value.code == T1ErrorCode.PATH_TRAVERSAL_REJECTED

    def test_sneaky_traversal_with_valid_prefix_rejected(self, tmp_path):
        base = tmp_path / "modules"
        base.mkdir()
        with pytest.raises(T1APIError):
            sanitize_module_path("safe_looking/../../outside", base)
