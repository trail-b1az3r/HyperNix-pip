"""Comprehensive tests for HyperNix v0.71.0 — Gatekeeper + Keymaster.

Covers:
  - T1 key format: generation, validation, deconstruction
  - Server ID parsing, formatting, incrementing (full cycle)
  - KeyMeta: serialisation round-trip, expiry helpers
  - Keymaster: create, get, revoke, rotate, list, export, import
  - Keymaster: auto-expiry detection (expires_soon / is_expired)
  - Keymaster: record_usage counters
  - Gatekeeper: authenticate (valid/invalid/revoked/expired keys)
  - Gatekeeper: quota enforcement (requests, tokens, sliding window)
  - Gatekeeper: record_usage + stats
  - Gatekeeper: permissions helper
  - gkey_cli: all subcommands exercised via main() in-process
"""
from __future__ import annotations

import json
import time
import uuid

import pytest

# ===========================================================================
# T1 Key Format
# ===========================================================================


class TestT1KeyFormat:
    """Tests for T1KeyGenerator."""

    def test_generate_returns_string(self):
        from hypernix.keymaster import T1KeyGenerator
        key = T1KeyGenerator.generate()
        assert isinstance(key, str)

    def test_generated_key_starts_with_prefix(self):
        from hypernix.keymaster import T1KeyGenerator
        key = T1KeyGenerator.generate()
        assert key.startswith("T1_")

    def test_generated_key_passes_validate(self):
        from hypernix.keymaster import T1KeyGenerator
        for _ in range(50):  # generate multiple to cover entropy
            key = T1KeyGenerator.generate()
            assert T1KeyGenerator.validate(key), f"Failed: {key!r}"

    def test_generated_key_custom_body_length(self):
        from hypernix.keymaster import T1KeyGenerator
        for body_len in (16, 24, 32, 48):
            key = T1KeyGenerator.generate(body_length=body_len)
            assert T1KeyGenerator.validate(key)
            # key = "T1_" + body(body_len) + suffix(9 chars)
            assert len(key) == 3 + body_len + 9

    def test_body_length_too_short_raises(self):
        from hypernix.keymaster import T1KeyGenerator
        with pytest.raises(ValueError):
            T1KeyGenerator.generate(body_length=15)

    def test_validate_rejects_wrong_prefix(self):
        from hypernix.keymaster import T1KeyGenerator
        # Must start with T1_
        assert not T1KeyGenerator.validate("T2_Aabc12@#$%&/1")
        assert not T1KeyGenerator.validate("t1_Aabc12@#$%&/1")

    def test_validate_rejects_missing_suffix_digit(self):
        from hypernix.keymaster import T1KeyGenerator
        # suffix digit must be 1-9 (not 0)
        base = T1KeyGenerator.generate()
        bad = base[:-1] + "0"
        assert not T1KeyGenerator.validate(bad)

    def test_validate_rejects_uppercase_in_ll(self):
        from hypernix.keymaster import T1KeyGenerator
        # positions -8 and -7 must be lowercase
        key = T1KeyGenerator.generate()
        # Corrupt the two lowercase letters
        bad = key[:-8] + "AB" + key[-6:]
        assert not T1KeyGenerator.validate(bad)

    def test_validate_rejects_alphanumeric_in_special(self):
        from hypernix.keymaster import T1KeyGenerator
        key = T1KeyGenerator.generate()
        # Replace special chars with letters
        bad = key[:-6] + "AAAAA" + key[-1:]
        assert not T1KeyGenerator.validate(bad)

    def test_validate_rejects_invalid_slash(self):
        from hypernix.keymaster import T1KeyGenerator
        key = T1KeyGenerator.generate()
        # Position -1 must be / or \  followed by 1-9
        bad = key[:-2] + "X" + key[-1:]
        assert not T1KeyGenerator.validate(bad)

    def test_deconstruct_valid_key(self):
        from hypernix.keymaster import T1KeyGenerator
        key = T1KeyGenerator.generate(body_length=24)
        parts = T1KeyGenerator.deconstruct(key)
        assert parts["prefix"] == "T1_"
        assert len(parts["body"]) == 24
        assert len(parts["lowercase_pair"]) == 2
        assert all(c.islower() for c in parts["lowercase_pair"])
        assert len(parts["special_chars"]) == 5
        assert parts["slash"] in ("/", "\\")
        assert parts["digit"] in "123456789"
        assert parts["suffix"] == (
            parts["lowercase_pair"]
            + parts["special_chars"]
            + parts["slash"]
            + parts["digit"]
        )

    def test_deconstruct_invalid_key_raises(self):
        from hypernix.keymaster import T1KeyGenerator
        with pytest.raises(ValueError):
            T1KeyGenerator.deconstruct("not-a-valid-key")

    def test_validate_example_keys_from_spec(self):
        """All example keys from the specification must pass validation."""
        from hypernix.keymaster import T1KeyGenerator
        examples = [
            "T1_A9fjP8Lm2Qx7Nz4Rw8kap@#%&*/4",
            "T1_zY3Km8PqL0Vf2An7tb$!^+=\\7",
            "T1_n8GhQ4LmW7XaP2Drre?><|~/2",
        ]
        for key in examples:
            assert T1KeyGenerator.validate(key), f"Spec example failed: {key!r}"


# ===========================================================================
# Server ID
# ===========================================================================


class TestServerID:
    """Tests for server-ID parsing, formatting, and incrementing."""

    def test_parse_standard(self):
        from hypernix.keymaster import _parse_server_id
        seq, letter, gen = _parse_server_id("00001-A1")
        assert seq == 1
        assert letter == "A"
        assert gen == 1

    def test_parse_large(self):
        from hypernix.keymaster import _parse_server_id
        seq, letter, gen = _parse_server_id("99999-Z9")
        assert seq == 99999
        assert letter == "Z"
        assert gen == 9

    def test_parse_invalid_raises(self):
        from hypernix.keymaster import _parse_server_id
        with pytest.raises(ValueError):
            _parse_server_id("bad")

    def test_format(self):
        from hypernix.keymaster import _format_server_id
        assert _format_server_id(1, "A", 1) == "00001-A1"
        assert _format_server_id(99999, "Z", 9) == "99999-Z9"

    def test_next_increments_seq(self):
        from hypernix.keymaster import _next_server_id
        assert _next_server_id("00001-A1") == "00002-A1"
        assert _next_server_id("00099-A1") == "00100-A1"

    def test_next_rolls_letter_at_99999(self):
        from hypernix.keymaster import _next_server_id
        result = _next_server_id("99999-A1")
        seq, letter, gen = result.split("-")[0], result.split("-")[1][0], int(result.split("-")[1][1:])
        assert int(seq) == 1
        assert letter == "B"
        assert gen == 1

    def test_next_rolls_generation_after_Z(self):
        from hypernix.keymaster import _next_server_id
        result = _next_server_id("99999-Z1")
        seq, letter, gen = result.split("-")[0], result.split("-")[1][0], int(result.split("-")[1][1:])
        assert int(seq) == 1
        assert letter == "A"
        assert gen == 2

    def test_full_cycle(self):
        """Check that the sequence A1→B1→…→Z1→A2 works."""
        from hypernix.keymaster import _next_server_id
        sid = "99999-A1"
        sid = _next_server_id(sid)
        # Should now be 00001-B1
        assert sid == "00001-B1"
        # Advance to 99999-B1
        from hypernix.keymaster import _format_server_id
        sid = _format_server_id(99999, "Z", 1)
        sid = _next_server_id(sid)
        assert sid == "00001-A2"


# ===========================================================================
# KeyMeta
# ===========================================================================


class TestKeyMeta:
    """Tests for KeyMeta serialisation and helpers."""

    def _make_meta(self, expires_at=None):
        from hypernix.keymaster import KeyMeta, KeyScope, KeyType, T1KeyGenerator
        return KeyMeta(
            key_id=str(uuid.uuid4()),
            key=T1KeyGenerator.generate(),
            key_type=KeyType.USER,
            scopes={KeyScope.READ, KeyScope.WRITE},
            created_at=time.time(),
            expires_at=expires_at,
            usage_cap=None,
            request_limit=None,
            prefix="test",
            tags={"env": "ci"},
            server_id="00001-A1",
        )

    def test_to_dict_roundtrip(self):
        from hypernix.keymaster import KeyMeta
        meta = self._make_meta()
        d = meta.to_dict()
        restored = KeyMeta.from_dict(d)
        assert restored.key_id == meta.key_id
        assert restored.key == meta.key
        assert restored.key_type == meta.key_type
        assert restored.scopes == meta.scopes
        assert restored.tags == meta.tags

    def test_is_expired_never(self):
        meta = self._make_meta(expires_at=None)
        assert not meta.is_expired

    def test_is_expired_past(self):
        meta = self._make_meta(expires_at=time.time() - 1)
        assert meta.is_expired

    def test_is_expired_future(self):
        meta = self._make_meta(expires_at=time.time() + 3600)
        assert not meta.is_expired

    def test_expires_soon_within_window(self):
        from hypernix.keymaster import KeyMeta, KeyScope, KeyType, T1KeyGenerator
        meta = KeyMeta(
            key_id=str(uuid.uuid4()),
            key=T1KeyGenerator.generate(),
            key_type=KeyType.USER,
            scopes={KeyScope.READ},
            created_at=time.time(),
            expires_at=time.time() + 3600,  # expires in 1 hour
            usage_cap=None,
            request_limit=None,
            prefix="",
            tags={},
            server_id="00001-A1",
            rotation_window=24,  # 24-hour window → key expires within window
        )
        assert meta.expires_soon  # 1h < 24h window → soon

    def test_expires_soon_not_within_window(self):
        from hypernix.keymaster import KeyMeta, KeyScope, KeyType, T1KeyGenerator
        meta = KeyMeta(
            key_id=str(uuid.uuid4()),
            key=T1KeyGenerator.generate(),
            key_type=KeyType.USER,
            scopes={KeyScope.READ},
            created_at=time.time(),
            expires_at=time.time() + 48 * 3600,  # expires in 48h
            usage_cap=None,
            request_limit=None,
            prefix="",
            tags={},
            server_id="00001-A1",
            rotation_window=24,  # 24h window → not soon (48h away)
        )
        assert not meta.expires_soon

    def test_is_valid_active_not_expired(self):
        meta = self._make_meta(expires_at=time.time() + 3600)
        assert meta.is_valid

    def test_is_valid_expired(self):
        meta = self._make_meta(expires_at=time.time() - 1)
        assert not meta.is_valid

    def test_display_string(self):
        meta = self._make_meta()
        s = meta.display()
        assert "user" in s
        assert "read" in s or "write" in s


# ===========================================================================
# Keymaster
# ===========================================================================


class TestKeymaster:
    """Tests for the full Keymaster lifecycle."""

    def _km(self, tmp_path):
        from hypernix.keymaster import Keymaster
        return Keymaster(store_dir=tmp_path, auto_rotate=False)

    def test_create_returns_keymeta(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType, T1KeyGenerator
        km = self._km(tmp_path)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.stop()
        assert meta.key_id
        assert T1KeyGenerator.validate(meta.key)
        assert meta.key_type.value == "user"
        assert meta.active is True

    def test_create_persists_to_disk(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.stop()
        assert (tmp_path / f"{meta.key_id}.json").exists()

    def test_get_by_id(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        meta = km.create(key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN})
        found = km.get(meta.key_id)
        km.stop()
        assert found is not None
        assert found.key_id == meta.key_id

    def test_get_by_key_string(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        found = km.get_by_key(meta.key)
        km.stop()
        assert found is not None
        assert found.key_id == meta.key_id

    def test_get_unknown_returns_none(self, tmp_path):
        km = self._km(tmp_path)
        result = km.get("no-such-id")
        km.stop()
        assert result is None

    def test_revoke_marks_inactive(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.revoke(meta.key_id, reason="test revoke")
        km.stop()
        # Active record gone; archived
        assert not (tmp_path / f"{meta.key_id}.json").exists()
        assert (tmp_path / "archive" / f"{meta.key_id}.json").exists()

    def test_revoke_unknown_raises(self, tmp_path):
        km = self._km(tmp_path)
        with pytest.raises(KeyError):
            km.revoke("nonexistent-id")
        km.stop()

    def test_rotate_creates_new_key(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType, T1KeyGenerator
        km = self._km(tmp_path)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        new_meta = km.rotate(meta.key_id)
        km.stop()
        assert new_meta.key_id != meta.key_id
        assert T1KeyGenerator.validate(new_meta.key)
        assert new_meta.rotated_from == meta.key_id

    def test_rotate_archives_old_key(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.rotate(meta.key_id)
        km.stop()
        assert (tmp_path / "archive" / f"{meta.key_id}.json").exists()

    def test_rotate_unknown_raises(self, tmp_path):
        km = self._km(tmp_path)
        with pytest.raises(KeyError):
            km.rotate("no-such-id")
        km.stop()

    def test_list_returns_active_keys(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        m1 = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        m2 = km.create(key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN})
        keys = km.list(active_only=True)
        km.stop()
        ids = {k.key_id for k in keys}
        assert m1.key_id in ids
        assert m2.key_id in ids

    def test_list_filters_by_type(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.create(key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN})
        admin_keys = km.list(key_type=KeyType.ADMIN)
        km.stop()
        assert all(k.key_type == KeyType.ADMIN for k in admin_keys)

    def test_list_filters_by_scope(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.create(key_type=KeyType.SERVICE, scopes={KeyScope.WRITE, KeyScope.SERVICE})
        write_keys = km.list(scope=KeyScope.WRITE)
        km.stop()
        assert all(KeyScope.WRITE in k.scopes for k in write_keys)

    def test_list_archived(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.revoke(meta.key_id)
        archived = km.list_archived()
        km.stop()
        assert any(a.key_id == meta.key_id for a in archived)

    def test_record_usage_updates_counters(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.record_usage(meta.key_id, tokens=100, requests=1)
        updated = km.get(meta.key_id)
        km.stop()
        assert updated.usage_count == 100
        assert updated.request_count == 1

    def test_export_all_keys(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.create(key_type=KeyType.SERVICE, scopes={KeyScope.WRITE})
        payload = km.export()
        km.stop()
        assert payload["version"] == "1"
        assert len(payload["keys"]) == 2

    def test_export_single_key(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        payload = km.export(key_id=meta.key_id)
        km.stop()
        assert len(payload["keys"]) == 1
        assert payload["keys"][0]["key_id"] == meta.key_id

    def test_export_to_file(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        out = tmp_path / "export.json"
        km.export(path=out)
        km.stop()
        assert out.exists()
        data = json.loads(out.read_text())
        assert "keys" in data

    def test_import_keys(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        # Create and export from one store
        src = tmp_path / "src"
        src.mkdir()
        km_src = self._km(src)
        meta = km_src.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        payload = km_src.export()
        km_src.stop()

        # Import into a fresh store
        dst = tmp_path / "dst"
        dst.mkdir()
        km_dst = self._km(dst)
        imported = km_dst.import_keys(payload)
        found = km_dst.get(meta.key_id)
        km_dst.stop()

        assert meta.key_id in imported
        assert found is not None
        assert found.key == meta.key

    def test_import_skips_duplicates(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        payload = km.export()
        # Import into same km (key already exists)
        imported = km.import_keys(payload)
        km.stop()
        assert meta.key_id not in imported  # skipped as duplicate

    def test_persistence_across_instances(self, tmp_path):
        """Keys created in one Keymaster instance should load in a new one."""
        from hypernix.keymaster import KeyScope, KeyType
        km1 = self._km(tmp_path)
        meta = km1.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km1.stop()

        km2 = self._km(tmp_path)
        found = km2.get(meta.key_id)
        km2.stop()
        assert found is not None
        assert found.key == meta.key

    def test_multiple_key_types(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        for kt in KeyType:
            m = km.create(key_type=kt, scopes={KeyScope.READ})
            assert m.key_type == kt
        km.stop()

    def test_tags_and_prefix_preserved(self, tmp_path):
        from hypernix.keymaster import KeyScope, KeyType
        km = self._km(tmp_path)
        meta = km.create(
            key_type=KeyType.USER,
            scopes={KeyScope.READ},
            prefix="myapp",
            tags={"env": "prod", "team": "ml"},
        )
        found = km.get(meta.key_id)
        km.stop()
        assert found.prefix == "myapp"
        assert found.tags == {"env": "prod", "team": "ml"}


# ===========================================================================
# Gatekeeper
# ===========================================================================


class TestGatekeeper:
    """Tests for authentication, quota, usage, and permissions."""

    def _setup(self, tmp_path):
        from hypernix.gatekeeper import Gatekeeper
        from hypernix.keymaster import Keymaster, KeyScope, KeyType
        km_dir = tmp_path / "km"
        gk_dir = tmp_path / "gk"
        km = Keymaster(store_dir=km_dir, auto_rotate=False)
        gk = Gatekeeper(km, data_dir=gk_dir)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE})
        return km, gk, meta

    def test_authenticate_valid_key(self, tmp_path):
        km, gk, meta = self._setup(tmp_path)
        found = gk.authenticate(meta.key)
        km.stop()
        gk.stop()
        assert found.key_id == meta.key_id

    def test_authenticate_invalid_format(self, tmp_path):
        km, gk, meta = self._setup(tmp_path)
        with pytest.raises(ValueError):
            gk.authenticate("not-a-t1-key")
        km.stop()
        gk.stop()

    def test_authenticate_unknown_key(self, tmp_path):
        from hypernix.keymaster import T1KeyGenerator
        km, gk, meta = self._setup(tmp_path)
        fake = T1KeyGenerator.generate()
        with pytest.raises(PermissionError):
            gk.authenticate(fake)
        km.stop()
        gk.stop()

    def test_authenticate_revoked_key(self, tmp_path):
        km, gk, meta = self._setup(tmp_path)
        km.revoke(meta.key_id)
        # Gatekeeper's keymaster no longer holds the key
        with pytest.raises(PermissionError):
            gk.authenticate(meta.key)
        km.stop()
        gk.stop()

    def test_authenticate_expired_key(self, tmp_path):
        from hypernix.gatekeeper import Gatekeeper
        from hypernix.keymaster import Keymaster, KeyScope, KeyType
        km_dir = tmp_path / "km2"
        gk_dir = tmp_path / "gk2"
        km = Keymaster(store_dir=km_dir, auto_rotate=False)
        gk = Gatekeeper(km, data_dir=gk_dir)
        meta = km.create(
            key_type=KeyType.USER,
            scopes={KeyScope.READ},
            expires_at=time.time() - 1,  # already expired
        )
        with pytest.raises(PermissionError):
            gk.authenticate(meta.key)
        km.stop()
        gk.stop()

    def test_check_quota_no_limit(self, tmp_path):
        """check_quota should pass when no quota is set."""
        km, gk, meta = self._setup(tmp_path)
        gk.check_quota(meta.key_id, endpoint="/v1/test")
        km.stop()
        gk.stop()

    def test_check_quota_request_limit_exceeded(self, tmp_path):
        from hypernix.gatekeeper import Quota, QuotaViolation
        km, gk, meta = self._setup(tmp_path)
        gk.set_quota(meta.key_id, Quota(max_requests=2, window_seconds=60))
        gk.record_usage(meta.key_id, endpoint="/v1/test")
        gk.record_usage(meta.key_id, endpoint="/v1/test")
        with pytest.raises(QuotaViolation) as exc_info:
            gk.check_quota(meta.key_id, endpoint="/v1/test")
        km.stop()
        gk.stop()
        assert "requests" in str(exc_info.value).lower()

    def test_check_quota_token_limit_exceeded(self, tmp_path):
        from hypernix.gatekeeper import Quota, QuotaViolation
        km, gk, meta = self._setup(tmp_path)
        gk.set_quota(meta.key_id, Quota(max_tokens=100, window_seconds=60))
        gk.record_usage(meta.key_id, endpoint="/v1/test", tokens_used=90)
        with pytest.raises(QuotaViolation) as exc_info:
            gk.check_quota(meta.key_id, endpoint="/v1/test", tokens_requested=20)
        km.stop()
        gk.stop()
        assert "token" in str(exc_info.value).lower()

    def test_check_quota_lifetime_request_limit(self, tmp_path):
        from hypernix.gatekeeper import QuotaViolation
        from hypernix.keymaster import Keymaster, KeyScope, KeyType
        km_dir = tmp_path / "km3"
        gk_dir = tmp_path / "gk3"
        km = Keymaster(store_dir=km_dir, auto_rotate=False)
        from hypernix.gatekeeper import Gatekeeper
        gk = Gatekeeper(km, data_dir=gk_dir)
        meta = km.create(
            key_type=KeyType.USER,
            scopes={KeyScope.READ},
            request_limit=1,  # lifetime cap of 1 request
        )
        km.record_usage(meta.key_id, tokens=0, requests=1)  # burn the cap
        with pytest.raises(QuotaViolation):
            gk.check_quota(meta.key_id)
        km.stop()
        gk.stop()

    def test_record_usage_updates_stats(self, tmp_path):
        km, gk, meta = self._setup(tmp_path)
        gk.record_usage(meta.key_id, endpoint="/v1/gen", model="qwen", tokens_used=50)
        stats = gk.get_stats(meta.key_id)
        km.stop()
        gk.stop()
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] == 50

    def test_record_usage_multiple(self, tmp_path):
        km, gk, meta = self._setup(tmp_path)
        for _ in range(5):
            gk.record_usage(meta.key_id, tokens_used=10)
        stats = gk.get_stats(meta.key_id)
        km.stop()
        gk.stop()
        assert stats["total_requests"] == 5
        assert stats["total_tokens"] == 50

    def test_get_all_stats_returns_list(self, tmp_path):
        km, gk, meta = self._setup(tmp_path)
        gk.record_usage(meta.key_id)
        all_stats = gk.get_all_stats()
        km.stop()
        gk.stop()
        assert isinstance(all_stats, list)
        assert len(all_stats) >= 1

    def test_get_usage_log(self, tmp_path):
        km, gk, meta = self._setup(tmp_path)
        gk.record_usage(meta.key_id, endpoint="/v1/a", tokens_used=10)
        gk.record_usage(meta.key_id, endpoint="/v1/b", tokens_used=20)
        log = gk.get_usage_log(key_id=meta.key_id, limit=10)
        km.stop()
        gk.stop()
        assert len(log) == 2
        endpoints = {r["endpoint"] for r in log}
        assert "/v1/a" in endpoints
        assert "/v1/b" in endpoints

    def test_get_permissions_returns_scopes(self, tmp_path):
        km, gk, meta = self._setup(tmp_path)
        perms = gk.get_permissions(meta.key_id)
        km.stop()
        gk.stop()
        assert "read" in perms
        assert "write" in perms

    def test_has_permission_true(self, tmp_path):
        from hypernix.keymaster import KeyScope
        km, gk, meta = self._setup(tmp_path)
        result = gk.has_permission(meta.key_id, KeyScope.READ)
        km.stop()
        gk.stop()
        assert result is True

    def test_has_permission_false(self, tmp_path):
        from hypernix.keymaster import KeyScope
        km, gk, meta = self._setup(tmp_path)
        result = gk.has_permission(meta.key_id, KeyScope.ADMIN)
        km.stop()
        gk.stop()
        assert result is False

    def test_set_quota_and_retrieve(self, tmp_path):
        from hypernix.gatekeeper import Quota
        km, gk, meta = self._setup(tmp_path)
        q = Quota(max_requests=200, max_tokens=5000, window_seconds=120)
        gk.set_quota(meta.key_id, q)
        got = gk.get_quota(meta.key_id)
        km.stop()
        gk.stop()
        assert got is not None
        assert got.max_requests == 200
        assert got.max_tokens == 5000
        assert got.window_seconds == 120

    def test_stats_includes_quota(self, tmp_path):
        from hypernix.gatekeeper import Quota
        km, gk, meta = self._setup(tmp_path)
        gk.set_quota(meta.key_id, Quota(max_requests=50, window_seconds=60))
        stats = gk.get_stats(meta.key_id)
        km.stop()
        gk.stop()
        assert stats["quota"] is not None
        assert stats["quota"]["max_requests"] == 50

    def test_usage_persisted_to_disk(self, tmp_path):
        km, gk, meta = self._setup(tmp_path)
        gk.record_usage(meta.key_id, tokens_used=77)
        km.stop()
        gk.stop()
        usage_path = tmp_path / "gk" / "usage.json"
        assert usage_path.exists()

    def test_access_log_written(self, tmp_path):
        km, gk, meta = self._setup(tmp_path)
        gk.record_usage(meta.key_id, endpoint="/test", tokens_used=5)
        km.stop()
        gk.stop()
        log_path = tmp_path / "gk" / "access.log"
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["key_id"] == meta.key_id


# ===========================================================================
# Rate Window
# ===========================================================================


class TestRateWindow:
    """Tests for the sliding-window counter."""

    def test_empty_window(self):
        from hypernix.gatekeeper import RateWindow
        w = RateWindow(window_seconds=60)
        req, tok = w.counts()
        assert req == 0
        assert tok == 0

    def test_add_and_count(self):
        from hypernix.gatekeeper import RateWindow
        w = RateWindow(window_seconds=60)
        w.add(10)
        w.add(20)
        req, tok = w.counts()
        assert req == 2
        assert tok == 30

    def test_events_evicted_after_window(self):
        from hypernix.gatekeeper import RateWindow
        w = RateWindow(window_seconds=0.05)
        w.add(100)
        req1, _ = w.counts()
        assert req1 == 1
        time.sleep(0.1)
        req2, tok2 = w.counts()
        assert req2 == 0
        assert tok2 == 0


# ===========================================================================
# Quota dataclass
# ===========================================================================


class TestQuota:
    def test_defaults(self):
        from hypernix.gatekeeper import Quota
        q = Quota()
        assert q.max_requests is None
        assert q.max_tokens is None
        assert q.window_seconds == 60.0

    def test_to_dict_roundtrip(self):
        from hypernix.gatekeeper import Quota
        q = Quota(max_requests=100, max_tokens=500, window_seconds=30)
        d = q.to_dict()
        q2 = Quota.from_dict(d)
        assert q2.max_requests == 100
        assert q2.max_tokens == 500
        assert q2.window_seconds == 30


# ===========================================================================
# gkey CLI
# ===========================================================================


class TestGkeyCLI:
    """Tests for the gkey_cli main() dispatcher."""

    def _run(self, *args, tmp_path=None):
        """Run gkey_cli.main with a temp store and capture return code."""

        from hypernix import gkey_cli

        if tmp_path:
            # Patch default store dirs via environment variable approach
            # We monkeypatch the module-level defaults in keymaster/gatekeeper
            import hypernix.gatekeeper as gk_mod
            import hypernix.keymaster as km_mod
            orig_km = km_mod._DEFAULT_STORE
            orig_gk = gk_mod._DEFAULT_DATA
            km_mod._DEFAULT_STORE = tmp_path / "km"
            gk_mod._DEFAULT_DATA = tmp_path / "gk"
            try:
                return gkey_cli.main(list(args))
            finally:
                km_mod._DEFAULT_STORE = orig_km
                gk_mod._DEFAULT_DATA = orig_gk
        return gkey_cli.main(list(args))

    def test_help_exits_zero(self, tmp_path):
        rc = self._run("--help", tmp_path=tmp_path)
        assert rc == 0

    def test_no_args_shows_help(self, tmp_path):
        rc = self._run(tmp_path=tmp_path)
        assert rc == 0

    def test_create_basic(self, tmp_path):
        rc = self._run("create", "--type", "user", "--scopes", "read",
                       tmp_path=tmp_path)
        assert rc == 0

    def test_create_with_all_options(self, tmp_path):
        rc = self._run(
            "create",
            "--type", "service",
            "--scopes", "read,write,service",
            "--expires", "2099-12-31",
            "--cap", "100000",
            "--limit", "5000",
            "--prefix", "testapp",
            "--tags", "env=test", "team=ci",
            "--body-len", "28",
            "--note", "CI test key",
            tmp_path=tmp_path,
        )
        assert rc == 0

    def test_create_bad_scope_exits_nonzero(self, tmp_path):
        rc = self._run("create", "--scopes", "invalid_scope", tmp_path=tmp_path)
        assert rc != 0

    def test_list_empty(self, tmp_path):
        rc = self._run("list", tmp_path=tmp_path)
        assert rc == 0

    def test_list_after_create(self, tmp_path):
        self._run("create", tmp_path=tmp_path)
        rc = self._run("list", tmp_path=tmp_path)
        assert rc == 0

    def test_list_json(self, tmp_path):
        self._run("create", tmp_path=tmp_path)
        rc = self._run("list", "--json", tmp_path=tmp_path)
        assert rc == 0

    def test_revoke_existing_key(self, tmp_path):
        import hypernix.keymaster as km_mod
        orig = km_mod._DEFAULT_STORE
        km_mod._DEFAULT_STORE = tmp_path / "km"
        from hypernix.keymaster import Keymaster, KeyScope, KeyType
        km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.stop()
        km_mod._DEFAULT_STORE = orig
        rc = self._run("revoke", meta.key_id, tmp_path=tmp_path)
        assert rc == 0

    def test_revoke_unknown_exits_nonzero(self, tmp_path):
        rc = self._run("revoke", "non-existent-id", tmp_path=tmp_path)
        assert rc != 0

    def test_stats_empty(self, tmp_path):
        rc = self._run("stats", tmp_path=tmp_path)
        assert rc == 0

    def test_stats_json(self, tmp_path):
        rc = self._run("stats", "--json", tmp_path=tmp_path)
        assert rc == 0

    def test_quota_no_key_required(self, tmp_path):
        # --key is required for quota
        import hypernix.keymaster as km_mod
        orig = km_mod._DEFAULT_STORE
        km_mod._DEFAULT_STORE = tmp_path / "km"
        from hypernix.keymaster import Keymaster, KeyScope, KeyType
        km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.stop()
        km_mod._DEFAULT_STORE = orig
        rc = self._run("quota", "--key", meta.key_id, tmp_path=tmp_path)
        assert rc == 0

    def test_quota_set(self, tmp_path):
        import hypernix.keymaster as km_mod
        orig = km_mod._DEFAULT_STORE
        km_mod._DEFAULT_STORE = tmp_path / "km"
        from hypernix.keymaster import Keymaster, KeyScope, KeyType
        km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.stop()
        km_mod._DEFAULT_STORE = orig
        rc = self._run(
            "quota", "--key", meta.key_id,
            "--set", "max-requests=100,max-tokens=5000,window=60",
            tmp_path=tmp_path,
        )
        assert rc == 0

    def test_permissions(self, tmp_path):
        import hypernix.keymaster as km_mod
        orig = km_mod._DEFAULT_STORE
        km_mod._DEFAULT_STORE = tmp_path / "km"
        from hypernix.keymaster import Keymaster, KeyScope, KeyType
        km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.stop()
        km_mod._DEFAULT_STORE = orig
        rc = self._run("permissions", "--key", meta.key_id, tmp_path=tmp_path)
        assert rc == 0

    def test_rotate_key(self, tmp_path):
        import hypernix.keymaster as km_mod
        orig = km_mod._DEFAULT_STORE
        km_mod._DEFAULT_STORE = tmp_path / "km"
        from hypernix.keymaster import Keymaster, KeyScope, KeyType
        km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.stop()
        km_mod._DEFAULT_STORE = orig
        rc = self._run("rotate", meta.key_id, tmp_path=tmp_path)
        assert rc == 0

    def test_export_to_stdout(self, tmp_path):
        self._run("create", tmp_path=tmp_path)
        rc = self._run("export", tmp_path=tmp_path)
        assert rc == 0

    def test_export_to_file(self, tmp_path):
        self._run("create", tmp_path=tmp_path)
        out = tmp_path / "export.json"
        rc = self._run("export", "--out", str(out), tmp_path=tmp_path)
        assert rc == 0
        assert out.exists()

    def test_import_from_file(self, tmp_path):
        # First create + export
        self._run("create", tmp_path=tmp_path)
        out = tmp_path / "export.json"
        self._run("export", "--out", str(out), tmp_path=tmp_path)

        # Import into a fresh store
        import hypernix.keymaster as km_mod
        km_mod._DEFAULT_STORE = tmp_path / "km2"
        rc = self._run("import", str(out), tmp_path=tmp_path)
        assert rc == 0

    def test_import_missing_file(self, tmp_path):
        rc = self._run("import", str(tmp_path / "no_file.json"), tmp_path=tmp_path)
        assert rc != 0

    def test_list_id_subcommand(self, tmp_path):
        import hypernix.keymaster as km_mod
        orig = km_mod._DEFAULT_STORE
        km_mod._DEFAULT_STORE = tmp_path / "km"
        from hypernix.keymaster import Keymaster, KeyScope, KeyType
        km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.stop()
        km_mod._DEFAULT_STORE = orig
        rc = self._run("list", "id", meta.key_id, tmp_path=tmp_path)
        assert rc == 0

    def test_unknown_subcommand_exits_nonzero(self, tmp_path):
        rc = self._run("notacommand", tmp_path=tmp_path)
        assert rc != 0

    def test_version_flag(self, tmp_path):
        rc = self._run("--version", tmp_path=tmp_path)
        assert rc == 0


# ===========================================================================
# Module-level smoke imports
# ===========================================================================


class TestModuleImports:
    """Confirm all public symbols are importable."""

    def test_keymaster_all(self):
        from hypernix.keymaster import (  # noqa: F401
            Keymaster,
            KeyMeta,
            KeyScope,
            KeyType,
            T1KeyGenerator,
            generate_t1_key,
            validate_t1_key,
        )

    def test_gatekeeper_all(self):
        from hypernix.gatekeeper import (  # noqa: F401
            Gatekeeper,
            Quota,
            QuotaViolation,
            RateWindow,
            UsageRecord,
        )

    def test_gkey_cli_main(self):
        from hypernix.gkey_cli import main  # noqa: F401
        assert callable(main)

    def test_lazy_loader_gatekeeper(self):
        import hypernix
        assert hasattr(hypernix, "gatekeeper") or "gatekeeper" in hypernix.__all__

    def test_lazy_loader_keymaster(self):
        import hypernix
        assert hasattr(hypernix, "keymaster") or "keymaster" in hypernix.__all__


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
