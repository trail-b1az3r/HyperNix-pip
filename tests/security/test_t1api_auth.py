"""Tests for hypernix.t1api.auth — the adapter over the real Keymaster /
Gatekeeper (not a mock; the whole point is that t1api doesn't duplicate
their logic, so these tests exercise the real thing end to end).
"""
from __future__ import annotations

import time

import pytest

from hypernix.gatekeeper import Gatekeeper
from hypernix.keymaster import Keymaster, KeyScope, KeyType
from hypernix.t1api.auth import T1AuthService
from hypernix.t1api.errors import T1APIError, T1ErrorCode


@pytest.fixture
def km(tmp_path) -> Keymaster:
    return Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)


@pytest.fixture
def gk(km: Keymaster, tmp_path) -> Gatekeeper:
    return Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper", log_to_file=False)


@pytest.fixture
def svc(km: Keymaster, gk: Gatekeeper) -> T1AuthService:
    return T1AuthService(km, gk, token_secret="test-secret", default_ttl_seconds=3600)


class TestValidateKey:
    def test_valid_key_returns_context(self, km, svc):
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        ctx = svc.validate_key(meta.key)
        assert ctx.key_id == meta.key_id
        assert ctx.is_admin is False

    def test_admin_key_is_admin(self, km, svc):
        meta = km.create(key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN})
        ctx = svc.validate_key(meta.key)
        assert ctx.is_admin is True

    def test_malformed_key_raises_auth_invalid_key(self, svc):
        with pytest.raises(T1APIError) as exc_info:
            svc.validate_key("garbage")
        assert exc_info.value.code == T1ErrorCode.AUTH_INVALID_KEY

    def test_unknown_but_well_formed_key_raises(self, svc):
        from hypernix.keymaster import T1KeyGenerator

        fake_key = T1KeyGenerator.generate()
        with pytest.raises(T1APIError) as exc_info:
            svc.validate_key(fake_key)
        assert exc_info.value.code == T1ErrorCode.AUTH_INVALID_KEY

    def test_revoked_key_raises_auth_invalid_key(self, km, svc):
        """A revoked key is indistinguishable from an unknown one at the
        Gatekeeper.authenticate() layer — see the NOTE in
        T1AuthService.validate_key for why AUTH_INVALID_KEY (not
        AUTH_REVOKED_KEY) is the correct, honest code here."""
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        km.revoke(meta.key_id)
        with pytest.raises(T1APIError) as exc_info:
            svc.validate_key(meta.key)
        assert exc_info.value.code == T1ErrorCode.AUTH_INVALID_KEY

    def test_expired_key_raises(self, km, svc):
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ}, expires_at=time.time() - 1)
        with pytest.raises(T1APIError) as exc_info:
            svc.validate_key(meta.key)
        assert exc_info.value.code == T1ErrorCode.AUTH_EXPIRED_KEY


class TestScopedTokens:
    def test_round_trip(self, km, svc):
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE})
        tok = svc.issue_scoped_token(meta.key)
        ctx = svc.verify_scoped_token(tok.token)
        assert ctx.key_id == meta.key_id
        assert ctx.via_scoped_token is True

    def test_narrowed_scopes_respected(self, km, svc):
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE})
        tok = svc.issue_scoped_token(meta.key, scopes=["read"])
        ctx = svc.verify_scoped_token(tok.token)
        assert ctx.scopes == {KeyScope.READ}

    def test_cannot_widen_scopes_beyond_key(self, km, svc):
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        with pytest.raises(T1APIError) as exc_info:
            svc.issue_scoped_token(meta.key, scopes=["admin"])
        assert exc_info.value.code == T1ErrorCode.AUTH_INSUFFICIENT_SCOPE

    def test_expired_token_rejected(self, km, svc):
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        tok = svc.issue_scoped_token(meta.key, ttl_seconds=1)
        time.sleep(1.2)
        with pytest.raises(T1APIError) as exc_info:
            svc.verify_scoped_token(tok.token)
        assert exc_info.value.code == T1ErrorCode.AUTH_EXPIRED_TOKEN

    def test_tampered_token_rejected(self, km, svc):
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        tok = svc.issue_scoped_token(meta.key)
        prefix, payload, sig = tok.token.split(".")
        tampered = f"{prefix}.{payload}.{sig[:-2]}zz"
        with pytest.raises(T1APIError) as exc_info:
            svc.verify_scoped_token(tampered)
        assert exc_info.value.code == T1ErrorCode.AUTH_INVALID_TOKEN

    def test_malformed_token_rejected(self, svc):
        with pytest.raises(T1APIError) as exc_info:
            svc.verify_scoped_token("not-a-token-at-all")
        assert exc_info.value.code == T1ErrorCode.AUTH_INVALID_TOKEN

    def test_token_invalid_after_underlying_key_revoked(self, km, svc):
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        tok = svc.issue_scoped_token(meta.key)
        km.revoke(meta.key_id)
        with pytest.raises(T1APIError):
            svc.verify_scoped_token(tok.token)

    def test_different_secrets_reject_each_others_tokens(self, km, gk):
        svc_a = T1AuthService(km, gk, token_secret="secret-a")
        svc_b = T1AuthService(km, gk, token_secret="secret-b")
        meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        tok = svc_a.issue_scoped_token(meta.key)
        with pytest.raises(T1APIError) as exc_info:
            svc_b.verify_scoped_token(tok.token)
        assert exc_info.value.code == T1ErrorCode.AUTH_INVALID_TOKEN


class TestAdminOperations:
    def test_non_admin_cannot_admin_rotate(self, km, svc):
        user_meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        target_meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        ctx = svc.validate_key(user_meta.key)
        with pytest.raises(T1APIError) as exc_info:
            svc.admin_rotate(requester=ctx, target_key_id=target_meta.key_id)
        assert exc_info.value.code == T1ErrorCode.AUTH_ADMIN_REQUIRED

    def test_admin_can_rotate_target_key(self, km, svc):
        admin_meta = km.create(key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN})
        target_meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        ctx = svc.validate_key(admin_meta.key)
        new_meta = svc.admin_rotate(requester=ctx, target_key_id=target_meta.key_id)
        assert new_meta.key_id != target_meta.key_id
        assert new_meta.rotated_from == target_meta.key_id

    def test_admin_can_promote_to_admin(self, km, svc):
        admin_meta = km.create(key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN})
        target_meta = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        ctx = svc.validate_key(admin_meta.key)
        promoted = svc.admin_rotate(requester=ctx, target_key_id=target_meta.key_id, promote_to_admin=True)
        assert promoted.key_type == KeyType.ADMIN
        assert KeyScope.ADMIN in promoted.scopes

    def test_admin_rotate_unknown_target_raises_not_found(self, km, svc):
        admin_meta = km.create(key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN})
        ctx = svc.validate_key(admin_meta.key)
        with pytest.raises(T1APIError) as exc_info:
            svc.admin_rotate(requester=ctx, target_key_id="does-not-exist")
        assert exc_info.value.code == T1ErrorCode.NOT_FOUND
