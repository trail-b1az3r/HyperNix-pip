"""Beta 3 security tests: network policy, mTLS, rate limiting, audit.

These exercise the pure core (no FastAPI needed), which is where the
enforcement decisions actually live. The HTTP-level counterparts are in
tests/t1api/test_t1api_beta3_http.py.

Each class pins behaviour that would be a real vulnerability if it
regressed, and says which one in the test name rather than only in a
comment.
"""
from __future__ import annotations

import time

import pytest

from hypernix.t1api.audit import (
    AuditCategory,
    AuditLog,
    AuditOutcome,
    mask_identifier,
    scrub_details,
)
from hypernix.t1api.db import SQLiteBackend
from hypernix.t1api.errors import T1APIError, T1ErrorCode
from hypernix.t1api.mtls import (
    ClientCertVerifier,
    TLSSettings,
    _normalize_fingerprint,
)
from hypernix.t1api.netpolicy import Decision, EntryKind, NetworkPolicy
from hypernix.t1api.ratelimit import RateLimiter, RateRule, Subject, endpoint_cost


@pytest.fixture
def backend(tmp_path) -> SQLiteBackend:
    return SQLiteBackend(tmp_path / "beta3.sqlite3")


# ===========================================================================
# Network policy
# ===========================================================================


class TestNetworkPolicy:
    def test_unlisted_allowed_when_server_permits(self, backend):
        policy = NetworkPolicy(backend, allow_unlisted=True)
        decision, matched = policy.evaluate("203.0.113.9")
        assert decision is Decision.ALLOWED_UNLISTED
        assert matched is None

    def test_unlisted_denied_when_server_requires_allowlisting(self, backend):
        policy = NetworkPolicy(backend, allow_unlisted=False)
        decision, _ = policy.evaluate("203.0.113.9")
        assert decision is Decision.DENIED_NOT_LISTED
        assert not decision.allowed

    def test_blocked_address_is_denied(self, backend):
        policy = NetworkPolicy(backend, allow_unlisted=True)
        policy.block("203.0.113.9", reason="abuse")
        decision, matched = policy.evaluate("203.0.113.9")
        assert decision is Decision.DENIED_BLOCKED
        assert matched == "203.0.113.9/32"

    def test_blocklist_beats_allowlist(self, backend):
        """A later allowlist entry must not silently un-block an address
        an operator explicitly blocked. Un-blocking is an appeal, and it
        has its own operation."""
        policy = NetworkPolicy(backend, allow_unlisted=False)
        policy.allow("203.0.113.0/24", reason="partner range")
        policy.block("203.0.113.9", reason="one bad host in that range")
        decision, _ = policy.evaluate("203.0.113.9")
        assert decision is Decision.DENIED_BLOCKED
        # A different address in the same allowed range still gets in.
        assert policy.evaluate("203.0.113.10")[0] is Decision.ALLOWED_LISTED

    def test_cidr_range_matches_members(self, backend):
        policy = NetworkPolicy(backend, allow_unlisted=False)
        policy.allow("10.0.0.0/8")
        assert policy.evaluate("10.1.2.3")[0].allowed
        assert not policy.evaluate("11.1.2.3")[0].allowed

    def test_tailscale_range_can_be_allowlisted(self, backend):
        policy = NetworkPolicy(backend, allow_unlisted=False)
        policy.allow("100.64.0.0/10", reason="tailnet")
        assert policy.evaluate("100.101.102.103")[0] is Decision.ALLOWED_LISTED

    def test_expired_entry_stops_matching(self, backend):
        policy = NetworkPolicy(backend, allow_unlisted=True)
        policy.block("203.0.113.9", expires_at=time.time() - 1)
        assert policy.evaluate("203.0.113.9")[0] is Decision.ALLOWED_UNLISTED

    def test_appeal_removes_a_block(self, backend):
        policy = NetworkPolicy(backend, allow_unlisted=True)
        policy.block("203.0.113.9")
        assert policy.remove("203.0.113.9") is True
        assert policy.evaluate("203.0.113.9")[0].allowed
        assert policy.remove("203.0.113.9") is False  # nothing left to appeal

    def test_re_adding_updates_rather_than_duplicating(self, backend):
        policy = NetworkPolicy(backend, allow_unlisted=True)
        policy.block("203.0.113.9", reason="first")
        policy.block("203.0.113.9", reason="second")
        entries = policy.list_entries()
        assert len(entries) == 1
        assert entries[0].reason == "second"

    def test_kind_change_is_a_move_not_a_duplicate(self, backend):
        policy = NetworkPolicy(backend, allow_unlisted=False)
        policy.block("203.0.113.9")
        policy.allow("203.0.113.9")
        entries = policy.list_entries()
        assert len(entries) == 1
        assert entries[0].kind is EntryKind.ALLOW
        assert policy.evaluate("203.0.113.9")[0] is Decision.ALLOWED_LISTED

    def test_invalid_cidr_rejected_at_write_time(self, backend):
        policy = NetworkPolicy(backend, allow_unlisted=True)
        with pytest.raises(T1APIError) as exc:
            policy.block("not-an-ip")
        assert exc.value.code is T1ErrorCode.VALIDATION_ERROR
        assert policy.list_entries() == []

    def test_missing_client_ip_follows_allow_unlisted(self, backend):
        """A deployment behind a proxy that forwards no address must not
        fail every request — it falls under the operator's explicit
        allow_unlisted setting."""
        assert NetworkPolicy(backend, allow_unlisted=True).evaluate(None)[0].allowed
        assert not NetworkPolicy(backend, allow_unlisted=False).evaluate(None)[0].allowed

    def test_require_allowed_raises_distinct_codes(self, backend):
        policy = NetworkPolicy(backend, allow_unlisted=False)
        policy.block("203.0.113.9")
        with pytest.raises(T1APIError) as blocked:
            policy.require_allowed("203.0.113.9")
        assert blocked.value.code is T1ErrorCode.IP_BLOCKED
        with pytest.raises(T1APIError) as unlisted:
            policy.require_allowed("198.51.100.4")
        assert unlisted.value.code is T1ErrorCode.IP_NOT_ALLOWLISTED

    def test_ipv6_is_matched_by_version(self, backend):
        policy = NetworkPolicy(backend, allow_unlisted=False)
        policy.allow("2001:db8::/32")
        assert policy.evaluate("2001:db8::1")[0].allowed
        assert not policy.evaluate("192.0.2.1")[0].allowed

    def test_entries_survive_a_restart(self, backend):
        NetworkPolicy(backend, allow_unlisted=True).block("203.0.113.9", reason="persisted")
        reopened = NetworkPolicy(backend, allow_unlisted=True)
        assert reopened.evaluate("203.0.113.9")[0] is Decision.DENIED_BLOCKED


class TestForcedLimits:
    def test_set_and_read_back(self, backend):
        policy = NetworkPolicy(backend)
        policy.set_limit("key", "abc123", requests_per_window=10, window_seconds=60)
        limit = policy.get_limit("key", "abc123")
        assert limit.requests_per_window == 10
        assert limit.window_seconds == 60

    def test_requires_at_least_one_bound(self, backend):
        policy = NetworkPolicy(backend)
        with pytest.raises(T1APIError) as exc:
            policy.set_limit("key", "abc123")
        assert exc.value.code is T1ErrorCode.VALIDATION_ERROR

    def test_rejects_unknown_subject_type(self, backend):
        policy = NetworkPolicy(backend)
        with pytest.raises(T1APIError):
            policy.set_limit("user", "abc123", requests_per_window=5)

    def test_clear(self, backend):
        policy = NetworkPolicy(backend)
        policy.set_limit("server", "srv-1", tokens_per_window=1000)
        assert policy.clear_limit("server", "srv-1") is True
        assert policy.get_limit("server", "srv-1") is None
        assert policy.clear_limit("server", "srv-1") is False

    def test_masks_subject_id_in_public_dict(self, backend):
        policy = NetworkPolicy(backend)
        limit = policy.set_limit("key", "0123456789abcdef0123", requests_per_window=5)
        assert "…" in limit.to_dict()["subject_id"]
        assert "0123456789abcdef0123" != limit.to_dict()["subject_id"]


# ===========================================================================
# Rate limiting
# ===========================================================================


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestRateLimiter:
    def test_allows_within_capacity(self):
        limiter = RateLimiter([RateRule("t", capacity=3, refill_per_second=0)], clock=_Clock())
        for _ in range(3):
            limiter.check([Subject("key", "k")], cost=1)

    def test_refuses_past_capacity(self):
        limiter = RateLimiter([RateRule("t", capacity=2, refill_per_second=0)], clock=_Clock())
        limiter.check([Subject("key", "k")], cost=1)
        limiter.check([Subject("key", "k")], cost=1)
        with pytest.raises(T1APIError) as exc:
            limiter.check([Subject("key", "k")], cost=1)
        assert exc.value.code is T1ErrorCode.RATE_LIMITED
        assert exc.value.details["rule"] == "t"

    def test_refills_over_time(self):
        clock = _Clock()
        limiter = RateLimiter([RateRule("t", capacity=2, refill_per_second=1.0)], clock=clock)
        limiter.check([Subject("key", "k")], cost=2)
        with pytest.raises(T1APIError):
            limiter.check([Subject("key", "k")], cost=1)
        clock.advance(1.5)
        limiter.check([Subject("key", "k")], cost=1)

    def test_subjects_have_separate_buckets(self):
        limiter = RateLimiter([RateRule("t", capacity=1, refill_per_second=0)], clock=_Clock())
        limiter.check([Subject("key", "a")], cost=1)
        limiter.check([Subject("key", "b")], cost=1)  # b is unaffected by a

    def test_ip_and_key_are_both_enforced(self):
        """A caller can't escape a per-IP limit by rotating keys, or a
        per-key limit by changing address."""
        limiter = RateLimiter(
            [
                RateRule("per_key", capacity=10, refill_per_second=0, applies_to=("key",)),
                RateRule("per_ip", capacity=1, refill_per_second=0, applies_to=("ip",)),
            ],
            clock=_Clock(),
        )
        limiter.check([Subject("key", "a"), Subject("ip", "1.2.3.4")], cost=1)
        with pytest.raises(T1APIError) as exc:
            limiter.check([Subject("key", "b"), Subject("ip", "1.2.3.4")], cost=1)
        assert exc.value.details["rule"] == "per_ip"

    def test_endpoint_scoped_rule_ignores_other_paths(self):
        limiter = RateLimiter(
            [RateRule("uploads", capacity=1, refill_per_second=0, endpoints=("/modules/upload",))],
            clock=_Clock(),
        )
        for _ in range(5):
            limiter.check([Subject("key", "k")], path="/models", cost=1)
        limiter.check([Subject("key", "k")], path="/modules/upload/local", cost=1)
        with pytest.raises(T1APIError):
            limiter.check([Subject("key", "k")], path="/modules/upload/local", cost=1)

    def test_method_scoped_rule(self):
        limiter = RateLimiter(
            [RateRule("writes", capacity=1, refill_per_second=0, methods=("POST",))], clock=_Clock()
        )
        limiter.check([Subject("key", "k")], method="GET", cost=1)
        limiter.check([Subject("key", "k")], method="POST", cost=1)
        with pytest.raises(T1APIError):
            limiter.check([Subject("key", "k")], method="POST", cost=1)

    def test_expensive_endpoints_cost_more(self):
        assert endpoint_cost("/models/route") > endpoint_cost("/health")
        assert endpoint_cost("/modules/upload/local") > endpoint_cost("/modules")
        assert endpoint_cost("/health") == 1.0

    def test_retry_after_is_reported(self):
        limiter = RateLimiter([RateRule("t", capacity=1, refill_per_second=0.5)], clock=_Clock())
        limiter.check([Subject("key", "k")], cost=1)
        with pytest.raises(T1APIError) as exc:
            limiter.check([Subject("key", "k")], cost=1)
        assert exc.value.details["retry_after_seconds"] > 0

    def test_disabled_limiter_allows_everything(self):
        limiter = RateLimiter([RateRule("t", capacity=0, refill_per_second=0)], enabled=False)
        for _ in range(100):
            limiter.check([Subject("key", "k")], cost=10)

    def test_forced_limit_tightens_beyond_the_rules(self, backend):
        """A -r forced limit applies in addition to the configured rules,
        so it can only ever narrow what a subject may do."""
        policy = NetworkPolicy(backend)
        policy.set_limit("key", "k", requests_per_window=2, window_seconds=60)
        limiter = RateLimiter(
            [RateRule("generous", capacity=1000, refill_per_second=1000)],
            forced_limit_lookup=policy.get_limit,
            clock=_Clock(),
        )
        limiter.check([Subject("key", "k")], cost=1)
        limiter.check([Subject("key", "k")], cost=1)
        with pytest.raises(T1APIError) as exc:
            limiter.check([Subject("key", "k")], cost=1)
        assert exc.value.details["rule"] == "forced_limit"

    def test_forced_token_limit(self, backend):
        policy = NetworkPolicy(backend)
        policy.set_limit("key", "k", tokens_per_window=100, window_seconds=60)
        limiter = RateLimiter(
            [RateRule("generous", capacity=1000, refill_per_second=1000)],
            forced_limit_lookup=policy.get_limit,
            clock=_Clock(),
        )
        limiter.check([Subject("key", "k")])
        limiter.record_tokens([Subject("key", "k")], 150)
        with pytest.raises(T1APIError) as exc:
            limiter.check([Subject("key", "k")])
        assert exc.value.details["rule"] == "forced_limit_tokens"

    def test_forced_limit_lookup_failure_does_not_break_requests(self):
        """A limit-store outage must not turn every request into a 500."""

        def broken(_type, _id):
            raise RuntimeError("limit store is down")

        limiter = RateLimiter(
            [RateRule("t", capacity=5, refill_per_second=0)],
            forced_limit_lookup=broken,
            clock=_Clock(),
        )
        limiter.check([Subject("key", "k")], cost=1)

    def test_snapshot_reports_remaining(self):
        limiter = RateLimiter([RateRule("t", capacity=10, refill_per_second=0)], clock=_Clock())
        limiter.check([Subject("key", "k")], cost=4)
        snapshot = limiter.snapshot(Subject("key", "k"))
        assert snapshot[0]["remaining"] == 6
        assert snapshot[0]["capacity"] == 10

    def test_admin_exempt_is_off_by_default(self):
        """An admin key has a bigger blast radius, not a smaller one, so
        it is subject to limits unless a rule explicitly says otherwise."""
        limiter = RateLimiter([RateRule("t", capacity=1, refill_per_second=0)], clock=_Clock())
        limiter.check([Subject("key", "k")], is_admin=True, cost=1)
        with pytest.raises(T1APIError):
            limiter.check([Subject("key", "k")], is_admin=True, cost=1)


# ===========================================================================
# mTLS
# ===========================================================================


class TestMTLS:
    def test_proxy_headers_ignored_from_untrusted_peer(self):
        """The core mTLS-bypass vulnerability: without the trusted-proxy
        check, any client that can reach the app directly could forge
        X-Client-Verify and be treated as certificate-authenticated."""
        verifier = ClientCertVerifier(
            TLSSettings(require_client_cert=True, behind_proxy=True, trusted_proxies=("10.0.0.1",))
        )
        cert = verifier.extract(
            headers={"x-client-verify": "SUCCESS", "x-client-subject-dn": "CN=attacker"},
            peer_ip="203.0.113.9",  # not the trusted proxy
        )
        assert cert.verified is False
        with pytest.raises(T1APIError) as exc:
            verifier.require(cert, path="/models")
        assert exc.value.code is T1ErrorCode.MTLS_REQUIRED

    def test_proxy_headers_accepted_from_trusted_peer(self):
        verifier = ClientCertVerifier(
            TLSSettings(require_client_cert=True, behind_proxy=True, trusted_proxies=("10.0.0.1",))
        )
        cert = verifier.extract(
            headers={"x-client-verify": "SUCCESS", "x-client-subject-dn": "CN=deploy-01"},
            peer_ip="10.0.0.1",
        )
        assert cert.verified is True
        assert cert.source == "proxy"
        verifier.require(cert, path="/models")

    def test_failed_proxy_verification_is_refused(self):
        verifier = ClientCertVerifier(
            TLSSettings(require_client_cert=True, behind_proxy=True, trusted_proxies=("10.0.0.1",))
        )
        cert = verifier.extract(headers={"x-client-verify": "FAILED"}, peer_ip="10.0.0.1")
        assert cert.verified is False

    def test_empty_trusted_proxies_fails_closed(self):
        """Proxy mTLS with no trusted-proxy list is a misconfiguration,
        and it must reject rather than trust every header it sees."""
        settings = TLSSettings(require_client_cert=True, behind_proxy=True, trusted_proxies=())
        problems = settings.validation_errors()
        assert any("T1_TRUSTED_PROXIES" in p for p in problems)
        verifier = ClientCertVerifier(settings)
        cert = verifier.extract(headers={"x-client-verify": "SUCCESS"}, peer_ip="10.0.0.1")
        assert cert.verified is False

    def test_subject_allowlist(self):
        verifier = ClientCertVerifier(
            TLSSettings(
                require_client_cert=True,
                behind_proxy=True,
                trusted_proxies=("10.0.0.1",),
                allowed_subjects=("CN=deploy-01",),
            )
        )
        allowed = verifier.extract(
            headers={"x-client-verify": "SUCCESS", "x-client-subject-dn": "CN=deploy-01,O=HyperNix"},
            peer_ip="10.0.0.1",
        )
        verifier.require(allowed)
        rejected = verifier.extract(
            headers={"x-client-verify": "SUCCESS", "x-client-subject-dn": "CN=someone-else"},
            peer_ip="10.0.0.1",
        )
        with pytest.raises(T1APIError) as exc:
            verifier.require(rejected)
        assert exc.value.code is T1ErrorCode.MTLS_INVALID

    def test_fingerprint_allowlist_normalizes_formats(self):
        """Proxies format fingerprints inconsistently; an allowlist that
        only matched one spelling would silently never match."""
        assert _normalize_fingerprint("AA:BB:CC") == "aabbcc"
        assert _normalize_fingerprint("sha256/AABBCC") == "aabbcc"
        assert _normalize_fingerprint("not-hex!") == ""
        verifier = ClientCertVerifier(
            TLSSettings(
                require_client_cert=True,
                behind_proxy=True,
                trusted_proxies=("10.0.0.1",),
                allowed_fingerprints=("AA:BB:CC",),
            )
        )
        cert = verifier.extract(
            headers={"x-client-verify": "SUCCESS", "x-client-fingerprint": "sha256/aabbcc"},
            peer_ip="10.0.0.1",
        )
        verifier.require(cert)

    def test_health_is_exempt(self):
        """Load balancers must be able to poll /health without a client
        certificate, and it exposes nothing but liveness."""
        verifier = ClientCertVerifier(TLSSettings(require_client_cert=True, behind_proxy=True,
                                                  trusted_proxies=("10.0.0.1",)))
        cert = verifier.extract(headers={}, peer_ip="203.0.113.9")
        verifier.require(cert, path="/health")
        with pytest.raises(T1APIError):
            verifier.require(cert, path="/models")

    def test_transport_cert_beats_headers(self):
        """A certificate the TLS stack itself verified can't be forged by
        the client, so it wins over any header."""
        verifier = ClientCertVerifier(TLSSettings(require_client_cert=True))
        cert = verifier.extract(
            headers={"x-client-verify": "FAILED"},
            peer_ip="203.0.113.9",
            transport_cert={"subject": ((("commonName", "real-client"),),)},
        )
        assert cert.verified is True
        assert cert.source == "transport"

    def test_uvicorn_kwargs(self):
        import ssl as _ssl

        settings = TLSSettings(
            certfile="/tmp/c.pem", keyfile="/tmp/k.pem", ca_certs="/tmp/ca.pem",
            require_client_cert=True,
        )
        kwargs = settings.uvicorn_kwargs()
        assert kwargs["ssl_certfile"] == "/tmp/c.pem"
        assert kwargs["ssl_cert_reqs"] == _ssl.CERT_REQUIRED
        assert TLSSettings().uvicorn_kwargs() == {}

    def test_public_dict_never_names_key_material(self):
        settings = TLSSettings(certfile="/secret/path/c.pem", keyfile="/secret/path/k.pem")
        public = settings.public_dict()
        assert "/secret/path" not in str(public)
        assert public["tls_enabled"] is True


# ===========================================================================
# Audit log
# ===========================================================================


class TestAuditLog:
    def test_records_and_queries(self, backend):
        log = AuditLog(backend)
        log.record("keys.assign", category=AuditCategory.ADMIN, actor_key_id="abcdef0123456789")
        records = log.query()
        assert len(records) == 1
        assert records[0].action == "keys.assign"
        assert records[0].category is AuditCategory.ADMIN

    def test_actor_key_id_is_masked(self, backend):
        log = AuditLog(backend)
        log.record("test", actor_key_id="0123456789abcdef0123456789")
        assert log.query()[0].actor_key_id == "01234567…"
        assert "0123456789abcdef" not in log.query()[0].actor_key_id

    def test_secret_shaped_details_are_scrubbed(self, backend):
        """Deny-by-name at write time: a future call site that hands us a
        raw key must not be able to write it to disk."""
        log = AuditLog(backend)
        log.record(
            "keys.import",
            details={
                "key": "T1_super_secret_value",
                "payment_token": "T1PAY_secret",
                "authorization": "Bearer abc",
                "imported": 3,
                "key_id": "abc123",
            },
        )
        details = log.query()[0].details
        assert details["key"] == "[redacted]"
        assert details["payment_token"] == "[redacted]"
        assert details["authorization"] == "[redacted]"
        assert details["imported"] == 3
        # key_id is an identifier, not a secret — it must survive.
        assert details["key_id"] == "abc123"

    def test_scrub_is_recursive(self):
        scrubbed = scrub_details({"outer": {"api_key": "secret", "count": 2}})
        assert scrubbed["outer"]["api_key"] == "[redacted]"
        assert scrubbed["outer"]["count"] == 2

    def test_scrub_caps_long_values(self):
        scrubbed = scrub_details({"note": "x" * 5000})
        assert len(scrubbed["note"]) < 600

    def test_mask_identifier(self):
        assert mask_identifier("abcdefghijkl", keep=4) == "abcd…"
        assert mask_identifier("ab", keep=4) == "ab"
        assert mask_identifier(None) == ""

    def test_filters(self, backend):
        log = AuditLog(backend)
        log.record("a", category=AuditCategory.ADMIN, outcome=AuditOutcome.SUCCESS)
        log.record("b", category=AuditCategory.SECURITY, outcome=AuditOutcome.DENIED)
        assert len(log.query(category="admin")) == 1
        assert len(log.query(outcome="denied")) == 1
        assert log.query(action="b")[0].category is AuditCategory.SECURITY

    def test_query_by_actor_accepts_the_full_key_id(self, backend):
        """Callers hold the full key_id; records store the mask. The
        obvious query has to work."""
        log = AuditLog(backend)
        log.record("x", actor_key_id="0123456789abcdef")
        assert len(log.query(actor_key_id="0123456789abcdef")) == 1

    def test_write_failure_never_raises(self, backend, monkeypatch):
        """An audit write must not turn a successful operation into a
        500 — that would make the audit log a liability."""
        log = AuditLog(backend)

        def boom(*args, **kwargs):
            raise RuntimeError("disk is full")

        monkeypatch.setattr(log, "_write", boom)
        assert log.record("test") is None  # swallowed
        with pytest.raises(RuntimeError):
            log.record_strict("test")  # ...but record_strict propagates

    def test_disabled_log_records_nothing(self, backend):
        log = AuditLog(backend, enabled=False)
        log.record("test")
        assert log.count() == 0

    def test_purge_before(self, backend):
        log = AuditLog(backend)
        log.record("old")
        time.sleep(0.01)
        cutoff = time.time()
        log.record("new")
        assert log.purge_before(cutoff) == 1
        assert [r.action for r in log.query()] == ["new"]
