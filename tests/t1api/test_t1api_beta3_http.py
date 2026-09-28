"""Beta 3 HTTP tests: the new endpoints and the middleware stack.

Requires `pip install hypernix[t1api-test]`.

The middleware order is the thing most worth pinning here, because it is
invisible from any single endpoint and wrong order is a security bug:
network policy runs before authentication (a blocked address is turned
away without a credential being examined), and rate limiting runs before
the route handler (so an over-limit caller never reaches model work).
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402
from hypernix.t1api.ratelimit import RateLimiter, RateRule  # noqa: E402
from hypernix.t1api.registry import (  # noqa: E402
    ModelEntry,
    ModelPricing,
    ModelRegistry,
    ModelStatus,
)


def _model(model_id: str, **overrides) -> ModelEntry:
    defaults = dict(
        model_id=model_id,
        display_name=model_id,
        version="1.0",
        total_parameters=7.0,
        active_parameters=None,
        architecture="dense",
        supported_tasks=["chat"],
        availability="public",
        minimum_plan="free",
        free_tier_available=True,
        api_available=True,
        local_available=True,
        remote_available=True,
        context_limit=8000,
        input_token_limit=8000,
        output_token_limit=2000,
        tool_call_limit=4,
        pricing=ModelPricing(input_price_per_1k=1.0, output_price_per_1k=2.0),
        routing_priority=10,
        fallback_model=None,
        license="apache-2.0",
        status=ModelStatus.AVAILABLE,
    )
    defaults.update(overrides)
    return ModelEntry(**defaults)


@pytest.fixture
def km(tmp_path) -> Keymaster:
    return Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)


@pytest.fixture
def gk(km, tmp_path) -> Gatekeeper:
    return Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper", log_to_file=False)


@pytest.fixture
def registry() -> ModelRegistry:
    reg = ModelRegistry()
    reg.register(_model("model-a", routing_priority=10))
    reg.register(_model("model-b", routing_priority=20))
    return reg


@pytest.fixture
def routing_table():
    """A cascade over the test registry.

    The shipped example policies reference the nanonix placeholder
    models, so a registry of test models has no policy that can route —
    every step is "not registered" and the router correctly raises
    ROUTING_EXHAUSTED. Routing tests need a table that matches their
    registry.
    """
    from hypernix.t1api.routing import CascadeStep, RoutingPolicy, RoutingTable

    policy = RoutingPolicy(
        name="test_default",
        steps=[CascadeStep(model_id="model-a"), CascadeStep(model_id="model-b")],
        description="model-a, falling back to model-b",
    )
    return RoutingTable(
        {policy.name: policy}, plan_to_policy={"free": "test_default"},
        default_policy="test_default",
    )


@pytest.fixture
def config(tmp_path) -> T1APIConfig:
    return T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough",
        db_path=str(tmp_path / "t1.sqlite3"),
        module_storage_dir=str(tmp_path / "modules"),
        default_plan="free",
    )


@pytest.fixture
def app(km, gk, registry, config, routing_table):
    return create_app(
        config=config, keymaster=km, gatekeeper=gk, registry=registry,
        routing_table=routing_table,
    )


@pytest.fixture
def client(app) -> TestClient:
    # A real peer address: TestClient's default is the literal string
    # "testclient", which is not an IP and so exercises the
    # network policy's "unparseable address" path rather than its normal
    # one. Tests that want that path set it explicitly.
    return TestClient(app, client=("127.0.0.1", 5000))


@pytest.fixture
def user_key(km) -> str:
    return km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key


@pytest.fixture
def admin_key(km) -> str:
    return km.create(
        key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}
    ).key


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _key_id(gk, key: str) -> str:
    return gk.authenticate(key).key_id


# ===========================================================================
# Status / config
# ===========================================================================


class TestStatusAndConfig:
    def test_status_reports_beta_and_protections(self, client):
        body = client.get("/status").json()
        assert body["beta"] == "t1-1.0"
        assert body["rate_limit_enabled"] is True
        assert body["audit_enabled"] is True
        assert body["network_policy_enabled"] is True

    def test_status_reports_secret_presence_never_values(self, client, config):
        response = client.get("/status")
        assert response.json()["secrets_configured"]["token_secret"] is True
        assert config.token_secret not in response.text

    def test_status_surfaces_production_warnings(self, client):
        warnings = client.get("/status").json()["production_warnings"]
        assert any("T1_DATABASE_URL" in w for w in warnings)

    def test_security_headers_on_every_response(self, client):
        headers = client.get("/health").headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert "X-Request-ID" in headers


# ===========================================================================
# Keys
# ===========================================================================


class TestKeysEndpoints:
    def test_list_requires_auth(self, client):
        assert client.get("/keys").status_code == 401

    def test_non_admin_sees_only_itself(self, client, km, user_key):
        km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        body = client.get("/keys", headers=_auth(user_key)).json()
        assert body["scope"] == "own"
        assert body["count"] == 1

    def test_admin_sees_all(self, client, km, admin_key):
        km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        body = client.get("/keys", headers=_auth(admin_key)).json()
        assert body["scope"] == "all"
        assert body["count"] >= 2

    def test_listing_never_returns_key_material(self, client, admin_key, user_key):
        response = client.get("/keys", headers=_auth(admin_key))
        assert user_key not in response.text
        assert admin_key not in response.text

    def test_assign_is_admin_only(self, client, gk, user_key):
        response = client.post(
            "/keys/assign",
            json={"key_id": _key_id(gk, user_key), "plan": "enterprise"},
            headers=_auth(user_key),
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "AUTH_ADMIN_REQUIRED"

    def test_assign_sets_the_plan(self, client, gk, admin_key, user_key):
        response = client.post(
            "/keys/assign",
            json={"key_id": _key_id(gk, user_key), "plan": "pro", "account_id": "acct1"},
            headers=_auth(admin_key),
        )
        assert response.status_code == 200
        assert response.json()["assignment"]["plan"] == "pro"

    def test_assign_rejects_an_unregistered_model(self, client, gk, admin_key, user_key):
        response = client.post(
            "/keys/assign",
            json={"key_id": _key_id(gk, user_key), "allowed_models": ["not-a-model"]},
            headers=_auth(admin_key),
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "MODEL_NOT_SUPPORTED"

    def test_import_is_admin_only(self, client, user_key):
        response = client.post(
            "/keys/import", json={"payload": {"keys": []}}, headers=_auth(user_key)
        )
        assert response.status_code == 403

    def test_import_rejects_a_malformed_payload(self, client, admin_key):
        response = client.post(
            "/keys/import", json={"payload": {"nope": []}}, headers=_auth(admin_key)
        )
        assert response.status_code == 422


# ===========================================================================
# Routing: the plan is the server's to decide
# ===========================================================================


class TestServerAuthoritativeRouting:
    def test_route_without_a_plan_uses_the_assigned_one(self, client, user_key):
        response = client.post(
            "/models/route", json={"input_tokens": 10}, headers=_auth(user_key)
        )
        assert response.status_code == 200
        assert response.json()["resolved_plan"] == "free"

    def test_a_client_cannot_claim_a_better_plan(self, client, user_key):
        """Beta 2 took `plan` from the request body. That let a client
        name the most generous plan it could think of."""
        response = client.post(
            "/models/route",
            json={"plan": "enterprise", "input_tokens": 10},
            headers=_auth(user_key),
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "AUTH_INSUFFICIENT_SCOPE"

    def test_asserting_the_correct_plan_is_accepted(self, client, user_key):
        response = client.post(
            "/models/route", json={"plan": "free", "input_tokens": 10}, headers=_auth(user_key)
        )
        assert response.status_code == 200

    def test_manual_selection_respects_key_model_narrowing(
        self, client, gk, admin_key, user_key
    ):
        client.post(
            "/keys/assign",
            json={"key_id": _key_id(gk, user_key), "allowed_models": ["model-a"]},
            headers=_auth(admin_key),
        )
        allowed = client.post(
            "/models/route", json={"model_id": "model-a"}, headers=_auth(user_key)
        )
        assert allowed.status_code == 200
        refused = client.post(
            "/models/route", json={"model_id": "model-b"}, headers=_auth(user_key)
        )
        assert refused.status_code == 403
        assert refused.json()["error"]["code"] == "AUTH_INSUFFICIENT_SCOPE"

    def test_unregistered_model_still_rejected(self, client, user_key):
        response = client.post(
            "/models/route", json={"model_id": "invented"}, headers=_auth(user_key)
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "MODEL_NOT_SUPPORTED"


# ===========================================================================
# Usage / cost
# ===========================================================================


class TestUsageEndpoints:
    def test_history_requires_auth(self, client):
        assert client.get("/usage/history").status_code == 401

    def test_history_is_scoped_to_the_caller(self, client, app, gk, user_key, admin_key):
        """A non-admin asking for someone else's key_id gets their own
        history, not a 403 and not someone else's rows."""
        store = app.state.t1_usage_store
        store.record(key_id=_key_id(gk, user_key), model_id="model-a", input_tokens=10)
        store.record(key_id=_key_id(gk, admin_key), model_id="model-a", input_tokens=99)

        body = client.get(
            "/usage/history", params={"key_id": _key_id(gk, admin_key)}, headers=_auth(user_key)
        ).json()
        assert body["count"] == 1
        assert body["events"][0]["input_tokens"] == 10

    def test_admin_can_filter_by_key(self, client, app, gk, user_key, admin_key):
        store = app.state.t1_usage_store
        store.record(key_id=_key_id(gk, user_key), model_id="model-a", input_tokens=10)
        body = client.get(
            "/usage/history", params={"key_id": _key_id(gk, user_key)}, headers=_auth(admin_key)
        ).json()
        assert body["count"] == 1

    def test_usage_by_dimension(self, client, app, gk, user_key):
        store = app.state.t1_usage_store
        key_id = _key_id(gk, user_key)
        store.record(key_id=key_id, model_id="model-a", input_tokens=10, server_id="s1")
        store.record(key_id=key_id, model_id="model-b", input_tokens=20, server_id="s1")
        body = client.get(
            "/usage/by", params={"group_by": "model_id"}, headers=_auth(user_key)
        ).json()
        assert {row["value"] for row in body["rows"]} == {"model-a", "model-b"}

    def test_usage_by_rejects_an_invalid_dimension(self, client, user_key):
        response = client.get(
            "/usage/by", params={"group_by": "1; DROP TABLE"}, headers=_auth(user_key)
        )
        assert response.status_code == 422

    def test_cost_totals_and_forecast(self, client, app, gk, user_key):
        store = app.state.t1_usage_store
        store.record(
            key_id=_key_id(gk, user_key), model_id="model-a", input_tokens=1000, output_tokens=1000
        )
        body = client.get(
            "/usage/cost", params={"forecast": "true"}, headers=_auth(user_key)
        ).json()
        assert body["total_cost"] == pytest.approx(3.0)  # 1.0 in + 2.0 out
        assert body["forecast"]["confidence"] in ("low", "medium", "high")

    def test_estimate(self, client, user_key):
        body = client.post(
            "/usage/estimate",
            json={"model_id": "model-a", "input_tokens": 1000, "output_tokens": 1000},
            headers=_auth(user_key),
        ).json()
        assert body["estimated_cost"] == pytest.approx(3.0)
        assert body["output_tokens_assumed"] is False

    def test_estimate_rejects_an_unregistered_model(self, client, user_key):
        response = client.post(
            "/usage/estimate",
            json={"model_id": "invented", "input_tokens": 10},
            headers=_auth(user_key),
        )
        assert response.status_code == 404

    def test_invalid_range_is_rejected(self, client, user_key):
        response = client.get(
            "/usage/history", params={"range": "42 fortnights"}, headers=_auth(user_key)
        )
        assert response.status_code == 422


# ===========================================================================
# Audit
# ===========================================================================


class TestAuditEndpoint:
    def test_admin_only(self, client, user_key):
        response = client.get("/audit", headers=_auth(user_key))
        assert response.status_code == 403

    def test_records_admin_actions(self, client, gk, admin_key, user_key):
        client.post(
            "/keys/assign",
            json={"key_id": _key_id(gk, user_key), "plan": "pro"},
            headers=_auth(admin_key),
        )
        body = client.get("/audit", params={"action": "keys.assign"}, headers=_auth(admin_key)).json()
        assert body["count"] >= 1
        assert body["events"][0]["outcome"] == "success"

    def test_records_a_refusal(self, client, gk, admin_key, user_key):
        client.post(
            "/keys/assign",
            json={"key_id": _key_id(gk, user_key), "plan": "enterprise"},
            headers=_auth(user_key),
        )
        body = client.get(
            "/audit", params={"action": "keys.assign", "outcome": "denied"}, headers=_auth(admin_key)
        ).json()
        assert body["count"] >= 1

    def test_records_security_events(self, client, admin_key):
        client.get("/keys", headers=_auth("T1_definitely-not-a-real-key"))
        body = client.get(
            "/audit", params={"category": "security"}, headers=_auth(admin_key)
        ).json()
        assert any("auth_invalid_key" in e["action"] for e in body["events"])

    def test_audit_never_contains_a_credential(self, client, admin_key, user_key):
        client.get("/keys", headers=_auth(user_key))
        response = client.get("/audit", headers=_auth(admin_key))
        assert user_key not in response.text
        assert admin_key not in response.text


# ===========================================================================
# Network policy
# ===========================================================================


class TestNetworkPolicyEndpoints:
    def test_admin_only(self, client, user_key):
        assert client.get("/security/network", headers=_auth(user_key)).status_code == 403

    def test_block_and_appeal(self, client, admin_key):
        blocked = client.post(
            "/security/network/blacklist",
            json={"cidr": "203.0.113.0/24", "reason": "abuse"},
            headers=_auth(admin_key),
        )
        assert blocked.status_code == 200
        assert any(e["cidr"] == "203.0.113.0/24" for e in blocked.json()["entries"])

        appealed = client.delete("/security/network/203.0.113.0/24", headers=_auth(admin_key))
        assert appealed.status_code == 200
        assert appealed.json()["count"] == 0

    def test_appealing_an_unknown_entry_is_404(self, client, admin_key):
        assert client.delete("/security/network/198.51.100.1", headers=_auth(admin_key)).status_code == 404

    def test_cannot_block_your_own_address(self, client, admin_key):
        """Locking yourself out has no undo through this API."""
        response = client.post(
            "/security/network/blacklist",
            json={"cidr": "127.0.0.1"},  # the test client's own address
            headers=_auth(admin_key),
        )
        assert response.status_code == 422
        assert "lock you out" in response.json()["error"]["message"]

    def test_can_block_a_range_that_covers_your_address_only_deliberately(
        self, client, admin_key
    ):
        """The guard is about the caller's own address, not about
        blocking ranges in general."""
        response = client.post(
            "/security/network/blacklist",
            json={"cidr": "203.0.113.0/24"},
            headers=_auth(admin_key),
        )
        assert response.status_code == 200

    def test_blocked_client_is_refused_before_authentication(self, km, gk, registry, tmp_path):
        """Middleware order: the network policy runs before the credential
        is examined, so a blocked address is turned away regardless of how
        good its key is."""
        config = T1APIConfig(
            token_secret="x" * 40,
            db_path=str(tmp_path / "t1.sqlite3"),
            module_storage_dir=str(tmp_path / "modules"),
        )
        app = create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry)
        app.state.t1_network_policy.block("198.51.100.7", reason="test")
        client = TestClient(app, client=("198.51.100.7", 5000))

        admin = km.create(key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN, KeyScope.READ}).key
        response = client.get("/keys", headers=_auth(admin))
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "IP_BLOCKED"

    def test_allowlist_only_mode(self, km, gk, registry, tmp_path):
        config = T1APIConfig(
            token_secret="x" * 40,
            db_path=str(tmp_path / "t1.sqlite3"),
            allow_unlisted_clients=False,
        )
        app = create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry)
        client = TestClient(app, client=("10.1.2.3", 5000))
        assert client.get("/health").status_code == 403
        assert client.get("/health").json()["error"]["code"] == "IP_NOT_ALLOWLISTED"

        app.state.t1_network_policy.allow("10.0.0.0/8")
        assert client.get("/health").status_code == 200


class TestForcedLimitEndpoints:
    def test_admin_only(self, client, user_key):
        assert client.get("/security/limits", headers=_auth(user_key)).status_code == 403

    def test_set_list_and_clear(self, client, admin_key):
        created = client.post(
            "/security/limits",
            json={"subject_type": "key", "subject_id": "abc123", "requests_per_window": 5},
            headers=_auth(admin_key),
        )
        assert created.status_code == 200
        assert client.get("/security/limits", headers=_auth(admin_key)).json()["count"] == 1
        assert client.delete("/security/limits/key/abc123", headers=_auth(admin_key)).status_code == 200
        assert client.get("/security/limits", headers=_auth(admin_key)).json()["count"] == 0

    def test_requires_at_least_one_bound(self, client, admin_key):
        response = client.post(
            "/security/limits",
            json={"subject_type": "key", "subject_id": "abc"},
            headers=_auth(admin_key),
        )
        assert response.status_code == 422

    def test_own_rate_limit_budget_is_self_service(self, client, user_key):
        """Seeing your remaining budget lets you back off before a 429
        rather than after one."""
        body = client.get("/security/rate-limits", headers=_auth(user_key)).json()
        assert body["enabled"] is True
        assert len(body["rules"]) >= 1


# ===========================================================================
# Rate limiting middleware
# ===========================================================================


class TestRateLimitMiddleware:
    def test_over_limit_returns_429_with_retry_after(self, km, gk, registry, tmp_path):
        config = T1APIConfig(token_secret="x" * 40, db_path=str(tmp_path / "t1.sqlite3"))
        limiter = RateLimiter(
            [RateRule("tiny", capacity=2, refill_per_second=0.1, applies_to=("ip", "key"))]
        )
        app = create_app(
            config=config, keymaster=km, gatekeeper=gk, registry=registry, rate_limiter=limiter
        )
        client = TestClient(app)

        seen = [client.get("/health").status_code for _ in range(6)]
        assert 429 in seen
        response = client.get("/health")
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "RATE_LIMITED"
        assert "Retry-After" in response.headers

    def test_limit_applies_before_the_handler(self, km, gk, registry, tmp_path):
        """An over-limit caller must not reach model work at all — which
        is what "apply rate limits before expensive model operations"
        means in practice."""
        config = T1APIConfig(token_secret="x" * 40, db_path=str(tmp_path / "t1.sqlite3"))
        limiter = RateLimiter(
            [RateRule("none", capacity=0, refill_per_second=0, applies_to=("ip",))]
        )
        app = create_app(
            config=config, keymaster=km, gatekeeper=gk, registry=registry, rate_limiter=limiter
        )
        client = TestClient(app)
        # Even an unregistered model returns RATE_LIMITED, not
        # MODEL_NOT_SUPPORTED: the registry lookup never happened.
        response = client.get("/models/whatever")
        assert response.status_code == 429

    def test_disabled_limiter_lets_everything_through(self, km, gk, registry, tmp_path):
        config = T1APIConfig(
            token_secret="x" * 40, db_path=str(tmp_path / "t1.sqlite3"), rate_limit_enabled=False
        )
        app = create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry)
        client = TestClient(app)
        assert all(client.get("/health").status_code == 200 for _ in range(30))


# ===========================================================================
# mTLS middleware
# ===========================================================================


class TestMTLSMiddleware:
    @pytest.fixture
    def mtls_app(self, km, gk, registry, tmp_path):
        config = T1APIConfig(
            token_secret="x" * 40,
            db_path=str(tmp_path / "t1.sqlite3"),
            mtls_require_client_cert=True,
            mtls_behind_proxy=True,
            trusted_proxies=("127.0.0.1",),
        )
        return create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry)

    def test_request_without_a_client_cert_is_refused(self, mtls_app):
        client = TestClient(mtls_app, client=("127.0.0.1", 5000))
        response = client.get("/models")
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "MTLS_REQUIRED"

    def test_verified_cert_from_the_trusted_proxy_is_accepted(self, mtls_app):
        client = TestClient(mtls_app, client=("127.0.0.1", 5000))
        response = client.get(
            "/models",
            headers={"X-Client-Verify": "SUCCESS", "X-Client-Subject-DN": "CN=deploy-01"},
        )
        assert response.status_code == 200

    def test_forged_header_from_an_untrusted_peer_is_ignored(self, mtls_app):
        """The whole point of the trusted-proxy check."""
        client = TestClient(mtls_app, client=("203.0.113.9", 5000))
        response = client.get("/models", headers={"X-Client-Verify": "SUCCESS"})
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "MTLS_REQUIRED"

    def test_health_stays_reachable_for_load_balancers(self, mtls_app):
        client = TestClient(mtls_app, client=("203.0.113.9", 5000))
        assert client.get("/health").status_code == 200


# ===========================================================================
# Destructive-operation confirmation
# ===========================================================================


class TestDestructiveConfirmation:
    def test_server_delete_needs_confirmation(self, client, admin_key):
        server_id = client.post(
            "/servers/register",
            json={"name": "s", "address": "https://s.example.com"},
            headers=_auth(admin_key),
        ).json()["server"]["server_id"]

        unconfirmed = client.delete(f"/servers/{server_id}", headers=_auth(admin_key))
        assert unconfirmed.status_code == 409
        assert unconfirmed.json()["error"]["code"] == "CONFIRMATION_REQUIRED"

        confirmed = client.delete(f"/servers/{server_id}?confirm=true", headers=_auth(admin_key))
        assert confirmed.status_code == 200

    def test_confirmation_can_be_turned_off_deliberately(self, km, gk, registry, tmp_path):
        config = T1APIConfig(
            token_secret="x" * 40,
            db_path=str(tmp_path / "t1.sqlite3"),
            require_destructive_confirmation=False,
        )
        app = create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry)
        client = TestClient(app)
        admin = km.create(
            key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}
        ).key
        server_id = client.post(
            "/servers/register",
            json={"name": "s", "address": "https://s.example.com"},
            headers=_auth(admin),
        ).json()["server"]["server_id"]
        assert client.delete(f"/servers/{server_id}", headers=_auth(admin)).status_code == 200


# ===========================================================================
# Module receive (server-to-server)
# ===========================================================================


class TestModuleReceive:
    @pytest.fixture
    def receiver(self, km, gk, registry, tmp_path):
        config = T1APIConfig(
            token_secret="x" * 40,
            db_path=str(tmp_path / "t1.sqlite3"),
            module_storage_dir=str(tmp_path / "modules"),
            deploy_secret="d" * 40,
        )
        return TestClient(create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry))

    def _signed(self, body: bytes, *, secret: str = "d" * 40, path: str = "/modules/receive"):
        import hashlib
        import time

        from hypernix.t1api.transport import compute_signature

        stamp = f"{time.time():.3f}"
        return {
            "X-T1-Signature": compute_signature(
                secret, method="POST", path=path, timestamp=stamp,
                body_digest=hashlib.sha256(body).hexdigest(),
            ),
            "X-T1-Timestamp": stamp,
            "X-T1-Content-SHA256": hashlib.sha256(body).hexdigest(),
            "X-T1-Module-ID": "mod-1",
            "Content-Type": "application/octet-stream",
        }

    def test_accepts_a_correctly_signed_transfer(self, receiver):
        import hashlib

        body = b"module-bytes"
        response = receiver.post("/modules/receive", content=body, headers=self._signed(body))
        assert response.status_code == 200
        assert response.json()["checksum"] == hashlib.sha256(body).hexdigest()

    def test_rejects_an_unsigned_transfer(self, receiver):
        response = receiver.post("/modules/receive", content=b"x")
        assert response.status_code == 401

    def test_rejects_a_wrong_signature(self, receiver):
        body = b"x"
        response = receiver.post(
            "/modules/receive", content=body, headers=self._signed(body, secret="wrong" * 8)
        )
        assert response.status_code == 401

    def test_rejects_a_tampered_body(self, receiver):
        headers = self._signed(b"original")
        response = receiver.post("/modules/receive", content=b"tampered", headers=headers)
        assert response.status_code == 401

    def test_is_idempotent_by_module_id(self, receiver):
        body = b"same-bytes"
        first = receiver.post("/modules/receive", content=body, headers=self._signed(body))
        second = receiver.post("/modules/receive", content=body, headers=self._signed(body))
        assert first.json()["module_id"] == second.json()["module_id"]
        assert second.status_code == 200

    def test_a_server_without_a_deploy_secret_refuses(self, km, gk, registry, tmp_path):
        config = T1APIConfig(token_secret="x" * 40, db_path=str(tmp_path / "t1.sqlite3"))
        client = TestClient(create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry))
        body = b"x"
        response = client.post("/modules/receive", content=body, headers=self._signed(body))
        assert response.status_code == 403


# ===========================================================================
# OpenAPI
# ===========================================================================


class TestOpenAPI:
    def test_every_spec_endpoint_is_documented(self, client):
        """The spec's PRIMARY ENDPOINT SET, checked against the served
        schema rather than against a doc that could drift."""
        paths = set(client.get("/openapi.json").json()["paths"])
        required = {
            "/health", "/status",
            "/auth/t1/validate", "/auth/t1/admin/rotate", "/auth/t1/rotate", "/auth/token",
            "/usage/remaining", "/usage/current", "/usage/history", "/usage/cost", "/usage/estimate",
            "/models", "/models/{model_id}", "/models/{model_id}/availability",
            "/models/{model_id}/usage",
            "/modules", "/modules/create", "/modules/upload/local", "/modules/upload/remote",
            "/modules/{module_id}", "/modules/{module_id}/sync",
            "/servers", "/servers/register", "/servers/{server_id}",
            "/keys", "/keys/import", "/keys/assign",
            "/billing/balance", "/billing/transactions", "/billing/payment-token",
            "/billing/redeem", "/billing/add-balance",
            "/jobs", "/jobs/{job_id}", "/jobs/{job_id}/cancel",
            "/events",
            "/config",
        }
        assert required <= paths, f"missing from the served schema: {sorted(required - paths)}"

    def test_docs_can_be_switched_off(self, km, gk, registry, tmp_path):
        config = T1APIConfig(
            token_secret="x" * 40, db_path=str(tmp_path / "t1.sqlite3"), expose_docs=False
        )
        client = TestClient(create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry))
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/docs").status_code == 404


# ===========================================================================
# Production startup validation
# ===========================================================================


class TestProductionStartup:
    def test_create_app_refuses_an_unsafe_production_config(self, km, gk, registry, tmp_path):
        """A bad production config should fail the deploy, not surface
        later as a puzzling 500."""
        from hypernix.t1api.errors import T1APIError

        config = T1APIConfig(environment="production", db_path=str(tmp_path / "t1.sqlite3"))
        with pytest.raises(T1APIError) as exc:
            create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry)
        assert exc.value.code.value == "CONFIG_INVALID"

    def test_validation_can_be_skipped_explicitly(self, km, gk, registry, tmp_path):
        config = T1APIConfig(environment="production", db_path=str(tmp_path / "t1.sqlite3"))
        app = create_app(
            config=config, keymaster=km, gatekeeper=gk, registry=registry,
            validate_production=False,
        )
        assert TestClient(app).get("/health").status_code == 200
