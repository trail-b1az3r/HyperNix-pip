"""Beta 4: ``POST /usage/report``.

The endpoint exists so a client that runs inference itself can push its
token counts back, which is what makes the quota cascade advance for
``hyped-pro`` against a remote T1 server. Because any authenticated key
can call it, the tests here are weighted towards what it must *refuse*:
reporting for another key, reporting an unregistered or disallowed model,
and above all reporting a negative count — a client that could subtract
usage could refund itself quota, which would make every limit advisory.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402
from hypernix.t1api.registry import (  # noqa: E402
    ModelEntry,
    ModelPricing,
    ModelRegistry,
    ModelStatus,
)
from hypernix.t1api.schemas import MAX_REPORTED_REQUESTS, MAX_REPORTED_TOKENS  # noqa: E402


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
    return TestClient(app, client=("127.0.0.1", 5000))


@pytest.fixture
def user_key(km) -> str:
    return km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key


@pytest.fixture
def other_key(km) -> str:
    return km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key


@pytest.fixture
def admin_key(km) -> str:
    return km.create(
        key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}
    ).key


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _report(client, key, **body):
    payload = {"model_id": "model-a", "input_tokens": 100, "output_tokens": 50}
    payload.update(body)
    return client.post("/usage/report", json=payload, headers=_auth(key))


# ===========================================================================
# The happy path
# ===========================================================================


class TestReportRecords:
    def test_requires_authentication(self, client):
        response = client.post("/usage/report", json={"model_id": "model-a"})
        assert response.status_code == 401

    def test_records_and_returns_remaining(self, client, user_key):
        body = _report(client, user_key).json()
        assert body["recorded"] is True
        assert body["model_id"] == "model-a"
        assert body["input_remaining"] == 8000 - 100
        assert body["output_remaining"] == 2000 - 50
        assert body["exhausted"] is False

    def test_reports_accumulate(self, client, user_key):
        _report(client, user_key, input_tokens=100, output_tokens=0)
        body = _report(client, user_key, input_tokens=250, output_tokens=0).json()
        assert body["input_remaining"] == 8000 - 350

    def test_shows_up_in_usage_current(self, client, user_key):
        _report(client, user_key, input_tokens=400, output_tokens=200)
        current = client.get("/usage/current", headers=_auth(user_key)).json()
        assert current["current_window"]["input_tokens"] == 400
        assert current["current_window"]["output_tokens"] == 200

    def test_shows_up_in_model_usage(self, client, user_key):
        _report(client, user_key, input_tokens=400, output_tokens=200)
        usage = client.get("/models/model-a/usage", headers=_auth(user_key)).json()
        assert usage["input_tokens_used"] == 400
        assert usage["requests"] == 1

    def test_zero_tokens_is_a_valid_report(self, client, user_key):
        """A request that produced no output still consumed a request slot."""
        body = _report(client, user_key, input_tokens=0, output_tokens=0).json()
        assert body["recorded"] is True
        usage = client.get("/models/model-a/usage", headers=_auth(user_key)).json()
        assert usage["requests"] == 1

    def test_optional_attribution_fields_are_accepted(self, client, user_key):
        response = _report(client, user_key, endpoint="/chat", server_id="srv-1",
                           module_id="mod-1")
        assert response.status_code == 200


# ===========================================================================
# What it must refuse
# ===========================================================================


class TestReportRefusals:
    def test_negative_input_tokens_refused(self, client, user_key):
        """The security-critical one: a client that could report negative
        usage could refund itself quota."""
        assert _report(client, user_key, input_tokens=-500).status_code == 422

    def test_negative_output_tokens_refused(self, client, user_key):
        assert _report(client, user_key, output_tokens=-500).status_code == 422

    def test_zero_requests_refused(self, client, user_key):
        assert _report(client, user_key, requests=0).status_code == 422

    def test_absurd_token_count_refused(self, client, user_key):
        assert _report(client, user_key,
                       input_tokens=MAX_REPORTED_TOKENS + 1).status_code == 422

    def test_absurd_request_count_refused(self, client, user_key):
        assert _report(client, user_key,
                       requests=MAX_REPORTED_REQUESTS + 1).status_code == 422

    def test_unregistered_model_refused(self, client, user_key):
        response = _report(client, user_key, model_id="model-that-does-not-exist")
        assert response.status_code >= 400
        assert response.json()["error"]["code"] == "MODEL_NOT_SUPPORTED"

    def test_model_excluded_by_the_key_assignment_refused(
        self, client, user_key, admin_key, gk
    ):
        key_id = gk.authenticate(user_key).key_id
        assign = client.post(
            "/keys/assign",
            json={"key_id": key_id, "plan": "free", "allowed_models": ["model-b"]},
            headers=_auth(admin_key),
        )
        assert assign.status_code == 200

        response = _report(client, user_key, model_id="model-a")
        assert response.status_code >= 400
        assert response.json()["error"]["code"] == "AUTH_INSUFFICIENT_SCOPE"

    def test_there_is_no_way_to_report_for_another_key(self, client, user_key, other_key, gk):
        """A body-supplied key_id must be ignored, not honoured — otherwise
        any valid key could burn any other key's quota."""
        victim = gk.authenticate(other_key).key_id
        response = client.post(
            "/usage/report",
            json={"model_id": "model-a", "input_tokens": 5000, "key_id": victim},
            headers=_auth(user_key),
        )
        assert response.status_code == 200

        # The victim's counters are untouched; the caller's moved.
        victim_usage = client.get("/usage/current", headers=_auth(other_key)).json()
        assert victim_usage["current_window"]["input_tokens"] == 0
        caller_usage = client.get("/usage/current", headers=_auth(user_key)).json()
        assert caller_usage["current_window"]["input_tokens"] == 5000


# ===========================================================================
# Interaction with the quota cascade — the reason the endpoint exists
# ===========================================================================


class TestReportDrivesTheCascade:
    def test_report_can_exhaust_a_model(self, client, user_key):
        body = _report(client, user_key, input_tokens=8000, output_tokens=0).json()
        assert body["input_remaining"] == 0
        assert body["exhausted"] is True

    def test_exhausting_report_still_succeeds(self, client, user_key):
        """The tokens really were spent. Refusing the accounting for work
        already done would just lose the record."""
        response = _report(client, user_key, input_tokens=8000)
        assert response.status_code == 200
        assert response.json()["recorded"] is True

    def test_routing_moves_to_the_fallback_after_a_report_exhausts_the_first(
        self, client, user_key
    ):
        first = client.post("/models/route", json={}, headers=_auth(user_key)).json()
        assert first["model_id"] == "model-a"

        _report(client, user_key, model_id="model-a", input_tokens=8000, output_tokens=0)

        second = client.post("/models/route", json={}, headers=_auth(user_key)).json()
        assert second["model_id"] == "model-b"

    def test_manual_selection_of_an_exhausted_model_is_refused_after_a_report(
        self, client, user_key
    ):
        _report(client, user_key, model_id="model-a", input_tokens=8000, output_tokens=0)
        response = client.post(
            "/models/route", json={"model_id": "model-a"}, headers=_auth(user_key)
        )
        assert response.status_code >= 400
        assert response.json()["error"]["code"] == "MODEL_QUOTA_EXHAUSTED"


# ===========================================================================
# Auditing
# ===========================================================================


class TestReportIsAudited:
    def test_report_writes_an_audit_record(self, client, user_key, admin_key):
        _report(client, user_key, input_tokens=123, output_tokens=45)
        events = client.get("/audit", headers=_auth(admin_key)).json()["events"]
        reports = [e for e in events if e["action"] == "usage.report"]
        assert reports
        assert reports[0]["resource_id"] == "model-a"

    def test_audit_record_carries_no_raw_key(self, client, user_key, admin_key):
        _report(client, user_key)
        response = client.get("/audit", headers=_auth(admin_key))
        assert user_key not in response.text
