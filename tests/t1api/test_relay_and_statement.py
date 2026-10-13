"""``POST /inference/relay`` and ``GET /usage/statement`` (0.72.8).

The relay reads a document longer than the model's context in hops; the
statement is the itemised bill that prices those hops alongside the
features billed in something other than tokens.
"""
from __future__ import annotations

from unittest import mock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402
from hypernix.t1api.cost import STATEMENT_FORMAT, feature_of  # noqa: E402
from hypernix.t1api.registry import ModelEntry, ModelPricing, ModelRegistry, ModelStatus  # noqa: E402

BRIDGE = "hypernix.t1api.routers.inference.LMStudioBridge"


def _model(model_id: str, *, context: int, unit_prices: dict | None = None) -> ModelEntry:
    return ModelEntry(
        model_id=model_id, display_name=model_id, version="1.0", total_parameters=1.0,
        active_parameters=None, architecture="dense", supported_tasks=["chat"],
        availability="public", minimum_plan="free", free_tier_available=True,
        api_available=True, local_available=True, remote_available=True,
        context_limit=context, input_token_limit=10_000_000, output_token_limit=10_000_000,
        tool_call_limit=4,
        pricing=ModelPricing(input_price_per_1k=1.0, output_price_per_1k=2.0,
                             unit_prices=unit_prices or {}),
        routing_priority=10, fallback_model=None, license="apache-2.0",
        status=ModelStatus.AVAILABLE,
    )


def _document(paragraphs: int) -> str:
    return "\n\n".join(" ".join(f"p{i}w{j}" for j in range(60)) + "." for i in range(paragraphs))


@pytest.fixture
def km(tmp_path) -> Keymaster:
    return Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)


@pytest.fixture
def config(tmp_path) -> T1APIConfig:
    return T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough",
        db_path=str(tmp_path / "t1.sqlite3"),
        module_storage_dir=str(tmp_path / "modules"),
        hyperlink_files_dir=str(tmp_path / "files"),
        lmstudio_url="http://lmstudio.test:1234",
        default_plan="free",
        feature_prices_json='{"code_second": 0.01, "web_search": 0.5}',
    )


@pytest.fixture
def app(km, config, tmp_path):
    gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper", log_to_file=False)
    registry = ModelRegistry()
    registry.register(_model("small", context=1024))
    registry.register(_model("big", context=128_000, unit_prices={"image": 0.25}))
    return create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app, client=("127.0.0.1", 5000))


@pytest.fixture
def key(km):
    return km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE})


def _auth(key) -> dict[str, str]:
    return {"Authorization": f"Bearer {key.key}"}


def _fake_backend(factory) -> list[dict]:
    """A backend that answers every hop and reports usage."""
    sent: list[dict] = []

    def chat(messages, **kwargs):
        sent.append({"messages": messages, **kwargs})
        prompt = sum(len(m["content"]) for m in messages) // 4
        return {"choices": [{"message": {"content": f"notes {len(sent)}"}}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": 10}}

    factory.return_value.chat.side_effect = chat
    factory.return_value.base_url = "http://lmstudio.test:1234"
    return sent


class TestRelay:
    def test_a_long_document_is_read_in_hops_and_each_hop_billed(self, client, key, app):
        doc = _document(40)
        with mock.patch(BRIDGE) as factory:
            sent = _fake_backend(factory)
            response = client.post("/inference/relay", headers=_auth(key), json={
                "model": "small", "question": "what is in p3?", "document": doc, "instances": 3,
            })
        assert response.status_code == 200, response.text
        body = response.json()
        plan = body["plan"]
        assert plan["chunks"] > 1 and plan["context_tokens"] == 1024
        assert body["requests"] == len(sent) == plan["requests"] == len(body["hops"])
        assert [h["instance"] for h in body["hops"]] == [i % 3 for i in range(len(sent))]
        assert body["content"] == f"notes {len(sent)}"
        # No call was larger than the model's context.
        assert all(len("".join(m["content"] for m in call["messages"])) // 4 + call["max_tokens"] <= 1024
                   for call in sent)
        # Priced from what the backend reported, at 1.0/1k in and 2.0/1k out.
        assert body["cost"] == pytest.approx(body["input_tokens"] / 1000 + body["output_tokens"] * 2 / 1000)
        assert body["estimated_cost"] >= body["cost"]
        assert body["direct_cost"] < body["estimated_cost"]
        assert plan["overhead_factor"] > 1
        history = app.state.t1_usage_meter.store.history(endpoint="/inference/relay", limit=1000)
        assert len(history) == len(sent) and {e["feature"] for e in history} == {"relay"}

    def test_estimate_only_runs_nothing_and_records_nothing(self, client, key, app):
        with mock.patch(BRIDGE) as factory:
            response = client.post("/inference/relay", headers=_auth(key), json={
                "model": "small", "question": "q", "document": _document(40), "estimate_only": True,
            })
            assert not factory.return_value.chat.called
        body = response.json()
        assert response.status_code == 200 and body["estimate_only"] is True
        assert body["requests"] == 0 and body["estimated_cost"] > 0
        assert app.state.t1_usage_meter.store.event_count() == 0

    def test_a_document_that_fits_is_one_call(self, client, key):
        with mock.patch(BRIDGE) as factory:
            sent = _fake_backend(factory)
            body = client.post("/inference/relay", headers=_auth(key), json={
                "model": "big", "question": "q", "document": _document(40),
            }).json()
        assert body["plan"]["single"] is True and len(sent) == 1

    def test_a_context_too_small_is_a_validation_error(self, client, key):
        response = client.post("/inference/relay", headers=_auth(key), json={
            "model": "small", "question": "q " * 900, "document": _document(40), "estimate_only": True,
        })
        assert response.status_code == 422 and "context" in response.json()["error"]["message"]

    def test_the_mode_is_checked(self, client, key):
        response = client.post("/inference/relay", headers=_auth(key), json={
            "model": "small", "question": "q", "mode": "sideways", "estimate_only": True,
        })
        assert response.status_code == 422

    def test_the_model_is_governed(self, client, key):
        response = client.post("/inference/relay", headers=_auth(key), json={
            "model": "nope", "question": "q", "estimate_only": True,
        })
        assert response.json()["error"]["code"] == "MODEL_NOT_SUPPORTED"

    def test_instances_are_capped_by_the_server(self, client, key, config):
        config.relay_max_instances = 2
        body = client.post("/inference/relay", headers=_auth(key), json={
            "model": "small", "question": "q", "document": _document(40), "instances": 50,
            "estimate_only": True,
        }).json()
        assert body["instances"] == 2


class TestStatement:
    def test_it_itemises_tokens_and_units_per_feature(self, client, key, app):
        meter = app.state.t1_usage_meter
        with mock.patch(BRIDGE) as factory:
            _fake_backend(factory)
            assert client.post("/inference/chat", headers=_auth(key), json={
                "model": "small", "messages": [{"role": "user", "content": "hi"}],
            }).status_code == 200
        meter.record_feature(key_id=key.key_id, feature="code", endpoint="/code/create",
                             unit="code_second", units=12.5)
        meter.record_feature(key_id=key.key_id, feature="web_search", endpoint="/web/v1/search",
                             unit="web_search", units=1)
        meter.record_feature(key_id=key.key_id, feature="compact", endpoint="/chat/compact",
                             model_id="not-registered", input_tokens=300, output_tokens=40)
        meter.record_feature(key_id=key.key_id, feature="inference", model_id="big",
                             unit="image", units=2)
        # A row from before 0.72.8: no feature, read from the endpoint.
        meter.store.record(key_id=key.key_id, model_id="small", input_tokens=1000,
                           endpoint="/inference/embeddings")

        response = client.get("/usage/statement", headers=_auth(key))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["format"] == STATEMENT_FORMAT
        features = {f["feature"]: f for f in body["features"]}
        assert list(features) == ["inference", "embeddings", "compact", "code", "web_search"]
        assert features["code"]["subtotal"] == pytest.approx(0.125)
        assert features["code"]["units"] == {"code_second": 12.5}
        assert features["web_search"]["subtotal"] == pytest.approx(0.5)
        assert features["embeddings"]["subtotal"] == pytest.approx(1.0)
        image = next(i for i in features["inference"]["items"] if i["unit"] == "image")
        assert image["unit_price"] == 0.25 and image["total"] == pytest.approx(0.5)
        assert {"feature": "compact", "model_id": "", "unit": ""} in body["unpriced"]
        assert body["total"] == pytest.approx(sum(f["subtotal"] for f in body["features"]))

    def test_another_keys_usage_is_not_on_it(self, client, key, km, app):
        other = km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        app.state.t1_usage_meter.record_feature(key_id=other.key_id, feature="code",
                                                unit="code_second", units=99)
        body = client.get("/usage/statement", headers=_auth(key)).json()
        assert body["features"] == [] and body["total"] == 0

    def test_history_carries_the_new_fields(self, client, key, app):
        app.state.t1_usage_meter.record_feature(key_id=key.key_id, feature="web_search",
                                                unit="web_search", units=1)
        event = client.get("/usage/history", headers=_auth(key)).json()["events"][0]
        assert (event["feature"], event["unit"], event["units"]) == ("web_search", "web_search", 1.0)

    def test_old_rows_are_given_a_feature_by_endpoint(self):
        assert feature_of("", "/inference/relay") == "relay"
        assert feature_of("", "/inference/chat/stream") == "inference"
        assert feature_of("", "/code/sandbox/run") == "code"
        assert feature_of("", "") == "other"
        assert feature_of("made-up", "/inference/chat") == "other"


def test_an_old_database_gains_the_columns(tmp_path):
    import sqlite3

    from hypernix.t1api.storage import UsageStore

    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE usage_events (id INTEGER PRIMARY KEY AUTOINCREMENT, key_id TEXT NOT NULL,"
                     " model_id TEXT NOT NULL, input_tokens INTEGER NOT NULL DEFAULT 0,"
                     " output_tokens INTEGER NOT NULL DEFAULT 0, requests INTEGER NOT NULL DEFAULT 1,"
                     " endpoint TEXT NOT NULL DEFAULT '', server_id TEXT NOT NULL DEFAULT '',"
                     " module_id TEXT NOT NULL DEFAULT '', ts REAL NOT NULL)")
        conn.execute("INSERT INTO usage_events (key_id, model_id, input_tokens, endpoint, ts)"
                     " VALUES ('k', 'm', 5, '/web/v1/search', 1.0)")
    store = UsageStore(path)
    rows = store.line_items()
    assert rows == [{"feature": "", "endpoint": "/web/v1/search", "model_id": "m", "unit": "",
                     "units": 0.0, "requests": 1, "input_tokens": 5, "output_tokens": 0,
                     "total_tokens": 5}]


def test_unit_prices_round_trip_and_stay_out_of_old_registries():
    plain = ModelPricing(input_price_per_1k=1.0)
    assert "unit_prices" not in plain.to_dict()
    priced = ModelPricing.from_dict({"unit_prices": {"image": "0.5"}})
    assert priced.unit_prices == {"image": 0.5}
    assert ModelPricing.from_dict(priced.to_dict()) == priced


def test_feature_prices_are_parsed_leniently(monkeypatch):
    monkeypatch.setenv("T1_FEATURE_PRICES", '{"code_second": "0.2", "web_search": "free"}')
    assert T1APIConfig().feature_prices() == {"code_second": 0.2}
    monkeypatch.setenv("T1_FEATURE_PRICES", "not json")
    assert T1APIConfig().feature_prices() == {}
