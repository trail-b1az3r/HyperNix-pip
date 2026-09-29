"""``/inference/*`` — generation, with the rules the bridge skips.

``/bridge/lmstudio/*`` hands the caller's model string straight to LM
Studio. The registry never sees it, the plan's cascade never runs, the
quota is never consulted and nothing is metered — correct for a window
onto someone else's server, and it left the one path that actually
spends money outside every rule the rest of the API enforces.

So most of what is worth testing here is the refusals, and that the
governed path cannot be talked out of them: an unregistered model, a
model the key's assignment excludes, an exhausted allowance, a silent
substitution nobody asked for.
"""
from __future__ import annotations

from unittest import mock

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.bridge.lmstudio import LMStudioError  # noqa: E402
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
from hypernix.t1api.version import INFERENCE_VERSION, T1_VERSION  # noqa: E402

BRIDGE = "hypernix.t1api.routers.inference.LMStudioBridge"


def _model(model_id: str, *, input_limit: int = 8000) -> ModelEntry:
    return ModelEntry(
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
        input_token_limit=input_limit,
        output_token_limit=2000,
        tool_call_limit=4,
        pricing=ModelPricing(input_price_per_1k=1.0, output_price_per_1k=2.0),
        routing_priority=10,
        fallback_model=None,
        license="apache-2.0",
        status=ModelStatus.AVAILABLE,
    )


def _envelope(content: str = "hello", *, prompt=11, completion=7) -> dict:
    return {
        "model": "model-a",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


@pytest.fixture
def km(tmp_path) -> Keymaster:
    return Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)


@pytest.fixture
def gk(km, tmp_path) -> Gatekeeper:
    return Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper", log_to_file=False)


@pytest.fixture
def config(tmp_path) -> T1APIConfig:
    return T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough",
        db_path=str(tmp_path / "t1.sqlite3"),
        module_storage_dir=str(tmp_path / "modules"),
        hyperlink_files_dir=str(tmp_path / "files"),
        lmstudio_url="http://lmstudio.test:1234",
        default_plan="free",
    )


@pytest.fixture
def app(km, gk, config):
    registry = ModelRegistry()
    registry.register(_model("model-a"))
    registry.register(_model("model-b"))
    return create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app, client=("127.0.0.1", 5000))


@pytest.fixture
def key(km) -> str:
    return km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


class TestTheVersionAdvertisesIt:
    def test_this_build_is_at_least_the_inference_release(self):
        assert T1_VERSION >= INFERENCE_VERSION

    def test_a_client_can_check_by_comparison(self):
        """Rather than probing the endpoint and catching the 404."""
        assert INFERENCE_VERSION.long == "1.0.2026.9.2.0"


class TestItIsGoverned:
    def test_an_unregistered_model_is_refused(self, client, key):
        response = client.post(
            "/inference/chat",
            json={"model": "not-registered", "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth(key),
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "MODEL_NOT_SUPPORTED"

    def test_it_does_not_reach_the_backend_for_an_unknown_model(self, client, key):
        """Refused before dispatch, not after — the point of the gate."""
        with mock.patch(BRIDGE) as bridge:
            client.post(
                "/inference/chat",
                json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth(key),
            )
        bridge.assert_not_called()

    def test_empty_messages_are_refused(self, client, key):
        response = client.post(
            "/inference/chat",
            json={"model": "model-a", "messages": []},
            headers=_auth(key),
        )
        assert response.status_code == 422

    def test_an_empty_prompt_is_refused(self, client, key):
        response = client.post(
            "/inference/completions",
            json={"model": "model-a", "prompt": "   "},
            headers=_auth(key),
        )
        assert response.status_code == 422

    def test_it_needs_a_key(self, client):
        response = client.post(
            "/inference/chat",
            json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code in (401, 403)


class TestAChatThatWorks:
    def test_it_returns_the_content_and_what_ran(self, client, key):
        with mock.patch(BRIDGE) as factory:
            factory.return_value.chat.return_value = _envelope("the answer")
            factory.return_value.base_url = "http://lmstudio.test:1234"
            response = client.post(
                "/inference/chat",
                json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth(key),
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["content"] == "the answer"
        assert body["model"] == "model-a"
        assert body["requested_model"] == "model-a"
        assert body["substituted"] is False
        assert body["backend"] == "http://lmstudio.test:1234"

    def test_it_prices_what_the_backend_reported(self, client, key):
        """Not the estimate. The estimate sizes; the backend bills."""
        with mock.patch(BRIDGE) as factory:
            factory.return_value.chat.return_value = _envelope(prompt=1000, completion=500)
            factory.return_value.base_url = "http://lmstudio.test:1234"
            body = client.post(
                "/inference/chat",
                json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth(key),
            ).json()
        assert body["input_tokens"] == 1000
        assert body["output_tokens"] == 500
        # 1.0/1k in, 2.0/1k out
        assert body["cost"] == pytest.approx(1.0 * 1 + 2.0 * 0.5)

    def test_a_backend_that_reports_no_usage_is_still_metered(self, client, key):
        """Recording zero would make the quota unenforceable against it."""
        envelope = _envelope("some words here")
        envelope["usage"] = {}
        with mock.patch(BRIDGE) as factory:
            factory.return_value.chat.return_value = envelope
            factory.return_value.base_url = "http://x:1234"
            body = client.post(
                "/inference/chat",
                json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth(key),
            ).json()
        assert body["input_tokens"] > 0
        assert body["output_tokens"] > 0

    def test_usage_lands_against_the_key(self, client, key):
        with mock.patch(BRIDGE) as factory:
            factory.return_value.chat.return_value = _envelope(prompt=40, completion=10)
            factory.return_value.base_url = "http://x:1234"
            client.post(
                "/inference/chat",
                json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth(key),
            )
        usage = client.get("/usage/current", headers=_auth(key))
        assert usage.status_code == 200, usage.text
        assert "model-a" in usage.text

    def test_a_completion_is_carried_as_one_user_turn(self, client, key):
        with mock.patch(BRIDGE) as factory:
            factory.return_value.chat.return_value = _envelope("done")
            factory.return_value.base_url = "http://x:1234"
            response = client.post(
                "/inference/completions",
                json={"model": "model-a", "prompt": "write a haiku"},
                headers=_auth(key),
            )
            sent = factory.return_value.chat.call_args[0][0]
        assert response.status_code == 200, response.text
        assert sent == [{"role": "user", "content": "write a haiku"}]


class TestFallbackIsOptIn:
    def test_it_defaults_to_off(self, client, key):
        """A silent substitution of an exhausted model is the one thing
        the spec forbids, so asking for it has to be explicit."""
        from hypernix.t1api.schemas import InferenceChatRequest

        assert InferenceChatRequest.model_fields["allow_fallback"].default is False

    def test_the_response_says_which_model_actually_ran(self, client, key):
        """`substituted` exists so a caller can tell, rather than
        assuming it got what it asked for."""
        with mock.patch(BRIDGE) as factory:
            factory.return_value.chat.return_value = _envelope()
            factory.return_value.base_url = "http://x:1234"
            body = client.post(
                "/inference/chat",
                json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth(key),
            ).json()
        assert "substituted" in body
        assert "requested_model" in body


class TestTheBackendIsReported:
    def test_a_missing_backend_says_what_to_configure(self, km, gk, tmp_path):
        config = T1APIConfig(
            token_secret="test-secret-value-that-is-long-enough",
            db_path=str(tmp_path / "t1.sqlite3"),
            module_storage_dir=str(tmp_path / "modules"),
            hyperlink_files_dir=str(tmp_path / "files"),
            lmstudio_enabled=False,
            default_plan="free",
        )
        registry = ModelRegistry()
        registry.register(_model("model-a"))
        app = create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry)
        client = TestClient(app, client=("127.0.0.1", 5000))
        key = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key

        response = client.post(
            "/inference/chat",
            json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth(key),
        )
        assert response.status_code == 501
        assert "T1_LMSTUDIO_URL" in response.json()["error"]["message"]

    def test_backends_distinguishes_down_from_unconfigured(self, client, key):
        """Two different answers, not one error string to parse."""
        with mock.patch(BRIDGE) as factory:
            factory.return_value.list_models.side_effect = LMStudioError("refused")
            factory.return_value.base_url = "http://lmstudio.test:1234"
            body = client.get("/inference/backends", headers=_auth(key)).json()
        entry = body["backends"][0]
        assert entry["name"] == "lmstudio"
        assert entry["reachable"] is False
        assert "refused" in entry["detail"]
        assert body["default"] == ""

    def test_backends_reports_a_reachable_one(self, client, key):
        with mock.patch(BRIDGE) as factory:
            factory.return_value.list_models.return_value = []
            factory.return_value.base_url = "http://lmstudio.test:1234"
            body = client.get("/inference/backends", headers=_auth(key)).json()
        assert body["backends"][0]["reachable"] is True
        assert body["default"] == "lmstudio"

    def test_an_unreachable_backend_is_a_503_not_a_500(self, client, key):
        with mock.patch(BRIDGE) as factory:
            factory.return_value.chat.side_effect = LMStudioError("timed out")
            factory.return_value.base_url = "http://x:1234"
            response = client.post(
                "/inference/chat",
                json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth(key),
            )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"


class TestTokenCounting:
    def test_it_sizes_without_running_anything(self, client, key):
        with mock.patch(BRIDGE) as bridge:
            response = client.post(
                "/inference/tokens",
                json={"model": "model-a", "text": "a" * 400},
                headers=_auth(key),
            )
        bridge.assert_not_called()
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["tokens"] == 100
        assert body["estimated_input_cost"] == pytest.approx(0.1)

    def test_it_says_the_count_is_an_estimate(self, client, key):
        """The tokenizer that matters is in the backend. Claiming
        exactness here produces a number people rely on."""
        body = client.post(
            "/inference/tokens",
            json={"model": "model-a", "text": "hello"},
            headers=_auth(key),
        ).json()
        assert "heuristic" in body["method"]

    def test_both_shapes_are_accepted(self, client, key):
        for payload in (
            {"model": "model-a", "text": "hello there"},
            {"model": "model-a", "messages": [{"role": "user", "content": "hello there"}]},
        ):
            response = client.post("/inference/tokens", json=payload, headers=_auth(key))
            assert response.status_code == 200, response.text

    def test_neither_or_both_is_refused(self, client, key):
        for payload in (
            {"model": "model-a"},
            {"model": "model-a", "text": "x", "messages": [{"role": "user", "content": "y"}]},
        ):
            response = client.post("/inference/tokens", json=payload, headers=_auth(key))
            assert response.status_code == 422, payload

    def test_it_reports_what_is_left(self, client, key):
        body = client.post(
            "/inference/tokens",
            json={"model": "model-a", "text": "hi"},
            headers=_auth(key),
        ).json()
        assert body["remaining_input_tokens"] == 8000

    def test_an_unregistered_model_cannot_be_priced(self, client, key):
        response = client.post(
            "/inference/tokens",
            json={"model": "invented", "text": "hi"},
            headers=_auth(key),
        )
        assert response.status_code == 404


class TestEmbeddings:
    def test_it_returns_vectors_and_meters_them(self, client, key):
        with mock.patch(BRIDGE) as factory:
            factory.return_value.embeddings.return_value = {
                "model": "model-a",
                "data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}],
                "usage": {"prompt_tokens": 12},
            }
            factory.return_value.base_url = "http://x:1234"
            response = client.post(
                "/inference/embeddings",
                json={"model": "model-a", "input": ["one", "two"]},
                headers=_auth(key),
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["dimensions"] == 3
        assert len(body["embeddings"]) == 2
        assert body["input_tokens"] == 12

    def test_empty_input_is_refused(self, client, key):
        response = client.post(
            "/inference/embeddings",
            json={"model": "model-a", "input": []},
            headers=_auth(key),
        )
        assert response.status_code == 422

    def test_an_unregistered_model_is_refused(self, client, key):
        response = client.post(
            "/inference/embeddings",
            json={"model": "nope", "input": ["x"]},
            headers=_auth(key),
        )
        assert response.status_code == 404


class TestStreaming:
    def test_every_gate_runs_before_the_first_byte(self, client, key):
        """A 429 cannot be sent once the response has begun."""
        response = client.post(
            "/inference/chat/stream",
            json={"model": "not-registered", "messages": [{"role": "user", "content": "hi"}]},
            headers=_auth(key),
        )
        assert response.status_code == 404

    def test_it_streams_and_terminates(self, client, key):
        with mock.patch(BRIDGE) as factory:
            factory.return_value.chat_stream.return_value = iter([
                {"choices": [{"delta": {"content": "he"}}]},
                {"choices": [{"delta": {"content": "llo"}}]},
            ])
            factory.return_value.base_url = "http://x:1234"
            response = client.post(
                "/inference/chat/stream",
                json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth(key),
            )
        assert response.status_code == 200
        assert "data: [DONE]" in response.text
        assert "hypernix inference open" in response.text

    def test_a_mid_stream_failure_is_delivered_as_a_frame(self, client, key):
        """HTTP status is long gone; dropping the connection silently is
        indistinguishable from the network failing."""
        def _boom():
            yield {"choices": [{"delta": {"content": "partial"}}]}
            raise LMStudioError("backend died")

        with mock.patch(BRIDGE) as factory:
            factory.return_value.chat_stream.return_value = _boom()
            factory.return_value.base_url = "http://x:1234"
            response = client.post(
                "/inference/chat/stream",
                json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth(key),
            )
        assert response.status_code == 200
        assert "error" in response.text
        assert "data: [DONE]" in response.text

    def test_tokens_produced_before_a_failure_are_still_metered(self, client, key):
        """Not metering them under-counts exactly the callers whose
        requests fail most."""
        def _boom():
            yield {"choices": [{"delta": {"content": "partial output"}}]}
            raise LMStudioError("died")

        with mock.patch(BRIDGE) as factory:
            factory.return_value.chat_stream.return_value = _boom()
            factory.return_value.base_url = "http://x:1234"
            client.post(
                "/inference/chat/stream",
                json={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth(key),
            )
        usage = client.get("/usage/current", headers=_auth(key))
        assert usage.status_code == 200, usage.text
        assert "model-a" in usage.text


class TestItDoesNotDuplicateTheBridge:
    def test_the_bridge_is_still_there(self, client, key):
        """This adds a governed path; it does not remove the raw one."""
        with mock.patch("hypernix.t1api.routers.bridge.LMStudioBridge") as factory:
            factory.return_value.chat.return_value = _envelope()
            factory.return_value.base_url = "http://x:1234"
            response = client.post(
                "/bridge/lmstudio/chat",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers=_auth(key),
            )
        assert response.status_code == 200, response.text

    def test_the_bridge_still_does_not_require_registration(self, client, key):
        """Recording the difference, not endorsing it: the bridge is a
        window onto another server and takes whatever model that server
        knows. /inference is the path with the rules."""
        with mock.patch("hypernix.t1api.routers.bridge.LMStudioBridge") as factory:
            factory.return_value.chat.return_value = _envelope()
            factory.return_value.base_url = "http://x:1234"
            response = client.post(
                "/bridge/lmstudio/chat",
                json={"model": "never-registered", "messages": [{"role": "user", "content": "hi"}]},
                headers=_auth(key),
            )
        assert response.status_code == 200
