"""HTTP tests for the T1 v1.0.26.8.0.1 endpoints.

Requires ``pip install hypernix[t1api-test]``.

Three things are pinned here that cannot be seen from the core tests:

* **The pairing endpoint is the only unauthenticated one**, and it is
  still bounded — a code is single-use, short-lived, and attempt-capped.
* **A device token is not an admin credential.** Whatever key paired a
  phone, the phone cannot mint pairing codes, list devices, or unpair
  another device. A stolen phone must not be able to enrol a second one.
* **Ownership is by the pairing key, not the device.** An operator's
  phone and their desktop client see one set of sessions; another
  operator's are invisible.
"""
from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest import mock

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
from hypernix.t1api.version import T1_VERSION  # noqa: E402


def _model(model_id: str) -> ModelEntry:
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
        input_token_limit=8000,
        output_token_limit=2000,
        tool_call_limit=4,
        pricing=ModelPricing(input_price_per_1k=1.0, output_price_per_1k=2.0),
        routing_priority=10,
        fallback_model=None,
        license="apache-2.0",
        status=ModelStatus.AVAILABLE,
    )


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
    return create_app(config=config, keymaster=km, gatekeeper=gk, registry=registry)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app, client=("127.0.0.1", 5000))


@pytest.fixture
def admin_key(km) -> str:
    return km.create(
        key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}
    ).key


@pytest.fixture
def user_key(km) -> str:
    return km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture
def device(client, admin_key) -> tuple[str, str]:
    """A paired device: ``(device_id, device_token)``."""
    minted = client.post("/hyperlink/pair", json={"label": "test phone"}, headers=_auth(admin_key))
    assert minted.status_code == 200, minted.text
    code = minted.json()["code"]
    redeemed = client.post(
        "/hyperlink/pair/redeem",
        json={"code": code, "device_name": "Test iPhone", "app_version": "1.0"},
    )
    assert redeemed.status_code == 200, redeemed.text
    body = redeemed.json()
    return body["device_id"], body["device_token"]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_status_reports_both_version_spellings(self, client):
        body = client.get("/status").json()
        assert body["t1_api_version"] == T1_VERSION.short
        assert body["t1_api_version_long"] == T1_VERSION.long
        assert body["t1_version"]["year"] == 2026
        assert body["t1_version"]["generation"] == "1.1"

    def test_the_beta_field_still_exists_for_beta_3_clients(self, client):
        # Renaming it would break every Beta 3 client for a cosmetic win.
        assert client.get("/status").json()["beta"] == "t1-1.0"

    def test_status_reports_the_new_subsystems(self, client):
        body = client.get("/status").json()
        assert body["lmstudio_bridge_enabled"] is True
        assert body["lmstudio_configured"] is True
        assert body["hyperlink_enabled"] is True

    def test_secrets_are_reported_as_set_or_unset_only(self, client):
        secrets = client.get("/status").json()["secrets_configured"]
        assert secrets["hf_token"] is False
        assert all(isinstance(v, bool) for v in secrets.values())


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


class TestPairingEndpoints:
    def test_minting_a_code_requires_an_admin_key(self, client, user_key):
        response = client.post("/hyperlink/pair", json={}, headers=_auth(user_key))
        assert response.status_code == 403

    def test_minting_a_code_requires_any_credential(self, client):
        assert client.post("/hyperlink/pair", json={}).status_code == 401

    def test_a_minted_code_carries_the_addresses_to_type_in(self, client, admin_key):
        body = client.post("/hyperlink/pair", json={"label": "phone"}, headers=_auth(admin_key)).json()
        assert len(body["code"]) == 6
        assert body["seconds_remaining"] > 0
        assert body["endpoints"], "the app needs somewhere to connect to"
        assert body["qr_payload"]["kind"] == "hypernix.hyperlink.pairing"
        assert body["qr_payload"]["code"] == body["code"]

    def test_redeeming_is_the_one_unauthenticated_endpoint(self, client, admin_key):
        code = client.post("/hyperlink/pair", json={}, headers=_auth(admin_key)).json()["code"]
        response = client.post(
            "/hyperlink/pair/redeem", json={"code": code, "device_name": "iPhone"}
        )
        assert response.status_code == 200
        assert response.json()["device_token"].startswith("HLNK_")

    def test_a_code_cannot_be_redeemed_twice(self, client, admin_key):
        code = client.post("/hyperlink/pair", json={}, headers=_auth(admin_key)).json()["code"]
        client.post("/hyperlink/pair/redeem", json={"code": code, "device_name": "first"})
        second = client.post("/hyperlink/pair/redeem", json={"code": code, "device_name": "second"})
        assert second.status_code == 409

    def test_an_unknown_code_is_a_404(self, client):
        response = client.post(
            "/hyperlink/pair/redeem", json={"code": "ZZZZZZ", "device_name": "phone"}
        )
        assert response.status_code == 404

    def test_redeeming_without_a_device_name_is_refused(self, client, admin_key):
        code = client.post("/hyperlink/pair", json={}, headers=_auth(admin_key)).json()["code"]
        response = client.post("/hyperlink/pair/redeem", json={"code": code, "device_name": "  "})
        assert response.status_code == 422

    def test_guessing_is_capped(self, client, app, admin_key):
        # Five wrong attempts burn the code, so a six-character code
        # with a ten-minute life cannot be walked through.
        code = client.post("/hyperlink/pair", json={}, headers=_auth(admin_key)).json()["code"]
        registry = app.state.t1_device_registry
        for _ in range(5):
            registry.note_failed_attempt(code)
        response = client.post(
            "/hyperlink/pair/redeem", json={"code": code, "device_name": "phone"}
        )
        assert response.status_code == 422
        assert "Too many failed attempts" in response.json()["error"]["message"]
        # And the code is gone rather than merely refused.
        assert (
            client.post(
                "/hyperlink/pair/redeem", json={"code": code, "device_name": "phone"}
            ).status_code
            == 404
        )

    def test_a_wrong_guess_counts_against_the_code_it_named(self, client, app, admin_key):
        code = client.post("/hyperlink/pair", json={}, headers=_auth(admin_key)).json()["code"]
        client.post("/hyperlink/pair/redeem", json={"code": "ZZZZZZ", "device_name": "guess"})
        # A guess at a code that does not exist must not consume the
        # real code's budget.
        listed = app.state.t1_device_registry.list_codes()
        assert [c.attempts for c in listed if c.code == code] == [0]

    def test_an_admin_can_cancel_an_unredeemed_code(self, client, admin_key):
        code = client.post("/hyperlink/pair", json={}, headers=_auth(admin_key)).json()["code"]
        assert client.delete(f"/hyperlink/pair/{code}", headers=_auth(admin_key)).json()["ok"]
        assert (
            client.post("/hyperlink/pair/redeem", json={"code": code, "device_name": "x"}).status_code
            == 404
        )

    def test_pairing_is_recorded_in_the_audit_log(self, client, admin_key, device):
        entries = client.get("/audit?limit=50", headers=_auth(admin_key)).json()["events"]
        actions = {entry["action"] for entry in entries}
        assert "hyperlink.pair.create" in actions
        assert "hyperlink.pair.redeem" in actions


class TestDeviceAuthority:
    def test_a_device_token_authenticates(self, client, device):
        _, token = device
        response = client.get("/hyperlink/devices/me", headers=_auth(token))
        assert response.status_code == 200
        assert response.json()["device"]["name"] == "Test iPhone"

    def test_a_device_cannot_mint_a_pairing_code(self, client, device):
        # A stolen phone must not be able to enrol another one.
        _, token = device
        assert client.post("/hyperlink/pair", json={}, headers=_auth(token)).status_code == 403

    def test_a_device_cannot_list_the_other_devices(self, client, device):
        _, token = device
        assert client.get("/hyperlink/devices", headers=_auth(token)).status_code == 403

    def test_a_device_can_unpair_itself(self, client, device):
        # This is the app's "sign out"; requiring an admin would leave a
        # wiped phone's token valid until someone noticed.
        device_id, token = device
        assert client.delete(f"/hyperlink/devices/{device_id}", headers=_auth(token)).status_code == 200
        assert client.get("/hyperlink/devices/me", headers=_auth(token)).status_code == 401

    def test_a_device_cannot_unpair_another_device(self, client, admin_key, device):
        _, token = device
        other_code = client.post("/hyperlink/pair", json={}, headers=_auth(admin_key)).json()["code"]
        other = client.post(
            "/hyperlink/pair/redeem", json={"code": other_code, "device_name": "iPad"}
        ).json()
        response = client.delete(f"/hyperlink/devices/{other['device_id']}", headers=_auth(token))
        assert response.status_code == 403

    def test_an_admin_sees_every_device(self, client, admin_key, device):
        body = client.get("/hyperlink/devices", headers=_auth(admin_key)).json()
        assert body["count"] == 1
        assert body["devices"][0]["name"] == "Test iPhone"

    def test_a_revoked_token_stops_working_immediately(self, client, admin_key, device):
        device_id, token = device
        client.delete(f"/hyperlink/devices/{device_id}", headers=_auth(admin_key))
        response = client.get("/hyperlink/sessions", headers=_auth(token))
        assert response.status_code == 401
        assert "pair it again" in response.json()["error"]["message"].lower()

    def test_a_t1_key_is_told_to_use_the_right_endpoint(self, client, admin_key):
        response = client.get("/hyperlink/devices/me", headers=_auth(admin_key))
        assert response.status_code == 422
        assert "whoami" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Endpoints advertisement
# ---------------------------------------------------------------------------


class TestEndpointsAdvertisement:
    def test_it_needs_a_credential(self, client):
        # A list of a machine's internal addresses is reconnaissance if
        # handed to anyone who asks.
        assert client.get("/hyperlink/endpoints").status_code == 401

    def test_it_lists_addresses_best_first(self, client, device):
        _, token = device
        body = client.get("/hyperlink/endpoints", headers=_auth(token)).json()
        priorities = [e["priority"] for e in body["endpoints"]]
        assert priorities == sorted(priorities)
        assert body["t1_version"] == T1_VERSION.short


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class TestSessionEndpoints:
    def test_a_session_round_trips(self, client, device):
        _, token = device
        created = client.post(
            "/hyperlink/sessions", json={"title": "Notes", "model_id": "qwen"}, headers=_auth(token)
        )
        assert created.status_code == 200
        session_id = created.json()["session"]["session_id"]
        fetched = client.get(f"/hyperlink/sessions/{session_id}", headers=_auth(token))
        assert fetched.json()["session"]["title"] == "Notes"

    def test_a_phone_and_the_key_that_paired_it_share_sessions(self, client, admin_key, device):
        # The whole point of server-side history: start at the desk,
        # carry on from the phone.
        _, token = device
        client.post("/hyperlink/sessions", json={"title": "From the phone"}, headers=_auth(token))
        listed = client.get("/hyperlink/sessions", headers=_auth(admin_key)).json()
        assert [s["title"] for s in listed["sessions"]] == ["From the phone"]

    def test_another_key_cannot_see_them(self, client, user_key, device):
        _, token = device
        client.post("/hyperlink/sessions", json={"title": "Private"}, headers=_auth(token))
        listed = client.get("/hyperlink/sessions", headers=_auth(user_key)).json()
        assert listed["count"] == 0

    def test_another_key_cannot_fetch_one_by_id(self, client, user_key, device):
        _, token = device
        session_id = client.post(
            "/hyperlink/sessions", json={}, headers=_auth(token)
        ).json()["session"]["session_id"]
        response = client.get(f"/hyperlink/sessions/{session_id}", headers=_auth(user_key))
        assert response.status_code == 404

    def test_a_session_can_be_renamed_and_archived(self, client, device):
        _, token = device
        session_id = client.post("/hyperlink/sessions", json={}, headers=_auth(token)).json()[
            "session"
        ]["session_id"]
        client.patch(
            f"/hyperlink/sessions/{session_id}",
            json={"title": "Renamed", "archived": True},
            headers=_auth(token),
        )
        listed = client.get("/hyperlink/sessions", headers=_auth(token)).json()
        assert listed["count"] == 0
        listed = client.get("/hyperlink/sessions?include_archived=true", headers=_auth(token)).json()
        assert listed["sessions"][0]["title"] == "Renamed"

    def test_deleting_a_session_works_and_is_final(self, client, device):
        _, token = device
        session_id = client.post("/hyperlink/sessions", json={}, headers=_auth(token)).json()[
            "session"
        ]["session_id"]
        assert client.delete(f"/hyperlink/sessions/{session_id}", headers=_auth(token)).json()["ok"]
        assert client.get(f"/hyperlink/sessions/{session_id}", headers=_auth(token)).status_code == 404

    def test_a_system_prompt_is_stored_as_the_first_message(self, client, device):
        _, token = device
        session_id = client.post(
            "/hyperlink/sessions", json={"system_prompt": "Be terse."}, headers=_auth(token)
        ).json()["session"]["session_id"]
        messages = client.get(
            f"/hyperlink/sessions/{session_id}/messages", headers=_auth(token)
        ).json()["messages"]
        assert [m["role"] for m in messages] == ["system"]


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------


class TestAttachmentEndpoints:
    def _session(self, client, token) -> str:
        return client.post("/hyperlink/sessions", json={}, headers=_auth(token)).json()["session"][
            "session_id"
        ]

    def test_an_image_uploads_and_comes_back(self, client, device):
        _, token = device
        session_id = self._session(client, token)
        png = b"\x89PNG\r\n\x1a\n" + b"x" * 64
        uploaded = client.post(
            "/hyperlink/files",
            files={"file": ("shot.png", png, "image/png")},
            data={"session_id": session_id},
            headers=_auth(token),
        )
        assert uploaded.status_code == 200, uploaded.text
        record = uploaded.json()["file"]
        assert record["is_image"] and record["content_type"] == "image/png"

        fetched = client.get(f"/hyperlink/files/{record['file_id']}", headers=_auth(token))
        assert fetched.content == png

    def test_downloads_are_never_rendered_inline(self, client, device):
        # This server can be reached from a WKWebView; a stored file that
        # renders as HTML in the app's own origin is stored XSS.
        _, token = device
        uploaded = client.post(
            "/hyperlink/files",
            files={"file": ("evil.html", b"<script>alert(1)</script>", "text/html")},
            data={"session_id": self._session(client, token)},
            headers=_auth(token),
        ).json()["file"]
        response = client.get(f"/hyperlink/files/{uploaded['file_id']}", headers=_auth(token))
        assert response.headers["content-disposition"].startswith("attachment")
        assert response.headers["x-content-type-options"] == "nosniff"

    def test_a_source_file_is_recognised_as_text(self, client, device):
        _, token = device
        uploaded = client.post(
            "/hyperlink/files",
            files={"file": ("main.swift", b"let x = 1\n", "application/octet-stream")},
            data={"session_id": self._session(client, token)},
            headers=_auth(token),
        ).json()["file"]
        assert uploaded["is_text"]
        assert uploaded["content_type"] == "text/x-swift"

    def test_the_upload_limit_is_enforced_on_the_bytes_read(self, client, app, device):
        # Not on Content-Length, which the client chooses.
        _, token = device
        app.state.t1_config.hyperlink_max_upload_bytes = 32
        response = client.post(
            "/hyperlink/files",
            files={"file": ("big.bin", b"x" * 1000, "application/octet-stream")},
            data={"session_id": self._session(client, token)},
            headers=_auth(token),
        )
        assert response.status_code == 413

    def test_another_owners_file_is_a_404(self, client, user_key, device):
        _, token = device
        uploaded = client.post(
            "/hyperlink/files",
            files={"file": ("a.txt", b"secret", "text/plain")},
            data={"session_id": self._session(client, token)},
            headers=_auth(token),
        ).json()["file"]
        assert client.get(f"/hyperlink/files/{uploaded['file_id']}", headers=_auth(user_key)).status_code == 404


# ---------------------------------------------------------------------------
# The LM Studio bridge
# ---------------------------------------------------------------------------


def _lmstudio_stub(payloads: dict[str, object]):
    """Patch the bridge's socket check and HTTP so no server is needed."""

    def opener(request, timeout=None):  # noqa: ARG001
        path = "/" + request.full_url.split("://", 1)[1].split("/", 1)[1].split("?")[0]
        if path not in payloads:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, BytesIO(b"{}"))
        body = json.dumps(payloads[path]).encode()

        class _Resp:
            headers: dict[str, str] = {}

            def read(self):
                return body

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _Resp()

    return mock.patch.multiple(
        "hypernix.bridge.lmstudio.LMStudioBridge",
        _check_connect=lambda self: None,
    ), mock.patch("urllib.request.urlopen", opener)


class TestBridgeEndpoints:
    def test_models_are_listed_with_the_loaded_ones_marked(self, client, admin_key):
        connect, http = _lmstudio_stub(
            {
                "/api/v0/models": {
                    "data": [
                        {"id": "hot", "state": "loaded", "type": "llm", "quantization": "Q4_K_M"},
                        {"id": "cold", "state": "not-loaded", "type": "llm"},
                    ]
                }
            }
        )
        with connect, http:
            body = client.get("/bridge/lmstudio/models", headers=_auth(admin_key)).json()
        assert body["count"] == 2
        assert body["loaded_count"] == 1
        assert body["base_url"] == "http://lmstudio.test:1234"

    def test_a_chat_reply_is_lifted_out_of_the_openai_envelope(self, client, admin_key):
        connect, http = _lmstudio_stub(
            {
                "/api/v0/models": {"data": [{"id": "hot", "state": "loaded"}]},
                "/v1/chat/completions": {
                    "model": "hot",
                    "choices": [
                        {"message": {"role": "assistant", "content": "42"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 1},
                },
            }
        )
        with connect, http:
            body = client.post(
                "/bridge/lmstudio/chat",
                json={"messages": [{"role": "user", "content": "answer?"}]},
                headers=_auth(admin_key),
            ).json()
        assert body["content"] == "42"
        assert body["output_tokens"] == 1
        assert body["raw"]["choices"], "the full envelope is still available"

    def test_an_unreachable_lm_studio_is_a_503_not_a_500(self, client, admin_key):
        # The app shows "your PC isn't answering" for a 503 and a
        # spinner forever for a 500.
        response = client.get("/bridge/lmstudio/models", headers=_auth(admin_key))
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "MODEL_UNAVAILABLE"

    def test_a_per_request_address_override_is_admin_only(self, client, user_key):
        response = client.get(
            "/bridge/lmstudio/models?base_url=http://attacker.test:1234", headers=_auth(user_key)
        )
        assert response.status_code == 403

    def test_discovery_is_off_unless_explicitly_enabled(self, client, admin_key):
        response = client.get("/bridge/lmstudio/status?discover=true", headers=_auth(admin_key))
        assert response.status_code == 501

    def test_the_bridge_can_be_switched_off_entirely(self, client, app, admin_key):
        app.state.t1_config.lmstudio_enabled = False
        assert client.get("/bridge/lmstudio/models", headers=_auth(admin_key)).status_code == 501


class TestChatTurn:
    def test_a_turn_persists_both_messages(self, client, device):
        _, token = device
        session_id = client.post("/hyperlink/sessions", json={}, headers=_auth(token)).json()[
            "session"
        ]["session_id"]
        connect, http = _lmstudio_stub(
            {
                "/api/v0/models": {"data": [{"id": "hot", "state": "loaded"}]},
                "/v1/chat/completions": {
                    "model": "hot",
                    "choices": [{"message": {"content": "hello back"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
            }
        )
        with connect, http:
            turn = client.post(
                f"/hyperlink/sessions/{session_id}/chat",
                json={"content": "hello"},
                headers=_auth(token),
            )
        assert turn.status_code == 200, turn.text
        body = turn.json()
        assert body["assistant_message"]["content"] == "hello back"
        assert body["assistant_message"]["output_tokens"] == 2

        messages = client.get(
            f"/hyperlink/sessions/{session_id}/messages", headers=_auth(token)
        ).json()["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant"]

    def test_the_users_message_survives_a_backend_failure(self, client, device):
        # It is persisted before inference runs, so a retry does not mean
        # retyping it.
        _, token = device
        session_id = client.post("/hyperlink/sessions", json={}, headers=_auth(token)).json()[
            "session"
        ]["session_id"]
        response = client.post(
            f"/hyperlink/sessions/{session_id}/chat",
            json={"content": "will fail"},
            headers=_auth(token),
        )
        assert response.status_code == 503
        messages = client.get(
            f"/hyperlink/sessions/{session_id}/messages", headers=_auth(token)
        ).json()["messages"]
        assert [m["content"] for m in messages] == ["will fail"]

    def test_the_session_is_titled_from_the_first_message(self, client, device):
        _, token = device
        session_id = client.post("/hyperlink/sessions", json={}, headers=_auth(token)).json()[
            "session"
        ]["session_id"]
        connect, http = _lmstudio_stub(
            {
                "/api/v0/models": {"data": [{"id": "hot", "state": "loaded"}]},
                "/v1/chat/completions": {
                    "model": "hot",
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                },
            }
        )
        with connect, http:
            client.post(
                f"/hyperlink/sessions/{session_id}/chat",
                json={"content": "What is a GGUF file?"},
                headers=_auth(token),
            )
        session = client.get(f"/hyperlink/sessions/{session_id}", headers=_auth(token)).json()
        assert session["session"]["title"] == "What is a GGUF file?"

    def test_an_empty_turn_is_refused(self, client, device):
        _, token = device
        session_id = client.post("/hyperlink/sessions", json={}, headers=_auth(token)).json()[
            "session"
        ]["session_id"]
        response = client.post(
            f"/hyperlink/sessions/{session_id}/chat", json={"content": "   "}, headers=_auth(token)
        )
        assert response.status_code == 422

    def test_an_image_attachment_reaches_the_model_as_a_vision_part(self, client, device):
        _, token = device
        session_id = client.post("/hyperlink/sessions", json={}, headers=_auth(token)).json()[
            "session"
        ]["session_id"]
        png = b"\x89PNG\r\n\x1a\n" + b"q" * 32
        file_id = client.post(
            "/hyperlink/files",
            files={"file": ("a.png", png, "image/png")},
            data={"session_id": session_id},
            headers=_auth(token),
        ).json()["file"]["file_id"]

        captured: dict[str, object] = {}

        def _chat(self, messages, **kwargs):  # noqa: ANN001, ARG001
            # The first call is the reply; a new chat is then titled by a
            # second, shorter one (0.72.5.post16).
            captured.setdefault("messages", messages)
            return {"model": "hot", "choices": [{"message": {"content": "an image"}}]}

        with mock.patch("hypernix.bridge.lmstudio.LMStudioBridge.chat", _chat):
            client.post(
                f"/hyperlink/sessions/{session_id}/chat",
                json={"content": "what is this?", "attachment_ids": [file_id]},
                headers=_auth(token),
            )
        parts = captured["messages"][-1]["content"]
        assert isinstance(parts, list)
        kinds = {p["type"] for p in parts}
        assert kinds == {"text", "image_url"}
        assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_a_code_attachment_reaches_the_model_as_a_fenced_block(self, client, device):
        _, token = device
        session_id = client.post("/hyperlink/sessions", json={}, headers=_auth(token)).json()[
            "session"
        ]["session_id"]
        file_id = client.post(
            "/hyperlink/files",
            files={"file": ("main.py", b"print(1)\n", "text/x-python")},
            data={"session_id": session_id},
            headers=_auth(token),
        ).json()["file"]["file_id"]

        captured: dict[str, object] = {}

        def _chat(self, messages, **kwargs):  # noqa: ANN001, ARG001
            # The first call is the reply; a new chat is then titled by a
            # second, shorter one (0.72.5.post16).
            captured.setdefault("messages", messages)
            return {"model": "hot", "choices": [{"message": {"content": "it prints 1"}}]}

        with mock.patch("hypernix.bridge.lmstudio.LMStudioBridge.chat", _chat):
            client.post(
                f"/hyperlink/sessions/{session_id}/chat",
                json={"content": "explain", "attachment_ids": [file_id]},
                headers=_auth(token),
            )
        text = captured["messages"][-1]["content"]
        assert "```python" in text
        assert "print(1)" in text
        assert "main.py" in text


# ---------------------------------------------------------------------------
# Hugging Face resolution
# ---------------------------------------------------------------------------


class TestResolveEndpoint:
    def _api(self, siblings, **extra):
        info = {"siblings": [{"rfilename": n, "size": s} for n, s in siblings]}
        info.update(extra)
        return mock.patch("hypernix.hyperlink.hfmerge.fetch_repo_info", return_value=info)

    def test_a_page_and_a_file_link_merge_into_one_plan(self, client, device):
        _, token = device
        with self._api([("m-Q4_K_M.gguf", 4_000_000)]):
            body = client.post(
                "/hyperlink/models/resolve",
                json={
                    "page_url": "https://huggingface.co/o/r",
                    "file_url": "https://huggingface.co/o/r/resolve/main/m-Q4_K_M.gguf?download=true",
                },
                headers=_auth(token),
            ).json()
        assert body["repo_id"] == "o/r"
        assert body["quantization"] == "Q4_K_M"
        assert body["file_count"] == 1
        assert body["total_size_human"]

    def test_a_repository_conflict_is_a_409(self, client, device):
        _, token = device
        response = client.post(
            "/hyperlink/models/resolve",
            json={
                "page_url": "https://huggingface.co/a/b",
                "file_url": "https://huggingface.co/c/d/resolve/main/m.gguf",
            },
            headers=_auth(token),
        )
        assert response.status_code == 409
        assert "a/b" in response.json()["error"]["message"]

    def test_prefer_file_resolves_the_conflict(self, client, device):
        _, token = device
        with self._api([("m.gguf", 10)]):
            body = client.post(
                "/hyperlink/models/resolve",
                json={
                    "page_url": "https://huggingface.co/a/b",
                    "file_url": "https://huggingface.co/c/d/resolve/main/m.gguf",
                    "prefer": "file",
                },
                headers=_auth(token),
            ).json()
        assert body["repo_id"] == "c/d"

    def test_a_split_model_returns_every_part(self, client, device):
        _, token = device
        siblings = [(f"m-0000{i}-of-00003.gguf", 100) for i in (1, 2, 3)]
        with self._api(siblings):
            body = client.post(
                "/hyperlink/models/resolve",
                json={"file_url": "https://huggingface.co/o/r/resolve/main/m-00002-of-00003.gguf"},
                headers=_auth(token),
            ).json()
        assert body["is_split"] and body["file_count"] == 3
        assert body["primary_file"] == "m-00001-of-00003.gguf"

    def test_an_invalid_prefer_value_is_refused(self, client, device):
        _, token = device
        response = client.post(
            "/hyperlink/models/resolve",
            json={"page_url": "o/r", "prefer": "whatever"},
            headers=_auth(token),
        )
        assert response.status_code == 422

    def test_it_needs_a_credential(self, client):
        assert client.post("/hyperlink/models/resolve", json={"page_url": "o/r"}).status_code == 401


class TestHyperLinkDisabled:
    def test_every_hyperlink_endpoint_reports_501_when_off(self, client, app, admin_key, device):
        _, token = device
        app.state.t1_config.hyperlink_enabled = False
        for method, path in [
            ("get", "/hyperlink/endpoints"),
            ("get", "/hyperlink/sessions"),
            ("get", "/hyperlink/devices"),
        ]:
            response = getattr(client, method)(path, headers=_auth(admin_key))
            assert response.status_code == 501, path
        assert client.post("/hyperlink/pair", json={}, headers=_auth(admin_key)).status_code == 501
