"""Integration tests for Beta 2's HTTP layer: servers, modules, jobs,
events, billing, and the POST /models/route addition.

Requires `pip install hypernix[t1api-test]`. Executed and passing as of
Beta 3 (the Beta 2 authoring sandbox had no network access to install
FastAPI, so this file shipped unexecuted at the time; it has since been
run and the two assertions that Beta 3 legitimately changed are marked
below).
"""
from __future__ import annotations

import hashlib
import json
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402
from hypernix.t1api.registry import ModelRegistry, ModelStatus  # noqa: E402
from hypernix.t1api.storage import UsageStore  # noqa: E402
from hypernix.t1api.transport import ModuleTransport  # noqa: E402

_FREE_TIER_MODELS = [
    "nanonix-mini-plus", "nanonanonano-n3", "nanonix-mini-lite",
    "nanonix-mini", "nanonix-nano", "nanonanonanonano-n4",
]


@pytest.fixture
def km(tmp_path) -> Keymaster:
    return Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)


@pytest.fixture
def gk(km, tmp_path) -> Gatekeeper:
    return Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper", log_to_file=False)


@pytest.fixture
def registry() -> ModelRegistry:
    reg = ModelRegistry.load(include_examples=True)
    for model_id in _FREE_TIER_MODELS:
        entry = reg.require(model_id)
        entry.status = ModelStatus.AVAILABLE
        entry.is_example_entry = False
        reg.register(entry)
    return reg


@pytest.fixture
def client(km, gk, registry, tmp_path) -> TestClient:
    cfg = T1APIConfig(token_secret="test-secret", db_path=str(tmp_path / "usage.sqlite3"))
    app = create_app(
        config=cfg,
        keymaster=km,
        gatekeeper=gk,
        registry=registry,
        usage_store=UsageStore(tmp_path / "usage.sqlite3"),
    )
    return TestClient(app)


@pytest.fixture
def user_key(km) -> str:
    return km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key


@pytest.fixture
def admin_key(km) -> str:
    return km.create(key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}).key


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _await_job(client, job_id: str, key: str, timeout: float = 5.0):
    """Poll a job to a terminal state. Jobs run on a real thread pool, so
    tests wait for the transition rather than assuming it happened."""
    deadline = time.time() + timeout
    job_resp = client.get(f"/jobs/{job_id}", headers=_auth(key))
    while time.time() < deadline:
        job_resp = client.get(f"/jobs/{job_id}", headers=_auth(key))
        if job_resp.json()["job"]["status"] in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.02)
    return job_resp


@pytest.fixture
def sent_transfers() -> list[dict]:
    """Records every transfer the stub opener was asked to send."""
    return []


@pytest.fixture
def deploy_client(km, gk, registry, tmp_path, sent_transfers) -> TestClient:
    """A client whose ModuleTransport talks to a stub instead of the network.

    The real ModuleTransport is used — signing, size caps, SSRF
    validation and checksum verification all run for real. Only the
    socket is replaced, so these tests exercise the actual transport code
    path rather than a mock of it.
    """

    class _StubResponse:
        status = 200

        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def read(self, *args):
            return self._payload

        def getcode(self):
            return 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _opener(request, timeout=None):
        body = request.data or b""
        # urllib capitalizes header names its own way; normalize so
        # assertions can look them up by the canonical spelling.
        sent_transfers.append(
            {
                "url": request.full_url,
                "body": body,
                "headers": {k.lower(): v for k, v in request.headers.items()},
            }
        )
        # Echo the digest back the way a real receiving server does, so
        # the sender's post-transfer integrity check has something to
        # compare against.
        return _StubResponse(
            json.dumps({"checksum": hashlib.sha256(body).hexdigest()}).encode("utf-8")
        )

    cfg = T1APIConfig(
        token_secret="test-secret",
        db_path=str(tmp_path / "usage.sqlite3"),
        deploy_secret="deploy-secret-value",
        module_storage_dir=str(tmp_path / "modules"),
    )
    app = create_app(
        config=cfg,
        keymaster=km,
        gatekeeper=gk,
        registry=registry,
        usage_store=UsageStore(tmp_path / "usage.sqlite3"),
        transport=ModuleTransport(deploy_secret="deploy-secret-value", opener=_opener),
    )
    return TestClient(app)


class TestModelRouting:
    def test_automatic_routing_returns_primary(self, client, user_key):
        resp = client.post(
            "/models/route", json={"plan": "free", "input_tokens": 100}, headers=_auth(user_key)
        )
        assert resp.status_code == 200
        assert resp.json()["model_id"] == "nanonix-mini-plus"

    def test_manual_routing_of_available_model_succeeds(self, client, user_key, registry):
        """There's no inference/chat endpoint in Beta 1/2 (the T1 API is a
        control plane, not an inference gateway), so there's no HTTP path
        that actually consumes token quota to drive a model into
        MODEL_QUOTA_EXHAUSTED through this router set. The exhaustion
        behavior itself is fully covered at the core level — see
        test_t1api_routing.py's TestManualSelection. This test only
        confirms manual selection of a non-exhausted model round-trips
        correctly over HTTP.
        """
        resp = client.post(
            "/models/route",
            json={"plan": "free", "model_id": "nanonix-nano"},
            headers=_auth(user_key),
        )
        assert resp.status_code == 200
        assert resp.json()["model_id"] == "nanonix-nano"

    def test_manual_routing_of_unregistered_model_returns_404(self, client, user_key):
        resp = client.post(
            "/models/route", json={"plan": "free", "model_id": "fake-model"}, headers=_auth(user_key)
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "MODEL_NOT_SUPPORTED"


class TestServers:
    def test_register_returns_untrusted_server(self, client, user_key):
        resp = client.post(
            "/servers/register",
            json={"name": "prod-1", "address": "https://prod1.example.com"},
            headers=_auth(user_key),
        )
        assert resp.status_code == 200
        body = resp.json()["server"]
        assert body["trust_level"] == "untrusted"

    def test_register_private_address_rejected_without_flag(self, client, user_key):
        resp = client.post(
            "/servers/register",
            json={"name": "local-1", "address": "http://192.168.1.5:9000"},
            headers=_auth(user_key),
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_list_servers(self, client, user_key):
        client.post(
            "/servers/register",
            json={"name": "a", "address": "https://a.example.com"},
            headers=_auth(user_key),
        )
        resp = client.get("/servers", headers=_auth(user_key))
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_non_admin_cannot_patch_trust_level(self, client, user_key):
        reg_resp = client.post(
            "/servers/register",
            json={"name": "a", "address": "https://a.example.com"},
            headers=_auth(user_key),
        )
        server_id = reg_resp.json()["server"]["server_id"]
        resp = client.patch(
            f"/servers/{server_id}", json={"trust_level": "trusted"}, headers=_auth(user_key)
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "AUTH_ADMIN_REQUIRED"

    def test_admin_can_promote_trust_level(self, client, admin_key):
        reg_resp = client.post(
            "/servers/register",
            json={"name": "a", "address": "https://a.example.com"},
            headers=_auth(admin_key),
        )
        server_id = reg_resp.json()["server"]["server_id"]
        resp = client.patch(
            f"/servers/{server_id}", json={"trust_level": "trusted"}, headers=_auth(admin_key)
        )
        assert resp.status_code == 200
        assert resp.json()["server"]["trust_level"] == "trusted"

    def test_delete_requires_admin(self, client, user_key):
        reg_resp = client.post(
            "/servers/register",
            json={"name": "a", "address": "https://a.example.com"},
            headers=_auth(user_key),
        )
        server_id = reg_resp.json()["server"]["server_id"]
        resp = client.delete(f"/servers/{server_id}", headers=_auth(user_key))
        assert resp.status_code == 403


class TestModules:
    def test_create_module(self, client, user_key):
        resp = client.post(
            "/modules/create",
            json={"name": "my-mod", "version": "1.0.0"},
            headers=_auth(user_key),
        )
        assert resp.status_code == 200
        assert resp.json()["module"]["status"] == "draft"

    def test_upload_local_activates_module(self, client, user_key):
        create_resp = client.post(
            "/modules/create", json={"name": "my-mod", "version": "1.0.0"}, headers=_auth(user_key)
        )
        module_id = create_resp.json()["module"]["module_id"]
        upload_resp = client.post(
            f"/modules/upload/local?module_id={module_id}",
            files={"file": ("mod.bin", b"module content bytes")},
            headers=_auth(user_key),
        )
        assert upload_resp.status_code == 200
        body = upload_resp.json()["module"]
        assert body["status"] == "active"
        assert body["checksum"] is not None

    def test_upload_remote_marks_pending_fetch(self, client, user_key):
        create_resp = client.post(
            "/modules/create", json={"name": "my-mod", "version": "1.0.0"}, headers=_auth(user_key)
        )
        module_id = create_resp.json()["module"]["module_id"]
        resp = client.post(
            f"/modules/upload/remote?module_id={module_id}",
            json={"source_url": "https://example.com/mod.tar.gz"},
            headers=_auth(user_key),
        )
        assert resp.status_code == 200
        assert resp.json()["module"]["status"] == "pending_fetch"

    def test_upload_remote_ssrf_blocked(self, client, user_key):
        create_resp = client.post(
            "/modules/create", json={"name": "my-mod", "version": "1.0.0"}, headers=_auth(user_key)
        )
        module_id = create_resp.json()["module"]["module_id"]
        resp = client.post(
            f"/modules/upload/remote?module_id={module_id}",
            json={"source_url": "http://169.254.169.254/secret"},
            headers=_auth(user_key),
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "SSRF_BLOCKED"

    def test_other_users_module_not_editable(self, client, user_key, km):
        other_key = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}).key
        create_resp = client.post(
            "/modules/create", json={"name": "my-mod", "version": "1.0.0"}, headers=_auth(user_key)
        )
        module_id = create_resp.json()["module"]["module_id"]
        resp = client.patch(
            f"/modules/{module_id}", json={"metadata": {"x": 1}}, headers=_auth(other_key)
        )
        assert resp.status_code == 403

    def test_sync_to_untrusted_server_job_fails(self, client, user_key):
        """Trust is checked before the target's address is ever dialled.

        Beta 3 note: the job now surfaces the *specific* SERVER_UNTRUSTED
        code rather than a generic transport failure — see
        DeploymentCoordinator.module_sync_handler, which preserves the
        underlying code when every target failed the same way.
        """
        create_resp = client.post(
            "/modules/create", json={"name": "my-mod", "version": "1.0.0"}, headers=_auth(user_key)
        )
        module_id = create_resp.json()["module"]["module_id"]
        server_resp = client.post(
            "/servers/register",
            json={"name": "s1", "address": "https://s1.example.com"},
            headers=_auth(user_key),
        )
        server_id = server_resp.json()["server"]["server_id"]
        sync_resp = client.post(
            f"/modules/{module_id}/sync", json={"server_id": server_id}, headers=_auth(user_key)
        )
        assert sync_resp.status_code == 200
        job_id = sync_resp.json()["job_id"]

        job_resp = _await_job(client, job_id, user_key)
        assert job_resp.json()["job"]["status"] == "failed"
        assert "SERVER_UNTRUSTED" in job_resp.json()["job"]["error"]

    def test_sync_to_trusted_server_transfers_bytes(self, deploy_client, admin_key, sent_transfers):
        """Beta 3: sync moves real bytes to a trusted server.

        The Beta 2 version of this test asserted only that the sync was
        *recorded*, because there was no transport behind it. Now there
        is, so the test asserts what actually crossed the wire: the
        signed request, the exact payload, and the checksum the target
        echoed back.
        """
        create_resp = deploy_client.post(
            "/modules/create", json={"name": "my-mod", "version": "1.0.0"}, headers=_auth(admin_key)
        )
        module_id = create_resp.json()["module"]["module_id"]
        upload_resp = deploy_client.post(
            f"/modules/upload/local?module_id={module_id}",
            files={"file": ("mod.bin", b"deployable-bytes", "application/octet-stream")},
            headers=_auth(admin_key),
        )
        assert upload_resp.status_code == 200

        server_resp = deploy_client.post(
            "/servers/register",
            json={"name": "s1", "address": "https://s1.example.com"},
            headers=_auth(admin_key),
        )
        server_id = server_resp.json()["server"]["server_id"]
        deploy_client.patch(
            f"/servers/{server_id}", json={"trust_level": "trusted"}, headers=_auth(admin_key)
        )

        sync_resp = deploy_client.post(
            f"/modules/{module_id}/sync", json={"server_id": server_id}, headers=_auth(admin_key)
        )
        job_id = sync_resp.json()["job_id"]

        job_resp = _await_job(deploy_client, job_id, admin_key)
        job = job_resp.json()["job"]
        assert job["status"] == "succeeded", job.get("error")
        assert job["result"]["deployed_servers"] == [server_id]
        assert job["result"]["delivered"][0]["bytes_transferred"] == len(b"deployable-bytes")

        # The transport actually ran: one signed POST carrying exactly the
        # uploaded bytes to the registry's address for that server.
        assert len(sent_transfers) == 1
        transfer = sent_transfers[0]
        assert transfer["url"] == "https://s1.example.com/modules/receive"
        assert transfer["body"] == b"deployable-bytes"
        assert transfer["headers"]["x-t1-signature"]
        assert transfer["headers"]["x-t1-content-sha256"] == hashlib.sha256(
            b"deployable-bytes"
        ).hexdigest()

    def test_deploy_to_multiple_servers(self, deploy_client, admin_key, sent_transfers):
        """The Beta 3 multi-target form: one job, N trusted servers."""
        module_id = deploy_client.post(
            "/modules/create", json={"name": "multi", "version": "1.0.0"}, headers=_auth(admin_key)
        ).json()["module"]["module_id"]
        deploy_client.post(
            f"/modules/upload/local?module_id={module_id}",
            files={"file": ("mod.bin", b"multi-bytes", "application/octet-stream")},
            headers=_auth(admin_key),
        )
        server_ids = []
        for name in ("s1", "s2"):
            resp = deploy_client.post(
                "/servers/register",
                json={"name": name, "address": f"https://{name}.example.com"},
                headers=_auth(admin_key),
            )
            server_id = resp.json()["server"]["server_id"]
            deploy_client.patch(
                f"/servers/{server_id}", json={"trust_level": "trusted"}, headers=_auth(admin_key)
            )
            server_ids.append(server_id)

        resp = deploy_client.post(
            f"/modules/{module_id}/deploy", json={"server_ids": server_ids}, headers=_auth(admin_key)
        )
        assert resp.status_code == 200
        job = _await_job(deploy_client, resp.json()["job_id"], admin_key).json()["job"]
        assert job["status"] == "succeeded", job.get("error")
        assert sorted(job["result"]["deployed_servers"]) == sorted(server_ids)
        assert len(sent_transfers) == 2

    def test_deploy_to_untrusted_server_rejected_before_queueing(self, deploy_client, admin_key):
        """An untrusted target fails the request, not a job you have to
        go looking for afterwards."""
        module_id = deploy_client.post(
            "/modules/create", json={"name": "nope", "version": "1.0.0"}, headers=_auth(admin_key)
        ).json()["module"]["module_id"]
        server_id = deploy_client.post(
            "/servers/register",
            json={"name": "untrusted", "address": "https://u.example.com"},
            headers=_auth(admin_key),
        ).json()["server"]["server_id"]

        resp = deploy_client.post(
            f"/modules/{module_id}/deploy", json={"server_ids": [server_id]}, headers=_auth(admin_key)
        )
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "SERVER_UNTRUSTED"

    def test_delete_module_requires_confirmation(self, client, user_key):
        """Destructive operations need ?confirm=true (Beta 3)."""
        module_id = client.post(
            "/modules/create", json={"name": "doomed", "version": "1.0.0"}, headers=_auth(user_key)
        ).json()["module"]["module_id"]

        resp = client.delete(f"/modules/{module_id}", headers=_auth(user_key))
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "CONFIRMATION_REQUIRED"
        # Still there.
        assert client.get(f"/modules/{module_id}", headers=_auth(user_key)).status_code == 200

        resp = client.delete(f"/modules/{module_id}?confirm=true", headers=_auth(user_key))
        assert resp.status_code == 200
        assert client.get(f"/modules/{module_id}", headers=_auth(user_key)).status_code == 404


class TestJobs:
    def test_create_job_with_unregistered_kind_returns_501(self, client, user_key):
        resp = client.post(
            "/jobs", json={"kind": "not-a-real-kind", "payload": {}}, headers=_auth(user_key)
        )
        assert resp.status_code == 501
        assert resp.json()["error"]["code"] == "NOT_SUPPORTED"

    def test_get_unknown_job_returns_404(self, client, user_key):
        resp = client.get("/jobs/nope", headers=_auth(user_key))
        assert resp.status_code == 404

    def test_cancel_unknown_job_returns_404(self, client, user_key):
        resp = client.post("/jobs/nope/cancel", headers=_auth(user_key))
        assert resp.status_code == 404


class TestEvents:
    def test_list_events_after_server_register(self, client, user_key):
        client.post(
            "/servers/register",
            json={"name": "a", "address": "https://a.example.com"},
            headers=_auth(user_key),
        )
        resp = client.get("/events", headers=_auth(user_key))
        assert resp.status_code == 200
        types = [e["type"] for e in resp.json()["events"]]
        assert "server.registered" in types

    def test_events_filterable_by_type(self, client, user_key):
        client.post(
            "/servers/register",
            json={"name": "a", "address": "https://a.example.com"},
            headers=_auth(user_key),
        )
        resp = client.get("/events?type=server.registered", headers=_auth(user_key))
        events = resp.json()["events"]
        assert len(events) >= 1
        assert all(e["type"] == "server.registered" for e in events)


class TestBilling:
    def test_fresh_balance_is_zero(self, client, user_key):
        resp = client.get("/billing/balance", headers=_auth(user_key))
        assert resp.status_code == 200
        assert resp.json()["balance"] == 0.0

    def test_non_admin_cannot_mint_payment_token(self, client, user_key):
        resp = client.post(
            "/billing/payment-token", json={"amount": 50.0}, headers=_auth(user_key)
        )
        assert resp.status_code == 403

    def test_admin_mint_then_user_redeem(self, client, admin_key, user_key):
        mint_resp = client.post(
            "/billing/payment-token", json={"amount": 50.0}, headers=_auth(admin_key)
        )
        assert mint_resp.status_code == 200
        raw_token = mint_resp.json()["token"]

        redeem_resp = client.post(
            "/billing/redeem", json={"token": raw_token}, headers=_auth(user_key)
        )
        assert redeem_resp.status_code == 200
        assert redeem_resp.json()["balance"] == 50.0

    def test_double_redeem_rejected(self, client, admin_key, user_key):
        mint_resp = client.post(
            "/billing/payment-token", json={"amount": 50.0}, headers=_auth(admin_key)
        )
        raw_token = mint_resp.json()["token"]
        client.post("/billing/redeem", json={"token": raw_token}, headers=_auth(user_key))
        second = client.post("/billing/redeem", json={"token": raw_token}, headers=_auth(user_key))
        assert second.status_code == 409

    def test_non_admin_cannot_add_balance(self, client, user_key):
        resp = client.post(
            "/billing/add-balance",
            json={"account_type": "user", "account_id": "someone", "amount": 10},
            headers=_auth(user_key),
        )
        assert resp.status_code == 403

    def test_admin_add_balance(self, client, admin_key):
        resp = client.post(
            "/billing/add-balance",
            json={"account_type": "user", "account_id": "target-user", "amount": 25.0},
            headers=_auth(admin_key),
        )
        assert resp.status_code == 200
        assert resp.json()["balance"] == 25.0

    def test_transactions_reflect_redemption(self, client, admin_key, user_key):
        mint_resp = client.post(
            "/billing/payment-token", json={"amount": 15.0}, headers=_auth(admin_key)
        )
        raw_token = mint_resp.json()["token"]
        client.post("/billing/redeem", json={"token": raw_token}, headers=_auth(user_key))
        txns = client.get("/billing/transactions", headers=_auth(user_key))
        assert txns.status_code == 200
        assert any(t["kind"] == "redeem" for t in txns.json()["transactions"])
