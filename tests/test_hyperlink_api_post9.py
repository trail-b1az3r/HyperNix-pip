"""The 0.72.4.post9 HyperLink endpoints, over real HTTP.

`/hyperlink/sync`, `/hyperlink/sync/claim`, `/hyperlink/push*` and
`/hyperlink/search` — driven through a TestClient against a real app, so
the dependency wiring, the auth path and the response schemas are all
the real ones. The store-level behaviour is covered in
`test_hyperlink_sync.py`, `test_hyperlink_notify.py` and
`test_hyperlink_search.py`; this file is about whether a phone can
actually reach any of it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402

TOKEN = "a" * 64
OTHER_TOKEN = "b" * 64


@pytest.fixture
def km(tmp_path) -> Keymaster:
    return Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)


@pytest.fixture
def gk(km, tmp_path) -> Gatekeeper:
    return Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper", log_to_file=False)


@pytest.fixture
def client(km, gk, tmp_path) -> TestClient:
    config = T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough",
        db_path=str(tmp_path / "t1.sqlite3"),
        module_storage_dir=str(tmp_path / "modules"),
        hyperlink_files_dir=str(tmp_path / "files"),
        default_plan="free",
    )
    app = create_app(config=config, keymaster=km, gatekeeper=gk)
    return TestClient(app, client=("127.0.0.1", 5000))


@pytest.fixture
def admin_key(km) -> str:
    return km.create(
        key_type=KeyType.ADMIN, scopes={KeyScope.ADMIN, KeyScope.READ, KeyScope.WRITE}
    ).key


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def _pair(client: TestClient, admin_key: str, label: str) -> dict[str, str]:
    minted = client.post(
        "/hyperlink/pair", json={"label": label}, headers=_auth(admin_key)
    )
    assert minted.status_code == 200, minted.text
    redeemed = client.post(
        "/hyperlink/pair/redeem",
        json={"code": minted.json()["code"], "device_name": label, "app_version": "1.0"},
    )
    assert redeemed.status_code == 200, redeemed.text
    return redeemed.json()


@pytest.fixture
def phone(client, admin_key) -> dict[str, str]:
    return _pair(client, admin_key, "Test iPhone")


def _dev(phone: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {phone['device_token']}"}


class TestSyncOverHttp:
    def test_a_new_device_gets_an_empty_page_and_a_head(self, client, phone):
        response = client.get("/hyperlink/sync", headers=_dev(phone))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["changes"] == []
        assert body["more"] is False
        assert body["resync_required"] is False
        assert "head" in body, "no head, so a new device cannot skip the history"

    def test_it_requires_authentication(self, client):
        assert client.get("/hyperlink/sync").status_code in (401, 403)

    def test_a_negative_cursor_is_refused_by_validation(self, client, phone):
        response = client.get("/hyperlink/sync?cursor=-1", headers=_dev(phone))
        assert response.status_code == 422

    def test_the_page_size_is_capped_at_the_boundary(self, client, phone):
        assert client.get(
            "/hyperlink/sync?limit=99999", headers=_dev(phone)
        ).status_code == 422

    def test_an_unknown_entity_filter_is_rejected(self, client, phone):
        """422, like every other VALIDATION_ERROR in this API.

        The filter is a free-text query parameter, so FastAPI cannot
        reject it and the store does — and the T1 error handler maps
        VALIDATION_ERROR to 422, which is what a client already has a
        branch for.
        """
        response = client.get(
            "/hyperlink/sync?entities=telepathy", headers=_dev(phone)
        )
        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"


class TestClaimOverHttp:
    """A retry must not produce a second reply."""

    def test_a_first_claim_is_fresh(self, client, phone):
        response = client.post(
            "/hyperlink/sync/claim", json={"client_msg_id": "abc"}, headers=_dev(phone)
        )
        assert response.status_code == 200, response.text
        assert response.json()["fresh"] is True

    def test_the_same_key_twice_is_not_fresh(self, client, phone):
        client.post(
            "/hyperlink/sync/claim", json={"client_msg_id": "abc"}, headers=_dev(phone)
        )
        second = client.post(
            "/hyperlink/sync/claim", json={"client_msg_id": "abc"}, headers=_dev(phone)
        )
        assert second.json()["fresh"] is False

    def test_two_devices_may_use_the_same_key(self, client, admin_key, phone):
        """Keys are per device, so two phones can both pick "1"."""
        other = _pair(client, admin_key, "Second iPad")
        first = client.post(
            "/hyperlink/sync/claim", json={"client_msg_id": "1"}, headers=_dev(phone)
        )
        second = client.post(
            "/hyperlink/sync/claim", json={"client_msg_id": "1"}, headers=_dev(other)
        )
        assert first.json()["fresh"] is True
        assert second.json()["fresh"] is True

    def test_an_empty_key_is_refused(self, client, phone):
        assert client.post(
            "/hyperlink/sync/claim", json={"client_msg_id": ""}, headers=_dev(phone)
        ).status_code == 422


class TestPushOverHttp:
    def test_registering_returns_a_fingerprint_and_never_the_token(self, client, phone):
        response = client.post(
            "/hyperlink/push", json={"token": TOKEN, "bundle_id": "org.hnx"},
            headers=_dev(phone),
        )
        assert response.status_code == 200, response.text
        assert TOKEN not in response.text, "the token came back over the wire"
        assert len(response.json()["registration"]["fingerprint"]) == 8

    def test_listing_never_carries_a_token(self, client, phone):
        client.post("/hyperlink/push", json={"token": TOKEN}, headers=_dev(phone))
        listing = client.get("/hyperlink/push", headers=_dev(phone))
        assert listing.status_code == 200, listing.text
        assert TOKEN not in listing.text
        assert listing.json()["count"] == 1

    def test_re_registering_does_not_duplicate(self, client, phone):
        """iOS hands the app a token on every launch."""
        for _ in range(3):
            client.post("/hyperlink/push", json={"token": TOKEN}, headers=_dev(phone))
        assert client.get("/hyperlink/push", headers=_dev(phone)).json()["count"] == 1

    def test_a_short_token_is_refused_by_validation(self, client, phone):
        assert client.post(
            "/hyperlink/push", json={"token": "abc"}, headers=_dev(phone)
        ).status_code == 422

    def test_the_available_events_are_served_not_hard_coded(self, client, phone):
        """So a server that gains an event kind does not need the app
        updated before it can be offered."""
        response = client.get("/hyperlink/push/events", headers=_dev(phone))
        assert response.status_code == 200, response.text
        assert "chat.reply" in response.json()["events"]

    def test_events_can_be_narrowed(self, client, phone):
        created = client.post(
            "/hyperlink/push", json={"token": TOKEN}, headers=_dev(phone)
        ).json()["registration"]
        response = client.patch(
            f"/hyperlink/push/{created['registration_id']}",
            json={"events": ["training.done"]}, headers=_dev(phone),
        )
        assert response.status_code == 200, response.text
        assert response.json()["registration"]["events"] == ["training.done"]

    def test_unregistering_works(self, client, phone):
        created = client.post(
            "/hyperlink/push", json={"token": TOKEN}, headers=_dev(phone)
        ).json()["registration"]
        assert client.delete(
            f"/hyperlink/push/{created['registration_id']}", headers=_dev(phone)
        ).status_code == 200
        assert client.get("/hyperlink/push", headers=_dev(phone)).json()["count"] == 0

    def test_it_requires_authentication(self, client):
        assert client.post(
            "/hyperlink/push", json={"token": TOKEN}
        ).status_code in (401, 403)


class TestOneDeviceCannotTouchAnothers:
    """A registration id is not a secret, so it cannot be the authority.

    Without an ownership check, knowing an id would let any
    authenticated caller silence or delete someone else's
    notifications — a public credential escalating into control of
    another account's devices.
    """

    @pytest.fixture
    def two_owners(self, client, km, admin_key, phone):
        mine = client.post(
            "/hyperlink/push", json={"token": TOKEN}, headers=_dev(phone)
        ).json()["registration"]
        stranger_key = km.create(
            key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}
        ).key
        return mine, stranger_key

    def test_a_stranger_cannot_delete_it(self, client, two_owners):
        mine, stranger_key = two_owners
        response = client.delete(
            f"/hyperlink/push/{mine['registration_id']}", headers=_auth(stranger_key)
        )
        assert response.status_code == 404, response.text

    def test_a_stranger_cannot_change_its_events(self, client, two_owners):
        mine, stranger_key = two_owners
        response = client.patch(
            f"/hyperlink/push/{mine['registration_id']}",
            json={"events": []}, headers=_auth(stranger_key),
        )
        assert response.status_code == 404, response.text

    def test_the_refusal_does_not_confirm_the_id_exists(self, client, two_owners):
        """404 rather than 403: telling an unauthorised caller that an
        id is real is itself something they should not learn."""
        mine, stranger_key = two_owners
        real = client.delete(
            f"/hyperlink/push/{mine['registration_id']}", headers=_auth(stranger_key)
        )
        invented = client.delete(
            "/hyperlink/push/push_doesnotexist", headers=_auth(stranger_key)
        )
        assert real.status_code == invented.status_code == 404

    def test_the_owner_still_can(self, client, phone, two_owners):
        mine, _ = two_owners
        assert client.delete(
            f"/hyperlink/push/{mine['registration_id']}", headers=_dev(phone)
        ).status_code == 200


class TestSearchOverHttp:
    def _seed(self, client, phone):
        created = client.post(
            "/hyperlink/sessions", json={"title": "CUDA memory notes"},
            headers=_dev(phone),
        )
        assert created.status_code == 200, created.text
        return created.json()["session"]["session_id"]

    def test_it_answers(self, client, phone):
        self._seed(client, phone)
        response = client.get("/hyperlink/search?q=cuda", headers=_dev(phone))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["terms"] == ["cuda"]
        assert body["hits"], "found nothing in a session titled 'CUDA memory notes'"

    def test_it_reports_whether_it_searched_everything(self, client, phone):
        """A silent partial answer reads as "the conversation is gone"."""
        self._seed(client, phone)
        body = client.get("/hyperlink/search?q=cuda", headers=_dev(phone)).json()
        assert body["capped"] is False
        assert "scanned" in body

    def test_a_snippet_carries_offsets_not_markup(self, client, phone):
        self._seed(client, phone)
        hit = client.get("/hyperlink/search?q=cuda", headers=_dev(phone)).json()["hits"][0]
        assert "<" not in hit["snippet"]["text"]
        assert hit["snippet"]["ranges"], "no match position"

    def test_a_wildcard_does_not_return_everything(self, client, phone):
        self._seed(client, phone)
        body = client.get("/hyperlink/search?q=%25", headers=_dev(phone)).json()
        assert body["hits"] == []

    def test_an_empty_query_is_refused_by_validation(self, client, phone):
        assert client.get("/hyperlink/search?q=", headers=_dev(phone)).status_code == 422

    def test_it_requires_authentication(self, client):
        assert client.get("/hyperlink/search?q=cuda").status_code in (401, 403)

    def test_one_owner_cannot_search_anothers_history(self, client, km, phone):
        self._seed(client, phone)
        stranger = km.create(
            key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE}
        ).key
        body = client.get("/hyperlink/search?q=cuda", headers=_auth(stranger)).json()
        assert body["hits"] == []


class TestTheOpenApiContract:
    """The app is what the Swift client generates against."""

    @pytest.mark.parametrize(
        ("path", "method"),
        [
            ("/hyperlink/sync", "get"),
            ("/hyperlink/sync/claim", "post"),
            ("/hyperlink/push", "get"),
            ("/hyperlink/push", "post"),
            ("/hyperlink/push/events", "get"),
            ("/hyperlink/push/{registration_id}", "patch"),
            ("/hyperlink/push/{registration_id}", "delete"),
            ("/hyperlink/search", "get"),
        ],
    )
    def test_the_endpoint_is_documented(self, client, path, method):
        spec = client.app.openapi()
        assert path in spec["paths"], f"{path} is not in the OpenAPI schema"
        assert method in spec["paths"][path]

    def test_no_response_schema_mentions_a_token(self, client):
        """A generated client must not be given a field for one."""
        spec = client.app.openapi()
        for name in ("PushRegistrationSummary", "PushRegistrationResponse"):
            schema = spec["components"]["schemas"][name]
            assert "token" not in str(schema.get("properties", {})), name
