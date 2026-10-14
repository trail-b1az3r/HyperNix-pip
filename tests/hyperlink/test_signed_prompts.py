"""Chat ownership, the ping before a prompt, and signed prompts (0.72.8)."""
from __future__ import annotations

import time

import pytest

from hypernix.hyperlink import ownership, signing

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypernix.security.gatekeeper import Gatekeeper  # noqa: E402
from hypernix.security.keymaster import Keymaster, KeyScope, KeyType  # noqa: E402
from hypernix.t1api.app import create_app  # noqa: E402
from hypernix.t1api.config import T1APIConfig  # noqa: E402


class TestSigning:
    def test_a_signature_verifies_once(self):
        sig = signing.sign("T1_secret", signer="key:k1", chat_session_id="c", server_session_id="s",
                           action="send", content="hello")
        nonces = signing.NonceCache()
        checked = signing.verify(sig, credential="T1_secret", signer="key:k1", server_session_id="s",
                                 action="send", chat_session_id="c", content="hello", nonces=nonces)
        assert checked.signer == "key:k1"
        with pytest.raises(signing.SignatureError) as replay:
            signing.verify(sig, credential="T1_secret", signer="key:k1", server_session_id="s",
                           action="send", chat_session_id="c", content="hello", nonces=nonces)
        assert replay.value.code == "replay"

    @pytest.mark.parametrize("change, code", [
        ({"credential": "other"}, "mac"),
        ({"signer": "key:k2"}, "signer"),
        ({"server_session_id": "restarted"}, "server_session"),
        ({"chat_session_id": "another"}, "chat_session"),
        ({"content": "tampered"}, "content"),
        ({"action": "edit"}, "action"),
    ])
    def test_anything_changed_is_refused(self, change, code):
        sig = signing.sign("T1_secret", signer="key:k1", chat_session_id="c", server_session_id="s",
                           action="send", content="hello")
        args = dict(credential="T1_secret", signer="key:k1", server_session_id="s", action="send",
                    chat_session_id="c", content="hello")
        args.update(change)
        with pytest.raises(signing.SignatureError) as refused:
            signing.verify(sig, **args)
        assert refused.value.code == code

    def test_old_signatures_and_bad_nonces(self):
        old = signing.sign("k", signer="key:a", action="ping", signed_at=int(time.time()) - 3600)
        with pytest.raises(signing.SignatureError, match="too long ago"):
            signing.verify(old, credential="k", signer="key:a", server_session_id="", action="ping",
                           allow_no_server_session=True)
        short = signing.sign("k", signer="key:a", action="ping", nonce="x" * 10)
        with pytest.raises(signing.SignatureError) as bad:
            signing.verify(short, credential="k", signer="key:a", server_session_id="", action="ping",
                           allow_no_server_session=True)
        assert bad.value.code == "nonce"

    def test_nonces_are_23_to_26_characters(self):
        for n in (23, 24, 26):
            assert signing.NONCE.match(signing.new_nonce(n))
        with pytest.raises(ValueError):
            signing.new_nonce(22)

    def test_the_canonical_form_is_one_field_per_line(self):
        sig = signing.sign("k", signer="account:a1", chat_session_id="c", server_session_id="s",
                           action="edit", content="x", signed_at=1_800_000_000, nonce="n" * 24)
        lines = signing.canonical(sig).split("\n")
        assert lines == ["hyperlink-sign/v1", "account:a1", "1800000000", "n" * 24, "c", "s", "edit",
                         signing.content_hash("x")]


# -- through the API -----------------------------------------------------------


@pytest.fixture
def server(tmp_path, monkeypatch):
    from conftest import clear_t1_config

    clear_t1_config(monkeypatch)
    km = Keymaster(store_dir=tmp_path / "keymaster", auto_rotate=False)
    gk = Gatekeeper(keymaster=km, data_dir=tmp_path / "gatekeeper", log_to_file=False)
    config = T1APIConfig(
        token_secret="test-secret-value-that-is-long-enough",
        db_path=str(tmp_path / "t1.sqlite3"),
        module_storage_dir=str(tmp_path / "modules"),
        hyperlink_files_dir=str(tmp_path / "files"),
        lmstudio_enabled=False,
    )
    app = create_app(config=config, keymaster=km, gatekeeper=gk)
    client = TestClient(app, client=("127.0.0.1", 5000))
    key = km.create(key_type=KeyType.USER, scopes={KeyScope.READ, KeyScope.WRITE})
    return app, client, key, config


def _auth(key):
    return {"Authorization": f"Bearer {key.key}"}


def _ping(client, key, *, server_session_id="", chat=""):
    now = time.time()
    sig = signing.sign(key.key, signer=f"key:{key.key_id}", action="ping", chat_session_id=chat,
                       server_session_id=server_session_id, signed_at=int(now))
    return client.post("/hyperlink/ping", headers=_auth(key),
                       json={"sent_at": now, "chat_session_id": chat, "signature": sig})


class TestThePing:
    def test_it_answers_without_a_model_and_names_the_server_session(self, server):
        _app, client, key, _ = server
        response = _ping(client, key)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["server_session_id"].startswith("srv_")
        assert body["signer"] == f"key:{key.key_id}"
        assert body["ping_timeout_seconds"] == 4.0
        assert body["transit_seconds"] < 5

    def test_an_empty_ping_is_refused(self, server):
        _app, client, key, _ = server
        assert client.post("/hyperlink/ping", headers=_auth(key), json={}).status_code == 422

    def test_a_ping_signed_with_another_key_is_refused(self, server):
        _app, client, key, _ = server
        now = time.time()
        sig = signing.sign("not-the-key", signer=f"key:{key.key_id}", action="ping", signed_at=int(now))
        response = client.post("/hyperlink/ping", headers=_auth(key),
                               json={"sent_at": now, "signature": sig})
        assert response.status_code == 401


class TestSignedChat:
    def _session(self, client, key):
        return client.post("/hyperlink/sessions", headers=_auth(key), json={"title": "t"}).json()["session"]["session_id"]

    def test_a_signed_prompt_gets_past_the_check(self, server):
        _app, client, key, _ = server
        sid = self._session(client, key)
        server_session = _ping(client, key).json()["server_session_id"]
        sig = signing.sign(key.key, signer=f"key:{key.key_id}", chat_session_id=sid,
                           server_session_id=server_session, action="send", content="hi")
        response = client.post(f"/hyperlink/sessions/{sid}/chat", headers=_auth(key),
                               json={"content": "hi", "signature": sig})
        # No model on this server: refused for that, after the signature passed.
        assert response.status_code == 501, response.text

    def test_a_signature_from_before_a_restart_says_ping_again(self, server):
        _app, client, key, _ = server
        sid = self._session(client, key)
        _ping(client, key)
        sig = signing.sign(key.key, signer=f"key:{key.key_id}", chat_session_id=sid,
                           server_session_id="srv_old", action="send", content="hi")
        response = client.post(f"/hyperlink/sessions/{sid}/chat", headers=_auth(key),
                               json={"content": "hi", "signature": sig})
        assert response.status_code == 409
        assert response.json()["error"]["details"]["server_session_id"].startswith("srv_")

    def test_a_tampered_message_is_refused(self, server):
        _app, client, key, _ = server
        sid = self._session(client, key)
        server_session = _ping(client, key).json()["server_session_id"]
        sig = signing.sign(key.key, signer=f"key:{key.key_id}", chat_session_id=sid,
                           server_session_id=server_session, action="send", content="hi")
        response = client.post(f"/hyperlink/sessions/{sid}/chat", headers=_auth(key),
                               json={"content": "something else", "signature": sig})
        assert response.status_code == 401

    def test_unsigned_is_allowed_unless_required(self, server):
        app, client, key, config = server
        sid = self._session(client, key)
        assert client.post(f"/hyperlink/sessions/{sid}/chat", headers=_auth(key),
                           json={"content": "hi"}).status_code == 501
        config.hyperlink_require_signed = True
        response = client.post(f"/hyperlink/sessions/{sid}/chat", headers=_auth(key), json={"content": "hi"})
        assert response.status_code == 401 and "signed prompts" in response.json()["error"]["message"]

    def test_an_edit_is_re_signed(self, server):
        app, client, key, _ = server
        sid = self._session(client, key)
        store = app.state.t1_session_store
        msg = store.append(sid, role="user", content="first", owner=key.key_id)
        server_session = _ping(client, key).json()["server_session_id"]
        sig = signing.sign(key.key, signer=f"key:{key.key_id}", chat_session_id=sid,
                           server_session_id=server_session, action="edit", content="second")
        response = client.patch(f"/hyperlink/sessions/{sid}/messages/{msg.message_id}", headers=_auth(key),
                                json={"content": "second", "signature": sig})
        assert response.status_code == 200, response.text
        wrong = signing.sign(key.key, signer=f"key:{key.key_id}", chat_session_id=sid,
                             server_session_id=server_session, action="send", content="third")
        assert client.patch(f"/hyperlink/sessions/{sid}/messages/{msg.message_id}", headers=_auth(key),
                            json={"content": "third", "signature": wrong}).status_code == 401


class TestOwnership:
    def test_chats_are_per_key(self, server):
        app, client, key, _ = server
        sid = client.post("/hyperlink/sessions", headers=_auth(key), json={"title": "mine"}).json()["session"]["session_id"]
        assert any(s["session_id"] == sid for s in client.get("/hyperlink/sessions", headers=_auth(key)).json()["sessions"])
        assert app.state.t1_session_store.get(sid).owner == key.key_id

    def test_an_accounts_keys_share_its_chats_and_old_chats_move_to_it(self, tmp_path):
        class Settings:
            enabled = True

        class Account:
            account_id = "acct1"
            key_ids = ["k1", "k2"]

        class Accounts:
            def account_for_key(self, key_id):
                return Account() if key_id in Account.key_ids else None

        from hypernix.hyperlink.sessions import ChatSessionStore
        from hypernix.t1api.db import SQLiteBackend

        sessions = ChatSessionStore(backend=SQLiteBackend(tmp_path / "s.sqlite3"))
        before = sessions.create(owner="k1", title="from before accounts")

        class State:
            t1_webauth = Settings()
            t1_accounts = Accounts()
            t1_session_store = sessions

        ownership._adopted.clear()
        assert ownership.owner_for_key(State, "k1") == "account:acct1"
        assert ownership.owner_for_key(State, "k2") == "account:acct1"
        assert ownership.owner_for_key(State, "lonely") == "lonely"
        assert [s.session_id for s in sessions.list_sessions(owner="account:acct1")] == [before.session_id]
        assert sessions.list_sessions(owner="k1") == []

    def test_accounts_off_means_the_key(self):
        class State:
            t1_webauth = None

        assert ownership.owner_for_key(State, "k1") == "k1"


def test_whoami_says_what_to_sign_as(server):
    _app, client, key, _ = server
    body = client.get("/hyperlink/whoami", headers=_auth(key)).json()
    assert body["signer"] == f"key:{key.key_id}" and body["owner_kind"] == "key"
    assert body["server_session_id"].startswith("srv_")


def test_the_web_apps_signature_verifies_on_the_server():
    """The signing section of hyperlink/web/app.js, run in a real browser,
    produces signatures signing.verify accepts."""
    import os
    from pathlib import Path

    sync_api = pytest.importorskip("playwright.sync_api")
    source = (Path(signing.__file__).parent / "web" / "app.js").read_text(encoding="utf-8")
    start = source.index("const SIGN_VERSION")
    end = source.index("// The ping, stamped")
    section = source[start:end]
    executable = os.environ.get("CHROMIUM_PATH") or next(
        (str(p) for p in Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome") if p.is_file()), None)
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=executable) if executable else pw.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no Chromium: {exc}")
        page = browser.new_page()
        # crypto.subtle exists only in a secure context: serve a page from https.
        page.route("https://hyperlink.test/", lambda route: route.fulfill(body="<html></html>", content_type="text/html"))
        page.goto("https://hyperlink.test/")
        result = page.evaluate("""async (code) => {
            const state = { token: 'T1_web_secret' };
            const run = new Function('state', code + '; return signature;');
            const signature = run(state);
            return await signature({ action: 'send', chatSessionId: 'chat1', content: 'héllo',
                                     serverSessionId: 'srv_x', signedAt: Math.floor(Date.now()/1000),
                                     signer: 'key:k1' });
        }""", section)
        browser.close()
    checked = signing.verify(result, credential="T1_web_secret", signer="key:k1", server_session_id="srv_x",
                             action="send", chat_session_id="chat1", content="héllo")
    assert 23 <= len(checked.nonce) <= 26


def test_the_reference_mac_the_ios_tests_pin():
    """ios/Tests/HyperLinkTests/ParsingTests.swift checks the app's MAC
    for these inputs against this value; it must be the server's."""
    sig = signing.sign("T1_secret", signer="key:k1", chat_session_id="chat1", server_session_id="srv_x",
                       action="send", content="héllo", signed_at=1_800_000_000, nonce="n" * 24)
    assert sig["signature"] == "c68a260d7c57d0465d001074bb025753789c9f4d53524b314fd974a8ac276ea7"


class TestSigningInFromTheApp:
    PASSWORD = "correct horse battery staple 42!"

    @pytest.fixture
    def accounts(self, tmp_path, monkeypatch):
        from conftest import clear_t1_config

        clear_t1_config(monkeypatch)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("T1_ACCOUNTS_ENABLED", "1")
        monkeypatch.setenv("T1_ACCOUNTS_MODE", "local")
        monkeypatch.setenv("T1_ACCOUNTS_REGISTRATION", "open")
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1.db"))
        monkeypatch.setenv("T1_KEYMASTER_DIR", str(tmp_path / "keys"))
        ownership._adopted.clear()
        from hypernix.t1api import create_app

        client = TestClient(create_app(), client=("127.0.0.1", 5000))
        reply = client.post("/accounts/register", json={"username": "mason", "password": self.PASSWORD})
        assert reply.status_code == 200, reply.text
        return client

    def test_two_devices_signed_in_share_the_accounts_chats(self, accounts):
        phone = accounts.post("/hyperlink/login", json={"username": "mason", "password": self.PASSWORD,
                                                        "device_name": "phone"})
        assert phone.status_code == 200, phone.text
        body = phone.json()
        assert body["owner"].startswith("account:") and body["key"]
        headers = {"Authorization": f"Bearer {body['key']}"}
        sid = accounts.post("/hyperlink/sessions", headers=headers, json={"title": "t"}).json()["session"]["session_id"]
        assert accounts.get("/hyperlink/whoami", headers=headers).json()["owner_kind"] == "account"

        tablet = accounts.post("/hyperlink/login", json={"username": "mason", "password": self.PASSWORD}).json()
        assert tablet["key"] != body["key"]
        listed = accounts.get("/hyperlink/sessions", headers={"Authorization": f"Bearer {tablet['key']}"}).json()
        assert sid in [s["session_id"] for s in listed["sessions"]]

    def test_a_wrong_password_mints_nothing(self, accounts):
        assert accounts.post("/hyperlink/login", json={"username": "mason", "password": "nope"}).status_code == 401
        assert accounts.post("/hyperlink/login", json={}).status_code == 422

    def test_no_accounts_is_a_404(self, server):
        _app, client, _key, _ = server
        assert client.post("/hyperlink/login", json={"username": "a", "password": "b"}).status_code == 404
