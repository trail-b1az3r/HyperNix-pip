"""T1 v1.0.26.9.2.3 — sign up for a key without already having one.

The T1 API authenticates with T1 keys, and the only way to get one was
for somebody who already had an admin key to mint it. That is right for a
hosted service with a billing relationship and absurd for one person on
their own machine wanting to point a client at their own server.

These tests are weighted towards the security properties, because an
account system that mints API keys is the part of this package where a
mistake costs the most. In rough order of how bad the failure would be:

* A route that works when accounts are off.
* A session that survives a password change.
* A key minted with admin scope by anyone who can sign up.
* A CSRF-able mint — somebody else's page minting a key in your name.
* Username enumeration by timing or by error message.
* A Secure cookie over http://, which produces a login that appears to
  work and forgets you on the next page.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi", reason="the T1 API needs its server extra")

from fastapi.testclient import TestClient  # noqa: E402

from hypernix.t1api.accounts import (  # noqa: E402
    PASSWORD_MIN_LENGTH,
    AccountError,
    AccountStore,
    hash_password,
    needs_rehash,
    verify_password,
)
from hypernix.t1api.db import make_backend  # noqa: E402
from hypernix.t1api.webauth import (  # noqa: E402
    CSRF_HEADER,
    MODES,
    SESSION_COOKIE,
    WebAuthError,
    WebAuthSettings,
    client_ip,
    cookie_kwargs,
    is_origin_allowed,
    resolve_mode,
    validate,
)

GOOD_PASSWORD = "a-perfectly-fine-passphrase"


@pytest.fixture
def store(tmp_path):
    backend = make_backend(db_path=str(tmp_path / "t1.db"), database_url=None)
    return AccountStore(backend, registration="open")


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    def test_a_password_verifies(self):
        assert verify_password(GOOD_PASSWORD, hash_password(GOOD_PASSWORD))

    def test_a_wrong_password_does_not(self):
        assert not verify_password("something else", hash_password(GOOD_PASSWORD))

    def test_the_same_password_hashes_differently_every_time(self):
        """Per-account salt. Without it, two people with the same password
        have the same hash, and one leaked hash is two accounts."""
        assert hash_password(GOOD_PASSWORD) != hash_password(GOOD_PASSWORD)

    def test_the_plaintext_is_not_in_the_hash(self):
        assert GOOD_PASSWORD not in hash_password(GOOD_PASSWORD)

    def test_the_hash_says_what_produced_it(self):
        """Cost parameters live in the string so raising them later does
        not invalidate every existing password."""
        scheme, n, r, p, salt, digest = hash_password(GOOD_PASSWORD).split("$")
        assert scheme == "scrypt"
        assert int(n) >= 1 << 14
        assert int(r) == 8 and int(p) == 1
        assert len(bytes.fromhex(salt)) == 16
        assert len(bytes.fromhex(digest)) == 32

    def test_a_malformed_hash_fails_closed(self):
        """A corrupt record should not be distinguishable from a wrong
        password by whoever is guessing, and should never raise."""
        for broken in ("", "nonsense", "scrypt$x$8$1$zz$zz", "bcrypt$1$2$3$4$5"):
            assert verify_password(GOOD_PASSWORD, broken) is False

    def test_an_empty_password_never_verifies(self):
        assert not verify_password("", hash_password(GOOD_PASSWORD))
        assert not verify_password(GOOD_PASSWORD, "")

    def test_an_old_cost_is_flagged_for_rehash(self):
        cheap = hash_password(GOOD_PASSWORD, n=1 << 14)
        assert needs_rehash(cheap)
        assert not needs_rehash(hash_password(GOOD_PASSWORD))
        # And it still verifies, which is the point of storing the cost.
        assert verify_password(GOOD_PASSWORD, cheap)


# ---------------------------------------------------------------------------
# Accounts
# ---------------------------------------------------------------------------


class TestCreatingAccounts:
    def test_the_first_account_is_an_admin(self, store):
        assert store.create("first", GOOD_PASSWORD).is_admin

    def test_the_second_is_not(self, store):
        store.create("first", GOOD_PASSWORD)
        assert not store.create("second", GOOD_PASSWORD).is_admin

    def test_a_username_is_normalised(self, store):
        assert store.create("  MaSoN  ", GOOD_PASSWORD).username == "mason"

    def test_a_taken_username_is_refused(self, store):
        store.create("mason", GOOD_PASSWORD)
        with pytest.raises(AccountError, match="taken"):
            store.create("MASON", GOOD_PASSWORD)

    @pytest.mark.parametrize("bad", ["", "a", "-leading", "has space",
                                     "x" * 40, "emoji😀", "has/slash"])
    def test_a_bad_username_is_refused(self, store, bad):
        with pytest.raises(AccountError):
            store.create(bad, GOOD_PASSWORD)

    def test_case_is_normalised_rather_than_refused(self, store):
        """"UPPER" is not a bad username, it is "upper" spelled loudly.
        Refusing it would be a rule with no reason behind it; folding it
        is what stops "Mason" and "mason" being two accounts."""
        assert store.create("UPPER", GOOD_PASSWORD).username == "upper"

    def test_a_short_password_is_refused(self, store):
        with pytest.raises(AccountError, match=str(PASSWORD_MIN_LENGTH)):
            store.create("mason", "short")

    def test_an_enormous_password_is_refused(self, store):
        """Not a strength rule — scrypt's cost does not depend on input
        length. It is work this server would do on an unauthenticated
        request."""
        with pytest.raises(AccountError, match="1024"):
            store.create("mason", "x" * 5000)

    def test_the_hash_is_never_serialised(self, store):
        """At any level. There is no `include_private` that returns it."""
        account = store.create("mason", GOOD_PASSWORD)
        for payload in (account.to_dict(), account.to_dict(include_private=True)):
            assert "password_hash" not in payload
            assert GOOD_PASSWORD not in str(payload)


class TestRegistrationPolicy:
    def _store(self, tmp_path, **kwargs):
        backend = make_backend(db_path=str(tmp_path / "t1.db"), database_url=None)
        return AccountStore(backend, **kwargs)

    def test_closed_refuses_everyone(self, tmp_path):
        store = self._store(tmp_path, registration="closed")
        with pytest.raises(AccountError, match="not accepting"):
            store.create("mason", GOOD_PASSWORD)

    def test_closed_is_the_default(self, tmp_path):
        backend = make_backend(db_path=str(tmp_path / "t1.db"), database_url=None)
        assert AccountStore(backend).registration == "closed"

    def test_first_user_allows_exactly_one(self, tmp_path):
        store = self._store(tmp_path, registration="first-user")
        store.create("mason", GOOD_PASSWORD)
        with pytest.raises(AccountError, match="already has it"):
            store.create("second", GOOD_PASSWORD)

    def test_invite_needs_the_code(self, tmp_path):
        store = self._store(
            tmp_path, registration="invite", invite_codes=("sesame",)
        )
        with pytest.raises(AccountError, match="not valid"):
            store.create("mason", GOOD_PASSWORD, invite_code="wrong")
        assert store.create("mason", GOOD_PASSWORD, invite_code="sesame")

    def test_invite_with_no_codes_lets_nobody_in(self, tmp_path):
        """Rather than letting everybody in, which is the other way a
        missing config could go."""
        store = self._store(tmp_path, registration="invite")
        allowed, reason = store.can_register("anything")
        assert not allowed
        assert "none configured" in reason

    def test_the_cli_path_bypasses_the_policy(self, tmp_path):
        """Shell access outranks a registration policy, and needing to
        open registration to create the first account would be theatre."""
        store = self._store(tmp_path, registration="closed")
        assert store.create(
            "mason", GOOD_PASSWORD, bypass_registration_check=True
        )

    def test_an_unknown_mode_is_refused_at_construction(self, tmp_path):
        with pytest.raises(AccountError, match="Unknown registration mode"):
            self._store(tmp_path, registration="maybe")


class TestAuthentication:
    def test_the_right_password_works(self, store):
        store.create("mason", GOOD_PASSWORD)
        assert store.authenticate("mason", GOOD_PASSWORD).username == "mason"

    def test_the_wrong_one_does_not(self, store):
        store.create("mason", GOOD_PASSWORD)
        with pytest.raises(AccountError):
            store.authenticate("mason", "wrong")

    def test_a_missing_user_and_a_wrong_password_say_the_same_thing(self, store):
        """Telling them apart hands an attacker a list of which usernames
        exist, which is the first thing they want and the cheapest thing
        to withhold."""
        store.create("mason", GOOD_PASSWORD)
        messages = set()
        for username, password in (("mason", "wrong"), ("nobody", "wrong")):
            with pytest.raises(AccountError) as caught:
                store.authenticate(username, password)
            messages.add(caught.value.message)
        assert len(messages) == 1

    def test_a_missing_user_is_not_faster_to_reject(self, store):
        """Returning early would be the same enumeration leak by a
        different route: an attacker times the response instead of
        reading it."""
        store.create("mason", GOOD_PASSWORD)

        def _timed(username):
            best = float("inf")
            for _ in range(3):
                started = time.perf_counter()
                with pytest.raises(AccountError):
                    store.authenticate(username, "wrong-password-here")
                best = min(best, time.perf_counter() - started)
            return best

        real, missing = _timed("mason"), _timed("nobody")
        assert max(real, missing) / min(real, missing) < 3.0, (real, missing)

    def test_too_many_failures_locks_the_account(self, store):
        from hypernix.t1api.accounts import MAX_FAILED_LOGINS

        store.create("mason", GOOD_PASSWORD)
        for _ in range(MAX_FAILED_LOGINS):
            with pytest.raises(AccountError):
                store.authenticate("mason", "wrong")
        # Now even the right password is refused, and says why.
        with pytest.raises(AccountError, match="Too many failed"):
            store.authenticate("mason", GOOD_PASSWORD)

    def test_a_success_clears_the_counter(self, store):
        store.create("mason", GOOD_PASSWORD)
        for _ in range(3):
            with pytest.raises(AccountError):
                store.authenticate("mason", "wrong")
        store.authenticate("mason", GOOD_PASSWORD)
        assert store.by_username("mason").failed_logins == 0

    def test_a_disabled_account_cannot_sign_in(self, store):
        account = store.create("mason", GOOD_PASSWORD)
        store.set_active(account.account_id, False)
        with pytest.raises(AccountError, match="disabled"):
            store.authenticate("mason", GOOD_PASSWORD)

    def test_an_old_hash_is_upgraded_on_login(self, store):
        """The only moment the plaintext is ever in hand. A migration
        that logs everyone out is a migration nobody runs."""
        account = store.create("mason", GOOD_PASSWORD)
        store._update(
            account.account_id,
            {"password_hash": hash_password(GOOD_PASSWORD, n=1 << 14)},
        )
        assert needs_rehash(store.by_username("mason").password_hash)
        store.authenticate("mason", GOOD_PASSWORD)
        assert not needs_rehash(store.by_username("mason").password_hash)


class TestSessions:
    def test_a_session_resolves_back(self, store):
        account = store.create("mason", GOOD_PASSWORD)
        session, token = store.start_session(account)
        found = store.resolve_session(token)
        assert found is not None
        assert found[1].username == "mason"
        assert found[0].session_id == session.session_id

    def test_only_the_hash_is_stored(self, store):
        """A copy of the session table should not be a set of usable
        cookies."""
        account = store.create("mason", GOOD_PASSWORD)
        session, token = store.start_session(account)
        assert token not in session.token_hash
        with store.backend.connect() as conn:
            rows = conn.execute("SELECT token_hash FROM t1_sessions").fetchall()
        assert all(token not in str(row[0]) for row in rows)

    def test_a_bad_token_resolves_to_nothing(self, store):
        store.create("mason", GOOD_PASSWORD)
        assert store.resolve_session("not-a-token") is None
        assert store.resolve_session("") is None

    def test_an_expired_session_is_gone(self, tmp_path):
        backend = make_backend(db_path=str(tmp_path / "t1.db"), database_url=None)
        store = AccountStore(backend, registration="open", session_ttl=-1)
        account = store.create("mason", GOOD_PASSWORD)
        _session, token = store.start_session(account)
        assert store.resolve_session(token) is None

    def test_a_session_is_bound_to_its_address(self, store):
        """A cookie that starts arriving from somewhere else is the shape
        of a stolen cookie."""
        account = store.create("mason", GOOD_PASSWORD)
        _session, token = store.start_session(account, client_ip="10.0.0.1")
        assert store.resolve_session(token, client_ip="10.0.0.1") is not None
        assert store.resolve_session(token, client_ip="10.0.0.9") is None
        # And it is ended, not merely refused for that one request.
        assert store.resolve_session(token, client_ip="10.0.0.1") is None

    def test_binding_can_be_turned_off(self, tmp_path):
        backend = make_backend(db_path=str(tmp_path / "t1.db"), database_url=None)
        store = AccountStore(
            backend, registration="open", bind_sessions_to_ip=False
        )
        account = store.create("mason", GOOD_PASSWORD)
        _session, token = store.start_session(account, client_ip="10.0.0.1")
        assert store.resolve_session(token, client_ip="10.0.0.9") is not None

    def test_a_password_change_ends_every_session(self, store):
        """On every device. A password change that left a stolen session
        alive would not be a password change."""
        account = store.create("mason", GOOD_PASSWORD)
        _s1, token1 = store.start_session(account)
        _s2, token2 = store.start_session(account)
        store.set_password(account.account_id, "another-fine-passphrase")
        assert store.resolve_session(token1) is None
        assert store.resolve_session(token2) is None

    def test_disabling_an_account_ends_its_sessions(self, store):
        account = store.create("mason", GOOD_PASSWORD)
        _session, token = store.start_session(account)
        store.set_active(account.account_id, False)
        assert store.resolve_session(token) is None

    def test_changing_a_password_needs_the_old_one(self, store):
        account = store.create("mason", GOOD_PASSWORD)
        with pytest.raises(AccountError, match="current password"):
            store.change_password(account.account_id, "wrong", "a-new-passphrase")

    def test_csrf_tokens_are_per_session_and_compared_exactly(self, store):
        account = store.create("mason", GOOD_PASSWORD)
        first, _t1 = store.start_session(account)
        second, _t2 = store.start_session(account)
        assert first.csrf_token != second.csrf_token
        assert store.check_csrf(first, first.csrf_token)
        assert not store.check_csrf(first, second.csrf_token)
        assert not store.check_csrf(first, "")
        assert not store.check_csrf(first, first.csrf_token[:-1])

    def test_expired_sessions_can_be_purged(self, tmp_path):
        backend = make_backend(db_path=str(tmp_path / "t1.db"), database_url=None)
        store = AccountStore(backend, registration="open", session_ttl=-1)
        account = store.create("mason", GOOD_PASSWORD)
        store.start_session(account)
        assert store.purge_expired_sessions() == 1


# ---------------------------------------------------------------------------
# Deployment modes
# ---------------------------------------------------------------------------


class TestWebAuthModes:
    def test_all_four_exist(self):
        assert set(MODES) == {"local", "tailscale", "site", "cloudflare"}

    def test_an_unknown_mode_lists_the_real_ones(self):
        with pytest.raises(WebAuthError, match="local"):
            resolve_mode("carrier-pigeon")

    @pytest.mark.parametrize("name", ["local", "tailscale"])
    def test_the_loopback_modes_do_not_set_secure(self, name):
        """A Secure cookie is not sent over plain HTTP, so setting it
        would produce a login that appears to work and forgets you on the
        next page."""
        assert not MODES[name].secure_cookies
        assert not cookie_kwargs(MODES[name], max_age=60)["secure"]

    @pytest.mark.parametrize("name", ["site", "cloudflare"])
    def test_the_network_modes_do(self, name):
        assert MODES[name].secure_cookies
        assert cookie_kwargs(MODES[name], max_age=60)["secure"]

    @pytest.mark.parametrize("name", list(MODES))
    def test_the_cookie_is_always_httponly(self, name):
        """Not configurable. It is the difference between an XSS that
        defaces a page and one that walks away with the session."""
        assert cookie_kwargs(MODES[name], max_age=60)["httponly"]

    def test_loopback_gets_the_stricter_samesite(self):
        assert MODES["local"].same_site == "strict"

    @pytest.mark.parametrize("name", ["local", "tailscale"])
    def test_a_forwarded_header_is_ignored_without_a_proxy(self, name):
        """Otherwise a client could tell a loopback deployment "my address
        is 10.0.0.1" — the address its own session binding and rate
        limits are keyed on."""
        headers = {"X-Forwarded-For": "1.2.3.4", "CF-Connecting-IP": "1.2.3.4"}
        assert client_ip(headers, "127.0.0.1", MODES[name]) == "127.0.0.1"

    def test_cloudflare_reads_its_own_header(self):
        headers = {"CF-Connecting-IP": "203.0.113.9"}
        assert client_ip(headers, "10.0.0.1", MODES["cloudflare"]) == "203.0.113.9"

    def test_a_proxy_header_takes_the_first_entry(self):
        headers = {"X-Forwarded-For": "203.0.113.9, 10.0.0.1, 10.0.0.2"}
        assert client_ip(headers, "10.0.0.2", MODES["site"]) == "203.0.113.9"

    def test_a_junk_forwarded_header_falls_back_to_the_socket(self):
        """A proxy that sets this sets an address. Anything else should
        not become a session binding."""
        headers = {"CF-Connecting-IP": "not-an-ip; drop table"}
        assert client_ip(headers, "10.0.0.1", MODES["cloudflare"]) == "10.0.0.1"


class TestConfigurationRefusals:
    """Each of these produces a working-looking server with a hole or a
    dead login in it, which is why they are refused at startup."""

    def test_a_network_mode_needs_a_public_url(self):
        with pytest.raises(WebAuthError, match="PUBLIC_URL"):
            validate(WebAuthSettings(mode=MODES["site"], enabled=True))

    def test_secure_cookies_over_http_are_refused(self):
        with pytest.raises(WebAuthError, match="never sent over http"):
            validate(WebAuthSettings(
                mode=MODES["cloudflare"],
                public_url="http://t1.example.com",
                enabled=True,
            ))

    def test_invite_mode_with_no_codes_is_refused(self):
        with pytest.raises(WebAuthError, match="nobody can register"):
            validate(WebAuthSettings(
                mode=MODES["local"], registration="invite", enabled=True
            ))

    def test_a_valid_local_configuration_passes(self):
        validate(WebAuthSettings(mode=MODES["local"], enabled=True))

    def test_open_registration_on_the_internet_warns(self, caplog):
        """A warning, not a refusal — somebody running a community server
        may mean it. But it is far more often a local default that
        followed the server outside."""
        import logging

        with caplog.at_level(logging.WARNING):
            validate(WebAuthSettings(
                mode=MODES["site"], public_url="https://t1.example.com",
                registration="open", enabled=True,
            ))
        assert any("OPEN" in record.message for record in caplog.records)

    def test_invite_codes_never_reach_the_wire(self):
        settings = WebAuthSettings(
            mode=MODES["local"], registration="invite",
            invite_codes=("sesame",), enabled=True,
        )
        assert "sesame" not in str(settings.to_dict())

    def test_origins_are_checked_when_configured(self):
        settings = WebAuthSettings(
            mode=MODES["site"], public_url="https://t1.example.com", enabled=True
        )
        assert is_origin_allowed("https://t1.example.com", settings)
        assert is_origin_allowed("https://t1.example.com/", settings)
        assert not is_origin_allowed("https://evil.example.com", settings)
        # A missing Origin passes: browsers omit it on some same-origin
        # posts, and the CSRF token is what actually stops the attack.
        assert is_origin_allowed("", settings)


# ---------------------------------------------------------------------------
# The HTTP surface
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A T1 app with accounts on, in local mode, open registration."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("T1_ACCOUNTS_ENABLED", "1")
    monkeypatch.setenv("T1_ACCOUNTS_MODE", "local")
    monkeypatch.setenv("T1_ACCOUNTS_REGISTRATION", "open")
    monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1.db"))
    monkeypatch.setenv("T1_KEYMASTER_DIR", str(tmp_path / "keys"))
    from hypernix.t1api import create_app

    return TestClient(create_app())


@pytest.fixture
def signed_in(client):
    """A registered, signed-in client and its CSRF token."""
    reply = client.post(
        "/accounts/register",
        json={"username": "mason", "password": GOOD_PASSWORD},
    )
    assert reply.status_code == 200, reply.text
    return client, reply.json()["csrf_token"]


class TestAccountsAreOffByDefault:
    def test_every_route_404s_when_disabled(self, tmp_path, monkeypatch):
        """404, not 403. A deployment without accounts should not
        advertise that it could have them."""
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("T1_ACCOUNTS_ENABLED", raising=False)
        monkeypatch.setenv("T1_DB_PATH", str(tmp_path / "t1.db"))
        from hypernix.t1api import create_app

        off = TestClient(create_app())
        for method, path in (
            ("get", "/accounts/config"), ("get", "/accounts/me"),
            ("get", "/accounts/login"), ("get", "/accounts/"),
            ("post", "/accounts/register"), ("post", "/accounts/login"),
            ("post", "/accounts/keys"),
        ):
            # TestClient.get takes no json=, so only the posts carry one.
            reply = (
                off.post(path, json={}) if method == "post" else off.get(path)
            )
            assert reply.status_code == 404, (path, reply.status_code)


class TestRegisteringOverHttp:
    def test_no_api_key_is_needed(self, client):
        """The whole point of the module."""
        reply = client.post(
            "/accounts/register",
            json={"username": "mason", "password": GOOD_PASSWORD},
        )
        assert reply.status_code == 200
        assert reply.json()["account"]["username"] == "mason"

    def test_it_sets_a_session_cookie(self, client):
        client.post(
            "/accounts/register",
            json={"username": "mason", "password": GOOD_PASSWORD},
        )
        assert SESSION_COOKIE in client.cookies

    def test_a_weak_password_is_refused(self, client):
        reply = client.post(
            "/accounts/register", json={"username": "mason", "password": "abc"}
        )
        assert reply.status_code >= 400

    def test_the_password_is_never_echoed(self, client):
        reply = client.post(
            "/accounts/register",
            json={"username": "mason", "password": GOOD_PASSWORD},
        )
        assert GOOD_PASSWORD not in reply.text


class TestMintingKeys:
    def test_a_signed_in_account_can_mint_one(self, signed_in):
        client, csrf = signed_in
        reply = client.post(
            "/accounts/keys", json={}, headers={CSRF_HEADER: csrf}
        )
        assert reply.status_code == 200
        assert reply.json()["key"].startswith("T1_")

    def test_the_minted_key_actually_authenticates(self, signed_in):
        """The payoff. A key that came out of the web flow has to be the
        same kind of key as one an admin minted."""
        client, csrf = signed_in
        key = client.post(
            "/accounts/keys", json={}, headers={CSRF_HEADER: csrf}
        ).json()["key"]
        reply = client.post(
            "/auth/t1/validate", json={"key": key},
            headers={"Authorization": f"Bearer {key}"},
        )
        assert reply.status_code == 200
        assert reply.json()["active"]

    def test_a_self_service_key_is_never_admin(self, signed_in):
        """An account can be created by anyone the registration policy
        lets in. "Anyone who can sign up can mint an admin key" would
        make that policy the only thing between a stranger and the whole
        API."""
        client, csrf = signed_in
        payload = client.post(
            "/accounts/keys", json={}, headers={CSRF_HEADER: csrf}
        ).json()
        assert "admin" not in payload["scopes"]
        assert set(payload["scopes"]) == {"read", "write"}

    def test_minting_needs_the_csrf_header(self, signed_in):
        """Without it, any page on the internet could mint a key in the
        signed-in user's name."""
        client, _csrf = signed_in
        assert client.post("/accounts/keys", json={}).status_code >= 400

    def test_a_wrong_csrf_token_is_refused(self, signed_in):
        client, _csrf = signed_in
        reply = client.post(
            "/accounts/keys", json={}, headers={CSRF_HEADER: "nope"}
        )
        assert reply.status_code >= 400

    def test_the_key_is_shown_once_and_not_listed(self, signed_in):
        """It is stored encrypted in the Keymaster and nothing here can
        read it back."""
        client, csrf = signed_in
        key = client.post(
            "/accounts/keys", json={}, headers={CSRF_HEADER: csrf}
        ).json()["key"]
        listing = client.get("/accounts/keys")
        assert listing.status_code == 200
        assert listing.json()["count"] == 1
        assert key not in listing.text

    def test_an_account_can_revoke_its_own_key(self, signed_in):
        client, csrf = signed_in
        key_id = client.post(
            "/accounts/keys", json={}, headers={CSRF_HEADER: csrf}
        ).json()["key_id"]
        reply = client.delete(
            f"/accounts/keys/{key_id}", headers={CSRF_HEADER: csrf}
        )
        assert reply.status_code == 200
        assert client.get("/accounts/keys").json()["count"] == 0

    def test_it_cannot_revoke_a_key_that_is_not_its_own(self, signed_in):
        """And the refusal does not say whether that key id exists."""
        client, csrf = signed_in
        reply = client.delete(
            "/accounts/keys/somebody-elses-key", headers={CSRF_HEADER: csrf}
        )
        assert reply.status_code == 404
        assert "does not belong" in reply.text


class TestSessionsOverHttp:
    def test_signing_out_ends_the_session(self, signed_in):
        client, _csrf = signed_in
        assert client.post("/accounts/logout").status_code == 200
        assert client.get("/accounts/me").status_code >= 400

    def test_me_needs_a_session(self, client):
        assert client.get("/accounts/me").status_code >= 400

    def test_login_works_after_logout(self, signed_in):
        client, _csrf = signed_in
        client.post("/accounts/logout")
        reply = client.post(
            "/accounts/login",
            json={"username": "mason", "password": GOOD_PASSWORD},
        )
        assert reply.status_code == 200
        assert client.get("/accounts/me").json()["account"]["username"] == "mason"

    def test_a_wrong_password_over_http_is_refused(self, signed_in):
        client, _csrf = signed_in
        client.post("/accounts/logout")
        reply = client.post(
            "/accounts/login", json={"username": "mason", "password": "wrong"}
        )
        assert reply.status_code >= 400

    def test_changing_the_password_signs_everything_out(self, signed_in):
        client, csrf = signed_in
        reply = client.post(
            "/accounts/password",
            json={
                "current_password": GOOD_PASSWORD,
                "new_password": "a-different-long-passphrase",
            },
            headers={CSRF_HEADER: csrf},
        )
        assert reply.status_code == 200
        assert client.get("/accounts/me").status_code >= 400


class TestThePages:
    def test_the_login_page_renders(self, client):
        reply = client.get("/accounts/login")
        assert reply.status_code == 200
        assert "text/html" in reply.headers["content-type"]
        assert "Sign in" in reply.text

    def test_the_register_page_renders_when_open(self, client):
        reply = client.get("/accounts/register")
        assert reply.status_code == 200
        assert "Create an account" in reply.text

    def test_the_account_page_renders(self, client):
        assert client.get("/accounts/").status_code == 200

    def test_the_config_endpoint_leaks_nothing(self, client):
        payload = client.get("/accounts/config").json()
        assert payload["enabled"]
        assert payload["mode"] == "local"
        assert "invite_codes" not in payload
        assert "password" not in str(payload).lower()


class TestTheCli:
    def test_modes_explains_all_four(self, capsys):
        from hypernix.t1api.accounts_cli import main

        assert main(["modes"]) == 0
        out = capsys.readouterr().out
        for name in MODES:
            assert name in out

    def test_list_on_an_empty_store(self, tmp_path, monkeypatch, capsys):
        from hypernix.t1api.accounts_cli import main

        monkeypatch.setenv("HOME", str(tmp_path))
        assert main(["--db", str(tmp_path / "t1.db"), "list"]) == 0
        assert "No accounts" in capsys.readouterr().out

    def test_a_password_is_never_a_command_line_argument(self):
        """It would be in the shell history, in ps, and in whatever
        collects either."""
        import inspect

        from hypernix.t1api import accounts_cli

        source = inspect.getsource(accounts_cli)
        assert "getpass" in source
        assert 'add_argument("--password"' not in source
        assert "add_argument('--password'" not in source
