"""atc (0.73.0 pre-alpha 1): the shield, traffic control and the e2e layer.

The parts on their own, before the middleware puts them in front of the
T1 API.
"""
from __future__ import annotations

import pytest

from hypernix.atc import e2e
from hypernix.atc.config import ATCSettings
from hypernix.atc.shield import Shield, is_probe
from hypernix.atc.traffic import Traffic, route_key


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class TestShield:
    @pytest.mark.parametrize("path,query", [
        ("/.env", ""), ("/.git/config", ""), ("/wp-admin/setup.php", ""), ("/phpmyadmin/", ""),
        ("/x/../../etc/passwd", ""), ("/%2e%2e/%2e%2e/etc/passwd", ""), ("/q", "u=${jndi:ldap://x}"),
    ])
    def test_probes_are_recognised(self, path, query):
        assert is_probe(path, query)

    @pytest.mark.parametrize("path", ["/models", "/hyperlink/sessions", "/web/v2/I", "/runner/status"])
    def test_real_routes_are_not(self, path):
        assert not is_probe(path)

    def test_three_probes_jail_and_repeats_double(self):
        clock = Clock()
        shield = Shield(probe_strikes=3, jail_base=60, jail_max=1000, clock=clock)
        for _ in range(3):
            shield.inspect("203.0.113.9", "/.env")
        assert shield.jailed("203.0.113.9") == pytest.approx(60)
        clock.now += 61
        assert shield.jailed("203.0.113.9") == 0
        for _ in range(3):
            shield.inspect("203.0.113.9", "/.git/HEAD")
        assert shield.jailed("203.0.113.9") == pytest.approx(120)

    def test_key_guessing_jails(self):
        shield = Shield(auth_fail_limit=5, clock=Clock())
        for _ in range(6):
            shield.observe("198.51.100.4", 401)
        assert shield.jailed("198.51.100.4") > 0

    def test_this_machine_is_never_jailed_unless_asked(self):
        shield = Shield(probe_strikes=1, clock=Clock())
        shield.inspect("127.0.0.1", "/.env")
        assert shield.jailed("127.0.0.1") == 0
        strict = Shield(probe_strikes=1, jail_loopback=True, clock=Clock())
        strict.inspect("127.0.0.1", "/.env")
        assert strict.jailed("127.0.0.1") > 0

    def test_an_operator_can_release(self):
        shield = Shield(clock=Clock())
        shield.jail("192.0.2.1")
        assert shield.release("192.0.2.1") and shield.jailed("192.0.2.1") == 0

    def test_memory_is_bounded(self):
        shield = Shield(max_clients=16, clock=Clock())
        for i in range(100):
            shield.observe(f"192.0.2.{i}", 404)
        assert shield.snapshot()["tracked_clients"] == 16


class TestTraffic:
    def test_per_client_and_global_limits(self):
        traffic = Traffic(max_inflight=3, per_client_inflight=2)
        assert traffic.enter("a", "/x") and traffic.enter("a", "/x")
        assert not traffic.enter("a", "/x")
        assert traffic.enter("b", "/x")
        assert not traffic.enter("c", "/x")
        assert traffic.enter("c", "/health")  # an operator asking is never turned away
        traffic.leave("c")  # ...but it is in flight until it finishes
        traffic.leave("a")
        assert traffic.enter("c", "/x")

    def test_routes_are_grouped_and_measured(self):
        traffic = Traffic()
        traffic.record("GET", "/hyperlink/sessions/abc/messages", 200, bytes_in=0, bytes_out=10, ms=5)
        traffic.record("GET", "/hyperlink/sessions/def/messages", 404, bytes_in=0, bytes_out=2, ms=7)
        routes = traffic.snapshot()["routes"]
        assert route_key("get", "/hyperlink/sessions/x") == "GET /hyperlink/sessions"
        assert routes["GET /hyperlink/sessions"]["requests"] == 2
        assert routes["GET /hyperlink/sessions"]["statuses"] == {"2xx": 1, "4xx": 1}


class TestSettings:
    def test_defaults(self, monkeypatch):
        for name in ("T1_ATC_ENABLED", "T1_ATC_MAX_BODY_MB", "T1_ATC_E2E_REQUIRED"):
            monkeypatch.delenv(name, raising=False)
        s = ATCSettings()
        assert s.enabled and s.e2e and not s.e2e_required
        assert s.max_body_bytes == 128 * 1024 * 1024

    def test_bad_numbers_keep_the_default(self, monkeypatch):
        monkeypatch.setenv("T1_ATC_MAX_INFLIGHT", "lots")
        assert ATCSettings().max_inflight == 256


crypto = pytest.importorskip("cryptography")


def _signer():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.public_key().public_bytes(serialization.Encoding.PEM,
                                        serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return e2e.RSASigner(lambda: key, lambda: pem, lambda: "fp"), pem


def _client_keys(store):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import x25519

    private = x25519.X25519PrivateKey.generate()
    public = e2e._b64(private.public_key().public_bytes(serialization.Encoding.Raw,
                                                        serialization.PublicFormat.Raw))
    answer = store.handshake(public)
    shared = private.exchange(x25519.X25519PublicKey.from_public_bytes(e2e._unb64(answer["server_public_key"])))
    return public, answer, e2e.derive_keys(shared, answer["session_id"])


class TestEndToEnd:
    def test_a_signed_handshake_both_sides_agree_on(self):
        signer, pem = _signer()
        store = e2e.SessionStore(signer)
        public, answer, (c2s, s2c) = _client_keys(store)
        e2e.verify_handshake(pem, answer, public)
        session = store.get(answer["session_id"])
        assert (session.c2s, session.s2c) == (c2s, s2c)

    def test_a_swapped_server_key_fails_the_signature(self):
        signer, pem = _signer()
        _, answer, _ = _client_keys(e2e.SessionStore(signer))
        public, *_ = _client_keys(e2e.SessionStore(signer))
        with pytest.raises(e2e.E2EError, match="signed"):
            e2e.verify_handshake(pem, answer, public)

    def test_a_request_round_trips_and_hides_everything(self):
        signer, _ = _signer()
        store = e2e.SessionStore(signer)
        _, answer, (c2s, _) = _client_keys(store)
        sid = answer["session_id"]
        blob = e2e.seal_request(c2s, sid, 1, method="post", path="/hyperlink/sessions",
                                headers={"Authorization": "Bearer T2_secret", "X-Evil": "1"},
                                body=b'{"title":"hi"}')
        assert b"T2_secret" not in blob and b"/hyperlink" not in blob
        opened = e2e.open_request(store.get(sid).c2s, sid, 1, blob)
        assert opened["method"] == "POST" and opened["body"] == b'{"title":"hi"}'
        assert opened["headers"] == {"authorization": "Bearer T2_secret"}  # X-Evil dropped
        with pytest.raises(e2e.E2EError):
            e2e.open_request(c2s, sid, 2, blob)  # another sequence number

    def test_replays_are_refused(self):
        session = e2e.Session("s", b"k" * 32, b"k" * 32, expires_at=10**12)
        assert session.accept(5) and session.accept(3)
        assert not session.accept(5) and not session.accept(0)
        assert session.accept(5000) and not session.accept(5000 - e2e.REPLAY_WINDOW)

    def test_reply_frames_stream_and_resist_tampering(self):
        key, sid = b"s" * 32, "sid"
        head = e2e.reply_head(200, [(b"content-type", b"text/event-stream"), (b"set-cookie", b"x")])
        data = (e2e.seal_frame(key, sid, 7, 0, head, final=False)
                + e2e.seal_frame(key, sid, 7, 1, b"data: one\n\n", final=False)
                + e2e.seal_frame(key, sid, 7, 2, b"data: two\n\n", final=True))
        status, headers, body = e2e.read_frames(key, sid, 7, data)
        assert status == 200 and body == b"data: one\n\ndata: two\n\n"
        assert headers == {"content-type": "text/event-stream"}  # set-cookie not carried
        truncated = data[: -len(e2e.seal_frame(key, sid, 7, 2, b"data: two\n\n", final=True))]
        with pytest.raises(e2e.E2EError, match="final"):
            e2e.read_frames(key, sid, 7, truncated)
        flipped = bytearray(data)
        flipped[-1] ^= 1
        with pytest.raises(e2e.E2EError):
            e2e.read_frames(key, sid, 7, bytes(flipped))

    def test_sessions_expire(self):
        clock = Clock()
        signer, _ = _signer()
        store = e2e.SessionStore(signer, ttl=10, clock=clock)
        _, answer, _ = _client_keys(store)
        clock.now += 11
        assert store.get(answer["session_id"]) is None


def test_a_local_proxy_is_held_only_to_the_server_wide_limit():
    import asyncio

    from hypernix.atc import ATC
    from hypernix.atc.middleware import ATCMiddleware

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    atc = ATC(ATCSettings(client_inflight=1, e2e=False))
    entered = []
    real_enter = atc.traffic.enter
    atc.traffic.enter = lambda client, path: entered.append(client) or real_enter(client, path)
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b""}

    scope = {"type": "http", "method": "GET", "path": "/models", "headers": [],
             "client": ("127.0.0.1", 1)}
    asyncio.run(ATCMiddleware(app, atc=atc)(scope, receive, send))
    assert entered == [""] and sent[0]["status"] == 200
