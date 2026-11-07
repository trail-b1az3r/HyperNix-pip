"""atc.e2e — end-to-end encryption between a client and the T1 API.

TLS protects a connection. It stops protecting it wherever TLS ends: at a
reverse proxy, a Cloudflare tunnel, Tailscale Funnel, a corporate
inspection box. Everything behind that point -- the bearer key, the
prompt, the reply -- is plaintext to whoever runs it. ATC seals the
request and the reply themselves, so they are readable at exactly two
places: the client, and this server's ATC layer.

Protocol ``hypernix-atc/1``
---------------------------
**Handshake** -- ``POST /atc/handshake`` with an X25519 public key. The
server answers with its own (fresh per session), a session id, an
expiry, and an RSA-PSS signature over all of them made with the server's
long-term identity key: the T2C key, whose fingerprint clients already
pin. A client that checks the signature against the pinned fingerprint
knows no one in the middle swapped the keys. Each side derives two keys
with HKDF-SHA256 (salt: the session id): one per direction.

**Requests** -- ``POST /atc/tunnel`` with headers ``X-ATC-Session`` and
``X-ATC-Seq`` and a ChaCha20-Poly1305 ciphertext body. Inside is the
real request: method, path, query, the headers that matter
(``Authorization`` among them) and the body. Nothing about the real
request is visible outside: not the route, not the key, not the size of
the parts. ``X-ATC-Seq`` is a number the client never reuses in a
session; the server keeps a window of the ones it has seen, so a
captured request cannot be replayed.

**Replies** -- a sequence of frames, each ``4-byte length || final flag
|| sealed`` (the flag is in clear for framing and bound into the seal).
Frame 0 carries the status and headers; the rest carry the body as the
application produced it, so streaming answers stream. Each frame is
bound by its associated data to the session, the request's sequence
number, its own index and whether it is the last, so frames cannot be
reordered, dropped from the end, or moved between replies unnoticed.

Requires ``cryptography`` (``pip install 'hypernix[security]'``); without
it the handshake says so and nothing else changes.
"""
from __future__ import annotations

import base64
import json
import secrets
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PROTOCOL", "ALGORITHM", "CONTENT_TYPE", "E2EError", "available",
    "SessionStore", "Session", "RSASigner", "transcript", "verify_handshake",
    "derive_keys", "seal_request", "open_request", "seal_frame", "read_frames",
    "INNER_HEADERS", "REPLY_HEADERS",
]

PROTOCOL = "hypernix-atc/1"
ALGORITHM = "x25519+hkdf-sha256+chacha20poly1305"
CONTENT_TYPE = "application/vnd.hypernix.atc"

#: The inner request headers that are passed on. Anything else a client
#: puts inside is dropped rather than trusted.
INNER_HEADERS = frozenset({
    "authorization", "content-type", "accept", "accept-language", "x-t1-signature",
    "x-hyperlink-device", "idempotency-key", "x-request-id",
})
#: The reply headers sealed into frame 0.
REPLY_HEADERS = frozenset({
    "content-type", "x-request-id", "retry-after", "deprecation", "link", "sunset",
    "cache-control", "content-disposition", "x-response-time-ms",
})

#: How many sequence numbers below the highest seen are still accepted,
#: so concurrent requests may arrive out of order.
REPLAY_WINDOW = 1024


class E2EError(ValueError):
    """A sealed message could not be accepted. ``reason`` is a short code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def available() -> bool:
    try:
        from cryptography.hazmat.primitives.asymmetric import x25519  # noqa: F401
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305  # noqa: F401
    except ImportError:
        return False
    return True


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (ValueError, TypeError) as exc:
        raise E2EError("bad_encoding", "not base64url") from exc


def transcript(session_id: str, client_public: str, server_public: str, expires_at: int) -> bytes:
    """What the server signs: every value both sides derive keys from."""
    return f"{PROTOCOL}|{ALGORITHM}|{session_id}|{client_public}|{server_public}|{int(expires_at)}".encode()


def derive_keys(shared: bytes, session_id: str) -> tuple[bytes, bytes]:
    """``(client-to-server, server-to-client)`` keys from an X25519 secret."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    def one(direction: str) -> bytes:
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=session_id.encode(),
                    info=f"{PROTOCOL} {direction}".encode()).derive(shared)

    return one("c2s"), one("s2c")


def _nonce(seq: int, index: int) -> bytes:
    return index.to_bytes(4, "big") + seq.to_bytes(8, "big")


def _aad(session_id: str, seq: int, direction: str, index: int, final: bool) -> bytes:
    return f"{PROTOCOL}|{session_id}|{seq}|{direction}|{index}|{int(final)}".encode()


def _aead(key: bytes):
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    return ChaCha20Poly1305(key)


# ---------------------------------------------------------------------------
# Requests (client -> server)
# ---------------------------------------------------------------------------

def seal_request(key: bytes, session_id: str, seq: int, *, method: str, path: str,
                 query: str = "", headers: dict[str, str] | None = None, body: bytes = b"") -> bytes:
    head = json.dumps({"m": method.upper(), "p": path, "q": query,
                       "h": {k.lower(): v for k, v in (headers or {}).items()}},
                      separators=(",", ":")).encode()
    return _aead(key).encrypt(_nonce(seq, 0), head + b"\n" + body,
                              _aad(session_id, seq, "c2s", 0, True))


def open_request(key: bytes, session_id: str, seq: int, blob: bytes) -> dict[str, Any]:
    from cryptography.exceptions import InvalidTag

    try:
        plain = _aead(key).decrypt(_nonce(seq, 0), blob, _aad(session_id, seq, "c2s", 0, True))
    except InvalidTag as exc:
        raise E2EError("bad_seal", "the request did not decrypt: wrong key, sequence or a changed byte") from exc
    head, _, body = plain.partition(b"\n")
    try:
        meta = json.loads(head)
    except ValueError as exc:
        raise E2EError("bad_request", "the sealed request has no readable header") from exc
    method, path = str(meta.get("m") or ""), str(meta.get("p") or "")
    if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS") or not path.startswith("/"):
        raise E2EError("bad_request", "the sealed request names no method or path")
    headers = {str(k).lower(): str(v) for k, v in (meta.get("h") or {}).items()
               if str(k).lower() in INNER_HEADERS}
    return {"method": method, "path": path, "query": str(meta.get("q") or ""),
            "headers": headers, "body": body}


# ---------------------------------------------------------------------------
# Replies (server -> client)
# ---------------------------------------------------------------------------

def seal_frame(key: bytes, session_id: str, seq: int, index: int, payload: bytes,
               *, final: bool) -> bytes:
    sealed = _aead(key).encrypt(_nonce(seq, index), payload,
                                _aad(session_id, seq, "s2c", index, final))
    return len(sealed).to_bytes(4, "big") + bytes([1 if final else 0]) + sealed


def read_frames(key: bytes, session_id: str, seq: int, data: bytes) -> tuple[int, dict[str, str], bytes]:
    """A whole sealed reply, opened: ``(status, headers, body)``.

    Refuses a reply that is truncated, reordered or altered.
    """
    from cryptography.exceptions import InvalidTag

    aead = _aead(key)
    offset, index, parts, status, headers, finished = 0, 0, [], 0, {}, False
    while offset < len(data):
        if finished:
            raise E2EError("bad_reply", "data after the final frame")
        if offset + 5 > len(data):
            raise E2EError("truncated", "the reply ends inside a frame header")
        size = int.from_bytes(data[offset:offset + 4], "big")
        final = data[offset + 4] == 1
        sealed = data[offset + 5:offset + 5 + size]
        if len(sealed) != size:
            raise E2EError("truncated", "the reply ends inside a frame")
        offset += 5 + size
        try:
            payload = aead.decrypt(_nonce(seq, index), sealed, _aad(session_id, seq, "s2c", index, final))
        except InvalidTag as exc:
            raise E2EError("bad_seal", f"reply frame {index} did not decrypt") from exc
        if index == 0:
            meta = json.loads(payload)
            status, headers = int(meta["s"]), {str(k): str(v) for k, v in meta.get("h", {}).items()}
        else:
            parts.append(payload)
        index += 1
        finished = final
    if not finished:
        raise E2EError("truncated", "the reply has no final frame")
    return status, headers, b"".join(parts)


def reply_head(status: int, headers: Iterable[tuple[bytes, bytes]]) -> bytes:
    kept = {}
    for name, value in headers:
        lowered = name.decode("latin-1").lower()
        if lowered in REPLY_HEADERS:
            kept[lowered] = value.decode("latin-1")
    return json.dumps({"s": int(status), "h": kept}, separators=(",", ":")).encode()


# ---------------------------------------------------------------------------
# The server's half
# ---------------------------------------------------------------------------

class RSASigner:
    """Signs handshakes with the server's identity key (RSA-PSS, SHA-256)."""

    def __init__(self, private_key_provider: Callable[[], Any], public_pem: Callable[[], str],
                 fingerprint: Callable[[], str]) -> None:
        self._key = private_key_provider
        self.public_pem = public_pem
        self.fingerprint = fingerprint

    def sign(self, data: bytes) -> bytes:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        return self._key().sign(
            data, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256())


def verify_handshake(public_pem: str, answer: dict[str, Any], client_public: str) -> None:
    """Raise :class:`E2EError` unless the server's signature is good."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    key = serialization.load_pem_public_key(public_pem.encode())
    if not isinstance(key, rsa.RSAPublicKey):
        raise E2EError("bad_identity", "the server's identity key is not an RSA key")
    data = transcript(answer["session_id"], client_public, answer["server_public_key"],
                      answer["expires_at"])
    try:
        key.verify(_unb64(answer["signature"]), data,
                   padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.AUTO),
                   hashes.SHA256())
    except InvalidSignature as exc:
        raise E2EError("bad_signature", "the handshake is not signed by this server's key") from exc


@dataclass
class Session:
    session_id: str
    c2s: bytes = field(repr=False)
    s2c: bytes = field(repr=False)
    expires_at: float
    top: int = 0
    seen: set[int] = field(default_factory=set)

    def accept(self, seq: int) -> bool:
        """True the first time *seq* is used, inside the replay window."""
        if seq < 1 or seq <= self.top - REPLAY_WINDOW or seq in self.seen:
            return False
        self.seen.add(seq)
        if seq > self.top:
            self.top = seq
            floor = self.top - REPLAY_WINDOW
            self.seen = {s for s in self.seen if s > floor}
        return True


class SessionStore:
    def __init__(self, signer: RSASigner | None, *, ttl: float = 3600.0, max_sessions: int = 10_000,
                 clock: Callable[[], float] = time.time) -> None:
        self.signer = signer
        self.ttl = ttl
        self.max_sessions = max(1, max_sessions)
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}

    @property
    def ready(self) -> bool:
        return self.signer is not None and available()

    def handshake(self, client_public: str) -> dict[str, Any]:
        if not self.ready:
            raise E2EError("unavailable",
                           "end-to-end encryption needs the cryptography package on the server "
                           "(pip install 'hypernix[security]')")
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import x25519

        raw = _unb64(client_public)
        if len(raw) != 32:
            raise E2EError("bad_key", "public_key must be a 32-byte X25519 key, base64url")
        server_private = x25519.X25519PrivateKey.generate()
        server_public = _b64(server_private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw))
        shared = server_private.exchange(x25519.X25519PublicKey.from_public_bytes(raw))
        session_id = secrets.token_urlsafe(18)
        expires_at = int(self._clock() + self.ttl)
        c2s, s2c = derive_keys(shared, session_id)
        assert self.signer is not None
        signature = self.signer.sign(transcript(session_id, client_public, server_public, expires_at))
        with self._lock:
            self._expire()
            while len(self._sessions) >= self.max_sessions:
                oldest = min(self._sessions.values(), key=lambda s: s.expires_at)
                del self._sessions[oldest.session_id]
            self._sessions[session_id] = Session(session_id, c2s, s2c, expires_at)
        return {
            "protocol": PROTOCOL, "algorithm": ALGORITHM, "session_id": session_id,
            "server_public_key": server_public, "expires_at": expires_at,
            "signature": _b64(signature), "server_fingerprint": self.signer.fingerprint(),
            "server_identity_pem": self.signer.public_pem(),
        }

    def _expire(self) -> None:
        now = self._clock()
        for sid in [sid for sid, s in self._sessions.items() if s.expires_at <= now]:
            del self._sessions[sid]

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.expires_at <= self._clock():
                self._sessions.pop(session_id, None)
                return None
            return session

    def accept(self, session: Session, seq: int) -> bool:
        with self._lock:
            return session.accept(seq)

    def count(self) -> int:
        with self._lock:
            self._expire()
            return len(self._sessions)
