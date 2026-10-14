"""Signed, timestamped HyperLink prompts (0.72.8).

When Send is pressed the app first *pings* the server -- ``POST
/hyperlink/ping``, answered without going near a model -- with the time
the button was pressed. If that round trip does not finish within four
seconds the app says the server is not answering, instead of waiting on
a request that may never come back; once the ping is answered, the
prompt itself has no timeout, because a model can take as long as it
takes. An empty request is not a ping and is refused.

Every send, resend and edit is then signed. What is signed, one field
per line::

    hyperlink-sign/v1
    signer              account:<account id>, or key:<key id> (T1, T2 or T2.1)
    signed_at           Unix time in seconds, when Send was pressed
    nonce               23 to 26 random characters, A-Z a-z 0-9 - _
    chat_session_id     the chat (or code) session
    server_session_id   this server process's id, from the ping
    action              ping | send | resend | edit
    content_sha256      hex SHA-256 of the message text ("" for a resend)

The key is SHA-256 of ``hyperlink-sign/v1:`` and the credential the
request carries -- the T1/T2 key or the paired device's token -- so the
server, which receives the credential anyway, can check the signature
and nothing new has to be shared. A signature names who sent it, when,
for which chat on which running server, and cannot be replayed: the
nonce is remembered, the time must be within five minutes, and a server
that restarted has a new session id, so the app pings again.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import time
from dataclasses import dataclass

__all__ = [
    "ACTIONS",
    "MAX_SKEW_SECONDS",
    "NONCE",
    "Signature",
    "SignatureError",
    "NonceCache",
    "canonical",
    "content_hash",
    "new_nonce",
    "new_server_session_id",
    "sign",
    "signing_key",
    "verify",
]

VERSION = "hyperlink-sign/v1"
ACTIONS = ("ping", "send", "resend", "edit")
MAX_SKEW_SECONDS = 300
NONCE = re.compile(r"^[A-Za-z0-9_-]{23,26}$")
_NONCE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


class SignatureError(ValueError):
    """A signature that does not check out; ``code`` says which way."""

    def __init__(self, message: str, code: str = "invalid") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Signature:
    signer: str
    signed_at: int
    nonce: str
    chat_session_id: str
    server_session_id: str
    action: str
    content_sha256: str
    signature: str

    @classmethod
    def from_dict(cls, data: dict) -> Signature:
        try:
            return cls(
                signer=str(data["signer"]), signed_at=int(data["signed_at"]), nonce=str(data["nonce"]),
                chat_session_id=str(data.get("chat_session_id") or ""),
                server_session_id=str(data.get("server_session_id") or ""),
                action=str(data["action"]), content_sha256=str(data.get("content_sha256") or ""),
                signature=str(data["signature"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SignatureError(f"the signature is missing a field: {exc}", "malformed") from exc


def new_nonce(length: int = 24) -> str:
    if not 23 <= length <= 26:
        raise ValueError("a nonce is 23 to 26 characters")
    return "".join(secrets.choice(_NONCE_CHARS) for _ in range(length))


def new_server_session_id() -> str:
    return "srv_" + secrets.token_urlsafe(18)


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest() if text else ""


def signing_key(credential: str) -> bytes:
    return hashlib.sha256(f"{VERSION}:{credential}".encode()).digest()


def canonical(sig: Signature | dict) -> str:
    s = sig if isinstance(sig, Signature) else Signature.from_dict({**sig, "signature": ""})
    return "\n".join([VERSION, s.signer, str(s.signed_at), s.nonce, s.chat_session_id,
                      s.server_session_id, s.action, s.content_sha256])


def sign(credential: str, *, signer: str, chat_session_id: str = "", server_session_id: str = "",
         action: str = "send", content: str = "", signed_at: int | None = None,
         nonce: str | None = None) -> dict:
    """What a client sends: every field, and the signature over them."""
    fields = {
        "signer": signer, "signed_at": int(signed_at if signed_at is not None else time.time()),
        "nonce": nonce or new_nonce(), "chat_session_id": chat_session_id,
        "server_session_id": server_session_id, "action": action,
        "content_sha256": content_hash(content),
    }
    mac = hmac.new(signing_key(credential), canonical(fields).encode(), hashlib.sha256).hexdigest()
    return {**fields, "signature": mac}


class NonceCache:
    """Nonces seen within the skew window, so a signature works once."""

    def __init__(self, ttl: float = MAX_SKEW_SECONDS * 2) -> None:
        self.ttl = ttl
        self._seen: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def claim(self, signer: str, nonce: str, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            if len(self._seen) > 50_000:
                self._seen = {k: t for k, t in self._seen.items() if now - t < self.ttl}
            key = (signer, nonce)
            seen = self._seen.get(key)
            if seen is not None and now - seen < self.ttl:
                return False
            self._seen[key] = now
            return True


def verify(data: dict, *, credential: str, signer: str, server_session_id: str,
           action: str, chat_session_id: str = "", content: str | None = None,
           nonces: NonceCache | None = None, now: float | None = None,
           allow_no_server_session: bool = False) -> Signature:
    """Check *data* against what the server knows. Raises SignatureError."""
    sig = Signature.from_dict(data)
    now = time.time() if now is None else now
    if sig.action != action:
        raise SignatureError(f"signed for {sig.action!r}, used for {action!r}", "action")
    if sig.signer != signer:
        raise SignatureError("signed as someone other than the sender of this request", "signer")
    if not NONCE.match(sig.nonce):
        raise SignatureError("the nonce must be 23 to 26 characters of A-Z a-z 0-9 - _", "nonce")
    if abs(now - sig.signed_at) > MAX_SKEW_SECONDS:
        raise SignatureError("signed too long ago (or in the future); check the device's clock", "time")
    if sig.server_session_id != server_session_id and not (allow_no_server_session and not sig.server_session_id):
        raise SignatureError("signed for a server session that has ended; ping again", "server_session")
    if chat_session_id and sig.chat_session_id != chat_session_id:
        raise SignatureError("signed for a different chat", "chat_session")
    if content is not None and sig.content_sha256 != content_hash(content):
        raise SignatureError("the message is not the one that was signed", "content")
    expected = hmac.new(signing_key(credential), canonical(sig).encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig.signature):
        raise SignatureError("the signature does not match", "mac")
    if nonces is not None and not nonces.claim(sig.signer, sig.nonce, now):
        raise SignatureError("this signature was already used", "replay")
    return sig
