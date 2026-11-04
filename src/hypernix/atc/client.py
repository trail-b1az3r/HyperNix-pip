"""atc.client — talk to a T1 server through its ATC tunnel.

    from hypernix.atc.client import ATCTunnel

    tunnel = ATCTunnel("https://t1.example.com", pin="<server fingerprint>")
    status, headers, body = tunnel.request(
        "GET", "/hyperlink/sessions", headers={"Authorization": "Bearer T2_..."})

The handshake runs on the first request and again when the session
expires. With ``pin`` set, a handshake signed by any other identity key is
refused, which is what makes the encryption end to end rather than
end-to-whoever-answered. Without it the key is trusted on first use and
its fingerprint kept on :attr:`ATCTunnel.fingerprint` for the caller to
store. Only the standard library is used for HTTP.
"""
from __future__ import annotations

import itertools
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from . import e2e

__all__ = ["ATCTunnel"]


class ATCTunnel:
    def __init__(self, base_url: str, *, pin: str | None = None, timeout: float = 60.0,
                 opener: Any = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.pin = pin
        self.timeout = timeout
        self.fingerprint: str | None = None
        self._open = opener or urllib.request.urlopen
        self._lock = threading.Lock()
        self._session: dict[str, Any] | None = None
        self._seq = itertools.count(1)

    def _post(self, path: str, body: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
        req = urllib.request.Request(self.base_url + path, data=body, method="POST", headers=headers)
        try:
            with self._open(req, timeout=self.timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def handshake(self) -> dict[str, Any]:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import x25519

        private = x25519.X25519PrivateKey.generate()
        public = e2e._b64(private.public_key().public_bytes(serialization.Encoding.Raw,
                                                            serialization.PublicFormat.Raw))
        status, raw = self._post("/atc/handshake", json.dumps({"public_key": public}).encode(),
                                 {"Content-Type": "application/json"})
        if status != 200:
            raise e2e.E2EError("handshake_failed", f"the handshake was answered {status}: {raw[:200]!r}")
        answer = json.loads(raw)
        fingerprint = str(answer.get("server_fingerprint") or "")
        if self.pin and fingerprint != self.pin:
            raise e2e.E2EError("pin_mismatch",
                               f"the server identified itself as {fingerprint}, not the pinned {self.pin}")
        e2e.verify_handshake(answer["server_identity_pem"], answer, public)
        if self.fingerprint and fingerprint != self.fingerprint:
            raise e2e.E2EError("pin_mismatch", "the server's identity key changed during this tunnel's life")
        self.fingerprint = fingerprint
        shared = private.exchange(x25519.X25519PublicKey.from_public_bytes(
            e2e._unb64(answer["server_public_key"])))
        c2s, s2c = e2e.derive_keys(shared, answer["session_id"])
        self._session = {"id": answer["session_id"], "c2s": c2s, "s2c": s2c,
                         "expires_at": int(answer["expires_at"])}
        self._seq = itertools.count(1)
        return answer

    def _current(self) -> dict[str, Any]:
        with self._lock:
            if self._session is None or self._session["expires_at"] - 30 <= time.time():
                self.handshake()
            assert self._session is not None
            return self._session

    def request(self, method: str, path: str, *, query: str = "",
                headers: dict[str, str] | None = None, body: bytes | str | dict | None = None,
                ) -> tuple[int, dict[str, str], bytes]:
        """One request through the tunnel: ``(status, headers, body)``."""
        headers = dict(headers or {})
        if isinstance(body, dict):
            body = json.dumps(body).encode()
            headers.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            body = body.encode()
        for attempt in (1, 2):
            session = self._current()
            with self._lock:
                seq = next(self._seq)
            blob = e2e.seal_request(session["c2s"], session["id"], seq, method=method, path=path,
                                    query=query, headers=headers, body=body or b"")
            status, raw = self._post("/atc/tunnel", blob, {
                "Content-Type": e2e.CONTENT_TYPE, "X-ATC-Session": session["id"],
                "X-ATC-Seq": str(seq)})
            if status == 401 and attempt == 1:  # the server forgot the session: shake again
                with self._lock:
                    self._session = None
                continue
            if status != 200:
                raise e2e.E2EError("tunnel_refused", f"the tunnel answered {status}: {raw[:200]!r}")
            return e2e.read_frames(session["s2c"], session["id"], seq, raw)
        raise e2e.E2EError("tunnel_refused", "the server refused two fresh sessions")
