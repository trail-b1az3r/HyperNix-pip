"""atc.middleware — ATC in front of the T1 API, as one ASGI layer.

It is the outermost layer, so it runs before anything else does:

1. **Jail.** A jailed client is answered 429 with ``Retry-After``.
2. **Probes.** A request for ``/.env``, ``/wp-admin`` and the rest is
   answered 404 without reaching the application, and counts as a strike.
3. **Body size.** A body over ``T1_ATC_MAX_BODY_MB`` is answered 413:
   when ``Content-Length`` says so before it is read, or once the count
   passes the limit while it is read.
4. **Tunnel.** ``POST /atc/tunnel`` is opened (``hypernix-atc/1``, see
   :mod:`.e2e`) and the request inside goes through the rest of this list
   and the application as if it had arrived on its own. Its reply goes
   back sealed, frame by frame, so streaming replies still stream.
5. **E2E required.** With ``T1_ATC_E2E_REQUIRED=1`` a request from
   another machine that did not come through the tunnel is refused, apart
   from the few routes a client needs before it can encrypt.
6. **In flight.** Past the server's or the client's in-flight limit the
   request is answered 503 with ``Retry-After``.
7. **Compression.** A complete JSON or text reply of 1 KiB or more is
   gzipped when the client accepts it. Streams are never buffered.
8. **Afterwards**, the status goes to the shield (401/403/404 counted)
   and the route's statistics.
"""
from __future__ import annotations

import gzip
import json
import time
from typing import Any

from . import e2e
from .config import ATCSettings
from .shield import Shield
from .traffic import Traffic

__all__ = ["ATCMiddleware", "E2E_OPEN_PATHS", "TUNNEL_PATH"]

TUNNEL_PATH = "/atc/tunnel"

#: What a client may ask in the clear when encryption is required: how to
#: start encrypting, and whether the server is there at all.
E2E_OPEN_PATHS = ("/atc/handshake", "/health", "/status", "/server/info",
                  "/auth/t2c/public-key")

_COMPRESS_MIN = 1024
_COMPRESS_MAX = 16 * 1024 * 1024
_COMPRESSIBLE = (b"application/json", b"text/", b"application/javascript", b"image/svg+xml")


class _BodyTooLarge(Exception):
    pass


def _header(scope: dict, name: bytes) -> bytes:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value
    return b""


async def _send_json(send: Any, status: int, code: str, message: str,
                     details: dict[str, Any] | None = None,
                     headers: list[tuple[bytes, bytes]] | None = None) -> None:
    body = json.dumps({"error": {"code": code, "message": message,
                                 "details": details or {}}}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                            *(headers or [])]})
    await send({"type": "http.response.body", "body": body})


class ATCMiddleware:
    def __init__(self, app: Any, *, atc: Any, prefix: str = "") -> None:
        self.app = app
        self.atc = atc
        self.settings: ATCSettings = atc.settings
        self.shield: Shield = atc.shield
        self.traffic: Traffic = atc.traffic
        self.sessions: e2e.SessionStore | None = atc.sessions
        self.prefix = prefix.rstrip("/")
        self.stats: dict[str, int] = atc.stats

    # -- helpers -------------------------------------------------------------

    def _route(self, path: str) -> str:
        if self.prefix and path.startswith(self.prefix):
            return path[len(self.prefix):] or "/"
        return path

    def _client(self, scope: dict) -> str:
        try:
            from starlette.requests import Request

            from ..t1api.deps import resolve_client_ip

            return resolve_client_ip(Request(scope))
        except Exception:  # no app state (tests, mounted elsewhere): the transport peer
            client = scope.get("client")
            return client[0] if client else ""

    # -- the layer -----------------------------------------------------------

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not self.settings.enabled:
            await self.app(scope, receive, send)
            return
        client = self._client(scope)
        route = self._route(scope.get("path") or "/")

        if self.settings.shield:
            left = self.shield.jailed(client)
            if left > 0:
                self.stats["jailed_refused"] += 1
                await _send_json(send, 429, "IP_BLOCKED",
                                 "This address is in ATC's jail for behaving like an attacker.",
                                 {"reason": "atc_jailed", "retry_after": int(left) + 1},
                                 [(b"retry-after", str(int(left) + 1).encode())])
                return

        if route == TUNNEL_PATH and scope.get("method") == "POST":
            await self._tunnel(scope, receive, send, client)
            return
        await self._serve(scope, receive, send, client, tunnelled=False)

    async def _serve(self, scope: dict, receive: Any, send: Any, client: str, *,
                     tunnelled: bool) -> None:
        route = self._route(scope.get("path") or "/")
        query = (scope.get("query_string") or b"").decode("latin-1")
        method = scope.get("method") or "GET"

        if self.settings.shield and self.shield.inspect(client, route, query):
            self.stats["probes"] += 1
            await _send_json(send, 404, "NOT_FOUND", "Not found.")
            return

        length = _header(scope, b"content-length")
        if length.isdigit() and int(length) > self.settings.max_body_bytes:
            self.stats["too_large"] += 1
            await self._too_large(send)
            return

        if (self.settings.e2e_required and not tunnelled and not self.shield.exempt(client)
                and not route.endswith(E2E_OPEN_PATHS)):
            self.stats["e2e_required_refused"] += 1
            await _send_json(send, 403, "AUTH_MISSING_CREDENTIALS",
                             "This server only answers end-to-end encrypted requests. Start with "
                             "POST /atc/handshake.",
                             {"reason": "e2e_required", "protocol": e2e.PROTOCOL})
            return

        # A loopback peer is usually a local reverse proxy speaking for
        # everyone, so it is held only to the server-wide limit.
        slot = "" if self.shield.exempt(client) else client
        if not self.traffic.enter(slot, route):
            await _send_json(send, 503, "RATE_LIMITED",
                             "The server is answering as many requests as it takes at once; try "
                             "again in a moment.", {"reason": "atc_busy"}, [(b"retry-after", b"1")])
            return

        started = time.monotonic()
        state = {"status": 500, "out": 0, "in": 0, "started": False}
        limit = self.settings.max_body_bytes

        async def counted_receive() -> dict:
            message = await receive()
            if message.get("type") == "http.request":
                state["in"] += len(message.get("body") or b"")
                if state["in"] > limit:
                    raise _BodyTooLarge
            return message

        async def counted_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                state["started"] = True
            elif message["type"] == "http.response.body":
                state["out"] += len(message.get("body") or b"")
            await send(message)

        outgoing = counted_send
        if self.settings.compress and not tunnelled and b"gzip" in _header(scope, b"accept-encoding"):
            outgoing = self._gzip(counted_send)
        try:
            await self.app(scope, counted_receive, outgoing)
        except _BodyTooLarge:
            self.stats["too_large"] += 1
            if not state["started"]:
                await self._too_large(send)
                state["status"] = 413
        finally:
            self.traffic.leave(slot)
            if self.settings.shield:
                self.shield.observe(client, int(state["status"]))
            self.traffic.record(method, route, int(state["status"]), bytes_in=int(state["in"]),
                                bytes_out=int(state["out"]),
                                ms=(time.monotonic() - started) * 1000)

    async def _too_large(self, send: Any) -> None:
        await _send_json(send, 413, "PAYLOAD_TOO_LARGE",
                         f"The request body is over this server's limit of "
                         f"{self.settings.max_body_bytes // (1024 * 1024)} MB (T1_ATC_MAX_BODY_MB).",
                         {"reason": "atc_body_limit", "limit_bytes": self.settings.max_body_bytes})

    # -- compression -----------------------------------------------------------

    def _gzip(self, send: Any) -> Any:
        """Gzip a reply whose length was declared up front.

        A declared ``Content-Length`` means the reply is complete, not a
        stream, even when the layers inside hand it over in pieces; those
        pieces are gathered (never past :data:`_COMPRESS_MAX`) and sent as
        one. A reply without one -- server-sent events, a download -- is
        never held back.
        """
        held: dict[str, Any] = {}
        parts: list[bytes] = []

        async def flush_plain(last: dict) -> None:
            start = held.pop("start")
            await send(start)
            await send({"type": "http.response.body", "body": b"".join(parts),
                        "more_body": bool(last.get("more_body"))})
            parts.clear()

        async def gzip_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = {k.lower(): v for k, v in message.get("headers") or ()}
                length = headers.get(b"content-length", b"")
                if (b"content-encoding" not in headers and length.isdigit()
                        and _COMPRESS_MIN <= int(length) <= _COMPRESS_MAX
                        and headers.get(b"content-type", b"").startswith(_COMPRESSIBLE)):
                    held["start"] = message
                    return
                await send(message)
                return
            if "start" not in held or message.get("type") != "http.response.body":
                await send(message)
                return
            parts.append(message.get("body") or b"")
            if message.get("more_body"):
                if sum(map(len, parts)) > _COMPRESS_MAX:  # it lied about its length
                    await flush_plain(message)
                return
            body = b"".join(parts)
            packed = gzip.compress(body, compresslevel=5)
            if len(packed) >= len(body):
                await flush_plain(message)
                return
            start = held.pop("start")
            parts.clear()
            out_headers = [(k, v) for k, v in start.get("headers") or ()
                       if k.lower() not in (b"content-length", b"vary")]
            vary = [v for k, v in start.get("headers") or () if k.lower() == b"vary"]
            out_headers += [(b"content-encoding", b"gzip"),
                        (b"content-length", str(len(packed)).encode()),
                        (b"vary", b", ".join([*vary, b"Accept-Encoding"]))]
            self.stats["compressed"] += 1
            self.stats["compressed_saved_bytes"] += len(body) - len(packed)
            await send({**start, "headers": out_headers})
            await send({"type": "http.response.body", "body": packed})

        return gzip_send

    # -- the tunnel ----------------------------------------------------------

    async def _tunnel_refused(self, send: Any, client: str, status: int, reason: str,
                              message: str) -> None:
        self.stats["tunnel_refused"] += 1
        if self.settings.shield:
            self.shield.observe(client, status)
        await _send_json(send, status, "AUTH_INVALID_TOKEN" if status == 401 else "VALIDATION_ERROR",
                         message, {"reason": reason, "protocol": e2e.PROTOCOL})

    async def _tunnel(self, scope: dict, receive: Any, send: Any, client: str) -> None:
        if not self.settings.e2e or self.sessions is None or not self.sessions.ready:
            await _send_json(send, 501, "NOT_SUPPORTED",
                             "End-to-end encryption is not available on this server.",
                             {"reason": "atc_e2e_off"})
            return
        session = self.sessions.get(_header(scope, b"x-atc-session").decode("latin-1"))
        if session is None:
            await self._tunnel_refused(send, client, 401, "atc_session_unknown",
                                       "No such encrypted session, or it has expired: run the "
                                       "handshake again.")
            return
        seq_raw = _header(scope, b"x-atc-seq")
        if not seq_raw.isdigit():
            await self._tunnel_refused(send, client, 400, "atc_seq_missing",
                                       "X-ATC-Seq must be a whole number.")
            return
        seq = int(seq_raw)

        chunks, size = [], 0
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            chunk = message.get("body") or b""
            size += len(chunk)
            if size > self.settings.max_body_bytes + 4096:
                self.stats["too_large"] += 1
                await self._too_large(send)
                return
            chunks.append(chunk)
            if not message.get("more_body"):
                break
        try:
            inner = e2e.open_request(session.c2s, session.session_id, seq, b"".join(chunks))
        except e2e.E2EError as exc:
            await self._tunnel_refused(send, client, 400, exc.reason, str(exc))
            return
        # Accepted only once it opens: a forged packet must not burn a number.
        if not self.sessions.accept(session, seq):
            await self._tunnel_refused(send, client, 409, "atc_replay",
                                       "This sequence number was already used in this session.")
            return
        self.stats["tunnelled"] += 1

        body = inner["body"]
        headers = [(k.encode("latin-1"), v.encode("latin-1")) for k, v in inner["headers"].items()]
        host = _header(scope, b"host")
        if host:
            headers.append((b"host", host))
        headers.append((b"content-length", str(len(body)).encode()))
        path = inner["path"]
        inner_scope = {**scope, "method": inner["method"], "path": path,
                       "raw_path": path.encode("utf-8"),
                       "query_string": inner["query"].encode("latin-1"),
                       "headers": headers, "atc.e2e": session.session_id}

        sent_body = False

        async def inner_receive() -> dict:
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        index = 0
        finished = False

        async def sealed_send(message: dict) -> None:
            nonlocal index, finished
            if message["type"] == "http.response.start":
                await send({"type": "http.response.start", "status": 200,
                            "headers": [(b"content-type", e2e.CONTENT_TYPE.encode()),
                                        (b"cache-control", b"no-store"),
                                        (b"x-atc-session", session.session_id.encode())]})
                head = e2e.reply_head(message["status"], message.get("headers") or ())
                frame = e2e.seal_frame(session.s2c, session.session_id, seq, 0, head, final=False)
                index = 1
                await send({"type": "http.response.body", "body": frame, "more_body": True})
            elif message["type"] == "http.response.body" and not finished:
                final = not message.get("more_body")
                frame = e2e.seal_frame(session.s2c, session.session_id, seq, index,
                                       message.get("body") or b"", final=final)
                index += 1
                finished = final
                await send({"type": "http.response.body", "body": frame, "more_body": not final})

        await self._serve(inner_scope, inner_receive, sealed_send, client, tunnelled=True)
