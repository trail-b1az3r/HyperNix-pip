"""t1sdk.transport — the SDK's HTTP layer.

Stdlib ``urllib`` only, deliberately: the SDK is a *client*, and
requiring ``requests``/``httpx`` to talk to a T1 server would mean every
machine running ``waiter`` needs a dependency the server-side extra
already covers. Everything below is the part of an HTTP client that
actually matters for an API like this, written once here rather than
per-call-site:

* **Retries with backoff.** Idempotent methods (GET/HEAD/PUT/DELETE) and
  explicitly-marked idempotent POSTs are retried on connection failures,
  502/503/504, and 429. The server's ``Retry-After`` wins over the
  backoff schedule when present — it is the server telling you what it
  knows and your exponential curve doesn't.
* **Never retrying a non-idempotent POST.** Retrying
  ``POST /billing/redeem`` after a timeout could redeem a payment token
  twice from the caller's point of view. The SDK would rather surface a
  timeout than risk that, so POST retries are opt-in per call.
* **mTLS.** Client certificate, key, and CA bundle, built into an
  ``ssl.SSLContext`` once and reused.
* **No credential ever logged.** The Authorization header is set at
  request time and never included in an exception message, a repr, or a
  debug log line.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import random
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import T1TransportError, exception_for

logger = logging.getLogger(__name__)

# Methods safe to replay without changing server state.
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})
# Status codes worth another attempt. 429 is included because the server
# is explicitly asking us to come back, not refusing outright.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass
class RetryPolicy:
    """How hard to try before giving up.

    Defaults are tuned for an interactive client: three attempts total,
    starting at a quarter second, so a transient blip is invisible and a
    real outage still fails in under two seconds rather than hanging.
    """

    max_attempts: int = 3
    initial_backoff: float = 0.25
    max_backoff: float = 8.0
    backoff_multiplier: float = 2.0
    jitter: float = 0.1

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait before *attempt* (1-based, so attempt 2 is the
        first retry). A server-sent ``Retry-After`` overrides the curve."""
        if retry_after is not None:
            return min(retry_after, self.max_backoff)
        base = min(self.initial_backoff * (self.backoff_multiplier ** (attempt - 1)), self.max_backoff)
        # Jitter so a fleet of clients retrying after one outage doesn't
        # come back in lockstep and cause the next one.
        return base + random.uniform(0, self.jitter * base)


@dataclass
class TLSConfig:
    """Client-side TLS, including mTLS.

    Args:
        client_certfile / client_keyfile: This client's certificate and
            key, for a server that requires mTLS.
        ca_certs: CA bundle used to verify the *server*. Point this at a
            private CA for an internal deployment rather than reaching
            for ``verify=False``.
        verify: Whether to verify the server's certificate at all.
            Defaults to True and should stay there — the SDK logs a
            warning every time it is turned off, because a disabled check
            in a config file outlives whatever debugging session
            justified it.
    """

    client_certfile: str | None = None
    client_keyfile: str | None = None
    ca_certs: str | None = None
    verify: bool = True

    def build_context(self) -> ssl.SSLContext | None:
        if not (self.client_certfile or self.ca_certs or not self.verify):
            return None  # stock verification is fine; let urllib use its default
        context = ssl.create_default_context(cafile=self.ca_certs)
        if not self.verify:
            logger.warning(
                "hypernix.t1sdk: TLS verification is DISABLED for this client. Traffic can be "
                "intercepted. Point ca_certs at your deployment's CA instead."
            )
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        if self.client_certfile:
            context.load_cert_chain(self.client_certfile, self.client_keyfile)
        return context


@dataclass
class Response:
    """A parsed T1 response."""

    status: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def request_id(self) -> str | None:
        return self.body.get("request_id") or self.headers.get("X-Request-ID")


class HTTPTransport:
    """Issues requests against one T1 server.

    Args:
        base_url: Server root, with or without a trailing slash.
        credential: Raw T1 key or scoped token, sent as ``Bearer``.
        timeout: Per-request socket timeout in seconds.
        retry: Retry policy; pass ``RetryPolicy(max_attempts=1)`` to
            disable retries.
        tls: Client TLS/mTLS configuration.
        user_agent: Overridable, so a deployment can tell its own
            services apart in server logs.
        opener: Test injection point — anything with urllib's
            ``urlopen(request, timeout=...)`` signature.
    """

    def __init__(
        self,
        base_url: str,
        *,
        credential: str | None = None,
        timeout: float = 15.0,
        retry: RetryPolicy | None = None,
        tls: TLSConfig | None = None,
        user_agent: str = "hypernix-t1sdk",
        opener: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.credential = credential
        self.timeout = timeout
        self.retry = retry or RetryPolicy()
        self.tls = tls or TLSConfig()
        self.user_agent = user_agent
        self._opener = opener
        self._ssl_context = self.tls.build_context()

    def wire_credential(self) -> str | None:
        """What goes in the Authorization header.

        The credential itself, except a v2.1 kit (``T2CK_``): a kit is
        what makes keys and never leaves this machine, so today's key is
        derived from it for each request instead (0.72.6).
        """
        credential = self.credential
        if credential and credential.startswith("T2CK_"):
            from ..security.t2c import T2CKit

            return T2CKit.from_text(credential).key_for()
        return credential

    # ------------------------------------------------------------------

    def url_for(self, path: str, query: dict[str, Any] | None = None) -> str:
        url = f"{self.base_url}{path}"
        if query:
            cleaned = {k: v for k, v in query.items() if v is not None}
            if cleaned:
                url += "?" + urllib.parse.urlencode(
                    {k: ("true" if v is True else "false" if v is False else v) for k, v in cleaned.items()}
                )
        return url

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        auth: bool = False,
        idempotent: bool | None = None,
        raw_body: bytes | None = None,
        content_type: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Response:
        """Issue one request, retrying per the policy.

        Args:
            idempotent: Override the method-based default. Set True on a
                POST that is safe to replay (submitting a job, asking for
                a routing decision); leave it alone for anything that
                moves money or consumes a single-use token.
        """
        url = self.url_for(path, query)
        can_retry = method.upper() in _IDEMPOTENT_METHODS if idempotent is None else idempotent

        headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        data = raw_body
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        if extra_headers:
            headers.update(extra_headers)
        if auth:
            if not self.credential:
                raise exception_for(
                    "This call requires a credential. Pass one to the client "
                    "(or run `waiter serv -A -I <server> -K <key>`).",
                    code="AUTH_MISSING_CREDENTIALS",
                    status=401,
                )
            headers["Authorization"] = f"Bearer {self.wire_credential()}"

        last_error: Exception | None = None
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                return self._attempt(method, url, data, headers)
            except _RetryableFailure as failure:
                last_error = failure.error
                if not can_retry or attempt == self.retry.max_attempts:
                    raise failure.error from None
                delay = self.retry.delay_for(attempt, failure.retry_after)
                logger.debug(
                    "hypernix.t1sdk: %s %s failed (%s); retry %d/%d in %.2fs",
                    method, path, failure.reason, attempt, self.retry.max_attempts - 1, delay,
                )
                time.sleep(delay)
        # Unreachable in practice: the loop either returns or raises.
        raise last_error or T1TransportError(f"{method} {path} failed with no recorded error")

    # ------------------------------------------------------------------

    def _attempt(self, method: str, url: str, data: bytes | None, headers: dict[str, str]) -> Response:
        request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with self._urlopen(request) as response:
                raw = response.read()
                status = getattr(response, "status", None) or response.getcode()
                response_headers = dict(getattr(response, "headers", {}) or {})
                return Response(status=status, body=_parse_json(raw, method, url), headers=response_headers)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw) if raw else {}
            except (ValueError, TypeError):
                payload = {}
            error = payload.get("error") if isinstance(payload, dict) else None
            error = error if isinstance(error, dict) else {}
            retry_after = _parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
            built = exception_for(
                error.get("message") or f"HTTP {exc.code} from {method} {_safe_path(url)}",
                code=error.get("code"),
                status=exc.code,
                request_id=payload.get("request_id") if isinstance(payload, dict) else None,
                details=error.get("details"),
                retry_after=retry_after,
            )
            if exc.code in _RETRYABLE_STATUS:
                raise _RetryableFailure(built, reason=f"HTTP {exc.code}", retry_after=retry_after) from exc
            raise built from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            built = T1TransportError(f"Could not reach {_safe_path(url)}: {reason}")
            raise _RetryableFailure(built, reason=str(reason)) from exc

    def _urlopen(self, request: urllib.request.Request):
        if self._opener is not None:
            return self._opener(request, timeout=self.timeout)
        if self._ssl_context is not None:
            return urllib.request.urlopen(request, timeout=self.timeout, context=self._ssl_context)
        return urllib.request.urlopen(request, timeout=self.timeout)

    # ------------------------------------------------------------------

    def upload_file(
        self,
        path: str,
        file_path: str | Path,
        *,
        field_name: str = "file",
        query: dict[str, Any] | None = None,
    ) -> Response:
        """Multipart file upload.

        Built by hand rather than with ``email.mime`` or a dependency: a
        single-part upload is a handful of lines, and the alternative is
        pulling in a library for one call. Never retried — an upload is
        expensive and a partially-consumed one on the server is worse
        than a clear failure.
        """
        file_path = Path(file_path)
        if not file_path.is_file():
            raise T1TransportError(f"No such file to upload: {file_path}")
        boundary = uuid.uuid4().hex
        filename = os.path.basename(str(file_path))
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        payload = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode(),
                f"Content-Type: {content_type}\r\n\r\n".encode(),
                file_path.read_bytes(),
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        return self.request(
            "POST",
            path,
            query=query,
            raw_body=payload,
            content_type=f"multipart/form-data; boundary={boundary}",
            auth=True,
            idempotent=False,
        )


class _RetryableFailure(Exception):
    """Internal: wraps an error the transport may retry."""

    def __init__(self, error: Exception, *, reason: str, retry_after: float | None = None) -> None:
        self.error = error
        self.reason = reason
        self.retry_after = retry_after
        super().__init__(reason)


def _parse_json(raw: bytes, method: str, url: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise T1TransportError(
            f"{method} {_safe_path(url)} returned a non-JSON body "
            f"({len(raw)} bytes). Is this really a T1 API server?"
        ) from exc
    return parsed if isinstance(parsed, dict) else {"data": parsed}


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        # HTTP-date form. Not worth parsing precisely for a backoff hint:
        # fall back to the SDK's own curve rather than guessing wrong.
        return None


def _safe_path(url: str) -> str:
    """Strip userinfo and query string before a URL enters an error
    message — either can carry a credential."""
    parsed = urllib.parse.urlparse(url)
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urllib.parse.urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))


__all__ = ["HTTPTransport", "Response", "RetryPolicy", "TLSConfig"]
