"""bridge.lmstudio — talk to an LM Studio server, wherever it is running.

LM Studio serves an OpenAI-compatible API (``/v1/...``) and, since 0.3.6,
a richer native one (``/api/v0/...``) that reports things the OpenAI
shape has no room for: whether a model is actually *loaded* into VRAM,
its quantisation, its architecture, its trained context length. This
bridge prefers the native surface for describing models and uses the
OpenAI surface for inference, which is the combination that gives the
most truthful answer to "what can I run right now".

The three deployments this has to work on
-----------------------------------------
1. **Same machine.** ``http://localhost:1234``. Nothing to configure;
   LM Studio binds loopback by default.
2. **Another machine on the LAN.** The operator has ticked *Serve on
   Local Network* and *Enable CORS* in LM Studio's server settings, and
   the address is ``http://192.168.x.y:1234``. CORS matters here because
   a browser or a WKWebView talking to LM Studio directly is subject to
   it — a Python client is not — and "it works from curl but not from
   the app" is otherwise a very long afternoon. :meth:`LMStudioBridge.probe`
   therefore reports the CORS state explicitly instead of leaving it to
   be discovered by failure.
3. **Another machine over Tailscale.** ``http://100.x.y.z:1234`` or
   ``http://desktop.tailnet-name.ts.net:1234``. Identical to the LAN
   case from LM Studio's point of view; the difference is only that the
   address happens to be routable from a phone on cellular. That is what
   makes the iOS app work off the home network, and it is why
   :func:`default_endpoints` looks at ``tailscale status`` when it is
   available.

What this module deliberately does not do
-----------------------------------------
Load, unload, or download models in LM Studio. LM Studio owns its
models; the bridge borrows whatever is loaded. Asking it to manage the
far side's lifetime would mean shelling out to ``lms`` on a machine we
may not be sitting at, and failing in ways HyperNix could not explain.
When nothing is loaded, :meth:`LMStudioBridge.chat` says exactly that,
with the address it asked and the models it saw.

Standard library only — see :mod:`hypernix.bridge`.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "LMStudioBridge",
    "LMStudioError",
    "LMStudioModel",
    "LMStudioProbe",
    "DEFAULT_PORT",
    "default_endpoints",
    "discover",
    "tailscale_peers",
]

DEFAULT_PORT = 1234

#: Set ``HYPERNIX_LMSTUDIO_URL=http://desktop:1234`` to skip discovery.
ENV_URL = "HYPERNIX_LMSTUDIO_URL"
ENV_API_KEY = "HYPERNIX_LMSTUDIO_API_KEY"


class LMStudioError(RuntimeError):
    """An LM Studio call failed, with enough context to act on it.

    ``code`` is a stable string (``unreachable``, ``no_model_loaded``,
    ``http_error``, ``bad_response``, ``timeout``) so callers — the T1
    API router, the waiter CLI, the iOS app — can branch without parsing
    English.
    """

    def __init__(self, message: str, *, code: str = "error", base_url: str = "", status: int = 0):
        super().__init__(message)
        self.code = code
        self.base_url = base_url
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "base_url": self.base_url,
            "status": self.status,
        }


@dataclass(frozen=True)
class LMStudioModel:
    """One model LM Studio knows about.

    ``loaded`` is the field the OpenAI surface cannot express and the one
    that actually decides whether a request will work: LM Studio lists
    every downloaded model at ``/v1/models``, loaded or not, and a chat
    against an unloaded one either stalls on a just-in-time load or
    fails outright depending on the server's settings.
    """

    model_id: str
    kind: str = "llm"                # llm | vlm | embeddings | unknown
    loaded: bool = False
    architecture: str = ""
    quantization: str = ""
    context_length: int = 0
    publisher: str = ""
    max_context_length: int = 0
    supports_vision: bool = False
    supports_tools: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_native(cls, data: dict[str, Any]) -> LMStudioModel:
        """Parse an ``/api/v0/models`` entry (the rich shape)."""
        kind = str(data.get("type") or "unknown").lower()
        state = str(data.get("state") or "").lower()
        ctx = _as_int(data.get("loaded_context_length"))
        max_ctx = _as_int(data.get("max_context_length"))
        model_id = str(data.get("id") or "")
        return cls(
            model_id=model_id,
            kind=kind,
            # "loaded" and "loaded-in-memory" have both been observed;
            # match on the prefix rather than an exact string so a new
            # spelling downgrades to "not loaded" only when it really is.
            loaded=state.startswith("loaded"),
            architecture=str(data.get("arch") or ""),
            quantization=str(data.get("quantization") or ""),
            context_length=ctx or max_ctx,
            publisher=str(data.get("publisher") or (model_id.split("/")[0] if "/" in model_id else "")),
            max_context_length=max_ctx,
            supports_vision=kind == "vlm",
            supports_tools=bool(data.get("supports_tool_use", False)),
            raw=data,
        )

    @classmethod
    def from_openai(cls, data: dict[str, Any]) -> LMStudioModel:
        """Parse a ``/v1/models`` entry (the thin shape).

        Everything the thin shape cannot tell us is left at its default
        rather than guessed. In particular ``loaded`` stays ``False``:
        the OpenAI listing does not distinguish, and claiming a model is
        resident because it appeared in a list is the kind of optimism
        that produces a sixty-second stall with no explanation.
        """
        return cls(
            model_id=str(data.get("id") or ""),
            kind="unknown",
            publisher=str(data.get("owned_by") or ""),
            raw=data,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "kind": self.kind,
            "loaded": self.loaded,
            "architecture": self.architecture,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "max_context_length": self.max_context_length,
            "publisher": self.publisher,
            "supports_vision": self.supports_vision,
            "supports_tools": self.supports_tools,
        }


@dataclass(frozen=True)
class LMStudioProbe:
    """The result of asking one address "are you an LM Studio server?"."""

    base_url: str
    reachable: bool
    latency_ms: float = 0.0
    model_count: int = 0
    loaded_count: int = 0
    native_api: bool = False
    cors_enabled: bool | None = None       # None = not tested
    cors_allow_origin: str = ""
    error: str = ""
    error_code: str = ""

    @property
    def usable(self) -> bool:
        """Reachable *and* something is loaded — the honest readiness bit."""
        return self.reachable and self.loaded_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "reachable": self.reachable,
            "usable": self.usable,
            "latency_ms": round(self.latency_ms, 2),
            "model_count": self.model_count,
            "loaded_count": self.loaded_count,
            "native_api": self.native_api,
            "cors_enabled": self.cors_enabled,
            "cors_allow_origin": self.cors_allow_origin,
            "error": self.error,
            "error_code": self.error_code,
        }


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------


class LMStudioBridge:
    """A client for one LM Studio server.

    ``timeout`` is split in two on purpose. A *connect* timeout wants to
    be short — an address that is not listening should be ruled out in a
    second, especially during discovery — while a *read* timeout wants to
    be long, because a 70B model on a busy GPU can legitimately take a
    minute to produce its first token. urllib exposes one timeout for
    both, so the connect phase is done separately with a raw socket and
    the urllib call gets the read timeout.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 300.0,
        connect_timeout: float = 3.0,
        user_agent: str = "hypernix-bridge/1.0.26.8.0.1",
    ) -> None:
        self.base_url = _normalise_url(base_url or os.environ.get(ENV_URL) or f"http://localhost:{DEFAULT_PORT}")
        self.api_key = api_key if api_key is not None else os.environ.get(ENV_API_KEY, "")
        self.timeout = float(timeout)
        self.connect_timeout = float(connect_timeout)
        self.user_agent = user_agent

    # -- low level ----------------------------------------------------

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        # LM Studio ignores the key, but a reverse proxy in front of it
        # (the usual way to expose it beyond a tailnet) very much does not.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            headers.update(extra)
        return headers

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path if path.startswith('/') else '/' + path}"

    def _check_connect(self) -> None:
        """Fail fast on a dead address, before urllib's long read timeout.

        Without this, a mistyped host on a network that blackholes
        packets makes every call hang for the full read timeout, which
        for a chat call is five minutes.
        """
        parsed = urllib.parse.urlparse(self.base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=self.connect_timeout):
                return
        except OSError as exc:
            raise LMStudioError(
                f"No LM Studio server answering at {self.base_url} ({exc}). "
                "Start LM Studio, open the Developer tab, and press Start Server; "
                "for another machine also tick 'Serve on Local Network'.",
                code="unreachable",
                base_url=self.base_url,
            ) from exc

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> Any:
        self._check_connect()
        data = None
        headers = self._headers()
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if stream:
            headers["Accept"] = "text/event-stream"
        req = urllib.request.Request(self._url(path), data=data, headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=timeout or self.timeout)  # noqa: S310
        except urllib.error.HTTPError as exc:
            detail = _read_error_detail(exc)
            raise LMStudioError(
                f"LM Studio returned HTTP {exc.code} for {method} {path}: {detail}",
                code="no_model_loaded" if _looks_like_no_model(detail) else "http_error",
                base_url=self.base_url,
                status=exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise LMStudioError(
                f"Could not reach LM Studio at {self.base_url}: {exc.reason}",
                code="unreachable",
                base_url=self.base_url,
            ) from exc
        except TimeoutError as exc:
            raise LMStudioError(
                f"LM Studio at {self.base_url} did not answer within {timeout or self.timeout:.0f}s",
                code="timeout",
                base_url=self.base_url,
            ) from exc
        if stream:
            return resp
        with resp:
            raw = resp.read().decode("utf-8", "replace")
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LMStudioError(
                f"LM Studio returned non-JSON from {path}: {raw[:200]!r}",
                code="bad_response",
                base_url=self.base_url,
            ) from exc

    # -- discovery / health -------------------------------------------

    def probe(self, *, check_cors: bool = True, origin: str = "http://hypernix.local") -> LMStudioProbe:
        """Ask this address what it is, without raising.

        Discovery probes many addresses and most of them are not LM
        Studio; an exception per dead address would make the caller's
        code a pile of try/except. Every failure mode is folded into the
        returned :class:`LMStudioProbe` instead.
        """
        started = time.monotonic()
        try:
            models, native = self._list_models_raw()
        except LMStudioError as exc:
            return LMStudioProbe(
                base_url=self.base_url,
                reachable=False,
                latency_ms=(time.monotonic() - started) * 1000,
                error=str(exc),
                error_code=exc.code,
            )
        latency = (time.monotonic() - started) * 1000
        cors_enabled: bool | None = None
        cors_origin = ""
        if check_cors:
            cors_enabled, cors_origin = self._probe_cors(origin)
        return LMStudioProbe(
            base_url=self.base_url,
            reachable=True,
            latency_ms=latency,
            model_count=len(models),
            loaded_count=sum(1 for m in models if m.loaded),
            native_api=native,
            cors_enabled=cors_enabled,
            cors_allow_origin=cors_origin,
        )

    def _probe_cors(self, origin: str) -> tuple[bool | None, str]:
        """Send a CORS preflight and report whether it was answered.

        A server with CORS off answers the OPTIONS (or 404s it) without
        an ``Access-Control-Allow-Origin`` header, and a browser then
        refuses the real request. Detecting that here turns a silent
        browser-side failure into a line of CLI output.
        """
        req = urllib.request.Request(
            self._url("/v1/models"),
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,authorization",
                "User-Agent": self.user_agent,
            },
            method="OPTIONS",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.connect_timeout) as resp:  # noqa: S310
                allow = resp.headers.get("Access-Control-Allow-Origin", "")
        except urllib.error.HTTPError as exc:
            allow = exc.headers.get("Access-Control-Allow-Origin", "") if exc.headers else ""
        except OSError:
            return None, ""
        return bool(allow), allow

    # -- models -------------------------------------------------------

    def _list_models_raw(self) -> tuple[list[LMStudioModel], bool]:
        """Native listing when available, OpenAI listing otherwise.

        Returns ``(models, used_native_api)``.
        """
        try:
            payload = self._request("GET", "/api/v0/models", timeout=self.connect_timeout * 4)
            items = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(items, list):
                return [LMStudioModel.from_native(m) for m in items if isinstance(m, dict)], True
        except LMStudioError as exc:
            # An unreachable host is fatal for both surfaces — don't
            # pay the connect timeout twice to prove it.
            if exc.code in ("unreachable", "timeout"):
                raise
        payload = self._request("GET", "/v1/models", timeout=self.connect_timeout * 4)
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise LMStudioError(
                f"{self.base_url}/v1/models did not return a model list — "
                "is this actually an LM Studio (or OpenAI-compatible) server?",
                code="bad_response",
                base_url=self.base_url,
            )
        return [LMStudioModel.from_openai(m) for m in items if isinstance(m, dict)], False

    def list_models(self) -> list[LMStudioModel]:
        """Every model LM Studio knows about, loaded or not."""
        models, _ = self._list_models_raw()
        return models

    def loaded_models(self) -> list[LMStudioModel]:
        """Only the models resident right now.

        On a server that only offers the OpenAI surface, ``loaded`` is
        never true and this returns empty — correct rather than
        convenient. Callers that just need *a* model should use
        :meth:`resolve_model`, which handles that case explicitly.
        """
        return [m for m in self.list_models() if m.loaded]

    def resolve_model(self, model_id: str | None = None) -> str:
        """Pick the model to use, or explain why none can be.

        With an explicit id, verify it exists (and warn loudly through
        the error path if it exists but is not loaded). Without one,
        prefer a loaded model; on a thin-API server, fall back to the
        first listed model, because there "loaded" is unknowable and
        refusing would make the bridge unusable against every non-LM
        Studio OpenAI server.
        """
        models, native = self._list_models_raw()
        if not models:
            raise LMStudioError(
                f"LM Studio at {self.base_url} has no models available. "
                "Load one in LM Studio (Developer tab → Select a model to load).",
                code="no_model_loaded",
                base_url=self.base_url,
            )
        if model_id:
            for m in models:
                if m.model_id == model_id:
                    if native and not m.loaded:
                        raise LMStudioError(
                            f"Model {model_id!r} is downloaded on {self.base_url} but not loaded. "
                            "Load it in LM Studio first, or enable Just-In-Time model loading "
                            "in the server settings.",
                            code="no_model_loaded",
                            base_url=self.base_url,
                        )
                    return m.model_id
            available = ", ".join(m.model_id for m in models[:8]) or "none"
            raise LMStudioError(
                f"Model {model_id!r} is not on {self.base_url}. Available: {available}",
                code="model_not_found",
                base_url=self.base_url,
            )
        for m in models:
            if m.loaded:
                return m.model_id
        if native:
            raise LMStudioError(
                f"LM Studio at {self.base_url} has {len(models)} model(s) downloaded but none loaded. "
                "Load one in LM Studio, then try again.",
                code="no_model_loaded",
                base_url=self.base_url,
            )
        return models[0].model_id

    # -- inference ----------------------------------------------------

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """One non-streaming chat completion. Returns the OpenAI envelope."""
        body = self._chat_body(
            messages, model=model, temperature=temperature, max_tokens=max_tokens,
            top_p=top_p, stop=stop, tools=tools, extra=extra, stream=False,
        )
        return self._request("POST", "/v1/chat/completions", body=body, timeout=timeout)

    def embeddings(
        self,
        inputs: Sequence[str],
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Embeddings for a batch of strings. Returns the OpenAI envelope.

        Sent as a batch in one request rather than one request per string:
        an embedding backend batches internally, and N round trips for N
        short strings is almost entirely latency.
        """
        body: dict[str, Any] = {"input": list(inputs)}
        if model:
            body["model"] = model
        return self._request("POST", "/v1/embeddings", body=body, timeout=timeout)

    def chat_stream(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: Sequence[str] | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream a chat completion, yielding parsed SSE chunks.

        Yields the decoded ``data:`` payloads and stops at
        ``[DONE]``. The response is closed when the generator is closed,
        including on an early ``break`` — a half-read stream left open is
        a socket LM Studio keeps a slot for.
        """
        body = self._chat_body(
            messages, model=model, temperature=temperature, max_tokens=max_tokens,
            top_p=top_p, stop=stop, tools=tools, extra=extra, stream=True,
        )
        resp = self._request("POST", "/v1/chat/completions", body=body, timeout=timeout, stream=True)
        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    # A chunk split across reads is possible in theory;
                    # skipping one malformed frame beats aborting a
                    # ninety-second generation over it.
                    continue
        finally:
            resp.close()

    def _chat_body(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        stop: Sequence[str] | None,
        tools: Sequence[dict[str, Any]] | None,
        extra: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.resolve_model(model),
            "messages": list(messages),
            "stream": stream,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_p is not None:
            body["top_p"] = top_p
        if stop:
            body["stop"] = list(stop)
        if tools:
            body["tools"] = list(tools)
        if extra:
            body.update(extra)
        return body

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Embeddings, for the callers that have an embedding model loaded."""
        body = {"model": model or self.resolve_model(model), "input": list(texts)}
        payload = self._request("POST", "/v1/embeddings", body=body)
        data = payload.get("data", []) if isinstance(payload, dict) else []
        return [list(item.get("embedding", [])) for item in data if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def tailscale_peers(*, timeout: float = 4.0) -> list[str]:
    """Hostnames/IPs of the online peers on this tailnet, or ``[]``.

    Used to find the desktop with the GPU from a laptop or a phone that
    is not on the same LAN. Returns empty — never raises — when
    Tailscale is not installed, not logged in, or slow to answer:
    discovery is a convenience, and a missing ``tailscale`` binary is
    the normal case on most machines.
    """
    exe = shutil.which("tailscale") or _macos_tailscale_path()
    if not exe:
        return []
    try:
        proc = subprocess.run(  # noqa: S603
            [exe, "status", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        status = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    hosts: list[str] = []
    for peer in (status.get("Peer") or {}).values():
        if not isinstance(peer, dict) or not peer.get("Online"):
            continue
        # DNSName is "desktop.tailnet.ts.net." — the trailing dot is
        # valid DNS but breaks URL parsing in some clients, so strip it.
        dns = str(peer.get("DNSName") or "").rstrip(".")
        if dns:
            hosts.append(dns)
        for addr in peer.get("TailscaleIPs") or []:
            if isinstance(addr, str) and ":" not in addr:   # IPv4 only; v6 needs brackets
                hosts.append(addr)
    return hosts


def _macos_tailscale_path() -> str:
    """The Mac App Store build does not put ``tailscale`` on PATH."""
    candidate = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
    return candidate if os.path.exists(candidate) else ""


def default_endpoints(*, include_tailscale: bool = True, port: int = DEFAULT_PORT) -> list[str]:
    """Addresses worth probing for an LM Studio server, best-first.

    An explicit ``HYPERNIX_LMSTUDIO_URL`` short-circuits everything —
    when the operator has said where the server is, probing a list of
    guesses first is just latency.
    """
    configured = os.environ.get(ENV_URL, "").strip()
    if configured:
        return [_normalise_url(configured)]
    endpoints = [f"http://localhost:{port}", f"http://127.0.0.1:{port}"]
    if include_tailscale:
        endpoints.extend(f"http://{host}:{port}" for host in tailscale_peers())
    # De-duplicate while keeping order; localhost and 127.0.0.1 are
    # intentionally both kept (they differ when IPv6 resolution is odd).
    seen: set[str] = set()
    return [e for e in endpoints if not (e in seen or seen.add(e))]


def discover(
    endpoints: Sequence[str] | None = None,
    *,
    port: int = DEFAULT_PORT,
    connect_timeout: float = 1.5,
    check_cors: bool = True,
    max_workers: int = 8,
) -> list[LMStudioProbe]:
    """Probe candidate addresses in parallel; return reachable ones first.

    Probing is parallel because a serial sweep of a tailnet with a dozen
    peers takes as long as the slowest dead address times twelve, and
    the whole point is to answer "where is my GPU" quickly enough that a
    CLI can do it on startup.
    """
    targets = list(endpoints) if endpoints is not None else default_endpoints(port=port)
    if not targets:
        return []

    def _one(url: str) -> LMStudioProbe:
        return LMStudioBridge(url, connect_timeout=connect_timeout).probe(check_cors=check_cors)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as pool:
        probes = list(pool.map(_one, targets))
    # Usable first, then reachable, then by latency — the order a
    # caller wants when it is going to take probes[0].
    return sorted(probes, key=lambda p: (not p.usable, not p.reachable, p.latency_ms))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_url(url: str) -> str:
    """``desktop:1234`` → ``http://desktop:1234``; strip trailing ``/`` and ``/v1``.

    Both mistakes are common enough to be worth absorbing: people paste
    the base URL LM Studio prints (which ends in ``/v1``) and people
    omit the scheme. Appending ``/v1/models`` to a base that already
    ends in ``/v1`` produces a 404 that reads like the server is broken.
    """
    url = url.strip()
    if not url:
        return f"http://localhost:{DEFAULT_PORT}"
    if "://" not in url:
        url = "http://" + url
    scheme = url.split("://", 1)[0].lower()
    if scheme not in ("http", "https"):
        # urlopen also speaks file:// and ftp://, and an admin-supplied
        # override is the one URL here that arrives over HTTP.
        raise LMStudioError(f"LM Studio's address must be http or https, not {scheme}://")
    url = url.rstrip("/")
    for suffix in ("/v1", "/api/v0"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - the body is a nicety, never the point
        return exc.reason or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:300]
    if isinstance(parsed, dict):
        err = parsed.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or raw[:300])
        if isinstance(err, str):
            return err
    return raw[:300]


def _looks_like_no_model(detail: str) -> bool:
    """Map LM Studio's 400 for an empty server onto a specific code.

    LM Studio answers "no models loaded" with a 400 and a message, not a
    dedicated status. Recognising it here is what lets the CLI print
    "load a model" instead of "HTTP 400".
    """
    lowered = detail.lower()
    return "no models loaded" in lowered or "model_not_found" in lowered or "no model" in lowered
