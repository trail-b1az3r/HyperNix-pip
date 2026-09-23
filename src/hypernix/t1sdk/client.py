"""t1sdk.client — :class:`T1Client`, the complete T1 API SDK.

One method per endpoint, plus the handful of composed helpers that every
caller would otherwise write themselves (:meth:`wait_for_job`,
:meth:`iter_usage_history`, :meth:`stream_events`,
:meth:`deploy_and_wait`).

Two return conventions, side by side on purpose:

* ``list_models()`` / ``get_job()`` / ``route()`` return **typed
  objects** (:mod:`t1sdk.models`) — the ergonomic path.
* every method also has a ``*_raw`` sibling, or accepts nothing and
  returns the parsed envelope, when you want the untouched JSON.

Typed objects keep their source payload in ``.raw``, so choosing the
typed path never costs you access to a field this SDK doesn't know about
yet.

Usage::

    from hypernix.t1sdk import T1Client

    client = T1Client("https://t1.example.com", credential="T1_...")
    for model in client.list_models():
        print(model.model_id, model.status)

    decision = client.route(input_tokens=1200)
    job = client.deploy_and_wait(module_id, ["srv-1", "srv-2"])
"""
from __future__ import annotations

import json
import time
import urllib.parse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .errors import T1Error, T1TransportError, T1ValidationError
from .models import (
    CostReport,
    Event,
    Job,
    KeyInfo,
    Model,
    ModelUsage,
    Module,
    RoutingDecision,
    Server,
    ServerStatus,
)
from .transport import HTTPTransport, Response, RetryPolicy, TLSConfig


def _q(value: str) -> str:
    """Percent-encode a path segment. Applied to every caller-supplied id
    that goes into a URL so a stray ``/`` or ``?`` can't restructure the
    request."""
    return urllib.parse.quote(str(value), safe="")


class T1Client:
    """A client for one T1 API server."""

    def __init__(
        self,
        base_url: str,
        *,
        credential: str | None = None,
        timeout: float = 15.0,
        retry: RetryPolicy | None = None,
        tls: TLSConfig | None = None,
        transport: HTTPTransport | None = None,
        user_agent: str = "hypernix-t1sdk",
    ) -> None:
        self.transport = transport or HTTPTransport(
            base_url,
            credential=credential,
            timeout=timeout,
            retry=retry,
            tls=tls,
            user_agent=user_agent,
        )

    # Convenience passthroughs so callers don't reach into .transport for
    # the two attributes they actually change at runtime.
    @property
    def base_url(self) -> str:
        return self.transport.base_url

    @property
    def credential(self) -> str | None:
        return self.transport.credential

    @credential.setter
    def credential(self, value: str | None) -> None:
        self.transport.credential = value

    def __repr__(self) -> str:
        # Never include the credential, not even masked — a repr ends up
        # in logs and tracebacks.
        return f"T1Client(base_url={self.transport.base_url!r}, authenticated={bool(self.credential)})"

    # ------------------------------------------------------------------
    # Raw access
    # ------------------------------------------------------------------

    def call(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        query: dict[str, Any] | None = None,
        auth: bool = False,
        idempotent: bool | None = None,
    ) -> dict[str, Any]:
        """Escape hatch for an endpoint this SDK version doesn't wrap.

        A deployment running a newer T1 API is never blocked on an SDK
        release: call the path directly and you get the parsed envelope
        with the same error handling and retries as every typed method.
        """
        return self.transport.request(
            method, path, body=body, query=query, auth=auth, idempotent=idempotent
        ).body

    def _get(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.transport.request("GET", path, **kwargs).body

    def _post(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return self.transport.request("POST", path, **kwargs).body

    # ------------------------------------------------------------------
    # Health / status / config
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def status(self) -> ServerStatus:
        return ServerStatus.from_dict(self._get("/status"))

    def config(self) -> dict[str, Any]:
        """The server's public configuration. Never contains a secret —
        the endpoint serves an explicit allowlist."""
        return self._get("/config").get("config", {})

    def ping(self) -> bool:
        """True if the server answers ``/health``. Swallows transport
        errors so a caller can poll without a try/except — anything other
        than "reachable and healthy" is False."""
        try:
            return self.health().get("status") == "ok"
        except T1Error:
            return False

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def validate(self, key: str | None = None) -> dict[str, Any]:
        if key is None:
            key = self.transport.wire_credential()
        elif key.startswith("T2CK_"):
            from ..security.t2c import T2CKit

            key = T2CKit.from_text(key).key_for()
        return self._post("/auth/t1/validate", body={"key": key})

    # ------------------------------------------------------------------
    # 0.72.6: server info, privacy, v2.1 (T2C) keys
    # ------------------------------------------------------------------

    def server_info(self) -> dict[str, Any]:
        """Name, description, owner, url, versions and features. Public."""
        return self._get("/server/info")

    def conceal_status(self) -> dict[str, Any]:
        return self._get("/privacy/conceal", auth=True)

    def conceal(self, enabled: bool = True) -> dict[str, Any]:
        """Turn conceal mode (masked address, 36-hour retention) on or off."""
        if enabled:
            return self._post("/privacy/conceal", auth=True)
        return self.transport.request("DELETE", "/privacy/conceal", auth=True).body

    def t2c_public_key(self) -> dict[str, Any]:
        return self._get("/auth/t2c/public-key")

    def t2c_devices(self) -> list[dict[str, Any]]:
        return list(self._get("/auth/t2c/devices", auth=True).get("devices", []))

    def t2c_revoke_device(self, device_id: str) -> dict[str, Any]:
        return self.transport.request("DELETE", f"/auth/t2c/devices/{_q(device_id)}", auth=True).body

    def seal_key(self, key: str | None = None, *, label: str = "", access_level: int = 1):
        """Turn a T1 or T2 key into a v2.1 kit, registered with this server.

        The secret crosses the network only encrypted for the server's
        public key, and the key itself only as the credential that proves
        the device is its to register. Returns a
        :class:`~hypernix.security.t2c.T2CKit`; keep its ``to_text()`` in
        place of the key, and use it as this client's credential.
        """
        import os as _os

        from ..security.t2c import T2CKit, build_inner, wrap_device_secret
        from ..security.t2keys import T2KeyGenerator, looks_like_t2

        key = key or self.credential
        if not key:
            raise ValueError("seal_key needs a T1 or T2 key")
        if key.startswith(("T2C_", "T2CK_")):
            raise ValueError("that key is already a v2.1 key")
        if looks_like_t2(key):
            t2 = T2KeyGenerator.parse(key)
        else:
            t2 = T2KeyGenerator.from_t1(key, access_level=access_level)
        published = self.t2c_public_key()
        secret = _os.urandom(32)
        registered = self._post("/auth/t2c/devices", body={
            "key": key,
            "wrapped_secret": wrap_device_secret(published["public_key_pem"], secret),
            "label": label,
            "access_level": t2.access_level,
        })
        return T2CKit(
            device_id=registered["device_id"],
            secret=secret,
            inner=build_inner(published["public_key_pem"], t2.raw),
            access_level=t2.access_level,
            server_fingerprint=published.get("fingerprint", ""),
            label=label,
        )

    def whoami(self) -> KeyInfo:
        """Identity of the configured credential, as a typed object."""
        return KeyInfo.from_dict(self.validate())

    def issue_token(
        self,
        key: str | None = None,
        *,
        ttl_seconds: int | None = None,
        scopes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Exchange a raw T1 key for a short-lived scoped token.

        A token can only ever *narrow* the underlying key's scopes; the
        server refuses a request for scopes the key doesn't hold.
        """
        body: dict[str, Any] = {"key": key or self.credential}
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        if scopes is not None:
            body["scopes"] = scopes
        return self._post("/auth/token", body=body)

    def rotate(self) -> dict[str, Any]:
        """Rotate the credential in use. The response carries the new raw
        key — the only place in the API where one is returned. Store it
        before discarding the response; the old key stops working."""
        return self._post("/auth/t1/rotate", auth=True, idempotent=False)

    def admin_rotate(self, target_key_id: str, *, promote_to_admin: bool = False) -> dict[str, Any]:
        """Rotate another key, optionally promoting it to admin. Requires
        the caller to already be admin — the permission check the spec
        asks for around admin-token conversion."""
        return self._post(
            "/auth/t1/admin/rotate",
            body={"target_key_id": target_key_id, "promote_to_admin": promote_to_admin},
            auth=True,
            idempotent=False,
        )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------

    def list_models(self) -> list[Model]:
        return [Model.from_dict(m) for m in self._get("/models").get("models", [])]

    def list_models_raw(self) -> dict[str, Any]:
        return self._get("/models")

    def get_model(self, model_id: str) -> Model:
        return Model.from_dict(self._get(f"/models/{_q(model_id)}")["model"])

    def model_availability(self, model_id: str) -> dict[str, Any]:
        return self._get(f"/models/{_q(model_id)}/availability")

    def model_usage(self, model_id: str) -> ModelUsage:
        return ModelUsage.from_dict(self._get(f"/models/{_q(model_id)}/usage", auth=True))

    def route(
        self,
        *,
        model_id: str | None = None,
        input_tokens: int = 0,
        automatic_fallback: bool = False,
        plan: str | None = None,
    ) -> RoutingDecision:
        """Ask the server which model to use.

        Omit *model_id* for automatic routing through the plan's cascade;
        supply one for manual selection, which raises
        :class:`~t1sdk.errors.T1QuotaError` (``MODEL_QUOTA_EXHAUSTED``) if
        that model is exhausted — unless *automatic_fallback* is set.
        Nothing is ever silently substituted.

        *plan* is an optional assertion, not a selection: the server
        resolves the real plan from the key's assignment and refuses a
        mismatch. Passing it is useful as a guard ("fail if this key
        isn't on the plan I expect"), never as a way to pick one.
        """
        body: dict[str, Any] = {
            "input_tokens": input_tokens,
            "automatic_fallback": automatic_fallback,
        }
        if model_id is not None:
            body["model_id"] = model_id
        if plan is not None:
            body["plan"] = plan
        # Routing decides but doesn't consume, so replaying it is safe.
        return RoutingDecision.from_dict(
            self._post("/models/route", body=body, auth=True, idempotent=True)
        )

    # ------------------------------------------------------------------
    # Usage / cost
    # ------------------------------------------------------------------

    def usage_current(self) -> dict[str, Any]:
        return self._get("/usage/current", auth=True)

    def usage_remaining(self, model_id: str) -> ModelUsage:
        return ModelUsage.from_dict(
            self._get("/usage/remaining", query={"model_id": model_id}, auth=True)
        )

    def usage_history(
        self,
        *,
        range: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
        offset: int = 0,
        **filters: Any,
    ) -> dict[str, Any]:
        query = {"range": range, "since": since, "until": until, "limit": limit, "offset": offset}
        query.update(filters)
        return self._get("/usage/history", query=query, auth=True)

    def iter_usage_history(
        self, *, page_size: int = 200, max_events: int | None = None, **kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        """Walk usage history transparently across pages.

        Stops when the server returns a short page, so it terminates
        naturally rather than needing a total count the endpoint doesn't
        promise.
        """
        offset = 0
        yielded = 0
        while True:
            page = self.usage_history(limit=page_size, offset=offset, **kwargs)
            events = page.get("events", [])
            for event in events:
                yield event
                yielded += 1
                if max_events is not None and yielded >= max_events:
                    return
            if len(events) < page_size:
                return
            offset += page_size

    def usage_by(self, group_by: str, **kwargs: Any) -> dict[str, Any]:
        """Usage grouped along one dimension: ``model_id``, ``key_id``,
        ``server_id``, ``module_id``, ``user_id``, ``account_id``,
        ``endpoint``."""
        return self._get("/usage/by", query={"group_by": group_by, **kwargs}, auth=True)

    def usage_cost(self, *, group_by: str = "model_id", forecast: bool = False, **kwargs: Any) -> CostReport:
        return CostReport.from_dict(
            self._get(
                "/usage/cost", query={"group_by": group_by, "forecast": forecast, **kwargs}, auth=True
            )
        )

    def estimate_cost(
        self,
        *,
        model_id: str,
        input_tokens: int,
        output_tokens: int | None = None,
        requests: int = 1,
    ) -> dict[str, Any]:
        """What an operation would cost. Records nothing, reserves
        nothing. Omitting *output_tokens* gives an upper bound based on
        the model's output limit."""
        body: dict[str, Any] = {
            "model_id": model_id,
            "input_tokens": input_tokens,
            "requests": requests,
        }
        if output_tokens is not None:
            body["output_tokens"] = output_tokens
        return self._post("/usage/estimate", body=body, auth=True, idempotent=True)

    def report_usage(
        self,
        *,
        model_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        requests: int = 1,
        endpoint: str = "",
        server_id: str = "",
        module_id: str = "",
    ) -> dict[str, Any]:
        """Report tokens actually consumed, so the server's quota advances.

        For clients that run inference themselves after the server routed
        them to a model — without this, the server's per-model counters
        never move and the quota cascade never advances past its first
        model.

        Usage is always recorded against *this client's* key; there is no
        way to report on behalf of another. Counts are additive only.

        **Not retried.** A report is not idempotent: replaying one after a
        timeout would double-count the tokens, and over-charging a key is
        worse than an occasionally-missed report.
        """
        body: dict[str, Any] = {
            "model_id": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "requests": requests,
        }
        for name, value in (("endpoint", endpoint), ("server_id", server_id),
                            ("module_id", module_id)):
            if value:
                body[name] = value
        return self._post("/usage/report", body=body, auth=True, idempotent=False)

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def list_keys(self, *, key_type: str | None = None, include_inactive: bool = False) -> list[KeyInfo]:
        """List keys. A non-admin caller sees only their own."""
        payload = self._get(
            "/keys", query={"key_type": key_type, "include_inactive": include_inactive}, auth=True
        )
        return [KeyInfo.from_dict(k) for k in payload.get("keys", [])]

    def import_keys(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Import a Keymaster export. Admin-only. The response carries
        counts and masked ids — never key material."""
        return self._post("/keys/import", body={"payload": payload}, auth=True, idempotent=False)

    def assign_key(self, key_id: str, **fields: Any) -> dict[str, Any]:
        """Bind a key to a plan / account / user / servers / model subset.

        Admin-only: this is where a key's plan, and therefore its quota
        and fallback chain, is decided.
        """
        return self._post("/keys/assign", body={"key_id": key_id, **fields}, auth=True, idempotent=True)

    # ------------------------------------------------------------------
    # Servers
    # ------------------------------------------------------------------

    def list_servers(self) -> list[Server]:
        return [Server.from_dict(s) for s in self._get("/servers", auth=True).get("servers", [])]

    def register_server(
        self,
        *,
        name: str,
        address: str,
        capabilities: list[str] | None = None,
        tags: dict[str, str] | None = None,
        allow_private_address: bool = False,
    ) -> Server:
        """Register a server. It starts *untrusted*; only an admin can
        promote it, and only a trusted server can receive a module.

        Set *allow_private_address* for Tailscale/LAN targets, where a
        private address is expected rather than suspicious.
        """
        body: dict[str, Any] = {
            "name": name,
            "address": address,
            "allow_private_address": allow_private_address,
        }
        if capabilities is not None:
            body["capabilities"] = capabilities
        if tags is not None:
            body["tags"] = tags
        return Server.from_dict(self._post("/servers/register", body=body, auth=True)["server"])

    def update_server(self, server_id: str, **fields: Any) -> Server:
        payload = self.transport.request(
            "PATCH", f"/servers/{_q(server_id)}", body=fields, auth=True
        ).body
        return Server.from_dict(payload["server"])

    def trust_server(self, server_id: str) -> Server:
        """Promote a server to ``trusted``. Admin-only, audited, and the
        gate every module push checks."""
        return self.update_server(server_id, trust_level="trusted")

    def delete_server(self, server_id: str, *, confirm: bool = False) -> dict[str, Any]:
        """Deregister a server. Destructive, so *confirm* must be True —
        the SDK surfaces the requirement as an argument rather than
        letting the server reject the call."""
        if not confirm:
            raise T1ValidationError(
                "delete_server() is destructive and requires confirm=True.",
                code="CONFIRMATION_REQUIRED",
            )
        return self.transport.request(
            "DELETE", f"/servers/{_q(server_id)}", query={"confirm": "true"}, auth=True
        ).body

    # ------------------------------------------------------------------
    # Modules
    # ------------------------------------------------------------------

    def list_modules(self) -> list[Module]:
        return [Module.from_dict(m) for m in self._get("/modules", auth=True).get("modules", [])]

    def create_module(self, *, name: str, version: str, metadata: dict[str, Any] | None = None) -> Module:
        body: dict[str, Any] = {"name": name, "version": version}
        if metadata is not None:
            body["metadata"] = metadata
        return Module.from_dict(self._post("/modules/create", body=body, auth=True)["module"])

    def get_module(self, module_id: str) -> Module:
        return Module.from_dict(self._get(f"/modules/{_q(module_id)}")["module"])

    def upload_module(self, module_id: str, file_path: str | Path) -> Module:
        """Upload local content. The server checksums it and stores it as
        an opaque blob — it is never executed, imported, or interpreted."""
        response = self.transport.upload_file(
            "/modules/upload/local", file_path, query={"module_id": module_id}
        )
        return Module.from_dict(response.body["module"])

    def register_remote_source(
        self, module_id: str, source_url: str, *, allow_private: bool = False
    ) -> Module:
        """Register (not fetch) a remote source URL. Validated against the
        SSRF guard now; call :meth:`fetch_module` to actually stage it."""
        return Module.from_dict(
            self._post(
                "/modules/upload/remote",
                query={"module_id": module_id},
                body={"source_url": source_url, "allow_private": allow_private},
                auth=True,
            )["module"]
        )

    def update_module(self, module_id: str, **fields: Any) -> Module:
        payload = self.transport.request(
            "PATCH", f"/modules/{_q(module_id)}", body=fields, auth=True
        ).body
        return Module.from_dict(payload["module"])

    def delete_module(self, module_id: str, *, confirm: bool = False) -> dict[str, Any]:
        if not confirm:
            raise T1ValidationError(
                "delete_module() is destructive and requires confirm=True.",
                code="CONFIRMATION_REQUIRED",
            )
        return self.transport.request(
            "DELETE", f"/modules/{_q(module_id)}", query={"confirm": "true"}, auth=True
        ).body

    def sync_module(self, module_id: str, server_id: str) -> dict[str, Any]:
        """Queue a push of this module to one trusted server."""
        return self._post(
            f"/modules/{_q(module_id)}/sync", body={"server_id": server_id}, auth=True, idempotent=True
        )

    def deploy_module(self, module_id: str, server_ids: list[str]) -> dict[str, Any]:
        """Queue a push to several trusted servers as one job."""
        return self._post(
            f"/modules/{_q(module_id)}/deploy",
            body={"server_ids": server_ids},
            auth=True,
            idempotent=True,
        )

    def fetch_module(self, module_id: str) -> dict[str, Any]:
        """Queue a fetch of this module's registered remote source."""
        return self._post(f"/modules/{_q(module_id)}/fetch", auth=True, idempotent=True)

    def deploy_and_wait(
        self, module_id: str, server_ids: list[str], *, timeout: float = 300.0
    ) -> Job:
        """Deploy to several servers and block until the job settles.

        Raises :class:`~t1sdk.errors.T1Error` if the job fails, carrying
        the job's own error text — the alternative is returning a
        "finished" job the caller has to remember to check.
        """
        queued = self.deploy_module(module_id, server_ids)
        job = self.wait_for_job(queued["job_id"], timeout=timeout)
        if not job.succeeded:
            raise T1Error(
                f"Deployment job {job.job_id[:8]} finished as '{job.status}': "
                f"{job.error or 'no error recorded'}",
                code="TRANSPORT_FAILED",
                details=job.result or {},
            )
        return job

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def submit_job(self, kind: str, payload: dict[str, Any] | None = None) -> Job:
        return Job.from_dict(
            self._post("/jobs", body={"kind": kind, "payload": payload or {}}, auth=True)["job"]
        )

    def get_job(self, job_id: str) -> Job:
        return Job.from_dict(self._get(f"/jobs/{_q(job_id)}", auth=True)["job"])

    def cancel_job(self, job_id: str) -> Job:
        return Job.from_dict(
            self._post(f"/jobs/{_q(job_id)}/cancel", auth=True, idempotent=True)["job"]
        )

    def wait_for_job(
        self, job_id: str, *, timeout: float = 300.0, poll_interval: float = 0.5
    ) -> Job:
        """Poll a job until it reaches a terminal state.

        Returns the job in whatever terminal state it reached — including
        ``failed`` — because "the job failed" is an answer, not an
        exception. A timeout *is* exceptional: it means the caller no
        longer knows the outcome.
        """
        deadline = time.monotonic() + timeout
        while True:
            job = self.get_job(job_id)
            if job.is_terminal:
                return job
            if time.monotonic() >= deadline:
                raise T1TransportError(
                    f"Job {job_id[:8]} was still '{job.status}' after {timeout:g}s. "
                    "It may still finish — poll get_job() to find out."
                )
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def list_events(
        self, *, limit: int = 50, since_id: str | None = None, event_type: str | None = None
    ) -> list[Event]:
        payload = self._get(
            "/events", query={"limit": limit, "since_id": since_id, "type": event_type}
        )
        return [Event.from_dict(e) for e in payload.get("events", [])]

    def stream_events(
        self, *, event_type: str | None = None, timeout: float = 300.0
    ) -> Iterator[Event]:
        """Live tail over Server-Sent Events.

        A generator, so the caller controls the lifetime: break out of
        the loop and the connection closes. Keep-alive comments are
        swallowed rather than surfaced as empty events.
        """
        url = self.transport.url_for("/events/stream", {"type": event_type})
        headers = {"Accept": "text/event-stream", "User-Agent": self.transport.user_agent}
        if self.credential:
            headers["Authorization"] = f"Bearer {self.credential}"
        import urllib.request

        request = urllib.request.Request(url, headers=headers, method="GET")
        with self.transport._urlopen(request) if self.transport._opener else urllib.request.urlopen(
            request, timeout=timeout
        ) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue  # keep-alive or blank separator
                if line.startswith("data:"):
                    try:
                        yield Event.from_dict(json.loads(line[5:].strip()))
                    except ValueError:
                        continue

    # ------------------------------------------------------------------
    # Billing
    # ------------------------------------------------------------------

    def balance(self) -> dict[str, Any]:
        return self._get("/billing/balance", auth=True)

    def transactions(self) -> list[dict[str, Any]]:
        return self._get("/billing/transactions", auth=True).get("transactions", [])

    def mint_payment_token(
        self, amount: float, *, currency: str = "USD", ttl_seconds: int | None = None
    ) -> dict[str, Any]:
        """Admin-only. The raw token is returned **once** and never
        retrievable again — the server stores only its hash. Capture it
        from the response or it is gone."""
        body: dict[str, Any] = {"amount": amount, "currency": currency}
        if ttl_seconds is not None:
            body["ttl_seconds"] = ttl_seconds
        return self._post("/billing/payment-token", body=body, auth=True, idempotent=False)

    def redeem_payment_token(
        self, token: str, *, account_type: str = "user", account_id: str | None = None
    ) -> dict[str, Any]:
        """Redeem a payment token. Single-use, and never retried by the
        transport — a replay after a timeout could look like a double
        redemption from the caller's side."""
        body: dict[str, Any] = {"token": token, "account_type": account_type}
        if account_id is not None:
            body["account_id"] = account_id
        return self._post("/billing/redeem", body=body, auth=True, idempotent=False)

    def add_balance(
        self, *, account_type: str, account_id: str, amount: float, note: str = ""
    ) -> dict[str, Any]:
        """Admin-only direct credit, with no payment-token trail."""
        return self._post(
            "/billing/add-balance",
            body={
                "account_type": account_type,
                "account_id": account_id,
                "amount": amount,
                "note": note,
            },
            auth=True,
            idempotent=False,
        )

    # ------------------------------------------------------------------
    # Audit / security (admin)
    # ------------------------------------------------------------------

    def audit_events(self, **filters: Any) -> dict[str, Any]:
        return self._get("/audit", query=filters, auth=True)

    def network_policy(self, *, kind: str | None = None) -> dict[str, Any]:
        return self._get("/security/network", query={"kind": kind}, auth=True)

    def blacklist_ip(self, cidr: str, *, reason: str = "", ttl_seconds: float | None = None) -> dict[str, Any]:
        return self._post(
            "/security/network/blacklist",
            body={"cidr": cidr, "reason": reason, "ttl_seconds": ttl_seconds},
            auth=True,
            idempotent=True,
        )

    def whitelist_ip(self, cidr: str, *, reason: str = "", ttl_seconds: float | None = None) -> dict[str, Any]:
        return self._post(
            "/security/network/whitelist",
            body={"cidr": cidr, "reason": reason, "ttl_seconds": ttl_seconds},
            auth=True,
            idempotent=True,
        )

    def appeal_ip(self, cidr: str) -> dict[str, Any]:
        """Remove an allow or block entry — the appeal path."""
        return self.transport.request("DELETE", f"/security/network/{cidr}", auth=True).body

    def set_allow_unlisted(self, enabled: bool) -> dict[str, Any]:
        return self._post(
            "/security/network/allow-unlisted", query={"enabled": enabled}, auth=True, idempotent=True
        )

    def list_forced_limits(self) -> list[dict[str, Any]]:
        return self._get("/security/limits", auth=True).get("limits", [])

    def set_forced_limit(
        self,
        *,
        subject_type: str,
        subject_id: str,
        requests_per_window: int | None = None,
        tokens_per_window: int | None = None,
        window_seconds: float = 60.0,
        reason: str = "",
    ) -> dict[str, Any]:
        """Force a ceiling on one key or server. Only ever tightens."""
        return self._post(
            "/security/limits",
            body={
                "subject_type": subject_type,
                "subject_id": subject_id,
                "requests_per_window": requests_per_window,
                "tokens_per_window": tokens_per_window,
                "window_seconds": window_seconds,
                "reason": reason,
            },
            auth=True,
            idempotent=True,
        )

    def clear_forced_limit(self, subject_type: str, subject_id: str) -> dict[str, Any]:
        return self.transport.request(
            "DELETE", f"/security/limits/{_q(subject_type)}/{_q(subject_id)}", auth=True
        ).body

    def rate_limit_status(self) -> dict[str, Any]:
        """The caller's own remaining budget per rule — enough to back off
        before hitting a 429 rather than after."""
        return self._get("/security/rate-limits", auth=True)


    # ------------------------------------------------------------------
    # T1 v1.0.26.8.0.1 — LM Studio bridge
    # ------------------------------------------------------------------

    def lmstudio_status(self, *, discover: bool = False) -> dict[str, Any]:
        """Is the server's LM Studio reachable, and is anything loaded?"""
        return self._get(
            "/bridge/lmstudio/status", query={"discover": discover} if discover else None, auth=True
        )

    def lmstudio_models(self, *, base_url: str | None = None) -> dict[str, Any]:
        """Every model LM Studio has, with the loaded ones marked."""
        return self._get(
            "/bridge/lmstudio/models",
            query={"base_url": base_url} if base_url else None,
            auth=True,
        )

    def lmstudio_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        """One completion through the bridge, with the reply lifted out."""
        return self._post(
            "/bridge/lmstudio/chat",
            body={
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "base_url": base_url,
            },
            auth=True,
        )

    # ------------------------------------------------------------------
    # T1 v1.0.26.8.0.1 — HyperLink
    # ------------------------------------------------------------------

    def hyperlink_endpoints(self) -> dict[str, Any]:
        """Addresses this server answers on, ranked best-first."""
        return self._get("/hyperlink/endpoints", auth=True)

    def hyperlink_pair(
        self, *, label: str = "", scopes: list[str] | None = None, ttl_seconds: float | None = None
    ) -> dict[str, Any]:
        """Mint a pairing code for a phone. Admin only."""
        return self._post(
            "/hyperlink/pair",
            body={"label": label, "scopes": scopes, "ttl_seconds": ttl_seconds},
            auth=True,
        )

    def hyperlink_redeem(
        self, code: str, *, device_name: str, platform: str = "ios", app_version: str = ""
    ) -> dict[str, Any]:
        """Redeem a pairing code. Unauthenticated — the code is the credential."""
        return self._post(
            "/hyperlink/pair/redeem",
            body={
                "code": code,
                "device_name": device_name,
                "platform": platform,
                "app_version": app_version,
            },
        )

    def hyperlink_devices(self, *, include_revoked: bool = False) -> dict[str, Any]:
        return self._get(
            "/hyperlink/devices", query={"include_revoked": include_revoked}, auth=True
        )

    def hyperlink_revoke_device(self, device_id: str) -> dict[str, Any]:
        return self.transport.request(
            "DELETE", f"/hyperlink/devices/{_q(device_id)}", auth=True
        ).body

    def hyperlink_sessions(self, *, include_archived: bool = False, limit: int = 50) -> dict[str, Any]:
        return self._get(
            "/hyperlink/sessions",
            query={"include_archived": include_archived, "limit": limit},
            auth=True,
        )

    def hyperlink_create_session(
        self, *, title: str = "", model_id: str = "", system_prompt: str = ""
    ) -> dict[str, Any]:
        return self._post(
            "/hyperlink/sessions",
            body={"title": title, "model_id": model_id, "system_prompt": system_prompt},
            auth=True,
        )

    def hyperlink_messages(self, session_id: str, *, after_seq: int = 0) -> dict[str, Any]:
        return self._get(
            f"/hyperlink/sessions/{_q(session_id)}/messages",
            query={"after_seq": after_seq},
            auth=True,
        )

    def hyperlink_chat(
        self,
        session_id: str,
        content: str,
        *,
        attachment_ids: list[str] | None = None,
        model_id: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """One chat turn. Both messages are persisted server-side."""
        return self._post(
            f"/hyperlink/sessions/{_q(session_id)}/chat",
            body={
                "content": content,
                "attachment_ids": attachment_ids or [],
                "model_id": model_id,
                "max_tokens": max_tokens,
            },
            auth=True,
        )

    def hyperlink_resolve_model(
        self,
        *,
        page_url: str = "",
        file_url: str = "",
        prefer: str = "strict",
        include_vision: bool = True,
        offline: bool = False,
    ) -> dict[str, Any]:
        """Merge a Hugging Face page link and/or file link into a plan."""
        return self._post(
            "/hyperlink/models/resolve",
            body={
                "page_url": page_url,
                "file_url": file_url,
                "prefer": prefer,
                "include_vision": include_vision,
                "offline": offline,
            },
            auth=True,
        )


__all__ = ["T1Client", "HTTPTransport", "Response", "RetryPolicy", "TLSConfig"]
