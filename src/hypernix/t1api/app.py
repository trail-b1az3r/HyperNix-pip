"""t1api.app — ``create_app()``, the T1 API's FastAPI factory.

Designed to be "easy to mount into an existing Python server" (spec,
HARD REQUIREMENTS)::

    from hypernix.t1api import create_app
    app = create_app()                     # standalone
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # OR mount into an existing FastAPI app:
    existing_app.mount("/t1", create_app(mount_prefix="/t1"))

    # OR pull the routers into an existing app's own router tree:
    t1 = create_app()
    for route in t1.routes:
        existing_app.router.routes.append(route)

Everything the routers need lives on ``app.state`` — see
``t1api/deps.py`` — so a caller mounting the routers directly can wire
the same attributes without any global singletons being involved.

The middleware order (Beta 3)
-----------------------------
Starlette runs middleware in reverse registration order, so the stack
below is registered bottom-up and *executes* in this order, which is the
order that matters:

1. **Request ID + timing** — first, so every later stage (including a
   rejection) has an id to report and log against.
2. **Network policy** — is this address allowed to talk to us at all?
   Cheapest possible check, and the one that should reject a blocked
   client before anything else spends work on it.
3. **mTLS** — does this connection carry an acceptable client
   certificate? Transport-level identity, checked before credential-level
   identity.
4. **Rate limiting** — before routing, before the registry, before any
   model work. This is the spec's "Apply rate limits before expensive
   model operations", made structural rather than left to each handler
   to remember.
5. **The route handler** — authentication, authorization, and the
   operation itself.

Each stage is skipped only by explicit configuration, never implicitly,
and a stage that is off says so in ``GET /status``.
"""
from __future__ import annotations

import logging
import math
import sys
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from hypernix.security.gatekeeper import Gatekeeper
from hypernix.security.keymaster import Keymaster

from ..hyperlink.files import AttachmentStore
from ..hyperlink.notify import NotificationStore
from ..hyperlink.pairing import DeviceRegistry
from ..hyperlink.search import SearchIndex
from ..hyperlink.sessions import ChatSessionStore
from ..hyperlink.sync import SyncStore
from ..security.t2keys import ServerKeyRegistry
from . import __t1api_version__
from .audit import AuditCategory, AuditLog, AuditOutcome
from .auth import T1AuthService
from .authhistory import AuthHistory
from .backup import BackupStore
from .billing import BillingLedger
from .billingkeys import BillingBindingStore
from .config import T1APIConfig
from .cost import CostCalculator
from .db import SQLBackend, make_backend
from .deploy import DeploymentCoordinator
from .errors import T1APIError, T1ErrorCode
from .events import EventBus
from .jobs import JobQueue
from .keys import KeyDirectory
from .modules import ModuleRegistry
from .mtls import ClientCertVerifier
from .netpolicy import NetworkPolicy
from .ratelimit import RateLimiter, Subject, rules_from_json
from .registry import ModelRegistry
from .routers import ALL_ROUTERS
from .routing import RoutingEngine, RoutingTable
from .servers import ServerRegistry
from .storage import UsageStore
from .transport import ModuleTransport
from .usage import UsageMeter

logger = logging.getLogger(__name__)

# Default HTTP status for each error code when the raiser didn't specify one.
_STATUS_FOR_CODE: dict[T1ErrorCode, int] = {
    T1ErrorCode.MODEL_NOT_SUPPORTED: 404,
    T1ErrorCode.MODEL_QUOTA_EXHAUSTED: 429,
    T1ErrorCode.MODEL_UNAVAILABLE: 503,
    T1ErrorCode.AUTH_MISSING_CREDENTIALS: 401,
    T1ErrorCode.AUTH_INVALID_KEY: 401,
    T1ErrorCode.AUTH_EXPIRED_KEY: 401,
    T1ErrorCode.AUTH_REVOKED_KEY: 401,
    T1ErrorCode.AUTH_INVALID_TOKEN: 401,
    T1ErrorCode.AUTH_EXPIRED_TOKEN: 401,
    T1ErrorCode.AUTH_INSUFFICIENT_SCOPE: 403,
    T1ErrorCode.AUTH_ADMIN_REQUIRED: 403,
    T1ErrorCode.RATE_LIMITED: 429,
    T1ErrorCode.QUOTA_EXCEEDED: 429,
    T1ErrorCode.NOT_SUPPORTED: 501,
    T1ErrorCode.NOT_FOUND: 404,
    T1ErrorCode.VALIDATION_ERROR: 422,
    T1ErrorCode.CONFLICT: 409,
    T1ErrorCode.INTERNAL_ERROR: 500,
    T1ErrorCode.MODULE_NOT_FOUND: 404,
    T1ErrorCode.MODULE_ALREADY_EXISTS: 409,
    T1ErrorCode.MODULE_UPLOAD_REJECTED: 400,
    T1ErrorCode.SERVER_NOT_FOUND: 404,
    T1ErrorCode.SERVER_UNTRUSTED: 403,
    T1ErrorCode.JOB_NOT_FOUND: 404,
    T1ErrorCode.JOB_NOT_CANCELLABLE: 409,
    T1ErrorCode.PATH_TRAVERSAL_REJECTED: 400,
    T1ErrorCode.SSRF_BLOCKED: 400,
    T1ErrorCode.PAYMENT_TOKEN_INVALID: 400,
    T1ErrorCode.PAYMENT_TOKEN_ALREADY_REDEEMED: 409,
    T1ErrorCode.INSUFFICIENT_BALANCE: 402,
    T1ErrorCode.ROUTING_EXHAUSTED: 429,
    # Beta 3
    T1ErrorCode.IP_BLOCKED: 403,
    T1ErrorCode.IP_NOT_ALLOWLISTED: 403,
    T1ErrorCode.MTLS_REQUIRED: 403,
    T1ErrorCode.MTLS_INVALID: 403,
    T1ErrorCode.TRANSPORT_FAILED: 502,
    T1ErrorCode.CHECKSUM_MISMATCH: 502,
    T1ErrorCode.PAYLOAD_TOO_LARGE: 413,
    T1ErrorCode.CONFIRMATION_REQUIRED: 409,
    T1ErrorCode.CONFIG_INVALID: 500,
}

# Error codes worth an audit record on their own. These are the ones that
# mean somebody was refused, as opposed to somebody made a typo — the
# spec's "Record security-relevant events in the audit log".
_AUDITED_ERROR_CODES = frozenset(
    {
        T1ErrorCode.AUTH_INVALID_KEY,
        T1ErrorCode.AUTH_EXPIRED_KEY,
        T1ErrorCode.AUTH_REVOKED_KEY,
        T1ErrorCode.AUTH_INVALID_TOKEN,
        T1ErrorCode.AUTH_EXPIRED_TOKEN,
        T1ErrorCode.AUTH_INSUFFICIENT_SCOPE,
        T1ErrorCode.AUTH_ADMIN_REQUIRED,
        T1ErrorCode.IP_BLOCKED,
        T1ErrorCode.IP_NOT_ALLOWLISTED,
        T1ErrorCode.MTLS_REQUIRED,
        T1ErrorCode.MTLS_INVALID,
        T1ErrorCode.SSRF_BLOCKED,
        T1ErrorCode.PATH_TRAVERSAL_REJECTED,
        T1ErrorCode.RATE_LIMITED,
        T1ErrorCode.PAYMENT_TOKEN_INVALID,
        T1ErrorCode.CHECKSUM_MISMATCH,
    }
)

# Security response headers applied to every response. Modest, and chosen
# for an API rather than copied from a website checklist: an API serves
# JSON, so the ones that matter are the sniffing and framing guards plus
# HSTS when TLS is actually in play.
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def _announce_bootstrap_key(key: Any, cfg: T1APIConfig) -> None:
    """Show the key once, on the terminal, and nowhere else.

    Printed rather than logged. A log line is a copy that outlives the
    three days and gets shipped wherever logs go; stdout on first start
    is seen by the person who just ran the command and by nobody else.
    Under a service manager stdout is the journal, which the banner says
    so the operator can judge it — and which is bounded by the expiry
    either way.
    """
    if key is None or not key.created:
        return
    from .bootstrap import bootstrap_banner

    # Only an address the deployment was actually configured with. The
    # bind address belongs to uvicorn, not to this config, so anything
    # else here would be a guess printed as an instruction.
    base_url = (cfg.hyperlink_public_url or "").rstrip("/")
    try:
        sys.stdout.write(bootstrap_banner(key, base_url=base_url))
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        # A server must not fail to start because it could not print.
        logger.info("t1api.bootstrap: minted a bootstrap key (banner not shown)")


def create_app(
    *,
    config: T1APIConfig | None = None,
    keymaster: Keymaster | None = None,
    gatekeeper: Gatekeeper | None = None,
    registry: ModelRegistry | None = None,
    usage_store: UsageStore | None = None,
    routing_table: RoutingTable | None = None,
    server_registry: ServerRegistry | None = None,
    module_registry: ModuleRegistry | None = None,
    job_queue: JobQueue | None = None,
    event_bus: EventBus | None = None,
    billing_ledger: BillingLedger | None = None,
    backend: SQLBackend | None = None,
    audit_log: AuditLog | None = None,
    network_policy: NetworkPolicy | None = None,
    rate_limiter: RateLimiter | None = None,
    key_directory: KeyDirectory | None = None,
    transport: ModuleTransport | None = None,
    # T1 v1.0.26.8.0.1 — injectable for the same reason as everything
    # above: a test wants an AttachmentStore in a tmpdir, not in
    # ~/.hypernix.
    device_registry: DeviceRegistry | None = None,
    session_store: ChatSessionStore | None = None,
    attachment_store: AttachmentStore | None = None,
    # 0.72.4.post9 -- injectable for the same reason as everything above.
    sync_store: SyncStore | None = None,
    notification_store: NotificationStore | None = None,
    search_index: SearchIndex | None = None,
    # T1 v1.0.26.8.1.0
    auth_history: AuthHistory | None = None,
    backup_store: BackupStore | None = None,
    server_key_registry: ServerKeyRegistry | None = None,
    mount_prefix: str | None = None,
    validate_production: bool | None = None,
) -> FastAPI:
    """Build a fully-wired T1 API FastAPI app.

    Every dependency is injectable so tests (and an embedding server) can
    swap in an in-memory Keymaster/Gatekeeper or a pre-populated registry
    without monkeypatching. Omit everything and you get a sane
    local/dev-ready default: SQLite under ``~/.hypernix/t1api/`` and the
    example model registry (invisible by default — see
    ``T1_ENABLE_EXAMPLE_MODELS``).

    Args:
        validate_production: Force the production-config check on or off.
            Default (``None``) validates iff ``T1_ENVIRONMENT`` says
            production, so this is safe to leave alone.

    Raises:
        T1APIError: ``CONFIG_INVALID`` when a production deployment is
            misconfigured. Raised at construction, not at first request —
            a bad production config should fail the deploy, not surface
            later as a puzzling 500.
    """
    cfg = config or T1APIConfig.from_env()
    cfg.validate_for_production(strict=validate_production)
    prefix = mount_prefix if mount_prefix is not None else cfg.mount_prefix

    km = keymaster or Keymaster(store_dir=cfg.keymaster_dir)
    gk = gatekeeper or Gatekeeper(keymaster=km)
    reg = registry or ModelRegistry.load(
        cfg.registry_path, include_examples=cfg.enable_example_models
    )

    # One backend, shared by every store: switching SQLite → PostgreSQL is
    # a single environment variable (T1_DATABASE_URL) and nothing else.
    db = backend or make_backend(database_url=cfg.database_url, db_path=cfg.db_path)

    store = usage_store or UsageStore(backend=db)
    meter = UsageMeter(store, reg, reset_period_seconds=cfg.usage_reset_period_seconds)
    auth_service = T1AuthService(
        km,
        gk,
        token_secret=cfg.token_secret,
        default_ttl_seconds=cfg.scoped_token_default_ttl_seconds,
        accept_t2_keys=cfg.accept_t2_keys,
        accept_t1_keys=cfg.accept_t1_keys,
    )

    table = routing_table or RoutingTable.load(cfg.routing_policy_path)
    routing_engine = RoutingEngine(table, reg, meter)
    servers = server_registry or ServerRegistry(db)
    modules = module_registry or ModuleRegistry(db, storage_dir=cfg.module_storage_dir)
    bus = event_bus or EventBus()
    jobs = job_queue or JobQueue(db, event_bus=bus)
    billing = billing_ledger or BillingLedger(db)

    # Beta 3 subsystems.
    audit = audit_log or AuditLog(db, enabled=cfg.audit_enabled)
    policy = network_policy or NetworkPolicy(db, allow_unlisted=cfg.allow_unlisted_clients)
    limiter = rate_limiter or RateLimiter(
        rules_from_json(cfg.rate_limit_rules()),
        forced_limit_lookup=policy.get_limit,
        enabled=cfg.rate_limit_enabled,
    )
    directory = key_directory or KeyDirectory(km, reg, db, default_plan=cfg.default_plan)
    cost_calculator = CostCalculator(store, reg)
    module_transport = transport or ModuleTransport(
        deploy_secret=cfg.deploy_secret,
        max_bytes=cfg.max_transfer_bytes,
        allow_private_targets=cfg.allow_private_deploy_targets,
    )
    deployment = DeploymentCoordinator(modules, servers, module_transport, audit=audit)
    jobs.register_handler("module_sync", deployment.module_sync_handler)
    jobs.register_handler("module_fetch", deployment.module_fetch_handler)
    jobs.register_handler("hf_download", _make_hf_download_handler(cfg))

    # T1 v1.0.26.8.1.0 subsystems. The auth history holds key material
    # for reversible rotations, so it gets Keymaster's Fernet when the
    # security extra is installed rather than sitting in plain JSON.
    history = auth_history or AuthHistory(db, fernet=getattr(km, "_fernet", None))
    backups = backup_store or BackupStore(cfg.backup_dir)
    # Beside this app's keys, taken from the Keymaster rather than from
    # the config: cfg.keymaster_dir is None whenever T1_KEYMASTER_DIR is
    # unset, and a bare ServerKeyRegistry() defaults to the operator's
    # real store — so every create_app() in a test suite would write
    # SSPKID assignments into ~/.hypernix, and two servers with different
    # key stores would share one registry. The Keymaster in use is the
    # authority on where these keys are.
    server_keys = server_key_registry or ServerKeyRegistry(store_dir=km.store_dir)

    # T1 v1.0.26.8.0.1 subsystems. All three share the same backend as
    # everything else, so a HyperLink deployment is still one database
    # file (or one PostgreSQL URL) and nothing extra to back up.
    devices = device_registry or DeviceRegistry(db)
    sessions = session_store or ChatSessionStore(db)
    attachments = attachment_store or AttachmentStore(
        cfg.hyperlink_files_dir,
        backend=db,
        max_bytes=cfg.hyperlink_max_upload_bytes,
    )
    # 0.72.4.post9. Same backend again, for the same reason: a phone
    # that can catch up, be notified, and search its own history should
    # not cost the operator a second thing to back up.
    sync = sync_store or SyncStore(db)
    notifications = notification_store or NotificationStore(db)
    searching = search_index or SearchIndex(db)

    tls = cfg.tls_settings()
    cert_verifier = ClientCertVerifier(tls)

    app = FastAPI(
        title="HyperNix T1 API",
        version=__t1api_version__,
        description=(
            "Controlled gateway into HyperNix-pip. The client requests an "
            "operation; the server decides what exists, what's available, "
            "and how much is left. See wiki/T1-API.md for the full contract."
        ),
        docs_url=f"{prefix}/docs" if cfg.expose_docs else None,
        redoc_url=f"{prefix}/redoc" if cfg.expose_docs else None,
        openapi_url=f"{prefix}/openapi.json" if cfg.expose_docs else None,
    )

    # Dependency wiring lives on app.state so t1api/deps.py never touches
    # a module-level global — this is what makes create_app() safe to call
    # more than once (e.g. once per test) without cross-talk.
    # The first credential this server has, when it has none. Loopback
    # only and three days — see hypernix.t1api.bootstrap for why those
    # two limits are what make printing it acceptable.
    if cfg.bootstrap_key_enabled:
        from .bootstrap import ensure_bootstrap_key

        app.state.t1_bootstrap_key = ensure_bootstrap_key(km)
        _announce_bootstrap_key(app.state.t1_bootstrap_key, cfg)
    else:
        app.state.t1_bootstrap_key = None

    app.state.t1_billing_bindings = BillingBindingStore(db)
    # A binding must not outlive the key it belongs to, and must follow a
    # key that is rotated rather than being dropped. The Keymaster owns
    # both events; this is where the two halves are introduced.
    app.state.t1_billing_bindings.attach_to(km)
    app.state.t1_config = cfg
    app.state.t1_keymaster = km
    app.state.t1_gatekeeper = gk
    app.state.t1_registry = reg
    app.state.t1_backend = db
    app.state.t1_usage_store = store
    app.state.t1_usage_meter = meter
    app.state.t1_auth_service = auth_service
    app.state.t1_routing_engine = routing_engine
    app.state.t1_server_registry = servers
    app.state.t1_module_registry = modules
    app.state.t1_job_queue = jobs
    app.state.t1_event_bus = bus
    app.state.t1_billing_ledger = billing
    app.state.t1_audit_log = audit
    app.state.t1_network_policy = policy
    app.state.t1_rate_limiter = limiter
    app.state.t1_key_directory = directory
    app.state.t1_cost_calculator = cost_calculator
    app.state.t1_transport = module_transport
    app.state.t1_deployment = deployment
    app.state.t1_cert_verifier = cert_verifier
    # T1 v1.0.26.8.0.1
    app.state.t1_device_registry = devices
    app.state.t1_session_store = sessions
    app.state.t1_attachment_store = attachments
    # 0.72.4.post9
    app.state.t1_sync_store = sync
    app.state.t1_notification_store = notifications
    app.state.t1_search_index = searching
    # T1 v1.0.26.8.1.0
    app.state.t1_auth_history = history
    app.state.t1_backup_store = backups
    app.state.t1_server_key_registry = server_keys

    if cfg.cors_allow_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors_allow_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # --- Middleware. Registered in reverse execution order; see the
    # module docstring for what the resulting order is and why. --------

    @app.middleware("http")
    async def _rate_limit(request: Request, call_next):
        """Runs before the route handler, so an over-limit caller is
        turned away before any registry lookup, routing decision, or
        model work happens."""
        if limiter.enabled:
            subjects = [Subject("ip", getattr(request.state, "client_ip", "") or "unknown")]
            # The key subject needs the caller's identity, which we don't
            # have yet — authentication is a route dependency. Rather than
            # authenticate twice, derive a stable pseudonym from the
            # credential itself: a caller's per-key bucket then tracks
            # them without this layer ever validating, storing, or logging
            # the credential. An invalid credential still gets a bucket,
            # which is the point — unauthenticated floods are exactly what
            # a limiter is for.
            credential = _credential_fingerprint(request)
            if credential:
                subjects.append(Subject("key", credential))
            try:
                limiter.check(
                    subjects,
                    path=_unprefixed(request.url.path, prefix),
                    method=request.method,
                )
            except T1APIError as exc:
                return _error_response(request, exc)
        return await call_next(request)

    @app.middleware("http")
    async def _mtls(request: Request, call_next):
        if tls.mtls_enabled:
            cert = cert_verifier.extract(
                headers=dict(request.headers),
                peer_ip=request.client.host if request.client else None,
                transport_cert=_transport_cert(request),
            )
            request.state.client_cert = cert
            try:
                cert_verifier.require(cert, path=_unprefixed(request.url.path, prefix))
            except T1APIError as exc:
                return _error_response(request, exc)
        return await call_next(request)

    @app.middleware("http")
    async def _network_policy(request: Request, call_next):
        """Resolve the client address once, then apply the allow/block
        decision. The resolved address is cached on ``request.state`` so
        audit records and rate limiting see the same value this decision
        was made on."""
        from .deps import resolve_client_ip

        client_ip = resolve_client_ip(request)
        request.state.client_ip = client_ip
        if cfg.network_policy_enabled:
            try:
                policy.require_allowed(client_ip)
            except T1APIError as exc:
                return _error_response(request, exc)
        return await call_next(request)

    @app.middleware("http")
    async def _request_id_and_timing(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        start = time.monotonic()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Response-Time-Ms"] = f"{(time.monotonic() - start) * 1000:.1f}"
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if tls.tls_enabled:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response

    def _error_response(request: Request, exc: T1APIError) -> JSONResponse:
        """Render a T1APIError as the documented JSON envelope.

        Shared by the exception handler *and* the middleware stack.
        Starlette only routes exceptions raised inside the application to
        ``@app.exception_handler``; one raised in an outer
        ``@app.middleware("http")`` propagates past it and surfaces as a
        bare 500 with no error code. Since every Beta 3 security check
        (network policy, mTLS, rate limiting) runs as middleware and
        signals refusal by raising, they each catch and render through
        this instead of relying on a handler that will never see them.
        """
        status_code = exc.http_status or _STATUS_FOR_CODE.get(exc.code, 500)
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        # Never log the credential itself — only the (already-safe) error
        # code, message, and request id.
        logger.info(
            "t1api.error code=%s status=%s path=%s request_id=%s",
            exc.code.value,
            status_code,
            request.url.path,
            request_id,
        )
        if exc.code in _AUDITED_ERROR_CODES:
            audit.record(
                f"security.{exc.code.value.lower()}",
                category=AuditCategory.SECURITY,
                outcome=AuditOutcome.DENIED,
                client_ip=getattr(request.state, "client_ip", ""),
                request_id=request_id,
                details={"path": request.url.path, "method": request.method},
            )
        headers = {"X-Request-ID": request_id}
        if exc.code == T1ErrorCode.RATE_LIMITED:
            retry_after = exc.details.get("retry_after_seconds")
            # None means "not by waiting" (a rule that never refills), and
            # a non-finite value would raise on the int() below — in both
            # cases the honest thing is to send no Retry-After at all
            # rather than a number the client would trust.
            if retry_after is not None and math.isfinite(float(retry_after)):
                headers["Retry-After"] = str(max(1, int(float(retry_after)) + 1))
        for header, value in _SECURITY_HEADERS.items():
            headers.setdefault(header, value)
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {"code": exc.code.value, "message": exc.message, "details": exc.details},
                "request_id": request_id,
            },
            headers=headers,
        )

    @app.exception_handler(T1APIError)
    async def _t1_error_handler(request: Request, exc: T1APIError) -> JSONResponse:
        return _error_response(request, exc)

    for router in ALL_ROUTERS:
        app.include_router(router, prefix=prefix)

    if cfg.is_production:
        logger.info(
            "t1api: production configuration validated (backend=%s, mtls=%s, rate_limit=%s, audit=%s)",
            db.describe,
            tls.public_dict()["mtls_mode"],
            cfg.rate_limit_enabled,
            cfg.audit_enabled,
        )

    return app


def _make_hf_download_handler(cfg: T1APIConfig):
    """Job handler for ``POST /hyperlink/models/download``.

    Built as a closure over the config rather than reaching into
    ``app.state`` at run time: a job outlives the request that queued it,
    and a handler that resolves its settings later is a handler that can
    observe a different configuration than the one the caller was told
    about.
    """
    from pathlib import Path as _Path

    def handler(payload: dict, cancel_event: object = None) -> dict:
        from ..hyperlink.hfdownload import HFDownloader

        repo_id = str(payload.get("repo_id") or "")
        if not repo_id:
            raise ValueError("hf_download job has no repo_id")
        root = _Path(
            payload.get("directory")
            or cfg.hf_download_dir
            or (_Path.home() / ".hypernix" / "models")
        )
        target = root / repo_id.replace("/", "__")
        # A multi-gigabyte download must be interruptible. The queue
        # signals cancellation through this event; the progress callback
        # is the only place that runs often enough to notice.
        def _check_cancelled(_progress) -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Download cancelled")

        downloader = HFDownloader(token=cfg.hf_token, progress=_check_cancelled)
        result = downloader.download(
            repo_id,
            target,
            revision=str(payload.get("revision") or "main"),
            filenames=list(payload.get("filenames") or []) or None,
            kind=str(payload.get("kind") or "auto"),
        )
        return result.to_dict()

    return handler


def _unprefixed(path: str, prefix: str) -> str:
    """Strip a mount prefix so middleware rules written against the
    canonical paths (``/models/route``) keep matching when the app is
    mounted at ``/t1``."""
    if prefix and path.startswith(prefix):
        return path[len(prefix) :] or "/"
    return path


def _credential_fingerprint(request: Request) -> str:
    """A stable, non-reversible per-credential bucket id.

    SHA-256 of the Authorization header, truncated. Never stored, never
    logged, never compared against anything but itself — its only job is
    to give the same caller the same rate-limit bucket without this layer
    handling the credential in any recoverable form.
    """
    header = request.headers.get("authorization", "")
    if not header:
        return ""
    import hashlib

    return hashlib.sha256(header.encode("utf-8")).hexdigest()[:32]


def _transport_cert(request: Request) -> dict | None:
    """The peer certificate uvicorn negotiated, when it terminated TLS.

    Exposed through the ASGI ``extensions['tls']`` scope extension. Absent
    behind a proxy (there is no TLS at this hop) and absent on servers
    that don't implement the extension — both handled by returning None,
    which sends the verifier down its header path instead.
    """
    extensions = request.scope.get("extensions") or {}
    tls_scope = extensions.get("tls") or {}
    peercert = tls_scope.get("client_cert_chain") or tls_scope.get("peercert")
    if isinstance(peercert, dict):
        return peercert
    return None


__all__ = ["create_app"]
