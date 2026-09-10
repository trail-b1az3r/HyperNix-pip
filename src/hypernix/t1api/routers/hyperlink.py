"""``/hyperlink`` — the surface the iOS app talks to.

New in **T1 v1.0.26.8.0.1**. Six groups of endpoints, one per thing the
app has to do:

======================  ==============================================
``/hyperlink/endpoints``  which addresses this PC answers on
``/hyperlink/pair``       mint / redeem / revoke pairing codes
``/hyperlink/devices``    list, rename and unpair phones
``/hyperlink/sessions``   conversations, and the chat turn itself
``/hyperlink/files``      images, documents and code
``/hyperlink/models``     resolve a Hugging Face link into a download
======================  ==============================================

Exactly one endpoint here is unauthenticated: ``POST
/hyperlink/pair/redeem``. It has to be — the phone has no credential yet,
that is what it is asking for — and the pairing code is the credential
for that one call: single-use, ten-minute life, five attempts, minted
only by an admin on the PC. Everything else requires either a device
token or a T1 key (see ``deps.get_hyperlink_principal``).

Ownership runs through ``principal.owner``, which for a device is the
key that paired it. So sessions and files are shared between an
operator's desktop client and their phone, and are invisible to another
operator's — the property that makes "carry on from the phone" work
without making "read the other user's chats" work too.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from ...bridge.lmstudio import LMStudioBridge, LMStudioError
from ...hyperlink.discovery import advertise
from ...hyperlink.files import AttachmentStore
from ...hyperlink.hfmerge import HFResolveError
from ...hyperlink.hfmerge import resolve as hf_resolve
from ...hyperlink.identity import fingerprint as server_fingerprint
from ...hyperlink.notify import EventKind, NotificationStore
from ...hyperlink.pairing import DeviceRegistry, pairing_payload
from ...hyperlink.search import SearchIndex
from ...hyperlink.sessions import ChatMessage, ChatSessionStore
from ...hyperlink.sync import SyncStore
from ..audit import AuditCategory, AuditOutcome
from ..config import T1APIConfig
from ..deps import (
    HyperLinkPrincipal,
    get_attachment_store,
    get_audit_log,
    get_client_ip,
    get_config,
    get_device_registry,
    get_hyperlink_principal,
    get_job_queue,
    get_notification_store,
    get_origin,
    get_request_id,
    get_search_index,
    get_session_store,
    get_sync_store,
    get_trust_policy,
    require_hyperlink_admin,
)
from ..errors import T1APIError, T1ErrorCode
from ..schemas import (
    AttachmentListResponse,
    AttachmentResponse,
    AttachmentSummary,
    DeviceListResponse,
    DeviceResponse,
    DeviceSummary,
    DownloadedModelsResponse,
    GenericOkResponse,
    HFDownloadRequest,
    HFDownloadResponse,
    HFFile,
    HFResolveRequest,
    HFResolveResponse,
    HyperLinkChatRequest,
    HyperLinkChatResponse,
    HyperLinkEndpoint,
    HyperLinkEndpointsResponse,
    HyperLinkPeer,
    HyperLinkPeersResponse,
    MessageListResponse,
    MessageSummary,
    PairingCodeResponse,
    PairingCreateRequest,
    PairingRedeemRequest,
    PairingRedeemResponse,
    PushEventsRequest,
    PushRegisterRequest,
    PushRegistrationListResponse,
    PushRegistrationResponse,
    PushRegistrationSummary,
    SearchHit,
    SearchResponse,
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
    SessionSummary,
    SessionUpdateRequest,
    SyncChange,
    SyncClaimRequest,
    SyncClaimResponse,
    SyncPageResponse,
)
from ..version import T1_VERSION

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hyperlink", tags=["hyperlink"])


def _require_enabled(config: T1APIConfig) -> None:
    if not config.hyperlink_enabled:
        raise T1APIError(
            T1ErrorCode.NOT_SUPPORTED,
            "HyperLink is disabled on this server (T1_HYPERLINK_ENABLED=0).",
            http_status=501,
        )


def _advertised_port(config: T1APIConfig, request: Request | None) -> int:
    """The port to hand a phone, which is not the one in the config.

    uvicorn owns the bind address; nothing passes it to the app. So the
    config's port was a default — 8000 — and a server started on any
    other port advertised endpoints pointing at 8000. The phone dutifully
    connected there and timed out, and the server log stayed empty
    because nothing ever arrived.

    The request knows. Whatever address a client just successfully
    reached this server on is, by construction, an address that works, so
    that is what gets advertised. An explicit T1_HYPERLINK_PORT still
    wins: a deployment behind a proxy that forwards to a different port
    is telling us something the request cannot.
    """
    if config.hyperlink_port_is_explicit:
        return config.hyperlink_advertised_port
    if request is not None:
        port = request.url.port
        if port:
            return int(port)
        # No port in the Host header means the scheme's default.
        return 443 if request.url.scheme == "https" else 80
    return config.hyperlink_advertised_port


def _endpoints(config: T1APIConfig, request: Request | None = None) -> dict[str, Any]:
    return advertise(
        port=_advertised_port(config, request),
        configured=config.hyperlink_public_url,
        t1_version=T1_VERSION.short,
    )


# ---------------------------------------------------------------------------
# Endpoints advertisement
# ---------------------------------------------------------------------------


@router.get("/endpoints", response_model=HyperLinkEndpointsResponse)
def list_endpoints(
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> HyperLinkEndpointsResponse:
    """Every address this machine can be reached on, best first.

    Authenticated, despite looking innocuous: the answer is a list of a
    machine's internal addresses and whether it is on a tailnet, which
    is reconnaissance if handed to anyone who asks.
    """
    _require_enabled(config)
    payload = _endpoints(config, request)
    policy = get_trust_policy(request)
    origin = get_origin(request)
    return HyperLinkEndpointsResponse(
        server_name=payload["server_name"],
        t1_version=payload["t1_version"],
        endpoints=[HyperLinkEndpoint(**e) for e in payload["endpoints"]],
        tailscale=payload["tailscale"],
        reachable_off_lan=payload["reachable_off_lan"],
        # The identity a client pins. Returned here rather than from an
        # open endpoint because this call is already authenticated, and
        # publishing a stable machine identifier to anyone who can reach
        # the port is reconnaissance for no gain.
        server_fingerprint=server_fingerprint(),
        trusted_network=policy.enabled,
        keyless_available_here=policy.enabled and policy.allows_keyless(origin),
        origin_trust=origin.trust.value,
        request_id=request_id,
    )


@router.get("/peers", response_model=HyperLinkPeersResponse)
def list_peers(
    request: Request,
    include_offline: bool = Query(False, description="Also list peers that are down."),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> HyperLinkPeersResponse:
    """Other HyperNix machines on this tailnet, as *candidates*.

    Someone with a desktop and a laptop should not have to go and look
    up the laptop's tailnet name to add it to their phone. This machine
    is on the same tailnet and already knows.

    Admin-only, and worth being explicit about why: the answer is a map
    of somebody's private network. A device token is a phone's
    credential for *this* server; it is not authority to enumerate every
    other machine its owner runs.

    Every result is ``verified: false``. Nothing here authenticates
    anything — the probe is a GET to ``/health`` and the only use made of
    the reply is to display two strings from it. A client that acts on
    one of these still authenticates against it, and compares the
    fingerprint that comes back with the one it pinned. The peer's
    advertised name is for a human choosing from a list, never for the
    decision to trust.
    """
    _require_enabled(config)
    require_hyperlink_admin(principal)

    from ...hyperlink import peers as _peers

    found = _peers.discover(
        port=_advertised_port(config, request),
        include_offline=include_offline,
    )
    if not found:
        from ...hyperlink.discovery import tailscale_diagnosis

        return HyperLinkPeersResponse(
            tailscale=False,
            detail=tailscale_diagnosis() or "no other machines on this tailnet.",
            request_id=request_id,
        )
    return HyperLinkPeersResponse(
        peers=[HyperLinkPeer(**p.to_dict()) for p in found],
        count=len(found),
        reachable=sum(1 for p in found if p.reachable),
        tailscale=True,
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


@router.post("/pair", response_model=PairingCodeResponse)
def create_pairing_code(
    payload: PairingCreateRequest,
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    registry: DeviceRegistry = Depends(get_device_registry),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
    audit=Depends(get_audit_log),
) -> PairingCodeResponse:
    """Mint a code for a phone to redeem. Admin only, on the PC."""
    _require_enabled(config)
    require_hyperlink_admin(principal)
    ttl = payload.ttl_seconds or config.hyperlink_pairing_ttl_seconds
    code = registry.create_code(
        created_by=principal.owner,
        scopes=payload.scopes,
        label=payload.label,
        ttl_seconds=ttl,
    )
    endpoints = _endpoints(config, request)
    audit.record(
        "hyperlink.pair.create",
        category=AuditCategory.ADMIN,
        actor_key_id=principal.owner,
        actor_is_admin=True,
        outcome=AuditOutcome.SUCCESS,
        client_ip=get_client_ip(request),
        request_id=request_id,
        details={"label": payload.label, "ttl_seconds": ttl},
    )
    return PairingCodeResponse(
        code=code.code,
        expires_at=code.expires_at,
        seconds_remaining=code.seconds_remaining,
        scopes=list(code.scopes),
        label=code.label,
        qr_payload=pairing_payload(
            code,
            endpoints=[e["url"] for e in endpoints["endpoints"]],
            server_name=endpoints["server_name"],
            t1_version=T1_VERSION.short,
        ),
        endpoints=[HyperLinkEndpoint(**e) for e in endpoints["endpoints"]],
        request_id=request_id,
    )


@router.post("/pair/redeem", response_model=PairingRedeemResponse)
def redeem_pairing_code(
    payload: PairingRedeemRequest,
    request: Request,
    registry: DeviceRegistry = Depends(get_device_registry),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
    audit=Depends(get_audit_log),
) -> PairingRedeemResponse:
    """Exchange a pairing code for this device's permanent token.

    The only unauthenticated endpoint in HyperLink. Every failure is
    recorded in the audit log with the client address, because a run of
    failures against this endpoint is exactly what an attempt to guess a
    code looks like.
    """
    _require_enabled(config)
    client_ip = get_client_ip(request)
    if not payload.device_name.strip():
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "device_name is required")
    try:
        record, token = registry.redeem(
            payload.code,
            device_name=payload.device_name,
            platform=payload.platform,
            app_version=payload.app_version,
            address=client_ip,
        )
    except T1APIError as exc:
        # Count a wrong code against the code that was tried, when it is
        # one that exists. An expired or already-used code is a user
        # mistake, not a guess, and consuming an attempt for it would
        # let a legitimate retry exhaust the budget.
        if exc.code is T1ErrorCode.NOT_FOUND:
            registry.note_failed_attempt(payload.code)
        audit.record(
            "hyperlink.pair.redeem",
            category=AuditCategory.SECURITY,
            actor_key_id=f"device:{payload.device_name}",
            outcome=AuditOutcome.DENIED,
            client_ip=client_ip,
            request_id=request_id,
            details={"reason": str(exc)},
        )
        raise

    endpoints = _endpoints(config, request)
    audit.record(
        "hyperlink.pair.redeem",
        category=AuditCategory.ADMIN,
        actor_key_id=record.paired_by,
        outcome=AuditOutcome.SUCCESS,
        resource_type="hyperlink_device",
        resource_id=record.device_id,
        client_ip=client_ip,
        request_id=request_id,
        details={"device_id": record.device_id, "device_name": record.name},
    )
    return PairingRedeemResponse(
        device_id=record.device_id,
        device_token=token,
        name=record.name,
        scopes=list(record.scopes),
        server_name=endpoints["server_name"],
        t1_version=T1_VERSION.short,
        request_id=request_id,
    )


@router.delete("/pair/{code}", response_model=GenericOkResponse)
def revoke_pairing_code(
    code: str,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    registry: DeviceRegistry = Depends(get_device_registry),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> GenericOkResponse:
    """Cancel a code that has not been redeemed yet."""
    _require_enabled(config)
    require_hyperlink_admin(principal)
    removed = registry.revoke_code(code)
    return GenericOkResponse(
        ok=removed,
        detail="Pairing code cancelled" if removed else "No such unredeemed pairing code",
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------


@router.get("/devices", response_model=DeviceListResponse)
def list_devices(
    include_revoked: bool = Query(default=False),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    registry: DeviceRegistry = Depends(get_device_registry),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> DeviceListResponse:
    _require_enabled(config)
    require_hyperlink_admin(principal)
    devices = registry.list_devices(include_revoked=include_revoked)
    return DeviceListResponse(
        devices=[DeviceSummary(**d.to_dict()) for d in devices],
        count=len(devices),
        request_id=request_id,
    )


@router.get("/devices/me", response_model=DeviceResponse)
def whoami(
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    registry: DeviceRegistry = Depends(get_device_registry),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> DeviceResponse:
    """What the caller's own device token resolves to.

    The app calls this on launch to decide between "signed in" and "show
    the pairing screen", so it must be cheap and must fail with a clear
    401 rather than an empty success when the token has been revoked.
    """
    _require_enabled(config)
    if not principal.is_device:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            "This endpoint describes a paired device; you authenticated with a T1 key. "
            "Use GET /auth/whoami instead.",
        )
    return DeviceResponse(
        device=DeviceSummary(**registry.get_device(principal.device_id).to_dict()),
        request_id=request_id,
    )


@router.patch("/devices/{device_id}", response_model=DeviceResponse)
def rename_device(
    device_id: str,
    payload: dict[str, Any],
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    registry: DeviceRegistry = Depends(get_device_registry),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> DeviceResponse:
    """Rename a device. A device may rename itself; only an admin may
    rename another one — otherwise a phone could relabel itself as
    somebody else's before doing something regrettable."""
    _require_enabled(config)
    if not (principal.is_admin or principal.device_id == device_id):
        require_hyperlink_admin(principal)
    name = str(payload.get("name") or "")
    record = registry.rename_device(device_id, name)
    return DeviceResponse(device=DeviceSummary(**record.to_dict()), request_id=request_id)


@router.delete("/devices/{device_id}", response_model=DeviceResponse)
def revoke_device(
    device_id: str,
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    registry: DeviceRegistry = Depends(get_device_registry),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
    audit=Depends(get_audit_log),
) -> DeviceResponse:
    """Unpair a device.

    A device may unpair *itself* — that is the app's "sign out", and
    requiring an admin for it would leave a wiped phone's token valid
    until somebody noticed. Unpairing anything else is an admin action.
    """
    _require_enabled(config)
    if not (principal.is_admin or principal.device_id == device_id):
        require_hyperlink_admin(principal)
    record = registry.revoke_device(device_id)
    audit.record(
        "hyperlink.device.revoke",
        category=AuditCategory.ADMIN,
        actor_key_id=principal.label,
        actor_is_admin=principal.is_admin,
        outcome=AuditOutcome.SUCCESS,
        resource_type="hyperlink_device",
        resource_id=device_id,
        client_ip=get_client_ip(request),
        request_id=request_id,
        details={"device_id": device_id, "device_name": record.name},
    )
    return DeviceResponse(device=DeviceSummary(**record.to_dict()), request_id=request_id)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@router.get("/sessions", response_model=SessionListResponse)
def list_sessions(
    include_archived: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: ChatSessionStore = Depends(get_session_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> SessionListResponse:
    _require_enabled(config)
    sessions = store.list_sessions(
        owner=principal.owner, include_archived=include_archived, limit=limit, offset=offset
    )
    return SessionListResponse(
        sessions=[SessionSummary(**s.to_dict()) for s in sessions],
        count=len(sessions),
        request_id=request_id,
    )


@router.post("/sessions", response_model=SessionResponse)
def create_session(
    payload: SessionCreateRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: ChatSessionStore = Depends(get_session_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> SessionResponse:
    _require_enabled(config)
    session = store.create(
        owner=principal.owner,
        title=payload.title,
        model_id=payload.model_id,
        backend=payload.backend,
        system_prompt=payload.system_prompt,
        device_id=principal.device_id,
    )
    return SessionResponse(
        session=SessionSummary(**store.get(session.session_id).to_dict()), request_id=request_id
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: str,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: ChatSessionStore = Depends(get_session_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> SessionResponse:
    _require_enabled(config)
    session = store.get(session_id, owner=principal.owner)
    return SessionResponse(session=SessionSummary(**session.to_dict()), request_id=request_id)


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
def update_session(
    session_id: str,
    payload: SessionUpdateRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: ChatSessionStore = Depends(get_session_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> SessionResponse:
    _require_enabled(config)
    session = store.update(
        session_id,
        owner=principal.owner,
        title=payload.title,
        model_id=payload.model_id,
        backend=payload.backend,
        system_prompt=payload.system_prompt,
        archived=payload.archived,
    )
    return SessionResponse(session=SessionSummary(**session.to_dict()), request_id=request_id)


@router.delete("/sessions/{session_id}", response_model=GenericOkResponse)
def delete_session(
    session_id: str,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: ChatSessionStore = Depends(get_session_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> GenericOkResponse:
    _require_enabled(config)
    store.delete(session_id, owner=principal.owner)
    return GenericOkResponse(ok=True, detail="Session deleted", request_id=request_id)


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
def list_messages(
    session_id: str,
    after_seq: int = Query(default=0, ge=0, description="Incremental sync: only newer messages"),
    limit: int = Query(default=0, ge=0, le=1000),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: ChatSessionStore = Depends(get_session_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> MessageListResponse:
    _require_enabled(config)
    messages = store.messages(
        session_id, owner=principal.owner, limit=limit, after_seq=after_seq
    )
    return MessageListResponse(
        session_id=session_id,
        messages=[MessageSummary(**_msg_dict(m)) for m in messages],
        count=len(messages),
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# The chat turn
# ---------------------------------------------------------------------------


def _msg_dict(message: ChatMessage) -> dict[str, Any]:
    return message.to_dict()


def _wire_messages(
    history: list[ChatMessage],
    store: AttachmentStore,
    owner: str,
) -> list[dict[str, Any]]:
    """Turn stored messages into the shape a model server accepts.

    Attachments are expanded here, at the last moment, and differently by
    type:

    * **Images** become ``image_url`` parts with an inline data URL —
      what every OpenAI-compatible vision model takes.
    * **Text and code** become a fenced block in the message text, with
      the filename in the fence info. A model reads that correctly, and
      unlike a base64 blob it costs the tokens it looks like it costs.
    * **Anything else** becomes a one-line note naming the file and its
      type, so the model can say "I can't read a 40 MB zip" instead of
      hallucinating its contents.

    A missing attachment is skipped with a note rather than failing the
    turn: the alternative is a conversation that can never be continued
    because one old screenshot was deleted.
    """
    wire: list[dict[str, Any]] = []
    for message in history:
        if not message.attachment_ids:
            wire.append({"role": message.role, "content": message.content})
            continue

        parts: list[dict[str, Any]] = []
        text = message.content
        for file_id in message.attachment_ids:
            try:
                record = store.get(file_id, owner=owner)
                if record.is_image:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": store.data_url(file_id, owner=owner)},
                        }
                    )
                elif record.is_text:
                    body = store.text_of(file_id, owner=owner)
                    lang = _fence_language(record.filename)
                    text += f"\n\n--- {record.filename} ---\n```{lang}\n{body}\n```"
                else:
                    text += (
                        f"\n\n[attached: {record.filename} "
                        f"({record.content_type}, {record.size_bytes} bytes) — "
                        "binary, contents not included]"
                    )
            except T1APIError:
                text += f"\n\n[attachment {file_id} is no longer available]"

        if parts:
            wire.append({"role": message.role, "content": [{"type": "text", "text": text}, *parts]})
        else:
            wire.append({"role": message.role, "content": text})
    return wire


def _fence_language(filename: str) -> str:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {
        "py": "python", "swift": "swift", "js": "javascript", "ts": "typescript",
        "rs": "rust", "go": "go", "c": "c", "h": "c", "cpp": "cpp", "java": "java",
        "sh": "bash", "yml": "yaml", "yaml": "yaml", "toml": "toml", "json": "json",
        "md": "markdown", "html": "html", "css": "css", "sql": "sql",
    }.get(suffix, "")


def _chat_bridge(config: T1APIConfig) -> LMStudioBridge:
    if not config.lmstudio_enabled:
        raise T1APIError(
            T1ErrorCode.NOT_SUPPORTED,
            "This server has no chat backend configured: the LM Studio bridge is disabled.",
            http_status=501,
        )
    return LMStudioBridge(
        config.lmstudio_url or None,
        api_key=config.lmstudio_api_key,
        timeout=config.lmstudio_timeout_seconds,
    )


def _bridge_error(exc: LMStudioError) -> T1APIError:
    status = {
        "unreachable": 503,
        "timeout": 504,
        "no_model_loaded": 503,
        "model_not_found": 404,
    }.get(exc.code, 502)
    code = (
        T1ErrorCode.MODEL_NOT_SUPPORTED
        if exc.code == "model_not_found"
        else T1ErrorCode.MODEL_UNAVAILABLE
    )
    return T1APIError(code, str(exc), details=exc.to_dict(), http_status=status)


@router.post("/sessions/{session_id}/chat", response_model=HyperLinkChatResponse)
def chat_turn(
    session_id: str,
    payload: HyperLinkChatRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: ChatSessionStore = Depends(get_session_store),
    files: AttachmentStore = Depends(get_attachment_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> HyperLinkChatResponse:
    """Append the user's message, run the model, append the reply.

    Both messages are persisted, in that order, and the user's message
    is persisted **before** inference runs. If the model call then fails
    — the desktop went to sleep, nothing is loaded — the question is
    still in the thread, and the app can retry it instead of the user
    retyping it.
    """
    _require_enabled(config)
    if not payload.content.strip() and not payload.attachment_ids:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR, "Send some text, an attachment, or both"
        )
    session = store.get(session_id, owner=principal.owner)

    user_message = store.append(
        session_id,
        role="user",
        content=payload.content,
        owner=principal.owner,
        attachment_ids=payload.attachment_ids,
        metadata={"device_id": principal.device_id} if principal.device_id else {},
    )

    bridge = _chat_bridge(config)
    history = store.context_for(
        session_id, owner=principal.owner, token_budget=payload.token_budget
    )
    wire = _wire_messages(history, files, principal.owner)

    started = time.monotonic()
    try:
        envelope = bridge.chat(
            wire,
            model=payload.model_id or session.model_id or None,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
    except LMStudioError as exc:
        raise _bridge_error(exc) from exc
    elapsed = time.monotonic() - started

    content, finish = _extract_reply(envelope)
    usage = envelope.get("usage") or {}
    model_id = str(envelope.get("model") or payload.model_id or session.model_id or "")
    assistant_message = store.append(
        session_id,
        role="assistant",
        content=content,
        owner=principal.owner,
        model_id=model_id,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        metadata={
            "finish_reason": finish,
            "backend": "lmstudio",
            "base_url": bridge.base_url,
            "elapsed_seconds": round(elapsed, 3),
        },
    )
    store.autotitle(session_id, owner=principal.owner)
    if model_id and not session.model_id:
        store.update(session_id, owner=principal.owner, model_id=model_id, backend="lmstudio")

    return HyperLinkChatResponse(
        session_id=session_id,
        user_message=MessageSummary(**_msg_dict(user_message)),
        assistant_message=MessageSummary(**_msg_dict(assistant_message)),
        model_id=model_id,
        backend="lmstudio",
        request_id=request_id,
    )


@router.post("/sessions/{session_id}/chat/stream")
def chat_turn_stream(
    session_id: str,
    payload: HyperLinkChatRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: ChatSessionStore = Depends(get_session_store),
    files: AttachmentStore = Depends(get_attachment_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> StreamingResponse:
    """The same turn, streamed token by token.

    Frames are HyperLink's own small shape rather than raw OpenAI
    chunks, because the app needs three things the OpenAI stream does
    not carry: the persisted ``message_id`` of the user's message (sent
    first, so an optimistic bubble can be reconciled), a terminal frame
    with the assistant ``message_id`` and token counts (so the thread
    matches what a later ``GET /messages`` will return), and errors
    delivered in-band.

    The assistant's reply is persisted from the accumulated text when the
    stream finishes — including when the client disconnects half way,
    which is the common case on a phone. A partial answer saved and
    marked ``truncated`` is worth more than a lost one.
    """
    _require_enabled(config)
    if not payload.content.strip() and not payload.attachment_ids:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "Send some text, an attachment, or both")
    session = store.get(session_id, owner=principal.owner)

    user_message = store.append(
        session_id,
        role="user",
        content=payload.content,
        owner=principal.owner,
        attachment_ids=payload.attachment_ids,
        metadata={"device_id": principal.device_id} if principal.device_id else {},
    )
    bridge = _chat_bridge(config)
    history = store.context_for(
        session_id, owner=principal.owner, token_budget=payload.token_budget
    )
    wire = _wire_messages(history, files, principal.owner)
    requested_model = payload.model_id or session.model_id or None

    def _frame(kind: str, **fields: Any) -> bytes:
        return f"data: {json.dumps({'type': kind, **fields})}\n\n".encode()

    def _events():
        yield _frame(
            "start",
            session_id=session_id,
            user_message_id=user_message.message_id,
            seq=user_message.seq,
        )
        collected: list[str] = []
        model_id = requested_model or ""
        finish = ""
        usage: dict[str, Any] = {}
        error: dict[str, Any] | None = None
        try:
            for chunk in bridge.chat_stream(
                wire,
                model=requested_model,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
            ):
                model_id = str(chunk.get("model") or model_id)
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(piece, str) and piece:
                        collected.append(piece)
                        yield _frame("delta", text=piece)
                    if choice.get("finish_reason"):
                        finish = str(choice["finish_reason"])
        except LMStudioError as exc:
            error = exc.to_dict()
            yield _frame("error", error=error)
        except GeneratorExit:
            # The phone closed the connection (backgrounded, tunnel
            # dropped). Persist what arrived, then let the exit
            # propagate — swallowing it would leak the generator.
            _persist(collected, model_id, finish or "disconnected", usage, truncated=True)
            raise

        message = _persist(collected, model_id, finish, usage, truncated=bool(error))
        yield _frame(
            "done",
            message_id=message.message_id,
            seq=message.seq,
            model_id=model_id,
            finish_reason=finish,
            input_tokens=message.input_tokens,
            output_tokens=message.output_tokens,
        )
        yield b"data: [DONE]\n\n"

    def _persist(
        pieces: list[str], model_id: str, finish: str, usage: dict[str, Any], *, truncated: bool
    ) -> ChatMessage:
        message = store.append(
            session_id,
            role="assistant",
            content="".join(pieces),
            owner=principal.owner,
            model_id=model_id,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            metadata={
                "finish_reason": finish,
                "backend": "lmstudio",
                "base_url": bridge.base_url,
                "truncated": truncated,
                "streamed": True,
            },
        )
        store.autotitle(session_id, owner=principal.owner)
        return message

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Request-Id": request_id,
            "X-Accel-Buffering": "no",
        },
    )


def _extract_reply(envelope: dict[str, Any]) -> tuple[str, str]:
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "", ""
    first = choices[0]
    message = first.get("message") or {}
    raw = message.get("content") if isinstance(message, dict) else ""
    if isinstance(raw, list):
        raw = "".join(
            str(p.get("text", "")) for p in raw if isinstance(p, dict) and p.get("type") == "text"
        )
    return (raw if isinstance(raw, str) else ""), str(first.get("finish_reason") or "")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


@router.post("/files", response_model=AttachmentResponse)
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(default=""),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: AttachmentStore = Depends(get_attachment_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> AttachmentResponse:
    """Upload one image, document or source file.

    The size limit is checked against the bytes actually read, not
    against ``Content-Length``: the header is client-supplied, and a
    limit enforced on a number the client chose is not a limit. Reading
    is capped at one byte over the limit so an oversized upload is
    rejected without buffering all of it.
    """
    limit = config.hyperlink_max_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            f"{file.filename or 'file'} exceeds the {limit}-byte upload limit for this server",
            http_status=413,
        )
    record = store.put(
        data,
        filename=file.filename or "upload",
        owner=principal.owner,
        device_id=principal.device_id,
        session_id=session_id,
        declared_type=file.content_type or "",
    )
    return AttachmentResponse(file=AttachmentSummary(**record.to_dict()), request_id=request_id)


@router.get("/files", response_model=AttachmentListResponse)
def list_attachments(
    session_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: AttachmentStore = Depends(get_attachment_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> AttachmentListResponse:
    _require_enabled(config)
    records = store.list_files(owner=principal.owner, session_id=session_id, limit=limit)
    return AttachmentListResponse(
        files=[AttachmentSummary(**r.to_dict()) for r in records],
        count=len(records),
        total_bytes=store.usage_bytes(owner=principal.owner),
        request_id=request_id,
    )


@router.get("/files/{file_id}")
def download_file(
    file_id: str,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: AttachmentStore = Depends(get_attachment_store),
    config: T1APIConfig = Depends(get_config),
) -> Response:
    """The raw bytes.

    ``Content-Disposition: attachment`` and ``X-Content-Type-Options:
    nosniff`` on every response, including images: this server may be
    reached from a WKWebView, and a stored file that renders inline as
    HTML in the app's own origin would be stored XSS.
    """
    _require_enabled(config)
    record = store.get(file_id, owner=principal.owner)
    return Response(
        content=store.read(file_id, owner=principal.owner),
        media_type=record.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{record.filename}"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, max-age=31536000, immutable",
        },
    )


@router.delete("/files/{file_id}", response_model=GenericOkResponse)
def delete_file(
    file_id: str,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: AttachmentStore = Depends(get_attachment_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> GenericOkResponse:
    _require_enabled(config)
    store.delete(file_id, owner=principal.owner)
    return GenericOkResponse(ok=True, detail="Attachment deleted", request_id=request_id)


# ---------------------------------------------------------------------------
# Hugging Face resolution
# ---------------------------------------------------------------------------


@router.post("/models/resolve", response_model=HFResolveResponse)
def resolve_model_link(
    payload: HFResolveRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> HFResolveResponse:
    """Turn a model page and/or a direct file link into a download plan.

    Resolution happens on the *server* rather than in the app for two
    reasons: the server may hold an ``HF_TOKEN`` for gated repositories,
    and the plan is what the server will act on when the download is
    started, so having the app compute one and send it over would mean
    trusting a client-supplied URL list.
    """
    _require_enabled(config)
    if payload.prefer not in ("strict", "file", "page"):
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR, "prefer must be one of: strict, file, page"
        )
    try:
        resolved = hf_resolve(
            payload.page_url,
            payload.file_url,
            prefer=payload.prefer,
            token=config.hf_token,
            include_vision=payload.include_vision,
            offline=payload.offline,
        )
    except HFResolveError as exc:
        status = {
            "repo_conflict": 409,
            "no_repo": 404,
            "file_not_found": 404,
            "no_gguf": 404,
            "offline": 503,
        }.get(exc.code, 400)
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR if status == 400 else T1ErrorCode.NOT_FOUND,
            str(exc),
            details=exc.to_dict(),
            http_status=status,
        ) from exc

    data = resolved.to_dict()
    return HFResolveResponse(
        **{k: v for k, v in data.items() if k != "files"},
        files=[HFFile(**f) for f in data["files"]],
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Hugging Face downloads
# ---------------------------------------------------------------------------


@router.post("/models/download", response_model=HFDownloadResponse)
def download_model(
    payload: HFDownloadRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
    jobs=Depends(get_job_queue),
) -> HFDownloadResponse:
    """Start downloading a model onto the **server**, not the phone.

    Queued as a job rather than run inline for the obvious reason: a
    70B download is not going to finish inside an HTTP request, and a
    phone that backgrounds mid-transfer must not cancel it. The response
    carries a job id; progress arrives over the live-stream channel and
    through ``GET /jobs/{id}``.

    The model lands on the machine with the GPU because that is the
    machine that will run it. Pulling gigabytes onto a phone to send them
    back would be the wrong direction on every axis.
    """
    _require_enabled(config)
    if not config.hf_downloads_enabled:
        raise T1APIError(
            T1ErrorCode.NOT_SUPPORTED,
            "Model downloads are disabled on this server (T1_HF_DOWNLOADS_ENABLED=0).",
            http_status=501,
        )
    if "write" not in principal.scopes:
        raise T1APIError(
            T1ErrorCode.AUTH_INSUFFICIENT_SCOPE,
            "Downloading a model writes to this server's disk and needs a write scope.",
            http_status=403,
        )
    if not payload.repo_id and not (payload.page_url or payload.file_url):
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            "Give a repo_id, or a page_url/file_url to resolve first.",
        )

    # Resolve links to a concrete file list, so the job records exactly
    # what it is going to fetch rather than deciding later.
    repo_id, revision, filenames = payload.repo_id, payload.revision, list(payload.filenames)
    if not repo_id:
        try:
            resolved = hf_resolve(
                payload.page_url, payload.file_url,
                prefer=payload.prefer, token=config.hf_token,
            )
        except HFResolveError as exc:
            raise T1APIError(
                T1ErrorCode.VALIDATION_ERROR, str(exc), details=exc.to_dict(), http_status=400
            ) from exc
        repo_id, revision = resolved.repo_id, resolved.revision
        filenames = [f.filename for f in resolved.files]

    queued = jobs.submit(
        "hf_download",
        payload={
            "repo_id": repo_id,
            "revision": revision or "main",
            "filenames": filenames,
            "kind": payload.kind,
            "directory": str(config.hf_download_dir or ""),
        },
        created_by=principal.owner,
    )
    return HFDownloadResponse(
        job_id=queued.job_id,
        repo_id=repo_id,
        revision=revision or "main",
        filenames=filenames,
        kind=payload.kind,
        request_id=request_id,
    )


@router.get("/models/downloaded", response_model=DownloadedModelsResponse)
def list_downloaded(
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> DownloadedModelsResponse:
    """What is already on this server's disk."""
    _require_enabled(config)
    root = Path(config.hf_download_dir or (Path.home() / ".hypernix" / "models"))
    models: list[dict[str, Any]] = []
    if root.exists():
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            files = [f for f in entry.rglob("*") if f.is_file() and not f.name.endswith(".part")]
            partial = [f for f in entry.rglob("*.part")]
            models.append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "file_count": len(files),
                    "total_bytes": sum(f.stat().st_size for f in files),
                    "has_gguf": any(f.suffix.lower() == ".gguf" for f in files),
                    # An in-flight or abandoned download is worth showing:
                    # "why is this model not loading" is usually this.
                    "incomplete_files": [f.name for f in partial],
                }
            )
    return DownloadedModelsResponse(
        models=models, count=len(models), directory=str(root), request_id=request_id
    )


# ---------------------------------------------------------------------------
# Catching up (0.72.4.post9)
# ---------------------------------------------------------------------------
#
# A phone loses the route mid-request, is suspended by the OS in the
# middle of a POST, and comes back hours later on a different network.
# Polling GET /sessions tells it the current state of everything, which
# costs a cellular download of conversations it already has and still
# does not reveal that a session was *deleted* -- an absence cannot be
# diffed against a list you no longer trust.


@router.get("/sync", response_model=SyncPageResponse)
def sync_changes(
    cursor: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    session_id: str | None = Query(default=None),
    entities: str | None = Query(
        default=None, description="Comma-separated: session,message,device,attachment"
    ),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: SyncStore = Depends(get_sync_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> SyncPageResponse:
    _require_enabled(config)
    wanted = (
        [e.strip() for e in entities.split(",") if e.strip()] if entities else None
    )
    page = store.since(
        cursor, owner=principal.owner, limit=limit,
        session_id=session_id, entities=wanted,
    )
    return SyncPageResponse(
        changes=[SyncChange(**c.to_dict()) for c in page.changes],
        cursor=page.cursor,
        more=page.more,
        resync_required=page.resync_required,
        head=store.head(owner=principal.owner),
        request_id=request_id,
    )


@router.post("/sync/claim", response_model=SyncClaimResponse)
def claim_client_message(
    payload: SyncClaimRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: SyncStore = Depends(get_sync_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> SyncClaimResponse:
    """Claim an idempotency key before sending a turn.

    The phone cannot tell "the server never saw it" from "the server saw
    it and the reply was lost", so it retries -- and produces two
    identical messages and two replies, one of which cost real tokens
    for nothing. A key minted before the first attempt and reused on
    every retry makes the second attempt return the first one's answer.

    Scoped to the calling device, so two phones cannot collide.
    """
    _require_enabled(config)
    claim = store.claim(
        device_id=principal.device_id or principal.owner,
        client_msg_id=payload.client_msg_id,
        owner=principal.owner,
    )
    return SyncClaimResponse(**claim.to_dict(), request_id=request_id)


# ---------------------------------------------------------------------------
# Push notifications (0.72.4.post9)
# ---------------------------------------------------------------------------


@router.post("/push", response_model=PushRegistrationResponse)
def register_push(
    payload: PushRegisterRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: NotificationStore = Depends(get_notification_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> PushRegistrationResponse:
    """Register this device's APNs token.

    Re-registering the same token updates rather than duplicates: iOS
    hands the app a token on every launch, and a row per launch would
    send every notification once per day the app had been opened.

    The response carries a fingerprint, never the token.
    """
    _require_enabled(config)
    registration = store.register(
        device_id=principal.device_id or principal.owner,
        owner=principal.owner,
        token=payload.token,
        platform=payload.platform,
        bundle_id=payload.bundle_id,
        environment=payload.environment,
        events=payload.events,
    )
    return PushRegistrationResponse(
        registration=PushRegistrationSummary(**registration.to_dict()),
        request_id=request_id,
    )


@router.get("/push", response_model=PushRegistrationListResponse)
def list_push(
    include_disabled: bool = Query(default=False),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: NotificationStore = Depends(get_notification_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> PushRegistrationListResponse:
    _require_enabled(config)
    registrations = store.list_registrations(
        owner=principal.owner, include_disabled=include_disabled
    )
    return PushRegistrationListResponse(
        registrations=[PushRegistrationSummary(**r.to_dict()) for r in registrations],
        count=len(registrations),
        pending=store.pending_count(owner=principal.owner),
        request_id=request_id,
    )


@router.patch("/push/{registration_id}", response_model=PushRegistrationResponse)
def set_push_events(
    registration_id: str,
    payload: PushEventsRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: NotificationStore = Depends(get_notification_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> PushRegistrationResponse:
    _require_enabled(config)
    _require_own_registration(store, registration_id, principal)
    registration = store.set_events(registration_id, payload.events)
    return PushRegistrationResponse(
        registration=PushRegistrationSummary(**registration.to_dict()),
        request_id=request_id,
    )


@router.delete("/push/{registration_id}", response_model=GenericOkResponse)
def unregister_push(
    registration_id: str,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: NotificationStore = Depends(get_notification_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> GenericOkResponse:
    _require_enabled(config)
    _require_own_registration(store, registration_id, principal)
    store.unregister(registration_id)
    return GenericOkResponse(ok=True, request_id=request_id)


@router.get("/push/events", response_model=dict)
def list_push_events(
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict:
    """What a device may subscribe to.

    Served rather than hard-coded in the app, so a server that gains an
    event kind does not need the client updated to offer it.
    """
    _require_enabled(config)
    return {"events": sorted(EventKind.ALL), "request_id": request_id}


def _require_own_registration(
    store: NotificationStore, registration_id: str, principal: HyperLinkPrincipal
) -> None:
    """A registration belongs to its owner.

    Without this, knowing a registration id -- which is not a secret --
    would let any authenticated device silence or delete another
    device's notifications. Checked before every mutation rather than
    trusted from the path.
    """
    registration = store.get_registration(registration_id)
    if registration.owner != principal.owner:
        # Not found, not forbidden: confirming the id exists tells an
        # unauthorised caller something they should not learn.
        raise T1APIError(
            T1ErrorCode.NOT_FOUND, f"No push registration {registration_id!r}"
        )


# ---------------------------------------------------------------------------
# Search (0.72.4.post9)
# ---------------------------------------------------------------------------


@router.get("/search", response_model=SearchResponse)
def search_history(
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(default=25, ge=1, le=200),
    session_id: str | None = Query(default=None),
    include_archived: bool = Query(default=False),
    roles: str | None = Query(default=None, description="Comma-separated"),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    index: SearchIndex = Depends(get_search_index),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> SearchResponse:
    """Search this owner's sessions and messages.

    Matching is done in Python over a bounded, SQL-narrowed candidate
    set rather than with FTS5, because the T1 API runs on SQLite *or*
    PostgreSQL. That also buys Unicode-correct case folding -- `LIKE` is
    case-insensitive for ASCII only, so it does not match "straße" for
    "STRASSE" -- and makes a query containing `%` a search for a percent
    sign rather than a request for every row.
    """
    _require_enabled(config)
    wanted = [r.strip() for r in roles.split(",") if r.strip()] if roles else None
    results = index.search(
        q, owner=principal.owner, limit=limit, session_id=session_id,
        include_archived=include_archived, roles=wanted,
    )
    return SearchResponse(
        hits=[SearchHit(**h.to_dict()) for h in results.hits],
        scanned=results.scanned,
        capped=results.capped,
        terms=results.terms,
        request_id=request_id,
    )


__all__ = ["router"]
