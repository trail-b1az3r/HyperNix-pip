"""``/bridge/lmstudio`` — the T1 API's window onto an LM Studio server.

New in **T1 v1.0.26.8.0.1**, and the reason it lives behind the T1 API
rather than being called directly by clients:

* **One address to configure.** The phone, ``hyped-pro`` and ``waiter``
  all point at the T1 API. Whether the actual inference is happening in
  LM Studio on the desktop, in ``neo_oven``, or on a remote HyperNix
  server is the API's business, not each client's.
* **The existing protections apply unchanged.** Authentication, scopes,
  rate limiting, the audit log and usage accounting all sit in front of
  this router because it is a router. LM Studio has none of those and is
  not trying to; exposing it directly to a phone over a tailnet would
  mean an unauthenticated inference endpoint on the home network.
* **LM Studio does not have to be reachable from the client.** It only
  has to be reachable from the T1 API. That is what lets LM Studio stay
  bound to loopback on the desktop while the phone still uses it.

Every endpoint here reports the address it used. When something is not
working the first question is always "which LM Studio did it ask", and
an answer that omits it costs a round of back-and-forth.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from ...bridge.lmstudio import LMStudioBridge, LMStudioError, discover
from ..auth import AuthContext
from ..config import T1APIConfig
from ..deps import get_auth_context, get_config, get_request_id
from ..disconnect import while_listened_to
from ..errors import T1APIError, T1ErrorCode
from ..schemas import (
    BridgeChatRequest,
    BridgeChatResponse,
    BridgeModel,
    BridgeModelsResponse,
    BridgeProbe,
    BridgeStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bridge/lmstudio", tags=["bridge"])


def _require_enabled(config: T1APIConfig) -> None:
    if not config.lmstudio_enabled:
        raise T1APIError(
            T1ErrorCode.NOT_SUPPORTED,
            "The LM Studio bridge is disabled on this server (T1_LMSTUDIO_ENABLED=0).",
            http_status=501,
        )


def _bridge_for(config: T1APIConfig, override: str | None = None) -> LMStudioBridge:
    """Build a bridge for the configured address, or an explicit override.

    The override is accepted from admins only — see :func:`_check_override`.
    A per-request base URL is genuinely useful (an operator moving a
    model between two machines wants to test both without restarting the
    API) and is also a server-side request forgery primitive if anyone
    may set it, so the two facts are handled separately: this function
    builds, the caller authorises.
    """
    url = (override or config.lmstudio_url or "").strip()
    return LMStudioBridge(
        url or None,
        api_key=config.lmstudio_api_key,
        timeout=config.lmstudio_timeout_seconds,
    )


def _check_override(ctx: AuthContext, override: str | None) -> None:
    if override and not ctx.is_admin:
        raise T1APIError(
            T1ErrorCode.AUTH_ADMIN_REQUIRED,
            "Overriding the LM Studio address per request is an admin operation.",
            http_status=403,
        )


def _as_t1_error(exc: LMStudioError) -> T1APIError:
    """Translate a bridge failure into the T1 error the client expects.

    The mapping matters for the app: ``MODEL_UNAVAILABLE`` (503) makes it
    show "your PC isn't answering", ``MODEL_NOT_SUPPORTED`` (404) makes
    it show "load a model", and a bare 500 makes it show a spinner
    forever.
    """
    mapping = {
        "unreachable": (T1ErrorCode.MODEL_UNAVAILABLE, 503),
        "timeout": (T1ErrorCode.MODEL_UNAVAILABLE, 504),
        "no_model_loaded": (T1ErrorCode.MODEL_UNAVAILABLE, 503),
        "model_not_found": (T1ErrorCode.MODEL_NOT_SUPPORTED, 404),
        "bad_response": (T1ErrorCode.INTERNAL_ERROR, 502),
        "http_error": (T1ErrorCode.INTERNAL_ERROR, 502),
    }
    code, status = mapping.get(exc.code, (T1ErrorCode.INTERNAL_ERROR, 502))
    return T1APIError(code, str(exc), details=exc.to_dict(), http_status=status)


# ---------------------------------------------------------------------------
# Status / discovery
# ---------------------------------------------------------------------------


@router.get("/status", response_model=BridgeStatusResponse)
def bridge_status(
    request: Request,
    discover_hosts: bool = Query(
        default=False,
        alias="discover",
        description="Also sweep localhost and the tailnet for other LM Studio servers (admin).",
    ),
    ctx: AuthContext = Depends(get_auth_context),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> BridgeStatusResponse:
    """Is LM Studio reachable, and is anything loaded in it?

    Discovery is opt-in and admin-only even though it only touches
    loopback and the tailnet: a port sweep triggered by any authenticated
    caller is a scanning primitive, and the answer is only interesting to
    whoever administers the machine anyway.
    """
    _require_enabled(config)
    probe = None
    if config.lmstudio_url:
        probe = BridgeProbe(**_bridge_for(config).probe().to_dict())

    discovered: list[BridgeProbe] = []
    if discover_hosts:
        if not ctx.is_admin:
            raise T1APIError(
                T1ErrorCode.AUTH_ADMIN_REQUIRED,
                "Discovery sweeps are an admin operation.",
                http_status=403,
            )
        if not config.lmstudio_discovery:
            raise T1APIError(
                T1ErrorCode.NOT_SUPPORTED,
                "Discovery is disabled on this server (set T1_LMSTUDIO_DISCOVERY=1).",
                http_status=501,
            )
        discovered = [BridgeProbe(**p.to_dict()) for p in discover()]

    return BridgeStatusResponse(
        enabled=config.lmstudio_enabled,
        configured_url=config.lmstudio_url,
        probe=probe,
        discovered=discovered,
        request_id=request_id,
    )


@router.get("/models", response_model=BridgeModelsResponse)
def bridge_models(
    base_url: str | None = Query(default=None, description="Admin-only per-request override"),
    ctx: AuthContext = Depends(get_auth_context),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> BridgeModelsResponse:
    """Everything LM Studio has, with the loaded ones marked."""
    _require_enabled(config)
    _check_override(ctx, base_url)
    bridge = _bridge_for(config, base_url)
    try:
        models = bridge.list_models()
    except LMStudioError as exc:
        raise _as_t1_error(exc) from exc
    return BridgeModelsResponse(
        base_url=bridge.base_url,
        models=[BridgeModel(**m.to_dict()) for m in models],
        count=len(models),
        loaded_count=sum(1 for m in models if m.loaded),
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=BridgeChatResponse)
def bridge_chat(
    payload: BridgeChatRequest,
    ctx: AuthContext = Depends(get_auth_context),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> BridgeChatResponse:
    """One completion from LM Studio, unwrapped into a flat response.

    The full OpenAI envelope is still returned in ``raw`` — a client that
    wants ``logprobs`` or a tool call has it — but the fields every
    caller actually reads are lifted to the top level so nobody has to
    write ``["choices"][0]["message"]["content"]`` again.
    """
    _require_enabled(config)
    _check_override(ctx, payload.base_url)
    if not payload.messages:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "messages must not be empty")
    if payload.stream:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            "Use POST /bridge/lmstudio/chat/stream for streaming responses.",
        )
    bridge = _bridge_for(config, payload.base_url)
    try:
        envelope = bridge.chat(
            [m.model_dump() for m in payload.messages],
            model=payload.model,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            top_p=payload.top_p,
            stop=payload.stop,
        )
    except LMStudioError as exc:
        raise _as_t1_error(exc) from exc

    content, finish = _first_choice(envelope)
    usage = envelope.get("usage") or {}
    return BridgeChatResponse(
        model=str(envelope.get("model") or payload.model or ""),
        content=content,
        finish_reason=finish,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        base_url=bridge.base_url,
        raw=envelope,
        request_id=request_id,
    )


@router.post("/chat/stream")
def bridge_chat_stream(
    payload: BridgeChatRequest,
    request: Request,
    ctx: AuthContext = Depends(get_auth_context),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> StreamingResponse:
    """Server-sent events, relayed from LM Studio essentially verbatim.

    The frames are re-emitted rather than transformed so an existing
    OpenAI-streaming client works unchanged. Two deliberate additions:
    a ``: hypernix`` comment frame is sent immediately so the client sees
    headers before the model's first token (otherwise a slow first token
    looks like a hung connection), and an error mid-stream is delivered
    as a final ``data:`` frame with an ``error`` object, because HTTP
    status is long gone by then and dropping the connection silently is
    indistinguishable from a network failure.
    """
    _require_enabled(config)
    _check_override(ctx, payload.base_url)
    if not payload.messages:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "messages must not be empty")
    bridge = _bridge_for(config, payload.base_url)
    messages = [m.model_dump() for m in payload.messages]

    def _events():
        import json as _json

        yield b": hypernix bridge open\n\n"
        try:
            for chunk in bridge.chat_stream(
                messages,
                model=payload.model,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
                top_p=payload.top_p,
                stop=payload.stop,
            ):
                yield f"data: {_json.dumps(chunk)}\n\n".encode()
        except LMStudioError as exc:
            logger.warning("t1api.bridge: stream failed against %s: %s", bridge.base_url, exc)
            yield f"data: {_json.dumps({'error': exc.to_dict()})}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        # Stops reading LM Studio when the caller goes; see t1api.disconnect.
        while_listened_to(request, _events(), what=f"LM Studio at {bridge.base_url}"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Request-Id": request_id,
            # Nginx buffers proxied responses by default, which turns a
            # token-by-token stream into one delivery at the end.
            "X-Accel-Buffering": "no",
        },
    )


def _first_choice(envelope: dict) -> tuple[str, str]:
    """``(content, finish_reason)`` from an OpenAI chat envelope."""
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", ""
    first = choices[0]
    if not isinstance(first, dict):
        return "", ""
    message = first.get("message")
    content = ""
    if isinstance(message, dict):
        raw = message.get("content")
        if isinstance(raw, str):
            content = raw
        elif isinstance(raw, list):
            # Multimodal replies come back as parts; concatenate the text.
            content = "".join(
                str(part.get("text", ""))
                for part in raw
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return content, str(first.get("finish_reason") or "")


__all__ = ["router"]
