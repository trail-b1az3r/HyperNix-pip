"""``/v1/messages`` — the Anthropic Messages API, answered by this server's models.

New in **T1 v1.1.26.10.2.0** (0.72.8a3), so Claude Code can run on a
model HyperNix is serving::

    ANTHROPIC_BASE_URL=http://127.0.0.1:8000 ANTHROPIC_AUTH_TOKEN=<T1 key> claude

or simply ``hnx claude``, which sets that up. Translation lives in
:mod:`hypernix.t1api.anthropic`; this file authenticates, decides which
model answers, and meters what it spent.

Which model answers
-------------------
1. A model in the registry is served exactly as ``/inference/chat``
   serves it: the key's plan and assignment, the quota checked before
   anything runs, the tokens metered and priced.
2. Otherwise the model this server's own runner has loaded, when it is
   asked for by name, or when the request names a Claude model (which a
   client like Claude Code always does): ``T1_CLAUDE_CODE_MODEL`` if the
   operator set one, else whatever the runner is serving. That usage is
   recorded against the key as unpriced, as HyperLink's is.
3. Anything else is a ``not_found_error`` that names what is available.

The response's ``model`` is the model that actually ran.

Authentication is the server's own, from ``Authorization: Bearer`` (what
Claude Code sends for ``ANTHROPIC_AUTH_TOKEN``) or ``x-api-key`` (what it
sends for ``ANTHROPIC_API_KEY``); keyless where trusted-network mode
allows it. Errors use Anthropic's envelope so the client reports them
as it reports Anthropic's own.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...bridge.lmstudio import LMStudioError
from .. import anthropic as wire
from ..deps import (
    get_auth_context,
    get_config,
    get_key_directory,
    get_registry,
    get_request_id,
    get_routing_engine,
    get_usage_meter,
)
from ..disconnect import while_listened_to
from ..errors import T1APIError
from .inference import _Dispatch, _loaded_model, _resolve, _runner_client, _select_backend, _vision

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["anthropic"])

#: Names a client uses for "your default model" rather than a model id.
_DEFAULT_NAMES = {"", "default", "hypernix", "local"}

_KINDS = {400: "invalid_request_error", 401: "authentication_error", 402: "billing_error",
          403: "permission_error", 404: "not_found_error", 409: "invalid_request_error",
          413: "request_too_large", 422: "invalid_request_error", 429: "rate_limit_error",
          501: "api_error", 503: "overloaded_error", 529: "overloaded_error"}


class _Refusal(Exception):
    def __init__(self, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status, self.kind, self.message = status, kind, message


def _error(status: int, kind: str, message: str, request_id: str = "") -> JSONResponse:
    headers = {"request-id": request_id} if request_id else {}
    return JSONResponse(wire.error_body(kind, message), status_code=status, headers=headers)


def _from_t1(exc: T1APIError) -> _Refusal:
    from ..app import _STATUS_FOR_CODE

    status = int(exc.http_status or _STATUS_FOR_CODE.get(exc.code, 500))
    return _Refusal(status, _KINDS.get(status, "api_error"), exc.message)


def _principal(request: Request):
    authorization = request.headers.get("authorization")
    api_key = request.headers.get("x-api-key")
    if not authorization and api_key:
        authorization = f"Bearer {api_key}"
    # Claude Code will not start without a token, so `hnx claude` sends
    # this one when there is no key. It means exactly "no credential":
    # the keyless trusted-network path decides, and outside it, a 401.
    if authorization and authorization.split()[-1] == "t1-keyless":
        authorization = None
    return get_auth_context(request, authorization)


def _runner(request: Request) -> Any:
    return getattr(request.app.state, "t1_runner", None)


def _choose(request: Request, requested: str, ctx: Any, input_tokens: int) -> tuple[str, bool, _Dispatch]:
    """``(model, governed, dispatch)`` for *requested*, or a refusal."""
    config = get_config(request)
    registry = get_registry(request)
    runner = _runner(request)
    loaded = _loaded_model(runner)
    target = requested
    if registry.get(requested) is None and (requested.lower().startswith("claude")
                                           or requested.lower() in _DEFAULT_NAMES):
        target = config.claude_code_model or loaded
        if not target:
            raise _Refusal(503, "overloaded_error",
                           f"{requested or 'The default model'} is served by this machine's own model, "
                           "and none is loaded. Load one (POST /runner/load, or `hypernix-t1 "
                           "built-in-runner start`), or set T1_CLAUDE_CODE_MODEL to a registered model.")
    if registry.get(target) is not None:
        model, _substituted = _resolve(
            ctx=ctx, keys=get_key_directory(request), engine=get_routing_engine(request),
            registry=registry, meter=get_usage_meter(request), model_id=target,
            input_tokens=input_tokens, allow_fallback=False,
        )
        return model, True, _select_backend(config, runner, model)
    if loaded and target == loaded:
        from ...hyperlink.inference import HYPERNIX

        return loaded, False, _Dispatch(_runner_client(runner, config), HYPERNIX)
    available = sorted(e.model_id for e in registry.list()) if hasattr(registry, "list") else []
    if loaded:
        available.append(f"{loaded} (loaded)")
    raise _Refusal(404, "not_found_error",
                   f"model: {requested} is not on this server. Available: {', '.join(available) or 'none'}.")


def _meter(request: Request, ctx: Any, model: str, governed: bool, usage: dict[str, int]) -> None:
    meter = get_usage_meter(request)
    try:
        if governed:
            meter.record(key_id=ctx.key_id, model_id=model, input_tokens=usage["input_tokens"],
                         output_tokens=usage["output_tokens"], endpoint="/v1/messages", feature="anthropic")
        else:
            meter.record_quietly(key_id=ctx.key_id, feature="anthropic", endpoint="/v1/messages",
                                 model_id=model, input_tokens=usage["input_tokens"],
                                 output_tokens=usage["output_tokens"])
    except Exception:  # noqa: BLE001 - the answer was given; accounting must not take it back
        logger.exception("t1api.anthropic: could not record usage")


def _body(raw: bytes) -> dict[str, Any]:
    try:
        body = json.loads(raw or b"{}")
    except json.JSONDecodeError as exc:
        raise _Refusal(400, "invalid_request_error", f"the body is not JSON: {exc.msg}") from exc
    if not isinstance(body, dict):
        raise _Refusal(400, "invalid_request_error", "the body must be a JSON object")
    return body


@router.post("/messages")
async def create_message(request: Request):
    """Create a message; with ``"stream": true``, as server-sent events."""
    request_id = get_request_id(request)
    try:
        if not get_config(request).anthropic_enabled:
            raise _Refusal(404, "not_found_error", "The Anthropic API is off on this server (T1_ANTHROPIC_ENABLED=0).")
        try:
            ctx = _principal(request)
        except T1APIError as exc:
            raise _from_t1(exc) from exc
        body = _body(await request.body())
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise _Refusal(400, "invalid_request_error", "messages: at least one message is required")
        if not isinstance(body.get("max_tokens"), int) or body["max_tokens"] < 1:
            raise _Refusal(400, "invalid_request_error", "max_tokens: an integer of at least 1 is required")
        converted = wire.to_openai(body)
        estimate = wire.estimate_tokens(wire.prompt_text(converted))
        try:
            model, governed, dispatch = _choose(request, str(body.get("model") or ""), ctx, estimate)
            messages = _vision(dispatch, _runner(request), converted["messages"])
        except T1APIError as exc:
            raise _from_t1(exc) from exc
    except _Refusal as refusal:
        return _error(refusal.status, refusal.kind, refusal.message, request_id)

    bridge = dispatch.client
    arguments = {"model": model, "temperature": converted["temperature"], "max_tokens": converted["max_tokens"],
                 "top_p": converted["top_p"], "stop": converted["stop"], "tools": converted["tools"] or None}
    headers = {"request-id": request_id, "x-hypernix-backend": dispatch.name, "x-hypernix-model": model}

    if not body.get("stream"):
        try:
            import anyio

            envelope = await anyio.to_thread.run_sync(
                lambda: bridge.chat(messages, extra=converted["extra"] or None, **arguments))
        except LMStudioError as exc:
            return _error(503, "overloaded_error", f"{dispatch.name} could not answer: {exc}", request_id)
        reply = wire.from_openai(envelope, model=model, thinking=converted["thinking"],
                                 input_estimate=estimate, stop=converted["stop"])
        _meter(request, ctx, model, governed, reply["usage"])
        return JSONResponse(reply, headers=headers)

    translator = wire.StreamTranslator(model=model, input_tokens=estimate, thinking=converted["thinking"])
    extra = {**converted["extra"], "stream_options": {"include_usage": True}}

    def _events():
        # Metered however the stream ends: what the model produced before
        # a failure (or before the client went) was still produced.
        try:
            yield from translator.start()
            for chunk in bridge.chat_stream(messages, extra=extra, **arguments):
                yield from translator.feed(chunk)
            yield from translator.end()
        except LMStudioError as exc:
            logger.warning("t1api.anthropic: stream failed against %s: %s", bridge.base_url, exc)
            yield from translator.error(f"{dispatch.name} stopped answering: {exc}")
        finally:
            usage = getattr(translator, "final_usage", None) or {
                "input_tokens": translator.input_tokens, "output_tokens": translator.output_chars // 4}
            _meter(request, ctx, model, governed, usage)

    return StreamingResponse(
        while_listened_to(request, _events(), what=f"{dispatch.name} at {bridge.base_url}"),
        media_type="text/event-stream",
        headers={**headers, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/messages/count_tokens")
async def count_tokens(request: Request):
    """How many input tokens a request would be. The runner counts when it can."""
    request_id = get_request_id(request)
    try:
        try:
            _principal(request)
        except T1APIError as exc:
            raise _from_t1(exc) from exc
        body = _body(await request.body())
    except _Refusal as refusal:
        return _error(refusal.status, refusal.kind, refusal.message, request_id)
    text = wire.prompt_text(wire.to_openai({**body, "messages": body.get("messages") or []}))
    counted = _runner_count(request, text)
    return JSONResponse({"input_tokens": counted if counted is not None else wire.estimate_tokens(text)})


def _runner_count(request: Request, text: str) -> int | None:
    runner = _runner(request)
    if not _loaded_model(runner):
        return None
    try:
        import urllib.request

        req = urllib.request.Request(runner.base_url.rstrip("/") + "/tokenize",
                                     data=json.dumps({"content": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310 - this server's own runner
            return len(json.loads(resp.read()).get("tokens") or [])
    except Exception:  # noqa: BLE001 - an estimate is the answer then
        return None


def _model_row(model_id: str, name: str = "") -> dict[str, Any]:
    return {"type": "model", "id": model_id, "display_name": name or model_id,
            "created_at": "2026-10-20T00:00:00Z"}


@router.get("/models")
async def list_models(request: Request):
    """The models a request may name, in Anthropic's list shape."""
    request_id = get_request_id(request)
    try:
        _principal(request)
    except T1APIError as exc:
        refusal = _from_t1(exc)
        return _error(refusal.status, refusal.kind, refusal.message, request_id)
    rows = [_model_row(e.model_id, getattr(e, "display_name", "")) for e in get_registry(request).list()]
    loaded = _loaded_model(_runner(request))
    if loaded and all(r["id"] != loaded for r in rows):
        rows.insert(0, _model_row(loaded, f"{loaded} (loaded)"))
    return {"data": rows, "has_more": False,
            "first_id": rows[0]["id"] if rows else None, "last_id": rows[-1]["id"] if rows else None}


__all__ = ["router"]
