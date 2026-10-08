"""``/inference`` — the governed inference surface.

New in **T1 v1.0.26.9.2.1**.

``/bridge/lmstudio/*`` is a pass-through. It hands the caller's model
string straight to LM Studio, which means the model registry, the plan's
routing cascade, the per-key quota and the cost ledger never see the
request. That is the correct shape for a *bridge* — a window onto
something else — but it left the one path that actually spends money as
the one path outside the rules the rest of the API is built on:

    the server determines which models exist, which are available,
    which limits apply, which fallbacks are allowed, how much usage
    remains, and what an operation costs

These endpoints are that principle applied to inference. Every call:

1. resolves the caller's plan from their **server-side assignment**, not
   from the request body,
2. requires the model through the registry, so an unregistered id is
   ``MODEL_NOT_SUPPORTED`` here exactly as everywhere else,
3. checks the key's assignment allows that model,
4. refuses an exhausted key **before** any inference runs — an over-quota
   request discovered afterwards has already cost the operator the work,
5. dispatches to whichever backend the server has, and
6. meters the tokens actually spent and prices them.

Fallback is opt-in. ``allow_fallback`` defaults to false because a silent
substitution of an exhausted model is precisely what the spec forbids; a
caller that wants the cascade has to say so, and the response says which
model really ran.

Two backends can answer, chosen per request for the model that is about
to run (after the cascade, never before):

* ``hypernix`` — this server's own runner (:mod:`hypernix.hyperlink.managed`),
  when the model it has loaded is *exactly* that model. It is the
  process the operator started on purpose and it is holding the VRAM.
* ``lmstudio`` — the LM Studio bridge, when it is enabled.

HyperLink chat can answer from "whatever the runner has loaded", because
a person is talking to the model on the status screen. This surface
cannot: the caller names a model, the registry, the quota and the price
all refer to that model, and replying from a different one because it
happened to be loaded would be the silent substitution this module exists
to forbid. So a runner serving another model is skipped, and if nothing
else can serve the request the refusal says what is loaded and how to
load the one that was asked for.

``GET /inference/backends`` reports both, probed rather than declared, so
a client can tell "no backend configured" from "the backend is down"
without parsing an error string. Every response names the backend that
answered in ``backend_name``; ``backend`` stays the address it always was.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ...bridge.lmstudio import LMStudioBridge, LMStudioError
from ...hyperlink.inference import HYPERNIX, LMSTUDIO
from ..auth import AuthContext
from ..config import T1APIConfig
from ..deps import (
    get_auth_context,
    get_config,
    get_cost_calculator,
    get_key_directory,
    get_registry,
    get_request_id,
    get_routing_engine,
    get_runner,
    get_usage_meter,
)
from ..disconnect import while_listened_to
from ..errors import T1APIError, T1ErrorCode
from ..schemas import (
    InferenceBackend,
    InferenceBackendsResponse,
    InferenceChatRequest,
    InferenceCompletionRequest,
    InferenceEmbeddingsRequest,
    InferenceEmbeddingsResponse,
    InferenceMessage,
    InferenceResponse,
    InferenceTokenCountRequest,
    InferenceTokenCountResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["inference"])

#: Rough characters-per-token for the pre-flight estimate. Only ever used
#: to *size* a request before it runs — never to bill one. What gets
#: metered is what the backend reports it actually consumed, because a
#: heuristic that decides the invoice is a heuristic that is wrong about
#: money.
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN) if text else 0


def _messages_text(messages: list[InferenceMessage]) -> str:
    def text(content) -> str:
        if isinstance(content, str):
            return content
        # The text parts, plus a token-ish allowance per image (~765 is
        # what a 1024px image costs most vision encoders).
        words = [str(p.get("text", "")) for p in content if p.get("type") == "text"]
        images = sum(1 for p in content if p.get("type") == "image_url")
        return " ".join(words) + " x" * (765 * 4 * images)

    return "\n".join(f"{m.role}: {text(m.content)}" for m in messages)


def _vision(dispatch, runner, messages: list[dict]) -> list[dict]:
    """Images re-encoded for the backend; a runner model without a vision
    projector refuses them with a 400 rather than a 500 from llama-server."""
    from ...hyperlink.imagecodec import ImagesNotSupported, vision_messages

    images_ok, who = True, "this model"
    if dispatch.name == "hypernix":
        try:
            current = runner.current if runner is not None else None
        except Exception:  # noqa: BLE001 -- a wedged runner fails later, by its own route
            current = None
        images_ok = bool(getattr(current, "supports_images", False))
        who = f"the runner's model ({getattr(current, 'model_id', '?')})"
    try:
        return vision_messages(messages, images_ok=images_ok, refuse=True, who=who)
    except ImagesNotSupported as exc:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, str(exc), http_status=400,
                         details={"backend": dispatch.name}) from exc


@dataclass(frozen=True)
class _Dispatch:
    """A client, and the honest name for where its answers come from."""

    client: LMStudioBridge
    #: The wire value: ``hypernix`` or ``lmstudio``.
    name: str

    @property
    def address(self) -> str:
        return self.client.base_url


def _loaded_model(runner: Any) -> str:
    """The model this server's own runner is serving, or ``""``.

    ``runner.current`` checks the llama.cpp process is still alive, so a
    runner whose model crashed out reads as empty here rather than as a
    socket nobody is listening on.
    """
    if runner is None:
        return ""
    try:
        current = runner.current
    except Exception:  # noqa: BLE001 - a broken runner must not hide LM Studio
        logger.debug("t1api.inference: the runner could not be asked", exc_info=True)
        return ""
    return str(getattr(current, "model_id", "") or "") if current is not None else ""


def _lmstudio(config: T1APIConfig, *, timeout: float | None = None) -> LMStudioBridge | None:
    if not config.lmstudio_enabled or not config.lmstudio_url:
        return None
    return LMStudioBridge(
        base_url=config.lmstudio_url,
        api_key=config.lmstudio_api_key or None,
        timeout=timeout if timeout is not None else config.lmstudio_timeout_seconds,
    )


def _runner_client(runner: Any, config: T1APIConfig, *, timeout: float | None = None) -> LMStudioBridge:
    # llama-server speaks the OpenAI API, so the bridge is already a
    # correct client for it (see hypernix.hyperlink.inference).
    return LMStudioBridge(
        base_url=runner.base_url,
        timeout=timeout if timeout is not None else config.lmstudio_timeout_seconds,
    )


def _select_backend(config: T1APIConfig, runner: Any, model_id: str) -> _Dispatch:
    """The backend that serves *model_id*, or a refusal that names both.

    The runner first, but only for the model it actually has loaded; then
    LM Studio; then nothing — and "nothing" distinguishes "the runner is
    serving something else" (load the right model) from "no backend at
    all" (set one up).
    """
    loaded = _loaded_model(runner)
    if loaded and loaded == model_id:
        return _Dispatch(_runner_client(runner, config), HYPERNIX)
    bridge = _lmstudio(config)
    if bridge is not None:
        return _Dispatch(bridge, LMSTUDIO)
    if loaded:
        raise T1APIError(
            T1ErrorCode.MODEL_UNAVAILABLE,
            f"{model_id} is not loaded. This server's HyperNix runner is serving "
            f"{loaded}. Load {model_id} with POST /runner/load, or point the server "
            "at LM Studio (T1_LMSTUDIO_URL and T1_LMSTUDIO_ENABLED=1).",
            details={"backends": [HYPERNIX], "loaded": loaded, "requested": model_id},
            http_status=503,
        )
    raise T1APIError(
        T1ErrorCode.NOT_SUPPORTED,
        "This server has no inference backend configured. Load a model on its "
        "built-in HyperNix runner (POST /runner/load, or `hypernix-t1 "
        "built-in-runner start`), or set T1_LMSTUDIO_URL (and "
        "T1_LMSTUDIO_ENABLED=1) to point it at LM Studio.",
        details={"backends": []},
        http_status=501,
    )


def _as_t1_error(exc: LMStudioError) -> T1APIError:
    return T1APIError(
        T1ErrorCode.MODEL_UNAVAILABLE,
        str(exc),
        details=exc.to_dict() if hasattr(exc, "to_dict") else {},
        http_status=503,
    )


def _resolve(
    *,
    ctx: AuthContext,
    keys,
    engine,
    registry,
    meter,
    model_id: str,
    input_tokens: int,
    allow_fallback: bool,
) -> tuple[str, bool]:
    """Which model may actually run, and whether that is a substitution.

    Every gate in one place so the chat, completion and stream paths
    cannot diverge on which of them they remembered to apply.
    """
    registry.require(model_id)
    plan = keys.resolve_plan(ctx.key_id)
    keys.assert_model_allowed(ctx.key_id, model_id)
    decision = engine.route_manual(
        key_id=ctx.key_id,
        plan=plan,
        model_id=model_id,
        input_tokens=input_tokens,
        automatic_fallback=allow_fallback,
    )
    chosen = decision.model_id
    if chosen != model_id:
        # A substitution still has to satisfy the caller's assignment: the
        # cascade is the server's policy, not an exemption from the key's.
        keys.assert_model_allowed(ctx.key_id, chosen)
    meter.assert_not_exhausted(ctx.key_id, chosen)
    return chosen, chosen != model_id


def _first_choice(envelope: dict) -> tuple[str, str]:
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", ""
    first = choices[0] or {}
    message = first.get("message") or {}
    content = message.get("content")
    if content is None:
        content = first.get("text") or ""
    return str(content), str(first.get("finish_reason") or "")


def _run_chat(
    *,
    bridge: LMStudioBridge,
    model: str,
    messages: list[dict],
    temperature,
    max_tokens,
    top_p,
    stop,
) -> dict:
    try:
        return bridge.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )
    except LMStudioError as exc:
        raise _as_t1_error(exc) from exc


def _complete(
    *,
    ctx: AuthContext,
    config: T1APIConfig,
    runner: Any,
    keys,
    engine,
    registry,
    meter,
    costs,
    request_id: str,
    requested_model: str,
    messages: list[InferenceMessage],
    temperature,
    max_tokens,
    top_p,
    stop,
    allow_fallback: bool,
    endpoint: str,
) -> InferenceResponse:
    """The whole governed path, shared by /chat and /completions."""
    if not messages:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "messages must not be empty")

    estimated = _estimate_tokens(_messages_text(messages))
    model, substituted = _resolve(
        ctx=ctx, keys=keys, engine=engine, registry=registry, meter=meter,
        model_id=requested_model, input_tokens=estimated,
        allow_fallback=allow_fallback,
    )

    dispatch = _select_backend(config, runner, model)
    envelope = _run_chat(
        bridge=dispatch.client, model=model,
        messages=_vision(dispatch, runner, [m.model_dump() for m in messages]),
        temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=stop,
    )

    content, finish = _first_choice(envelope)
    usage = envelope.get("usage") or {}
    # What the backend says it spent, not what we guessed. The estimate
    # sized the request; this is the bill.
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    if not input_tokens and not output_tokens:
        # A backend that reports no usage still consumed something, and
        # recording zero would make the quota unenforceable against it.
        input_tokens = estimated
        output_tokens = _estimate_tokens(content)

    meter.record(
        key_id=ctx.key_id, model_id=model,
        input_tokens=input_tokens, output_tokens=output_tokens,
        endpoint=endpoint,
    )
    input_cost, output_cost, currency = costs.price_tokens(
        model, input_tokens, output_tokens
    )

    return InferenceResponse(
        model=model,
        requested_model=requested_model,
        content=content,
        finish_reason=finish,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=round(input_cost + output_cost, 8),
        currency=currency,
        backend=dispatch.address,
        backend_name=dispatch.name,
        substituted=substituted,
        raw=envelope,
        request_id=request_id,
    )


@router.get("/backends", response_model=InferenceBackendsResponse)
def list_backends(
    ctx: AuthContext = Depends(get_auth_context),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
    runner=Depends(get_runner),
) -> InferenceBackendsResponse:
    """What this server can actually dispatch to, and whether it answers.

    Probed rather than declared, so "no backend configured" and "the
    backend is down" are two different answers instead of one error
    string a client has to parse.

    LM Studio stays first in the list, where clients written before the
    runner could answer have always found it. ``default`` follows the
    real preference: the runner, for the model it has loaded, when it
    answers; otherwise LM Studio.
    """
    probe_timeout = min(10.0, float(config.lmstudio_timeout_seconds or 10))
    backends: list[InferenceBackend] = []
    bridge = _lmstudio(config, timeout=probe_timeout)
    if bridge is not None:
        try:
            bridge.list_models()
            reachable, detail = True, "answered"
        except LMStudioError as exc:
            reachable, detail = False, str(exc)
        backends.append(InferenceBackend(
            name=LMSTUDIO, kind="openai-compatible",
            reachable=reachable, detail=detail, address=bridge.base_url,
        ))
    else:
        backends.append(InferenceBackend(
            name=LMSTUDIO, kind="openai-compatible", reachable=False,
            detail="disabled (T1_LMSTUDIO_ENABLED=0 or no T1_LMSTUDIO_URL)",
        ))

    loaded = _loaded_model(runner)
    if loaded:
        client = _runner_client(runner, config, timeout=probe_timeout)
        try:
            client.list_models()
            reachable, detail = True, f"serving {loaded}"
        except LMStudioError as exc:
            reachable, detail = False, str(exc)
        backends.append(InferenceBackend(
            name=HYPERNIX, kind="hypernix-runner", reachable=reachable,
            detail=detail, address=client.base_url, model_id=loaded,
        ))
    else:
        backends.append(InferenceBackend(
            name=HYPERNIX, kind="hypernix-runner", reachable=False,
            detail="nothing loaded (POST /runner/load, or `hypernix-t1 built-in-runner start`)",
            address=getattr(runner, "base_url", "") if runner is not None else "",
        ))

    reachable_names = {b.name for b in backends if b.reachable}
    default = next((n for n in (HYPERNIX, LMSTUDIO) if n in reachable_names), "")
    return InferenceBackendsResponse(
        backends=backends,
        default=default,
        request_id=request_id,
    )


@router.post("/chat", response_model=InferenceResponse)
def inference_chat(
    payload: InferenceChatRequest,
    ctx: AuthContext = Depends(get_auth_context),
    config: T1APIConfig = Depends(get_config),
    registry=Depends(get_registry),
    meter=Depends(get_usage_meter),
    engine=Depends(get_routing_engine),
    keys=Depends(get_key_directory),
    costs=Depends(get_cost_calculator),
    request_id: str = Depends(get_request_id),
    runner=Depends(get_runner),
) -> InferenceResponse:
    """A chat completion, through the registry, the cascade and the meter."""
    return _complete(
        ctx=ctx, config=config, runner=runner, keys=keys, engine=engine, registry=registry,
        meter=meter, costs=costs, request_id=request_id,
        requested_model=payload.model, messages=payload.messages,
        temperature=payload.temperature, max_tokens=payload.max_tokens,
        top_p=payload.top_p, stop=payload.stop,
        allow_fallback=payload.allow_fallback,
        endpoint="/inference/chat",
    )


@router.post("/completions", response_model=InferenceResponse)
def inference_completions(
    payload: InferenceCompletionRequest,
    ctx: AuthContext = Depends(get_auth_context),
    config: T1APIConfig = Depends(get_config),
    registry=Depends(get_registry),
    meter=Depends(get_usage_meter),
    engine=Depends(get_routing_engine),
    keys=Depends(get_key_directory),
    costs=Depends(get_cost_calculator),
    request_id: str = Depends(get_request_id),
    runner=Depends(get_runner),
) -> InferenceResponse:
    """A plain prompt, sent as a single user turn.

    Carried over the chat shape rather than the legacy completions one,
    because every backend worth reaching still speaks chat and several no
    longer speak completions at all.
    """
    if not payload.prompt.strip():
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "prompt must not be empty")
    return _complete(
        ctx=ctx, config=config, runner=runner, keys=keys, engine=engine, registry=registry,
        meter=meter, costs=costs, request_id=request_id,
        requested_model=payload.model,
        messages=[InferenceMessage(role="user", content=payload.prompt)],
        temperature=payload.temperature, max_tokens=payload.max_tokens,
        top_p=payload.top_p, stop=payload.stop,
        allow_fallback=payload.allow_fallback,
        endpoint="/inference/completions",
    )


@router.post("/chat/stream")
def inference_chat_stream(
    payload: InferenceChatRequest,
    request: Request,
    ctx: AuthContext = Depends(get_auth_context),
    config: T1APIConfig = Depends(get_config),
    registry=Depends(get_registry),
    meter=Depends(get_usage_meter),
    engine=Depends(get_routing_engine),
    keys=Depends(get_key_directory),
    request_id: str = Depends(get_request_id),
    runner=Depends(get_runner),
) -> StreamingResponse:
    """The same governed path, streamed.

    Every gate runs **before** the first byte, because a stream that
    starts and then discovers the key is exhausted has already spent the
    work — and a 429 cannot be sent once the response has begun.

    Usage is metered when the stream ends, including when it ends badly:
    tokens the backend produced before a mid-stream failure were still
    produced, and not recording them makes the quota under-count exactly
    for the callers whose requests fail most.
    """
    if not payload.messages:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "messages must not be empty")

    estimated = _estimate_tokens(_messages_text(payload.messages))
    model, substituted = _resolve(
        ctx=ctx, keys=keys, engine=engine, registry=registry, meter=meter,
        model_id=payload.model, input_tokens=estimated,
        allow_fallback=payload.allow_fallback,
    )
    dispatch = _select_backend(config, runner, model)
    bridge = dispatch.client
    messages = _vision(dispatch, runner, [m.model_dump() for m in payload.messages])
    key_id = ctx.key_id

    def _events():
        produced: list[str] = []
        header = {
            "model": model,
            "requested_model": payload.model,
            "substituted": substituted,
            "backend": bridge.base_url,
            "backend_name": dispatch.name,
        }
        yield f": hypernix inference open {json.dumps(header)}\n\n".encode()
        try:
            for chunk in bridge.chat_stream(
                messages,
                model=model,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
                top_p=payload.top_p,
                stop=payload.stop,
            ):
                for choice in (chunk.get("choices") or []):
                    delta = (choice or {}).get("delta") or {}
                    if delta.get("content"):
                        produced.append(str(delta["content"]))
                yield f"data: {json.dumps(chunk)}\n\n".encode()
        except LMStudioError as exc:
            logger.warning(
                "t1api.inference: stream failed against %s: %s", bridge.base_url, exc
            )
            yield f"data: {json.dumps({'error': exc.to_dict()})}\n\n".encode()
        finally:
            try:
                meter.record(
                    key_id=key_id, model_id=model,
                    input_tokens=estimated,
                    output_tokens=_estimate_tokens("".join(produced)),
                    endpoint="/inference/chat/stream",
                )
            except Exception:  # pragma: no cover - metering must not break the stream
                logger.exception("t1api.inference: could not record stream usage")
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        # Stops reading the backend when the caller goes; see t1api.disconnect.
        while_listened_to(request, _events(), what=f"{dispatch.name} at {bridge.base_url}"),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Request-Id": request_id,
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/embeddings", response_model=InferenceEmbeddingsResponse)
def inference_embeddings(
    payload: InferenceEmbeddingsRequest,
    ctx: AuthContext = Depends(get_auth_context),
    config: T1APIConfig = Depends(get_config),
    registry=Depends(get_registry),
    meter=Depends(get_usage_meter),
    keys=Depends(get_key_directory),
    request_id: str = Depends(get_request_id),
    runner=Depends(get_runner),
) -> InferenceEmbeddingsResponse:
    """Embeddings, under the same registry and quota rules as generation.

    No routing cascade: substituting a different embedding model would
    return vectors from a different space, which is not a fallback but a
    silently wrong answer.
    """
    if not payload.input:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, "input must not be empty")
    registry.require(payload.model)
    keys.assert_model_allowed(ctx.key_id, payload.model)
    meter.assert_not_exhausted(ctx.key_id, payload.model)

    dispatch = _select_backend(config, runner, payload.model)
    bridge = dispatch.client
    if not hasattr(bridge, "embeddings"):
        raise T1APIError(
            T1ErrorCode.NOT_SUPPORTED,
            "This server's inference backend does not expose embeddings.",
            http_status=501,
        )
    try:
        envelope = bridge.embeddings(payload.input, model=payload.model)
    except LMStudioError as exc:
        raise _as_t1_error(exc) from exc

    vectors = [
        [float(x) for x in (item or {}).get("embedding") or []]
        for item in (envelope.get("data") or [])
    ]
    usage = envelope.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0) or _estimate_tokens(
        "\n".join(payload.input)
    )
    meter.record(
        key_id=ctx.key_id, model_id=payload.model,
        input_tokens=input_tokens, output_tokens=0,
        endpoint="/inference/embeddings",
    )
    return InferenceEmbeddingsResponse(
        model=str(envelope.get("model") or payload.model),
        embeddings=vectors,
        dimensions=len(vectors[0]) if vectors else 0,
        input_tokens=input_tokens,
        backend=bridge.base_url,
        backend_name=dispatch.name,
        request_id=request_id,
    )


@router.post("/tokens", response_model=InferenceTokenCountResponse)
def inference_token_count(
    payload: InferenceTokenCountRequest,
    ctx: AuthContext = Depends(get_auth_context),
    registry=Depends(get_registry),
    meter=Depends(get_usage_meter),
    costs=Depends(get_cost_calculator),
    request_id: str = Depends(get_request_id),
) -> InferenceTokenCountResponse:
    """Size a request before committing to it.

    A caller working against a spend cap or a token allowance needs to
    know what a prompt will cost *before* sending it; discovering it
    afterwards is a refund, not a budget. Runs no inference and records
    no usage.

    The count is an estimate and says so in ``method`` — the tokenizer
    that matters lives in the backend, and claiming exactness here would
    be a number people would then rely on.
    """
    if (payload.text is None) == (payload.messages is None):
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            "Send exactly one of 'text' or 'messages'.",
        )
    entry = registry.require(payload.model)
    text = payload.text if payload.text is not None else _messages_text(payload.messages or [])
    tokens = _estimate_tokens(text)
    input_cost, _, currency = costs.price_tokens(payload.model, tokens, 0)

    snapshot = meter.snapshot_for_model(ctx.key_id, payload.model)
    remaining = None
    if entry.input_token_limit:
        remaining = max(0, entry.input_token_limit - snapshot.input_tokens_used)

    return InferenceTokenCountResponse(
        model=payload.model,
        tokens=tokens,
        estimated_input_cost=round(input_cost, 8),
        currency=currency,
        remaining_input_tokens=remaining,
        method=f"heuristic:{_CHARS_PER_TOKEN}-chars-per-token",
        request_id=request_id,
    )
