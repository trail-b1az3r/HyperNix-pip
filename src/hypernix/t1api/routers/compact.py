"""``/chat/compact/*`` — making a long conversation fit again.

Five endpoints, one implementation. What differs is *what* gets
summarised, and the reason there are five rather than one flag is that
the answer differs by person: somebody pasting logs into a model wants
their own messages compacted, somebody reading long answers wants the
replies, and somebody whose system prompt grew by accretion wants
neither.

``/chat/compact/dynamic`` is the one to reach for. It measures which side
is actually using the budget and compacts that, so nobody has to have
already worked out where their tokens went.

Nothing is deleted — see :mod:`hypernix.hyperlink.compaction`. A
compacted message is marked; the transcript a person scrolls is
unchanged and only the model sees the shorter version.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ...hyperlink.compaction import (
    SUMMARY_MARKER,
    plan,
    summarise_extractively,
    summary_prompt,
)
from ...hyperlink.sessions import ChatSessionStore
from ..config import T1APIConfig
from ..deps import (
    HyperLinkPrincipal,
    get_config,
    get_hyperlink_principal,
    get_request_id,
    get_session_store,
)
from ..schemas import CompactRequest, CompactResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat/compact", tags=["compaction"])


def _summarise(config: T1APIConfig, messages: list) -> tuple[str, str]:
    """``(summary, how)``. A model when one answers, quotes when not.

    The fallback is extractive on purpose. A server with no reachable
    model could refuse instead — but compaction is most wanted exactly
    when a conversation has got long, and refusing leaves somebody stuck
    with a session that no longer fits and no way to shrink it. Quoting
    is lossy and honest; inventing prose would be neither.
    """
    from .hyperlink import _chat_bridge

    try:
        bridge = _chat_bridge(config)
        envelope = bridge.chat(summary_prompt(messages), temperature=0.2)
        choices = envelope.get("choices") or []
        text = ""
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            text = str(message.get("content") or "").strip()
        if text:
            return text, "model"
        logger.info("compaction: the model returned nothing; quoting instead")
    except Exception as exc:  # noqa: BLE001 - any backend failure is the same here
        logger.info("compaction: no model available (%s); quoting instead", exc)
    return summarise_extractively(messages), "extractive"


def _compact(
    scope: str,
    payload: CompactRequest,
    principal: HyperLinkPrincipal,
    store: ChatSessionStore,
    config: T1APIConfig,
    request_id: str,
) -> CompactResponse:
    store.get(payload.session_id, owner=principal.owner)
    history = store.messages(payload.session_id, owner=principal.owner)
    proposed = plan(history, scope, keep_recent=payload.keep_recent)

    if payload.dry_run or not proposed.viable:
        return CompactResponse(
            session_id=payload.session_id,
            scope=proposed.scope,
            plan=proposed.to_dict(),
            applied=False,
            request_id=request_id,
        )

    text, how = _summarise(config, proposed.targets)
    if not text:
        # Nothing to write down. Not an error: an empty summary would
        # replace real messages with nothing, which is deletion wearing
        # compaction's name.
        return CompactResponse(
            session_id=payload.session_id,
            scope=proposed.scope,
            plan=proposed.to_dict(),
            applied=False,
            request_id=request_id,
        )

    summary = store.append(
        payload.session_id,
        role="system",
        content=text,
        owner=principal.owner,
        metadata={
            SUMMARY_MARKER: True,
            "scope": proposed.scope,
            "summarised_by": how,
            "replaced": len(proposed.targets),
        },
    )
    # Marked *after* the summary exists, and pointing at it. The other
    # order would leave a window in which the originals are hidden and
    # nothing has replaced them — a conversation that briefly forgot
    # itself if the write failed in between.
    marked = store.mark_compacted(
        [m.message_id for m in proposed.targets],
        session_id=payload.session_id,
        owner=principal.owner,
        summary_id=summary.message_id,
    )
    return CompactResponse(
        session_id=payload.session_id,
        scope=proposed.scope,
        plan=proposed.to_dict(),
        applied=True,
        summary_message_id=summary.message_id,
        summary=text,
        summarised_by=how,
        messages_compacted=marked,
        request_id=request_id,
    )


@router.post("/prompts", response_model=CompactResponse)
def compact_prompts(
    payload: CompactRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: ChatSessionStore = Depends(get_session_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> CompactResponse:
    """Summarise what the person sent, keeping the answers intact.

    For a thread where somebody has been pasting logs, files or stack
    traces in: the inputs are most of the weight and the replies are what
    the conversation is worth keeping.
    """
    return _compact("prompts", payload, principal, store, config, request_id)


@router.post("/prompts/system", response_model=CompactResponse)
def compact_system_prompt(
    payload: CompactRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: ChatSessionStore = Depends(get_session_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> CompactResponse:
    """Summarise the system prompt alone.

    The cheapest win when one applies, because a system prompt is paid
    for on every single turn — a thousand tokens of accumulated
    instructions is a thousand tokens a hundred times over.
    """
    return _compact("system", payload, principal, store, config, request_id)


@router.post("/responses", response_model=CompactResponse)
def compact_responses(
    payload: CompactRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: ChatSessionStore = Depends(get_session_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> CompactResponse:
    """Summarise the model's messages, keeping the questions intact."""
    return _compact("responses", payload, principal, store, config, request_id)


@router.post("/all", response_model=CompactResponse)
def compact_all(
    payload: CompactRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: ChatSessionStore = Depends(get_session_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> CompactResponse:
    """Summarise both sides, oldest first."""
    return _compact("all", payload, principal, store, config, request_id)


@router.post("/dynamic", response_model=CompactResponse)
def compact_dynamic(
    payload: CompactRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: ChatSessionStore = Depends(get_session_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> CompactResponse:
    """Compact whichever part is actually using the budget.

    The one to reach for. It measures rather than guessing, and reports
    what it decided and why in `plan.reason`, so the answer is
    inspectable instead of magic.
    """
    return _compact("dynamic", payload, principal, store, config, request_id)
