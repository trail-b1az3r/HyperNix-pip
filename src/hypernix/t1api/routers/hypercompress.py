"""``/chat/hypercompress`` — a system prompt that fits the loaded model.

When the prompt a HyperLink turn would send is bigger than the loaded
model's context, the system prompt is compressed to a share of it before
the turn runs (see :mod:`hypernix.hyperlink.hypercompress`). The share is
per owner:

    GET /chat/hypercompress/size            what it is now
    GET /chat/hypercompress/size?s?=1       a third of the context (default)
    GET /chat/hypercompress/size?s?=2       a fifth of the context

``s=2`` works too; POST does the same as GET with ``s``. ``POST
/chat/hypercompress`` compresses a text by hand, to see what a turn
would get.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Request

from ...hyperlink.hypercompress import (
    DEFAULT_SIZE,
    METADATA_KEY,
    SIZES,
    Compressed,
    compress,
    estimate_tokens,
    maybe_compress,
    parse_size,
    target_tokens,
)
from ..deps import HyperLinkPrincipal, get_hyperlink_principal, get_preference_store, get_request_id
from ..errors import T1APIError, T1ErrorCode
from ..schemas import HypercompressRequest, HypercompressResponse, HypercompressSizeResponse

router = APIRouter(prefix="/chat/hypercompress", tags=["compaction"])

_S = re.compile(r"(?:^|&)s\??=([^&]*)", re.IGNORECASE)


def _owner_size(preferences, owner: str) -> int:
    size = (preferences.get(owner=owner).metadata or {}).get(METADATA_KEY, DEFAULT_SIZE)
    return size if size in SIZES else DEFAULT_SIZE


def _size(request: Request, principal: HyperLinkPrincipal, preferences, request_id: str):
    match = _S.search(request.url.query or "")
    changed = False
    if match:
        try:
            wanted = parse_size(match.group(1))
        except ValueError as exc:
            raise T1APIError(T1ErrorCode.VALIDATION_ERROR, str(exc)) from exc
        current = preferences.get(owner=principal.owner)
        metadata = {**(current.metadata or {}), METADATA_KEY: wanted}
        preferences.save(owner=principal.owner, metadata=metadata)
        changed = True
    size = _owner_size(preferences, principal.owner)
    return HypercompressSizeResponse(size=size, share=f"1/{SIZES[size]}", changed=changed,
                                     request_id=request_id)


@router.get("/size", response_model=HypercompressSizeResponse)
def hypercompress_size(
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    preferences=Depends(get_preference_store),
    request_id: str = Depends(get_request_id),
) -> HypercompressSizeResponse:
    """The share of the context a too-big system prompt is cut to; ``s?=1|2`` sets it."""
    return _size(request, principal, preferences, request_id)


@router.post("/size", response_model=HypercompressSizeResponse)
def set_hypercompress_size(
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    preferences=Depends(get_preference_store),
    request_id: str = Depends(get_request_id),
) -> HypercompressSizeResponse:
    """The same as GET with ``s``."""
    return _size(request, principal, preferences, request_id)


@router.post("", response_model=HypercompressResponse)
def hypercompress_text(
    payload: HypercompressRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    preferences=Depends(get_preference_store),
    request_id: str = Depends(get_request_id),
) -> HypercompressResponse:
    """Compress *text* as a turn would, or to the share with ``force``."""
    try:
        size = parse_size(payload.size) if payload.size is not None else _owner_size(
            preferences, principal.owner)
    except ValueError as exc:
        raise T1APIError(T1ErrorCode.VALIDATION_ERROR, str(exc)) from exc
    if payload.force:
        target = target_tokens(payload.context_length, size)
        text = compress(payload.text, target)
        before = estimate_tokens(payload.text)
        result = Compressed(text, before, estimate_tokens(text), target, size, text != payload.text)
    else:
        result = maybe_compress(payload.text, context_length=payload.context_length,
                                other_tokens=payload.other_tokens, size=size)
    info = result.to_dict()
    return HypercompressResponse(text=result.text, request_id=request_id, **info)
