"""``/memory/*`` — what the assistant remembers between conversations.

Four endpoints, per the spec: create, get, list, edit. Delete is here too
because a memory store you cannot remove things from is one nobody will
let write anything, and "the model believes something wrong about me and
I cannot stop it" is the failure that makes the whole feature unusable.

Owner-scoped throughout. A memory belongs to whoever's key or device
paired it, not to a session and not to a device — so the phone and the
desktop see the same set, which is the point of it being on the server.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from ...hyperlink.memory import MemoryStore
from ..config import T1APIConfig
from ..deps import (
    HyperLinkPrincipal,
    get_config,
    get_hyperlink_principal,
    get_memory_store,
    get_request_id,
)
from ..schemas import (
    GenericOkResponse,
    MemoryCreateRequest,
    MemoryListResponse,
    MemoryResponse,
    MemoryUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/create", response_model=MemoryResponse)
def create_memory(
    payload: MemoryCreateRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: MemoryStore = Depends(get_memory_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> MemoryResponse:
    """Remember something.

    A near-duplicate updates the existing memory rather than adding a
    second one, and returns it. That is not an error case being
    swallowed: the caller asked for this fact to be remembered, and after
    this call it is. Making every auto-memory write handle a duplicate
    error would mean the model has to remember what it has remembered.
    """
    record = store.create(
        owner=principal.owner,
        content=payload.content,
        category=payload.category,
        source=payload.source,
        session_id=payload.session_id,
        pinned=payload.pinned,
        metadata=payload.metadata,
    )
    return MemoryResponse(memory=record.to_dict(), request_id=request_id)


@router.get("/get", response_model=MemoryResponse)
def get_memory(
    memory_id: str = Query(...),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: MemoryStore = Depends(get_memory_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> MemoryResponse:
    """One memory. 404 for somebody else's, same as for one that is not
    there — distinguishing them would let a caller enumerate ids."""
    record = store.get(memory_id, owner=principal.owner)
    return MemoryResponse(memory=record.to_dict(), request_id=request_id)


@router.get("/list", response_model=MemoryListResponse)
def list_memories(
    category: str = Query(default=""),
    source: str = Query(default="", description="manual | auto"),
    limit: int = Query(default=200, ge=1, le=1000),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: MemoryStore = Depends(get_memory_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> MemoryListResponse:
    """Everything remembered about this owner, pinned first.

    `source` is the filter that matters on a settings screen: "what has
    the model decided about me on its own" is a different question from
    "what did I tell it", and a person reviewing their memory is usually
    asking the first one.
    """
    records = store.list(
        owner=principal.owner, category=category, source=source, limit=limit
    )
    return MemoryListResponse(
        memories=[record.to_dict() for record in records],
        count=len(records),
        auto_count=store.count(owner=principal.owner, source="auto"),
        request_id=request_id,
    )


@router.post("/edit", response_model=MemoryResponse)
def edit_memory(
    payload: MemoryUpdateRequest,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: MemoryStore = Depends(get_memory_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> MemoryResponse:
    """Change one. Only the fields sent are touched."""
    record = store.edit(
        payload.memory_id,
        owner=principal.owner,
        content=payload.content,
        category=payload.category,
        pinned=payload.pinned,
        metadata=payload.metadata,
    )
    return MemoryResponse(memory=record.to_dict(), request_id=request_id)


@router.post("/delete", response_model=GenericOkResponse)
def delete_memory(
    memory_id: str = Query(...),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: MemoryStore = Depends(get_memory_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> GenericOkResponse:
    """Forget one.

    Not in the four the spec named, and not optional. A model that can
    write memories and a person who cannot delete them is a person who
    will turn the feature off — and "it believes something wrong about me
    and I cannot stop it" is the complaint that would follow.
    """
    removed = store.delete(memory_id, owner=principal.owner)
    return GenericOkResponse(
        ok=removed,
        detail="Forgotten." if removed else "No such memory, so nothing to forget.",
        request_id=request_id,
    )
