"""``/memory/*`` — what the assistant remembers between conversations.

Four endpoints, per the spec: create, get, list, edit. Delete is here too
because a memory store you cannot remove things from is one nobody will
let write anything, and "the model believes something wrong about me and
I cannot stop it" is the failure that makes the whole feature unusable.

Owner-scoped throughout. A memory belongs to whoever's key or device
paired it, not to a session and not to a device — so the phone and the
desktop see the same set, which is the point of it being on the server.

``/memory/sync`` (0.72.6) keeps a client's copy current from a cursor,
deletions included, rather than refetching the whole list.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from ...hyperlink.memory import MemoryStore, rename_category
from ...hyperlink.memory import categories as memory_categories
from ...hyperlink.memory import organise as organise_memories
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


@router.get("/sync")
def sync_memories(
    cursor: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: MemoryStore = Depends(get_memory_store),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """What changed since *cursor*, so the phone keeps a copy current.

    Send 0 the first time and the returned ``cursor`` after that. With
    ``full`` false the answer is a delta: upsert ``memories`` by id and
    drop the ids in ``deleted``. With ``full`` true it is the whole set
    and replaces the copy: a first sync, a cursor older than the
    tombstones, or one this server never issued. Ask again while
    ``more`` is true. A cursor that is already current costs one small
    answer with nothing in it.
    """
    page = store.sync(owner=principal.owner, cursor=cursor, limit=limit)
    return {
        **page.to_dict(),
        "count": store.count(owner=principal.owner),
        "auto_count": store.count(owner=principal.owner, source="auto"),
        "request_id": request_id,
    }


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


# ---------------------------------------------------------------------------
# Organising (0.72.5.post16)
# ---------------------------------------------------------------------------


@router.get("/categories")
def list_categories(
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: MemoryStore = Depends(get_memory_store),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """Each category with how many memories it holds."""
    return {"categories": memory_categories(store, owner=principal.owner), "request_id": request_id}


@router.post("/categories/rename")
def rename_memory_category(
    payload: dict[str, Any],
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: MemoryStore = Depends(get_memory_store),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """Move every memory in one category to another; merges into an existing one."""
    moved = rename_category(
        store, owner=principal.owner, old=str(payload.get("from", "")), new=str(payload.get("to", ""))
    )
    return {"moved": moved, "request_id": request_id}


@router.post("/organise")
def organise(
    payload: dict[str, Any] | None = None,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    store: MemoryStore = Depends(get_memory_store),
    request_id: str = Depends(get_request_id),
) -> dict[str, Any]:
    """File memories that are not under a topic under one.

    Categories a person chose are left alone. ``dry_run`` shows the moves
    without making them.
    """
    dry_run = bool((payload or {}).get("dry_run", False))
    changes = organise_memories(store, owner=principal.owner, dry_run=dry_run)
    return {"changes": changes, "applied": not dry_run, "request_id": request_id}
