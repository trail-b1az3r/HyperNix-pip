"""``/counter`` — the public-server counter's own end (0.72.8a3).

Off unless ``T1_COUNTER_HOST=1``: one server counts for everyone, and
nobody else needs these routes. No key is needed — the senders are other
people's servers — so a heartbeat is a tiny fixed-shape body, and the
sender's address is never stored. See :mod:`hypernix.t1api.counter`.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..deps import get_config

router = APIRouter(prefix="/counter", tags=["counter"])

MAX_BODY = 512


def _counter(request: Request):
    if not get_config(request).counter_host:
        return None
    return getattr(request.app.state, "t1_counter", None)


@router.post("/heartbeat")
async def counter_heartbeat(request: Request):
    """One public server saying it is up. No key; nothing about the sender is kept."""
    counter = _counter(request)
    if counter is None:
        return JSONResponse({"error": "this server is not a counter"}, status_code=404)
    raw = await request.body()
    if len(raw) > MAX_BODY:
        return JSONResponse({"error": "a heartbeat is under 512 bytes"}, status_code=413)
    try:
        counter.record(json.loads(raw or b"{}"))
    except (ValueError, TypeError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"ok": True}


@router.get("")
async def counter_summary(request: Request):
    """How many public T1 servers were heard from in the last two hours, by version."""
    counter = _counter(request)
    if counter is None:
        return JSONResponse({"error": "this server is not a counter"}, status_code=404)
    return JSONResponse(counter.summary(), headers={"Access-Control-Allow-Origin": "*",
                                                     "Cache-Control": "max-age=300"})
