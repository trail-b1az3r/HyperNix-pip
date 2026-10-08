"""Knowing when a streaming caller has gone.

T1's ``@app.middleware("http")`` stack hides the ``http.disconnect`` a
departing client produces from the endpoints below it: Starlette's
``BaseHTTPMiddleware`` keeps iterating a streaming response to the end
whether or not anybody is still reading, and ``request.is_disconnected()``
in the endpoint never turns true. A relayed model stream then read the
backend to the end of the reply after its caller had left — a client
that timed out, a closed tab — and llama.cpp's single slot, or LM
Studio's, kept writing it, with every later request queued behind.

:class:`ClientGoneWatch` is a pure ASGI middleware, registered outside
that stack, that watches the connection itself and sets an event in the
scope; :func:`while_listened_to` relays a stream until the caller goes and
then closes it, which closes the backend stream with it.
"""
from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import anyio

logger = logging.getLogger(__name__)

#: Where the "caller has gone" event lives in the ASGI scope.
SCOPE_KEY = "hypernix.client_gone"
#: What to run when the caller goes: closers registered by streams.
ON_GONE_KEY = "hypernix.on_client_gone"


class ClientGoneWatch:
    """Sets ``scope[SCOPE_KEY]`` when the client disconnects.

    Once the request body has been read, the next message ASGI servers
    deliver on ``receive`` is ``http.disconnect``; a background task waits
    for it, so nothing in the app has to keep asking. Reading the body is
    left to the app: the watcher starts only after it, so it never takes a
    body chunk the app is waiting for.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        gone = anyio.Event()
        body_read = anyio.Event()
        scope[SCOPE_KEY] = gone
        scope[ON_GONE_KEY] = on_gone = []

        async def tracked_receive() -> dict:
            message = await receive()
            if message.get("type") == "http.disconnect":
                gone.set()
            elif message.get("type") == "http.request" and not message.get("more_body"):
                body_read.set()
            return message

        async def watch() -> None:
            await body_read.wait()
            while not gone.is_set():
                if (await receive()).get("type") == "http.disconnect":
                    gone.set()
            # The middleware stack has stopped iterating the response by now,
            # so a stream cannot notice for itself: close what it registered.
            with anyio.CancelScope(shield=True):
                for close in list(on_gone):
                    await anyio.to_thread.run_sync(close)

        async with anyio.create_task_group() as group:
            group.start_soon(watch)
            try:
                await self.app(scope, tracked_receive, send)
            finally:
                group.cancel_scope.cancel()


def client_gone(request: Any) -> bool:
    event = request.scope.get(SCOPE_KEY)
    return bool(event is not None and event.is_set())


def _close(events: Iterator[bytes], what: str) -> None:
    """Close a generator that may be mid-``next`` in another thread: wait
    for that step to finish (a streaming backend sends the next token
    within moments), then close it, which closes the backend stream."""
    deadline = time.monotonic() + 30.0
    while True:
        try:
            events.close()  # type: ignore[attr-defined]
        except ValueError:  # "generator already executing"
            if time.monotonic() > deadline:
                logger.warning("t1api: could not stop %s after its caller left", what or "a stream")
                return
            time.sleep(0.05)
            continue
        logger.info("t1api: the caller left mid-stream; stopped %s", what or "the stream")
        return


async def while_listened_to(request: Any, events: Iterator[bytes], *, what: str = "") -> AsyncIterator[bytes]:
    """Relay *events* (a sync generator, run off the event loop) until it
    ends or the caller has gone; either way it is closed, and with it the
    backend stream it was reading."""
    closers = request.scope.get(ON_GONE_KEY)
    if closers is not None:
        closers.append(lambda: _close(events, what))
    try:
        while not client_gone(request) and not await request.is_disconnected():
            chunk = await anyio.to_thread.run_sync(next, events, None)
            if chunk is None:
                return
            yield chunk
    finally:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(events.close)
