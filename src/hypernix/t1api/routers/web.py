"""``/web/v1`` — keyless web search for HyperLink and hyperchat.

Three endpoints, in the shapes the request asked for:

* ``GET /web/v1/I?=query&q?=maxdepth`` — search
* ``POST /web/v1/summarize`` — summarise a page, or a search's results
* ``GET /web/v1/config/<directives>`` — read and change the three
  settings

Everything about the grammar lives in :mod:`hypernix.t1api.websearch`,
which has no FastAPI in it. This module is the HTTP part and the two
things that can only be got wrong here.

The first is that the documented forms are not URLs in the ordinary
sense. ``/web/v1/I?=query`` puts the search text in a *nameless* query
parameter, and ``config/s1?=k|s2?:=x`` puts settings after a ``?``,
which makes everything from there on the query string rather than the
path. So both are read from ``request.url.query`` as raw text rather
than through FastAPI's parameter binding — which would see one empty
key and one value and hand over nothing. Conventional forms
(``?q=cats&depth=2``) work too, because no HTTP client library will
build the documented one for you.

Authenticated the way every HyperLink route is: a paired device's
``HLNK_`` token, a T2S key, or any T1 credential. It used to take only T1
keys and scoped tokens, which meant a paired phone — the client these
endpoints were built for — was refused on every one of them.

The second is that an API key arrives **in the URL**, and a URL is the
one part of a request that everything logs by default. Every path out
of here redacts before logging, and no response ever contains the key.
"""
from __future__ import annotations

import logging
import urllib.parse

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from .. import websearch as ws
from ..config import T1APIConfig
from ..deps import (
    HyperLinkPrincipal,
    get_config,
    get_hyperlink_principal,
    get_request_id,
)
from ..errors import T1APIError, T1ErrorCode

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/web/v1", tags=["web"])

#: Longest page body accepted for summarising. Past this the summary is
#: not better and the request is somebody's disk.
MAX_SUMMARIZE_CHARS = 400_000


def _settings(request: Request) -> ws.WebSettings:
    """The server's settings, held on the app rather than per request."""
    current = getattr(request.app.state, "t1_web_settings", None)
    if current is None:
        current = ws.WebSettings()
        request.app.state.t1_web_settings = current
    return current


def _store(request: Request, settings: ws.WebSettings) -> None:
    request.app.state.t1_web_settings = settings


def _grammar_error(exc: ws.GrammarError, raw: str) -> T1APIError:
    # The message is the operator's only guide to a grammar with three
    # sigils in it, so it is passed through -- but the string it was
    # reading is redacted first, since a rejected config still carried
    # a key and error details are logged.
    return T1APIError(
        T1ErrorCode.CONFIG_INVALID,
        str(exc),
        http_status=400,
        details={"received": ws.redact(raw)[:200]},
    )


@router.get("/I")
@router.get("/i")
@router.get("/search")
async def web_search(
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
):
    """``/web/v1/I?=query&q?=maxdepth`` — or ``?q=cats&depth=2``.

    Three paths for one endpoint: ``I`` is the documented name, ``i``
    because nobody can tell a capital I from a lowercase l in a URL,
    and ``search`` because that is what people type.
    """
    raw = request.url.query
    query, depth = ws.parse_search_query(raw)
    if not query:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            "no search text. Use `/web/v1/I?=your+query&q?=2` or the "
            "ordinary `/web/v1/search?q=your+query&depth=2`.",
            http_status=400,
        )

    settings = _settings(request)
    outcome = ws.search(query, depth=depth, settings=settings)
    logger.info("web: search %r depth=%d -> %d hit(s) [%s]",
                query[:80], depth, len(outcome.hits), outcome.status)
    body = outcome.to_dict()
    body["request_id"] = request_id
    return JSONResponse(body)


@router.post("/summarize")
@router.post("/summarise")
async def web_summarize(
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    config: T1APIConfig = Depends(get_config),
    request_id: str = Depends(get_request_id),
):
    """Summarise text, a URL, or the results of a search.

    Works with no model loaded. The summary is extractive unless the
    server has a model to hand — not as a placeholder but because a
    phone asking for the gist of a page should not have to wait for a
    model load, and most servers have not loaded one.
    """
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    text = str(payload.get("text") or "")
    url = str(payload.get("url") or "")
    if url:
        url = _validated_summary_url(url)
    query = str(payload.get("query") or "")
    sentences = payload.get("sentences", 5)
    try:
        sentences = int(sentences)
    except (TypeError, ValueError):
        sentences = 5

    if not text and url:
        text = _fetch_for_summary(url)

    if not text:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            "nothing to summarise — send `text` or a `url`.",
            http_status=400,
        )
    if len(text) > MAX_SUMMARIZE_CHARS:
        text = text[:MAX_SUMMARIZE_CHARS]

    result = ws.summarize(text, sentences=sentences, query=query,
                          model=_model_for(request))
    result["request_id"] = request_id
    result["chars_in"] = len(text)
    return JSONResponse(result)


def _validated_summary_url(url: str) -> str:
    """Validate and normalize a URL accepted by /summarize."""
    candidate = url.strip()
    parts = urllib.parse.urlsplit(candidate)
    if parts.scheme not in {"http", "https"}:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            "url must start with http:// or https://",
            http_status=400,
        )
    if not parts.hostname:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            "url must include a hostname",
            http_status=400,
        )
    if parts.username is not None or parts.password is not None:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            "url must not include user info",
            http_status=400,
        )
    if parts.fragment:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            "url fragments are not allowed",
            http_status=400,
        )
    return candidate


def _fetch_failure(page) -> str:
    """Why a fetch failed, in this server's words only.

    Never the exception's text: a network error's message can carry
    addresses, paths and internals the caller has no need for. The
    specific cause is in the server log.
    """
    if page.status and page.status >= 400:
        return f"the page answered HTTP {int(page.status)}"
    if page.error.startswith("refused"):
        return "it leads to an address that is not public"
    if page.error.startswith("not text"):
        return "it is not a text page"
    if page.error.startswith("gzip"):
        return "its compressed body was damaged"
    return "the page could not be reached"


def _fetch_for_summary(url: str) -> str:
    """The text of a public page, fetched on a caller's behalf.

    The server must not be a way into places only it can reach: itself,
    the LAN, a cloud metadata service. The address is checked before
    anything is fetched — robots.txt included — and again on every
    redirect (``public_only``).
    """
    from hypernix.data import gather
    from hypernix.interfaces import websearch as iw

    problem = gather.public_address_problem(url)
    if problem is not None:
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            f"will not fetch that URL: {problem}"[:300],
            http_status=400,
        )
    if not iw._shared_robots().allows(url):
        raise T1APIError(
            T1ErrorCode.VALIDATION_ERROR,
            "that site's robots.txt asks not to be fetched",
            http_status=400,
        )
    page = gather.fetch(url, timeout=15.0, limiter=iw._shared_limiter(), public_only=True)
    if not page.ok:
        logger.info("web: fetching %s for a summary failed: %s", url, page.error or page.status)
        raise T1APIError(
            T1ErrorCode.TRANSPORT_FAILED,
            f"could not fetch {url}: {_fetch_failure(page)}"[:300],
            http_status=502,
        )
    return page.text[:MAX_SUMMARIZE_CHARS]


def _model_for(request: Request):
    """The loaded model, if there is one, as a plain prompt->text call.

    Deliberately never *loads* one. Summarising is a small favour and
    loading a model to do it would take the GPU from whatever the
    server is actually for.
    """
    runner = getattr(request.app.state, "t1_runner", None)
    if runner is None or not getattr(runner, "is_loaded", lambda: False)():
        return None

    def generate(prompt: str) -> str:
        return str(runner.generate(prompt, max_tokens=512, temperature=0.3))

    return generate


@router.get("/config")
async def web_config_read(
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    request_id: str = Depends(get_request_id),
):
    """The three settings, and the browsers actually installed.

    Never the API key. ``s3`` reports whether one is set and which
    provider it belongs to, which is everything a caller needs to know
    and nothing it needs to hold.
    """
    body = _settings(request).to_dict()
    body["grammar"] = (
        "s1?=k|s2?:=duckduckgo|s3?:={key}?[provider]  "
        "— `?` marks a setting, `=k` keeps it, `?:=` sets it, `|` divides"
    )
    body["request_id"] = request_id
    return JSONResponse(body)


@router.get("/config/{directives:path}")
async def web_config_write(
    directives: str,
    request: Request,
    principal: HyperLinkPrincipal = Depends(get_hyperlink_principal),
    request_id: str = Depends(get_request_id),
):
    """``/web/v1/config/s1?=k|s2?:=wikipedia|s3?:={key}?[auto]``

    Everything from the first ``?`` onwards is the query string as far
    as HTTP is concerned, so the clause is rebuilt from the path
    segment and the raw query before it is parsed. Changing settings
    over GET is unusual; it is what the documented form is, and the
    route is admin-gated instead.
    """
    if not principal.is_admin:
        raise T1APIError(
            T1ErrorCode.AUTH_ADMIN_REQUIRED,
            "changing the web search settings needs an admin key.",
            http_status=403,
        )

    raw = directives
    if request.url.query:
        # `s1?=k` arrives as path "s1" + query "=k". Rejoin them with
        # the `?` HTTP took out, or the grammar sees a clause with no
        # setting marker in it and refuses something valid.
        raw = f"{directives}?{request.url.query}"

    settings = _settings(request)
    try:
        updated, changed = ws.apply_config(settings, raw)
    except ws.GrammarError as exc:
        logger.info("web: config refused: %s [%s]", exc, ws.redact(raw)[:120])
        raise _grammar_error(exc, raw) from exc

    _store(request, updated)
    logger.info("web: config changed %s [%s]", changed or "nothing",
                ws.redact(raw, updated)[:120])
    body = updated.to_dict()
    body["changed"] = changed
    body["request_id"] = request_id
    return JSONResponse(body)


__all__ = ["router"]
