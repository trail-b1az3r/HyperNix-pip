"""t1tools — the T1 API's endpoints, offered to a model as tools.

A model running under HyperLink can ask the server things — what is
loaded, how busy the GPU is, what it remembers about you — and, with the
right key, change them. Each tool here is one real route, called over
HTTP with the **caller's own credential**, so a model can never do more
than the person it is talking to could do by hand. That rule is the
design: there is no in-process shortcut that would skip the API's
authentication, scopes, rate limits or audit log.

Mutating tools are marked, and :func:`catalogue` leaves them out unless
asked. A model that can unload the model it is running on is useful to
exactly one person, and surprising to everybody else.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..system import errorcatalogue as codes

__all__ = ["T1Tool", "TOOLS", "catalogue", "openai_tools", "call_tool"]


@dataclass(frozen=True)
class T1Tool:
    name: str
    method: str
    path: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    #: Sent as the JSON body (POST) rather than the query string.
    body: bool = False
    #: Changes the server's state. Off by default in :func:`catalogue`.
    mutating: bool = False

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.required),
                    "additionalProperties": False,
                },
            },
        }


def _s(desc: str, **extra) -> dict[str, Any]:
    return {"type": "string", "description": desc, **extra}


def _i(desc: str, **extra) -> dict[str, Any]:
    return {"type": "integer", "description": desc, **extra}


TOOLS: tuple[T1Tool, ...] = (
    T1Tool("server_status", "GET", "/status",
           "The server's version, how many models it knows, and its storage backend."),
    T1Tool("server_version", "GET", "/version",
           "The hyperNix version the server is running, and whether an "
           "installed upgrade is waiting for a restart."),
    T1Tool("hardware", "GET", "/hyperlink/hardware",
           "Current GPU, CPU and memory use on the server."),
    T1Tool("uptime", "GET", "/hyperlink/uptime",
           "How long the server process and the machine have been up."),
    T1Tool("runner_status", "GET", "/runner/status",
           "Which model the built-in runner has loaded, where its layers "
           "are, and for how long."),
    T1Tool("runner_queue", "GET", "/runner/hyperchat",
           "Whether prompts run in parallel or queue, and how many are waiting."),
    T1Tool("list_models", "GET", "/hyperlink/models",
           "Every model this server can serve, from every source it knows."),
    T1Tool("list_downloaded_models", "GET", "/hyperlink/models/downloaded",
           "GGUF files already on the server's disk."),
    T1Tool("web_search", "GET", "/web/v1/search",
           "Search the web without an API key.",
           {"q": _s("What to search for."),
            "depth": _i("Result pages to read, 1 to 5.", minimum=1, maximum=5)},
           required=("q",)),
    T1Tool("web_summarize", "POST", "/web/v1/summarize",
           "Summarise a web page or a piece of text.",
           {"url": _s("A page to fetch and summarise."),
            "text": _s("Text to summarise instead of a URL."),
            "query": _s("What the summary should focus on."),
            "sentences": _i("Roughly how many sentences.", minimum=1, maximum=12)},
           body=True),
    T1Tool("memory_list", "GET", "/memory/list",
           "What the assistant remembers about this person.",
           {"category": _s("Only this category."),
            "limit": _i("At most this many.", minimum=1, maximum=1000)}),
    T1Tool("memory_get", "GET", "/memory/get",
           "One memory by id.", {"memory_id": _s("The memory's id.")},
           required=("memory_id",)),
    T1Tool("memory_create", "POST", "/memory/create",
           "Remember something about this person for later conversations. "
           "Only for things they would want remembered.",
           {"content": _s("What to remember, as one sentence."),
            "category": _s("A short category, e.g. 'preferences'.")},
           required=("content",), body=True, mutating=True),
    T1Tool("runner_load", "POST", "/runner/load",
           "Load a different model into the built-in runner. This replaces "
           "the model currently answering.",
           {"model_id": _s("The model to load, as list_models names it."),
            "context_length": _i("Context window to load it with.", minimum=256)},
           required=("model_id",), body=True, mutating=True),
    T1Tool("runner_unload", "POST", "/runner/unload",
           "Unload the model from the built-in runner, freeing its memory.",
           body=True, mutating=True),
)

_BY_NAME = {tool.name: tool for tool in TOOLS}


def catalogue(*, allow_mutating: bool = False,
              only: list[str] | None = None) -> list[T1Tool]:
    """The tools to offer. Read-only unless *allow_mutating*."""
    chosen = [t for t in TOOLS if allow_mutating or not t.mutating]
    if only is not None:
        wanted = set(only)
        chosen = [t for t in chosen if t.name in wanted]
    return chosen


def openai_tools(**kwargs) -> list[dict[str, Any]]:
    return [t.to_openai() for t in catalogue(**kwargs)]


def call_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    base_url: str,
    token: str = "",
    allow_mutating: bool = False,
    timeout: float = 60.0,
    max_chars: int = 12_000,
) -> str:
    """Call one T1 endpoint and return its reply as text for the model.

    Never raises for an HTTP error: a 403 is information the model can
    relay ("I'm not allowed to load models with your key"), and an
    exception here would end the turn instead.
    """
    tool = _BY_NAME.get(name)
    if tool is None:
        return f"{codes.TOOLCALL_UNKNOWN.code}: no T1 tool called {name!r}"
    if tool.mutating and not allow_mutating:
        # Refused here as well as hidden from the catalogue: a model can
        # name a tool it was never shown, and hiding is not enforcing.
        return (f"{codes.TOOLCALL_REFUSED.code}: {name} changes the server "
                f"and is not enabled for this conversation")

    args = {k: v for k, v in (arguments or {}).items()
            if k in tool.parameters and v not in (None, "")}
    url = base_url.rstrip("/") + tool.path
    data = None
    if tool.body:
        data = json.dumps(args).encode()
    elif args:
        url += "?" + urllib.parse.urlencode(args)

    request = urllib.request.Request(url, data=data, method=tool.method)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return f"HTTP {exc.code} from {tool.path}: {_error_text(body)}"
    except (urllib.error.URLError, OSError) as exc:
        return (f"{codes.T1_CODES['TRANSPORT_FAILED'].code}: could not reach "
                f"the server: {exc}")
    return _compact(payload, max_chars)


def _error_text(body: str) -> str:
    try:
        error = json.loads(body).get("error") or {}
        return f"{error.get('hx_code', error.get('code', ''))} {error.get('message', '')}".strip()
    except (ValueError, AttributeError):
        return body[:300]


def _compact(payload: str, max_chars: int) -> str:
    """Pretty enough for a model to read; short enough to fit its context."""
    try:
        data = json.loads(payload)
        if isinstance(data, dict):
            data.pop("request_id", None)
        text = json.dumps(data, indent=1, ensure_ascii=False)
    except ValueError:
        text = payload
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n…[{len(text) - max_chars} characters cut]"
    return text
