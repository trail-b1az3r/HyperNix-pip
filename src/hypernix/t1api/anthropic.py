"""t1api.anthropic — the Anthropic Messages API over HyperNix's own models.

Claude Code (and any Anthropic SDK client) speaks ``POST /v1/messages``:
a system prompt, messages made of content blocks, tools described by a
JSON schema, and — streamed — a sequence of typed server-sent events.
llama.cpp and LM Studio speak the OpenAI chat API. This module is the
translation between the two, both ways, with no FastAPI in it, so every
case can be tested against a dict; ``routers/anthropic.py`` is the
transport.

What maps to what
-----------------
* ``system`` (a string or text blocks) -> the first, ``system`` message.
* A ``text`` block -> text; an ``image`` block (base64 or URL) -> an
  ``image_url`` part; a ``document`` with a text source -> its text.
* An assistant's ``tool_use`` block -> an OpenAI ``tool_call``, and the
  user's ``tool_result`` -> a ``tool`` message, *before* any text in the
  same turn (OpenAI requires tool messages to follow their call).
* ``thinking`` blocks a client sends back are dropped: they belong to
  the model that wrote them, and a local model's reasoning is not one.
* ``tools`` -> ``functions``; ``tool_choice`` ``auto``/``any``/``tool``/
  ``none`` -> ``auto``/``required``/that function/``none``.
* Back: text and tool calls become ``text`` and ``tool_use`` blocks;
  ``finish_reason`` ``stop``/``length``/``tool_calls`` become
  ``end_turn``/``max_tokens``/``tool_use``.

Tool arguments from a small local model are not always JSON. They are
parsed if they are, recovered from the first ``{...}`` if that parses,
and otherwise passed as ``{}`` with the raw text kept, so the client gets
a tool call it can answer with an error the model can read, rather than
a 500.
"""
from __future__ import annotations

import json
import re
import secrets
from collections.abc import Iterable
from typing import Any

__all__ = [
    "ANTHROPIC_VERSION",
    "StreamTranslator",
    "error_body",
    "estimate_tokens",
    "from_openai",
    "message_id",
    "sse",
    "to_openai",
]

#: The ``anthropic-version`` this surface implements.
ANTHROPIC_VERSION = "2023-06-01"

_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                 "function_call": "tool_use", "content_filter": "refusal"}


def message_id() -> str:
    return "msg_" + secrets.token_hex(12)


def _tool_id() -> str:
    return "toolu_" + secrets.token_hex(12)


def error_body(kind: str, message: str) -> dict[str, Any]:
    """Anthropic's error envelope. *kind* is e.g. ``invalid_request_error``."""
    return {"type": "error", "error": {"type": kind, "message": message}}


def sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI
# ---------------------------------------------------------------------------

def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content if isinstance(content, list) else [content]:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif block.get("type") == "document":
                parts.append(_document_text(block))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(p for p in parts if p)


def _document_text(block: dict[str, Any]) -> str:
    source = block.get("source") or {}
    title = block.get("title") or ""
    if source.get("type") == "text":
        return (f"[{title}]\n" if title else "") + str(source.get("data", ""))
    if source.get("type") == "content":
        return _text_of(source.get("content"))
    return f"[a {source.get('media_type', 'document')} was attached; this model reads text only]"


def _image_part(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source") or {}
    if source.get("type") == "base64":
        return {"type": "image_url", "image_url": {
            "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"}}
    if source.get("type") == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _user_turn(content: Any) -> list[dict[str, Any]]:
    """One Anthropic user turn as OpenAI messages: tool results first."""
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    tool_messages: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    for block in content or []:
        kind = block.get("type") if isinstance(block, dict) else "text"
        if kind == "tool_result":
            text = _text_of(block.get("content"))
            if block.get("is_error"):
                text = f"Error: {text}" if text else "Error"
            tool_messages.append({"role": "tool", "tool_call_id": str(block.get("tool_use_id", "")),
                                  "content": text or "(no output)"})
        elif kind == "text":
            parts.append({"type": "text", "text": str(block.get("text", "") if isinstance(block, dict) else block)})
        elif kind == "image":
            image = _image_part(block)
            if image:
                parts.append(image)
        elif kind == "document":
            parts.append({"type": "text", "text": _document_text(block)})
    out = list(tool_messages)
    if parts:
        if all(p["type"] == "text" for p in parts):
            out.append({"role": "user", "content": "\n".join(p["text"] for p in parts)})
        else:
            out.append({"role": "user", "content": parts})
    return out


def _assistant_turn(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    texts, calls = [], []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            texts.append(str(block.get("text", "")))
        elif block.get("type") in ("tool_use", "server_tool_use"):
            calls.append({"id": str(block.get("id") or _tool_id()), "type": "function",
                          "function": {"name": str(block.get("name", "")),
                                       "arguments": json.dumps(block.get("input") or {})}})
    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(t for t in texts if t) or None}
    if calls:
        message["tool_calls"] = calls
    return message


def to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """An Anthropic Messages request as OpenAI chat-completions arguments.

    Returns ``messages``, ``tools`` (possibly empty) and ``extra`` (the
    keys :class:`~hypernix.bridge.lmstudio.LMStudioBridge` passes through
    as they are), plus ``max_tokens``, ``temperature``, ``top_p``, ``stop``.
    """
    messages: list[dict[str, Any]] = []
    system = _text_of(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for turn in body.get("messages") or []:
        role = turn.get("role")
        if role == "assistant":
            messages.append(_assistant_turn(turn.get("content")))
        else:
            messages.extend(_user_turn(turn.get("content")))
    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        if tool.get("type") and not str(tool["type"]).startswith("custom") and "input_schema" not in tool:
            continue  # Anthropic's own server tools (web_search_...) have no schema to give a local model
        tools.append({"type": "function", "function": {
            "name": tool["name"], "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}}}})
    extra: dict[str, Any] = {}
    choice = body.get("tool_choice") or {}
    if tools and isinstance(choice, dict) and choice.get("type"):
        mapped = {"auto": "auto", "any": "required", "none": "none"}.get(choice["type"])
        if choice["type"] == "tool" and choice.get("name"):
            extra["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        elif mapped:
            extra["tool_choice"] = mapped
        if choice.get("disable_parallel_tool_use"):
            extra["parallel_tool_calls"] = False
    if body.get("top_k") is not None:
        extra["top_k"] = body["top_k"]
    return {
        "messages": messages,
        "tools": tools,
        "extra": extra,
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "stop": body.get("stop_sequences") or None,
        "thinking": (body.get("thinking") or {}).get("type") == "enabled",
    }


def prompt_text(converted: dict[str, Any]) -> str:
    """Everything the model will read, for estimating its size."""
    out = []
    for message in converted["messages"]:
        content = message.get("content")
        out.append(content if isinstance(content, str) else
                   " ".join(p.get("text", "") for p in content or [] if isinstance(p, dict)))
        for call in message.get("tool_calls") or []:
            out.append(call["function"]["arguments"])
    out.append(json.dumps(converted["tools"]))
    return "\n".join(o for o in out if o)


# ---------------------------------------------------------------------------
# OpenAI -> Anthropic
# ---------------------------------------------------------------------------

def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                value = json.loads(match.group(0))
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
    return {"_unparsed_arguments": text}


def _usage(usage: dict[str, Any] | None, *, input_estimate: int = 0, output_estimate: int = 0) -> dict[str, int]:
    usage = usage or {}
    return {"input_tokens": int(usage.get("prompt_tokens") or input_estimate),
            "output_tokens": int(usage.get("completion_tokens") or output_estimate),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)}


def from_openai(envelope: dict[str, Any], *, model: str, thinking: bool = False,
                input_estimate: int = 0, stop: Iterable[str] | None = None) -> dict[str, Any]:
    """An OpenAI chat completion as an Anthropic ``message``."""
    choice = (envelope.get("choices") or [{}])[0] or {}
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if thinking and reasoning:
        content.append({"type": "thinking", "thinking": str(reasoning), "signature": ""})
    text = message.get("content")
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    if text:
        content.append({"type": "text", "text": str(text)})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        content.append({"type": "tool_use", "id": str(call.get("id") or _tool_id()),
                        "name": str(function.get("name", "")), "input": _arguments(function.get("arguments"))})
    finish = str(choice.get("finish_reason") or "stop")
    stop_reason = _STOP_REASONS.get(finish, "end_turn")
    if any(block["type"] == "tool_use" for block in content):
        stop_reason = "tool_use"
    stop_sequence = None
    if stop and finish == "stop" and text:
        stop_sequence = next((s for s in stop if str(text).endswith(s)), None)
        if stop_sequence:
            stop_reason = "stop_sequence"
    if not content:
        content.append({"type": "text", "text": ""})
    output_estimate = estimate_tokens(str(text or "")) + sum(
        estimate_tokens(json.dumps(b.get("input"))) for b in content if b["type"] == "tool_use")
    return {"id": message_id(), "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": stop_reason, "stop_sequence": stop_sequence,
            "usage": _usage(envelope.get("usage"), input_estimate=input_estimate,
                            output_estimate=output_estimate)}


class StreamTranslator:
    """OpenAI stream chunks in, Anthropic stream events out.

    Anthropic's blocks are sequential: a block is started, filled and
    stopped before the next starts. OpenAI interleaves text and the
    arguments of several tool calls freely, so the open block is closed
    whenever something of another kind (or another tool call) arrives.
    """

    def __init__(self, *, model: str, input_tokens: int = 0, thinking: bool = False) -> None:
        self.model = model
        self.thinking = thinking
        self.id = message_id()
        self.input_tokens = input_tokens
        self.output_chars = 0
        self.usage: dict[str, Any] | None = None
        self.finish: str = ""
        self._index = -1
        self._open: tuple[str, Any] | None = None     # ("text"|"thinking"|"tool", openai tool index)
        self._started = False
        self._blocks = 0
        self._tool_seen = False
        self.text: list[str] = []

    def start(self) -> list[bytes]:
        self._started = True
        return [sse("message_start", {"type": "message_start", "message": {
            "id": self.id, "type": "message", "role": "assistant", "model": self.model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": 0,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}}),
            sse("ping", {"type": "ping"})]

    def _close(self) -> list[bytes]:
        if self._open is None:
            return []
        self._open = None
        return [sse("content_block_stop", {"type": "content_block_stop", "index": self._index})]

    def _begin(self, kind: str, key: Any, block: dict[str, Any]) -> list[bytes]:
        out = self._close()
        self._index += 1
        self._blocks += 1
        self._open = (kind, key)
        out.append(sse("content_block_start", {"type": "content_block_start", "index": self._index,
                                               "content_block": block}))
        return out

    def feed(self, chunk: dict[str, Any]) -> list[bytes]:
        out = [] if self._started else self.start()
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = (choice or {}).get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning and self.thinking:
                if self._open is None or self._open[0] != "thinking":
                    out += self._begin("thinking", None, {"type": "thinking", "thinking": "", "signature": ""})
                out.append(sse("content_block_delta", {"type": "content_block_delta", "index": self._index,
                                                       "delta": {"type": "thinking_delta", "thinking": reasoning}}))
            text = delta.get("content")
            if text:
                if self._open is None or self._open[0] != "text":
                    out += self._begin("text", None, {"type": "text", "text": ""})
                self.output_chars += len(text)
                self.text.append(text)
                out.append(sse("content_block_delta", {"type": "content_block_delta", "index": self._index,
                                                       "delta": {"type": "text_delta", "text": text}}))
            for call in delta.get("tool_calls") or []:
                key = call.get("index", 0)
                function = call.get("function") or {}
                if self._open is None or self._open != ("tool", key):
                    self._tool_seen = True
                    out += self._begin("tool", key, {"type": "tool_use", "id": str(call.get("id") or _tool_id()),
                                                     "name": str(function.get("name") or ""), "input": {}})
                arguments = function.get("arguments")
                if arguments:
                    self.output_chars += len(arguments)
                    out.append(sse("content_block_delta", {"type": "content_block_delta", "index": self._index,
                                                           "delta": {"type": "input_json_delta",
                                                                     "partial_json": arguments}}))
            if choice.get("finish_reason"):
                self.finish = str(choice["finish_reason"])
        return out

    def error(self, message: str) -> list[bytes]:
        return [sse("error", error_body("api_error", message))]

    def end(self) -> list[bytes]:
        out = [] if self._started else self.start()
        if self._blocks == 0:
            out += self._begin("text", None, {"type": "text", "text": ""})
        out += self._close()
        stop_reason = "tool_use" if self._tool_seen else _STOP_REASONS.get(self.finish or "stop", "end_turn")
        usage = _usage(self.usage, input_estimate=self.input_tokens, output_estimate=max(0, self.output_chars // 4))
        out.append(sse("message_delta", {"type": "message_delta",
                                         "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                         "usage": {"output_tokens": usage["output_tokens"]}}))
        out.append(sse("message_stop", {"type": "message_stop"}))
        self.final_usage = usage
        return out
