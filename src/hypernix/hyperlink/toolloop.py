"""Letting a HyperLink model actually use the tools.

``/noodle/run`` lets a *caller* run one tool. This is the other half:
the model asks, the server runs it, and the answer goes back into the
conversation — so "zip up the logs and tell me what is in them" is one
message rather than five.

The shape is OpenAI's tool-calling protocol, because that is what LM
Studio, llama.cpp's server and every hosted backend speak. Nothing here
invents a format.

Three rules, and each one is a failure that has to be designed out
rather than handled.

**Bounded.** A model that calls a tool, reads the result, and calls it
again can do that forever — and each round is a full inference on a
growing transcript. :data:`MAX_ROUNDS` ends it, and the model is *told*
it has been ended, so the last reply is written knowing it could not
finish rather than looking like it chose to stop.

**Every call is answered.** A ``tool_calls`` message with no matching
``tool`` reply is a malformed conversation, and the next turn either
errors or quietly drops it. So a tool that does not exist, one that
raises, and one that is switched off all produce a tool message saying
so. The model can then apologise or try something else, which is a much
better outcome than a 500.

**A refusal is a result.** ``allow_execute`` off is not an error: it is
the operator's answer. Returning it to the model as a normal tool
result — "execution is disabled on this server" — lets the model say
that to the person, instead of the request failing with a stack trace
they cannot act on.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from ..interfaces.noodle.tools import TOOLS, ToolContext, run_tool
from ..runtime import t1tools
from ..runtime.toolcalls import (
    ToolCall,
    extract_calls,
    tool_prompt,
    validate_arguments,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_ROUNDS",
    "MAX_RESULT_CHARS",
    "T1Access",
    "ToolRound",
    "openai_tools",
    "run_tool_loop",
]

#: How many times the model may call tools before it has to answer.
#:
#: Each round is a whole inference over a transcript that just grew by a
#: tool result, so this is a wall-clock and a cost limit as much as a
#: safety one. Eight is enough for "look at these files, zip the ones
#: that match, tell me what you did" and far short of a loop that runs
#: until somebody notices the GPU is busy.
MAX_ROUNDS = 8

#: Longest tool result handed back to the model.
#:
#: A directory listing or a file read can be megabytes. Past this it is
#: not context, it is the whole context — the conversation gets evicted
#: to make room for a listing nobody asked to see.
MAX_RESULT_CHARS = 8_000


class T1Access:
    """How a model reaches the T1 API: the server, and the caller's key.

    The caller's own key, never a server one — so a model can do exactly
    what the person it is talking to could do by hand, and no more.
    """

    __slots__ = ("base_url", "token", "allow_mutating", "only")

    def __init__(self, base_url: str, token: str = "", *,
                 allow_mutating: bool = False, only: list[str] | None = None) -> None:
        self.base_url = base_url
        self.token = token
        self.allow_mutating = allow_mutating
        self.only = only


class ToolRound:
    """One call and its result, for showing what happened."""

    __slots__ = ("tool", "arguments", "ok", "content", "code")

    def __init__(
        self, tool: str, arguments: dict[str, Any], ok: bool, content: str, code: str = ""
    ) -> None:
        self.tool = tool
        self.arguments = arguments
        self.ok = ok
        self.content = content
        self.code = code

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "ok": self.ok,
            "content": self.content,
            "code": self.code,
        }


def openai_tools(context: ToolContext) -> list[dict[str, Any]]:
    """The tool list in the shape every backend expects.

    Tools the context has switched off are left out rather than offered
    and refused. A model that is told it can execute files and then told
    it cannot, every time, spends its rounds finding that out.
    """
    offered: list[dict[str, Any]] = []
    for tool in TOOLS.values():
        if tool.name == "execute_file" and not context.allow_execute:
            continue
        if tool.name == "web_search" and not context.allow_web_search:
            continue
        if tool.name in ("update_memory", "read_memory") and not context.memory_enabled:
            continue
        parameters = dict(tool.parameters)
        offered.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        })
    return offered


def _arguments(raw: Any) -> dict[str, Any]:
    """Parse a call's arguments, whatever the model produced.

    Models emit this field as a JSON *string*, and sometimes as an empty
    string, and occasionally as an object because a backend already
    parsed it. All three have to work, and a malformed one has to come
    back as a tool error rather than an exception — the model can fix
    its own JSON if it is told what was wrong with it.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the arguments were not valid JSON: {exc}") from exc
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _truncate(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    dropped = len(text) - MAX_RESULT_CHARS
    return (
        text[:MAX_RESULT_CHARS]
        + f"\n\n[{dropped} more characters were cut. Ask for a narrower "
        f"range if you need them.]"
    )


def run_tool_loop(
    wire: list[dict[str, Any]],
    context: ToolContext,
    *,
    ask: Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]],
    extract: Callable[[dict[str, Any]], tuple[str, str]],
    max_rounds: int = MAX_ROUNDS,
    on_round: Callable[[ToolRound], None] | None = None,
    t1: T1Access | None = None,
    teach_format: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[ToolRound]]:
    """Run the model until it answers rather than calls a tool.

    *ask* takes ``(messages, tools)`` and returns the raw envelope;
    *extract* pulls ``(content, finish_reason)`` out of one. Both are
    passed in rather than imported so this works for any backend, and so
    a test can drive it without a model.

    Returns ``(final_messages, final_envelope, rounds)``. The messages
    include the assistant's tool calls and every tool reply, because
    that transcript is what the *next* turn has to be built from — a
    conversation missing its tool replies is one the model cannot
    continue.
    """
    messages = list(wire)
    tools = openai_tools(context)
    t1_names: set[str] = set()
    if t1 is not None:
        # Existing tools win a name clash: noodle's `web_search` and the
        # T1 one both exist, and a model shown two tools with one name
        # calls whichever it saw last.
        taken = {t["function"]["name"] for t in tools}
        for tool in t1tools.openai_tools(allow_mutating=t1.allow_mutating, only=t1.only):
            name = tool["function"]["name"]
            if name not in taken:
                tools.append(tool)
                t1_names.add(name)
    if teach_format and tools:
        # For a model without native tool support: one format, one worked
        # example. Prepended as system text so it survives templates that
        # drop the `tools` field entirely.
        messages.insert(0, {"role": "system", "content": tool_prompt(tools)})
    rounds: list[ToolRound] = []
    envelope: dict[str, Any] = {}
    retried_format = False

    for round_number in range(max_rounds):
        envelope = ask(messages, tools)
        calls, failures = _calls_in(envelope, tools)
        if not calls:
            if failures and not retried_format:
                # It tried to call a tool and the JSON was beyond repair.
                # Told once, rather than the person being shown the raw
                # markup as if it were the answer.
                retried_format = True
                messages.append(_assistant_message(envelope, []))
                messages.append({
                    "role": "system",
                    "content": (
                        "Your tool call could not be read: "
                        + "; ".join(why for _, why in failures[:3])
                        + ". Send it again as <tool_call>{\"name\": ..., "
                          "\"arguments\": {...}}</tool_call> with valid JSON, "
                          "or answer without a tool."
                    ),
                })
                continue
            return messages, envelope, rounds

        assistant = _assistant_message(envelope, calls)
        messages.append(assistant)

        for call in calls:
            name = (call.get("function") or {}).get("name", "")
            if name in t1_names:
                record = _run_t1(call, tools, t1)
            else:
                record = _run_one(call, context, tools)
            rounds.append(record)
            if on_round is not None:
                try:
                    on_round(record)
                except Exception:  # noqa: BLE001 - a listener must not fail the turn
                    logger.debug("hyperlink: tool listener raised", exc_info=True)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id") or "",
                "name": record.tool,
                "content": _truncate(record.content),
            })

        if round_number == max_rounds - 1:
            # Told, not silently cut off. A model that does not know it
            # ran out of rounds writes its last reply as if it were
            # about to do more, and the person is left with a half
            # sentence about what it was going to check next.
            messages.append({
                "role": "system",
                "content": (
                    f"You have used all {max_rounds} tool rounds for this "
                    f"turn. Answer now with what you have, and say what you "
                    f"did not get to."
                ),
            })
            envelope = ask(messages, [])

    return messages, envelope, rounds


def _calls_in(
    envelope: dict[str, Any], tools: list[dict[str, Any]] | None = None
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Tool calls in *envelope*, structured or written as text.

    Structured `tool_calls` first. When there are none, the reply's text
    is read for the conventions local models actually use — Hermes/Qwen
    `<tool_call>`, Llama's `<|python_tag|>`, Mistral's `[TOOL_CALLS]`, a
    fenced JSON block. Without this, a GGUF that calls a tool the way it
    was trained to has its call shown to the person as the answer, and
    nothing runs. Returns ``(calls, unreadable)``.
    """
    choices = envelope.get("choices") or []
    if not choices:
        return [], []
    message = (choices[0] or {}).get("message") or {}
    structured = [c for c in (message.get("tool_calls") or []) if isinstance(c, dict)]
    if structured:
        return structured, []
    found = extract_calls(message, tools=tools)
    return [c.to_openai() for c in found.calls], found.failures


def _assistant_message(
    envelope: dict[str, Any], calls: list[dict[str, Any]]
) -> dict[str, Any]:
    choices = envelope.get("choices") or []
    message = (choices[0] or {}).get("message") or {}
    # `content` is kept even when it is empty: some backends reject an
    # assistant message without the key, and some models write a
    # sentence *and* call a tool, which the person should see.
    return {
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": calls,
    }


def _validated(call: dict[str, Any], tools: list[dict[str, Any]] | None):
    """``(name, arguments, error_round)`` — error_round set when unusable."""
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    try:
        arguments = _arguments(function.get("arguments"))
    except ValueError as exc:
        return name, {}, ToolRound(name, {}, False, str(exc), code="bad_arguments")
    if tools:
        arguments, problem = validate_arguments(ToolCall(name, arguments), tools)
        if problem:
            # `bad_arguments` means the JSON did not parse; this is JSON
            # that parsed and does not fit the tool — a different thing
            # for the model to fix, so a different code.
            code = "no_such_tool" if "no tool called" in problem else "invalid_arguments"
            return name, arguments, ToolRound(name, arguments, False, problem, code=code)
    return name, arguments, None


def _run_t1(call: dict[str, Any], tools: list[dict[str, Any]], t1: T1Access) -> ToolRound:
    """A T1 endpoint, called with the caller's own key."""
    name, arguments, failed = _validated(call, tools)
    if failed is not None:
        return failed
    content = t1tools.call_tool(
        name, arguments, base_url=t1.base_url, token=t1.token,
        allow_mutating=t1.allow_mutating,
    )
    ok = not (content.startswith("HTTP ") or content[:2] in ("R2", "T1"))
    return ToolRound(name, arguments, ok, content, code="" if ok else "t1_error")


def _run_one(call: dict[str, Any], context: ToolContext,
             tools: list[dict[str, Any]] | None = None) -> ToolRound:
    """One call, with every failure turned into a result the model reads."""
    name, arguments, failed = _validated(call, tools)
    if failed is not None:
        return failed

    if name not in TOOLS:
        return ToolRound(
            name, arguments, False,
            f"There is no tool called {name!r}. Available: "
            f"{', '.join(sorted(TOOLS))}.",
            code="no_such_tool",
        )

    try:
        result = run_tool(context, name, arguments)
    except Exception as exc:  # noqa: BLE001
        # Including this in the transcript rather than raising: the
        # model can apologise or try another way, where a 500 loses the
        # whole turn and the person's question with it.
        logger.info("hyperlink: tool %s raised: %s", name, exc)
        return ToolRound(
            name, arguments, False, f"The tool failed: {exc}", code="tool_raised"
        )

    return ToolRound(
        name, arguments, bool(result.ok), result.content or "", code=result.code or ""
    )
