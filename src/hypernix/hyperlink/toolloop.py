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

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_ROUNDS",
    "MAX_RESULT_CHARS",
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
    rounds: list[ToolRound] = []
    envelope: dict[str, Any] = {}

    for round_number in range(max_rounds):
        envelope = ask(messages, tools)
        calls = _calls_in(envelope)
        if not calls:
            return messages, envelope, rounds

        assistant = _assistant_message(envelope, calls)
        messages.append(assistant)

        for call in calls:
            record = _run_one(call, context)
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


def _calls_in(envelope: dict[str, Any]) -> list[dict[str, Any]]:
    choices = envelope.get("choices") or []
    if not choices:
        return []
    message = (choices[0] or {}).get("message") or {}
    calls = message.get("tool_calls") or []
    return [c for c in calls if isinstance(c, dict)]


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


def _run_one(call: dict[str, Any], context: ToolContext) -> ToolRound:
    """One call, with every failure turned into a result the model reads."""
    function = call.get("function") or {}
    name = str(function.get("name") or "")

    try:
        arguments = _arguments(function.get("arguments"))
    except ValueError as exc:
        return ToolRound(name, {}, False, str(exc), code="bad_arguments")

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
