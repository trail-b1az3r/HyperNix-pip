"""noodle.agent — one agent: a model, a tool loop, and a self-correction cycle.

The loop is the usual shape — send, run whatever tools come back, send
the results, repeat — with three things that are not.

**Compaction is done by the loop, not the tool.** ``compact_context``
returns a *request*; this class performs it, because the loop owns the
transcript and is the only thing that can replace it. A tool that
returned a summary would be handing the model a string and hoping.

**Self-correction is bounded and evidenced.** After the model says it is
finished, an optional verifier runs (syntax check, tests, whatever the
caller supplies). A failure is fed back as a tool result rather than a
new instruction, because that is the channel the model is already
reading, and the cycle is capped: an agent that cannot fix its own work
in three attempts is not going to fix it in thirty, and saying so beats
looping until the budget runs out.

**Every turn is observable.** :attr:`Agent.on_event` receives structured
events for thoughts, tool calls, results and corrections. That is what
the live-stream TUI renders; without it a swarm is a progress bar
attached to nothing.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .providers import ChatResult, ModelClient, ProviderError
from .tools import ToolContext, ToolResult, run_tool, tool_schemas

logger = logging.getLogger(__name__)

__all__ = ["Agent", "AgentResult", "AgentEvent", "Verifier"]

#: A verifier is given the workspace and returns ``(ok, feedback)``.
Verifier = Callable[[ToolContext], "tuple[bool, str]"]


@dataclass
class AgentEvent:
    kind: str                # thought | tool_call | tool_result | correction | done | error
    agent: str
    detail: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "agent": self.agent, "at": self.at, "detail": dict(self.detail)}


@dataclass
class AgentResult:
    agent: str
    ok: bool
    content: str
    turns: int = 0
    tool_calls: int = 0
    corrections: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: str = ""
    events: list[AgentEvent] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent, "ok": self.ok, "content": self.content,
            "turns": self.turns, "tool_calls": self.tool_calls,
            "corrections": self.corrections,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "seconds": round(self.seconds, 2), "provider": self.provider,
            "model": self.model, "error": self.error,
        }


class Agent:
    """One model working a task with tools."""

    def __init__(
        self,
        name: str,
        client: ModelClient,
        context: ToolContext,
        *,
        system: str = "",
        tools: Sequence[str] | None = None,
        max_turns: int = 12,
        max_corrections: int = 3,
        verifier: Verifier | None = None,
        on_event: Callable[[AgentEvent], None] | None = None,
        temperature: float | None = None,
    ) -> None:
        self.name = name
        self.client = client
        self.context = context
        self.system = system or _DEFAULT_SYSTEM
        self.tool_names = list(tools) if tools else None
        self.max_turns = int(max_turns)
        self.max_corrections = int(max_corrections)
        self.verifier = verifier
        self.on_event = on_event
        self.temperature = temperature

    # -- events -------------------------------------------------------

    def _emit(self, kind: str, **detail: Any) -> AgentEvent:
        event = AgentEvent(kind=kind, agent=self.name, detail=detail)
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # noqa: BLE001 - a broken listener must not fail the run
                logger.debug("noodle.agent: on_event raised", exc_info=True)
        return event

    # -- the loop -----------------------------------------------------

    def run(self, task: str) -> AgentResult:
        """Work *task* to completion, or to a stated limit."""
        started = time.monotonic()
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        schemas = tool_schemas(self.tool_names)
        result = AgentResult(
            agent=self.name, ok=False, content="",
            provider=self.client.spec.provider.value, model=self.client.model,
        )
        corrections = 0

        while result.turns < self.max_turns:
            result.turns += 1
            try:
                reply = self.client.chat(
                    messages, system=self.system, tools=schemas,
                    temperature=self.temperature,
                )
            except ProviderError as exc:
                result.error = str(exc)
                result.events.append(self._emit("error", **exc.to_dict()))
                result.seconds = time.monotonic() - started
                return result

            result.input_tokens += reply.input_tokens
            result.output_tokens += reply.output_tokens
            if reply.content.strip():
                result.events.append(
                    self._emit("thought", text=reply.content, turn=result.turns)
                )

            if reply.wants_tools:
                messages.append(_assistant_turn(reply))
                compact_request: dict[str, Any] | None = None
                for call in reply.tool_calls:
                    result.tool_calls += 1
                    result.events.append(
                        self._emit("tool_call", tool=call.name, arguments=call.arguments)
                    )
                    outcome = run_tool(self.context, call.name, call.arguments)
                    result.events.append(
                        self._emit(
                            "tool_result", tool=call.name, ok=outcome.ok,
                            code=outcome.code, preview=outcome.content[:400],
                        )
                    )
                    messages.append(_tool_turn(call.call_id, call.name, outcome))
                    if outcome.code == "compact_requested":
                        compact_request = outcome.data
                if compact_request is not None:
                    messages = self._compact(messages, compact_request)
                continue

            # No tools requested: the model considers itself done.
            result.content = reply.content
            if self.verifier is None:
                result.ok = True
                break

            ok, feedback = self._verify()
            if ok:
                result.ok = True
                break
            corrections += 1
            result.corrections = corrections
            if corrections > self.max_corrections:
                result.error = (
                    f"Verification still failing after {self.max_corrections} correction "
                    f"attempts. Last feedback: {feedback[:400]}"
                )
                result.events.append(self._emit("error", reason="corrections_exhausted"))
                break
            result.events.append(
                self._emit("correction", attempt=corrections, feedback=feedback[:400])
            )
            # Fed back as a user turn rather than a system one: the model
            # is already reading this channel, and a mid-conversation
            # system message is treated differently (or dropped) by
            # several providers.
            messages.append({"role": "assistant", "content": reply.content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"That did not pass verification (attempt {corrections} of "
                        f"{self.max_corrections}):\n\n{feedback}\n\n"
                        "Fix it using the tools, then stop."
                    ),
                }
            )
        else:
            result.error = (
                f"Hit the {self.max_turns}-turn limit without finishing. "
                "Either the task needs splitting or the agent is looping."
            )
            result.events.append(self._emit("error", reason="turn_limit"))

        result.seconds = time.monotonic() - started
        result.events.append(
            self._emit("done", ok=result.ok, turns=result.turns, error=result.error)
        )
        return result

    def _verify(self) -> tuple[bool, str]:
        if self.verifier is None:
            raise AssertionError('self.verifier is not None')
        try:
            return self.verifier(self.context)
        except Exception as exc:  # noqa: BLE001
            # A verifier that crashes is a broken verifier, not a failed
            # task. Say which, or the agent spends its correction budget
            # trying to fix someone else's bug.
            logger.exception("noodle.agent: verifier raised")
            return False, f"The verifier itself failed ({type(exc).__name__}: {exc})"

    def _compact(
        self, messages: list[dict[str, Any]], request: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Summarise the transcript, keeping the most recent turns verbatim.

        The summary is mechanical rather than model-generated: asking the
        model to summarise costs another round trip at exactly the moment
        the context is too big, and what actually needs preserving — which
        files were touched, which tools failed — is structured data the
        loop already has.
        """
        keep = max(2, int(request.get("keep_last") or 6))
        if len(messages) <= keep + 1:
            return messages
        head, tail = messages[:-keep], messages[-keep:]
        touched = sorted(
            {
                str(entry.get("path", ""))
                for entry in self.context.audit
                if entry.get("path")
            }
        )
        open_todos = [t.text for t in self.context.todos.values() if t.status != "done"]
        summary = (
            f"[compacted: {len(head)} earlier messages removed, {request.get('reason', '')}]\n"
            f"Files touched so far: {', '.join(touched) or 'none'}\n"
            f"Tool calls so far: {len(self.context.audit)}\n"
            f"Still open: {'; '.join(open_todos) or 'nothing tracked'}"
        )
        self._emit("correction", compacted=len(head), kept=keep)
        return [{"role": "user", "content": summary}, *tail]


_DEFAULT_SYSTEM = (
    "You are a Noodle agent working inside a sandboxed workspace.\n"
    "\n"
    "Use the tools to do the work rather than describing what you would do. Read a file "
    "before editing it. When you believe the task is complete, stop calling tools and say "
    "what you did — an automated check may then run, and if it fails you will be told why "
    "and asked to fix it.\n"
    "\n"
    "Every path is relative to the workspace root and paths outside it are refused. If a "
    "tool returns an error, read it: it says what was wrong and usually what to do instead."
)


def _assistant_turn(reply: ChatResult) -> dict[str, Any]:
    """The assistant message carrying tool calls, in OpenAI's shape.

    The transcript is kept in one format and translated per provider at
    send time (see ``providers.ModelClient``), rather than being stored
    in whatever shape the last provider used — otherwise a swarm that
    hands a task from one model to another hands over a transcript the
    second one cannot read.
    """
    return {
        "role": "assistant",
        "content": reply.content or None,
        "tool_calls": [
            {
                "id": call.call_id or call.name,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in reply.tool_calls
        ],
    }


def _tool_turn(call_id: str, name: str, outcome: ToolResult) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id or name,
        "name": name,
        "content": outcome.content if outcome.ok else f"ERROR ({outcome.code}): {outcome.content}",
    }
