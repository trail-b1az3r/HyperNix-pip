"""hyperlink.compaction — making a long conversation fit again.

A session that has run for a week does not fit in a context window, and
the thing that happens by default — dropping the oldest turns — loses
exactly what a long conversation is *for*. The decision made on Tuesday
falls out first.

Compaction replaces a run of messages with a summary of them. Five
shapes, because what is worth losing differs:

``prompts``
    The person's own messages. Useful where somebody has been pasting
    large inputs — a log, a file, a stack trace — and the *answers* are
    the thread's value.
``prompts/system``
    The system prompt alone. For a system prompt that grew by accretion
    and is now a third of the budget before anybody says anything.
``responses``
    The model's messages. The mirror case: long answers, and what
    matters is what was asked.
``all``
    Both sides, oldest-first.
``dynamic``
    Look at what is actually using the budget and compact that. The
    default, and the only one that does not require somebody to have
    already worked out where their tokens went.

Nothing is deleted
------------------
A compacted message is *marked*, not removed. ``messages()`` still
returns it, so the transcript a person scrolls is the one they had;
``context_for`` skips it in favour of the summary, so the model sees the
shorter version. Deleting would make compaction a destructive operation
somebody could not undo, over data they may care about more than the
model does.

That is also why :func:`plan` exists separately from :func:`apply`: a
caller can ask what would happen, and to how many tokens, before anything
does.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .sessions import ChatMessage, estimate_tokens

logger = logging.getLogger(__name__)

__all__ = [
    "SCOPES",
    "CompactionPlan",
    "plan",
    "summarise_extractively",
    "SUMMARY_MARKER",
    "COMPACTED_KEY",
]

#: What a caller may ask to compact. The endpoint names map onto these.
SCOPES = ("prompts", "system", "responses", "all", "dynamic")

#: Metadata key marking a message as superseded by a summary. On the
#: message rather than in a side table so it travels with the row: a
#: message that is compacted in one place and not another is a
#: conversation that means two different things depending on who asks.
COMPACTED_KEY = "compacted_by"

#: Marks a message as a summary rather than something anybody said. What
#: lets the app render it differently, and what stops a second compaction
#: summarising the summaries.
SUMMARY_MARKER = "compaction_summary"

#: Never compact the last few turns, whatever the scope says. The recent
#: exchange is what the next reply is a reply *to*; summarising it is how
#: compaction turns into "the assistant stopped following the thread".
KEEP_RECENT = 4

#: Below this, compaction costs more than it saves — a summary has its
#: own overhead, and three short messages do not become two.
MIN_MESSAGES = 3


@dataclass
class CompactionPlan:
    """What compaction would do, before it does it."""

    scope: str
    #: Messages that would be replaced by a summary.
    targets: list[ChatMessage] = field(default_factory=list)
    #: Messages left alone, and why, for a caller that wants to explain
    #: the result rather than just show it.
    kept_recent: int = 0
    tokens_before: int = 0
    #: What the summary is expected to cost. An estimate: the real figure
    #: is known only once a model has written it.
    tokens_after: int = 0
    reason: str = ""

    @property
    def viable(self) -> bool:
        """Worth doing. A plan that saves nothing is not an error — the
        session is simply already short — so callers check this rather
        than catching something."""
        return bool(self.targets) and self.tokens_after < self.tokens_before

    @property
    def saving(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "message_count": len(self.targets),
            "message_ids": [m.message_id for m in self.targets],
            "kept_recent": self.kept_recent,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "estimated_saving": self.saving,
            "viable": self.viable,
            "reason": self.reason,
        }


def _is_summary(message: ChatMessage) -> bool:
    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    return bool(metadata.get(SUMMARY_MARKER))


def _is_compacted(message: ChatMessage) -> bool:
    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    return bool(metadata.get(COMPACTED_KEY))


def _eligible(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Messages a compaction may touch at all.

    Already-compacted ones are gone from the model's view, and summaries
    are the *output* of a previous compaction — summarising them again
    is how a conversation becomes a summary of a summary of a summary,
    losing a little more each time for less and less saving.
    """
    return [m for m in messages if not _is_compacted(m) and not _is_summary(m)]


def _dynamic_scope(messages: list[ChatMessage]) -> tuple[str, str]:
    """Which side is actually using the budget.

    Measured, not guessed. "Compact the responses" is the right answer
    for somebody whose model writes essays and the wrong one for somebody
    pasting logs into it, and the server can tell which is which without
    being asked.
    """
    prompts = sum(estimate_tokens(m.content) for m in messages if m.role == "user")
    replies = sum(estimate_tokens(m.content) for m in messages if m.role == "assistant")
    system = sum(estimate_tokens(m.content) for m in messages if m.role == "system")
    total = prompts + replies + system

    if not total:
        return "all", "there is nothing here to compact yet."
    # A system prompt that is a third of everything before anybody has
    # said a word is its own problem and the cheapest thing to fix.
    if system and system / total > 0.33:
        return "system", (
            f"the system prompt is {system / total:.0%} of this conversation, "
            "which is a cost paid on every single turn."
        )
    if prompts > replies * 1.5:
        return "prompts", (
            f"what has been sent ({prompts} tokens) is well over what came "
            f"back ({replies}), so the inputs are the weight here."
        )
    if replies > prompts * 1.5:
        return "responses", (
            f"the replies ({replies} tokens) are well over what was asked "
            f"({prompts}), so the answers are the weight here."
        )
    return "all", "both sides are contributing about equally."


def plan(
    messages: list[ChatMessage],
    scope: str = "dynamic",
    *,
    keep_recent: int = KEEP_RECENT,
) -> CompactionPlan:
    """What compacting *scope* would replace, and what it would save."""
    if scope not in SCOPES:
        raise ValueError(f"Unknown compaction scope {scope!r}; expected {list(SCOPES)}")

    eligible = _eligible(messages)
    reason = ""
    if scope == "dynamic":
        scope, reason = _dynamic_scope(eligible)

    # The recent exchange is what the next reply answers. Protected
    # before the scope filter, so "compact the prompts" cannot quietly
    # take the question that was just asked.
    recent = {m.message_id for m in eligible[-keep_recent:]} if keep_recent else set()
    # A system message is never "recent" — it sits at the top of every
    # request regardless of age, so the protection means nothing for it
    # and would exempt it from the one scope that names it.
    recent -= {m.message_id for m in eligible if m.role == "system"}

    if scope == "prompts":
        roles = {"user"}
    elif scope == "responses":
        roles = {"assistant"}
    elif scope == "system":
        roles = {"system"}
    else:
        roles = {"user", "assistant"}

    targets = [
        m for m in eligible
        if m.role in roles and m.message_id not in recent and m.content.strip()
    ]

    before = sum(estimate_tokens(m.content) for m in targets)
    built = CompactionPlan(
        scope=scope,
        targets=targets,
        kept_recent=len(recent),
        tokens_before=before,
        # A summary of N messages, estimated. Deliberately conservative:
        # promising a saving that does not materialise is worse than
        # under-promising one that does.
        tokens_after=min(before, 60 + 12 * len(targets)),
        reason=reason,
    )
    if len(targets) < MIN_MESSAGES and scope != "system":
        built.targets = []
        built.reason = reason or (
            f"only {len(targets)} message(s) to work with; a summary has its "
            "own overhead and would not save anything."
        )
    return built


def summarise_extractively(messages: list[ChatMessage], *, limit: int = 1200) -> str:
    """A summary written without a model.

    The fallback for a server with no reachable model, and it is
    deliberately *extractive* — first lines of what was actually said,
    attributed — rather than an attempt at prose. A bad abstractive
    summary invents things the conversation did not contain, and a model
    reading an invented fact cannot tell it from a real one. Quoting is
    lossy and honest.
    """
    if not messages:
        return ""
    lines = ["[Earlier in this conversation, summarised:]"]
    for message in messages:
        speaker = {
            "user": "They asked", "assistant": "You answered",
            "system": "Instructions",
        }.get(message.role, message.role)
        first = " ".join(message.content.split())
        if len(first) > 160:
            first = first[:157].rstrip() + "…"
        lines.append(f"- {speaker}: {first}")
        if sum(len(line) for line in lines) > limit:
            lines.append(f"- (and {len(messages) - len(lines) + 2} more exchanges)")
            break
    return "\n".join(lines)


def summary_prompt(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """What to send a model to get a summary worth keeping.

    Explicit about the failure mode it is trying to avoid: a summary that
    reads well and drops the decision is worse than a clumsy one that
    keeps it, because nothing downstream can tell that something is
    missing.
    """
    transcript = "\n\n".join(
        f"{m.role.upper()}: {m.content}" for m in messages if m.content.strip()
    )
    return [
        {
            "role": "system",
            "content": (
                "Summarise the conversation below so it can replace the "
                "original in a context window. Keep every decision, every "
                "fact established, every name, number, file path and piece "
                "of code that was agreed on, and anything the user asked to "
                "be remembered. Drop pleasantries, restatements and "
                "reasoning that led nowhere. Write it as notes, not prose. "
                "If you are unsure whether something matters, keep it: what "
                "is lost here cannot be recovered, and nothing downstream "
                "will be able to tell that it is missing."
            ),
        },
        {"role": "user", "content": transcript},
    ]
