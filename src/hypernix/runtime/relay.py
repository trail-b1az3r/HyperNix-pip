"""Read prompts longer than a model's context, with small-context instances in series.

A model with a 4k context cannot read a 40k-token document. Several
copies of it can, one part at a time, if each hands the next what it has
learned so far. That is a *relay*::

    part 1 ──► instance 0 ──notes──► part 2 ──► instance 1 ──notes──► …
                                                             └──► answer

Each hop sees the question, the notes so far and one part of the
document -- never more than the context holds -- and writes the notes
the next hop starts from. The last hop answers from the notes alone.
Hops rotate through the instances, so N copies of the model share the
work; in the default ``chain`` mode they run strictly in series, because
each hop needs the notes the previous one wrote.

``map`` mode trades that dependency for speed: every part is digested on
its own, in parallel across the instances (each instance still reads its
own parts one at a time), and the digests are then folded together in
series and answered. It reads faster and costs about the same; what it
gives up is a reader that has the earlier parts in mind when it reads
the later ones.

When the whole prompt fits in one context, a relay is one ordinary call.

The relay is model-agnostic: an *instance* is any callable
``(messages, max_tokens) -> Reply``. :func:`openai_instance` wraps an
OpenAI-compatible client (llama-server, LM Studio, the T1 API's own
backends) and :func:`load_instances` starts N copies of a GGUF on
consecutive ports with a :class:`~hypernix.hyperlink.managed.ManagedPool`
and returns one instance per copy::

    from hypernix.runtime.relay import Relay, load_instances

    pool, instances = load_instances("small-4k.gguf", instances=4, context_tokens=4096)
    try:
        result = Relay(instances, context_tokens=4096).run(document, "What changed in v2?")
        print(result.answer, result.input_tokens, result.output_tokens)
    finally:
        pool.unload()

Reading in hops costs more than one big read would: the question, the
instructions and the notes are re-read on every hop. :func:`plan` says
how much more before anything runs (``overhead_factor``), and the T1
API's ``POST /inference/relay`` prices both.
"""
from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CHAIN",
    "MAP",
    "MODES",
    "ContextTooSmall",
    "Hop",
    "Instance",
    "Relay",
    "RelayPlan",
    "RelayResult",
    "Reply",
    "estimate_tokens",
    "load_instances",
    "openai_instance",
    "plan",
    "split",
]

CHAIN, MAP = "chain", "map"
MODES = (CHAIN, MAP)

#: Characters per token for sizing. The same rough figure the T1 API uses
#: to size requests -- and, as there, only to size them: what a hop is
#: billed is what the backend says it spent.
CHARS_PER_TOKEN = 4

#: What the instructions and chat framing of one hop cost, in tokens.
#: Measured against the prompts below and rounded up.
OVERHEAD_TOKENS = 160

#: Share of the context kept free: the token estimate is a guess, and a
#: guess that is low by a few percent must not overflow the context.
MARGIN = 0.10

READ_PROMPT = (
    "You are reading a long document one part at a time, because it does not fit "
    "in your memory at once. You will be given a question, your notes from the parts "
    "read so far, and the next part. Rewrite the notes so they keep everything from "
    "the old notes and this part that could help answer the question: facts, names, "
    "numbers, quotes worth keeping word for word. Drop what cannot matter. Write only "
    "the notes, as short as they can be while losing nothing useful."
)
DIGEST_PROMPT = (
    "You are reading one part of a long document. Note everything in it that could "
    "help answer the question: facts, names, numbers, quotes worth keeping word for "
    "word. If nothing in it is relevant, say so in one line. Write only the notes."
)
ANSWER_PROMPT = (
    "Answer the question from the notes below, which were taken while reading a "
    "document too long to show you whole. If the notes do not hold the answer, say "
    "what they do say and what is missing."
)


def estimate_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN if text else 0


class ContextTooSmall(ValueError):
    """The context cannot hold the instructions, the notes and any document."""


# -- instances ---------------------------------------------------------------


@dataclass(frozen=True)
class Reply:
    """One instance's answer, and what it cost."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


#: One copy of the model: ``(messages, max_tokens) -> Reply``.
Instance = Callable[[list[dict[str, Any]], int], Reply]


def openai_instance(client: Any, model: str = "", *, temperature: float | None = 0.2) -> Instance:
    """An instance over an OpenAI-compatible client.

    *client* needs ``chat(messages, model=, max_tokens=, temperature=)``
    returning the OpenAI envelope -- :class:`~hypernix.bridge.lmstudio.LMStudioBridge`
    does, for llama-server and LM Studio alike. Usage is read from the
    envelope; a backend that reports none is estimated rather than
    recorded as free.
    """
    def complete(messages: list[dict[str, Any]], max_tokens: int) -> Reply:
        envelope = client.chat(messages, model=model or None, max_tokens=max_tokens,
                               temperature=temperature)
        choices = envelope.get("choices") or [{}]
        first = choices[0] or {}
        text = str((first.get("message") or {}).get("content") or first.get("text") or "")
        usage = envelope.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        if not input_tokens and not output_tokens:
            input_tokens = estimate_tokens("".join(str(m.get("content", "")) for m in messages))
            output_tokens = estimate_tokens(text)
        return Reply(text, input_tokens, output_tokens)

    return complete


def load_instances(
    model_path: str,
    *,
    instances: int = 2,
    context_tokens: int = 4096,
    port: int = 8790,
    avoid_ports: Sequence[int] = (),
    model_id: str = "",
    **load_kwargs: Any,
) -> tuple[Any, list[Instance]]:
    """Start *instances* copies of a GGUF, each with *context_tokens* of
    context, and return ``(pool, instances)``.

    All or nothing: if one copy fails to start, the others are stopped
    and the error says which. Stop them with ``pool.unload()``.
    """
    from ..bridge.lmstudio import LMStudioBridge
    from ..hyperlink.managed import ManagedPool

    pool = ManagedPool(instances=instances, port=port, avoid_ports=avoid_ports)
    pool.load(model_path, context_length=context_tokens, model_id=model_id, **load_kwargs)
    return pool, [openai_instance(LMStudioBridge(base_url=url), model_id) for url in pool.base_urls]


# -- planning ----------------------------------------------------------------


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def split(text: str, chunk_tokens: int) -> list[str]:
    """*text* in parts of at most *chunk_tokens* (estimated), cut at
    paragraph breaks where it can, then sentences, then anywhere."""
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be at least 1")
    limit = chunk_tokens * CHARS_PER_TOKEN
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        if not paragraph.strip():
            continue
        if len(paragraph) <= limit:
            pieces.append(paragraph)
            continue
        for sentence in _SENTENCE.split(paragraph):
            while len(sentence) > limit:
                pieces.append(sentence[:limit])
                sentence = sentence[limit:]
            if sentence.strip():
                pieces.append(sentence)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        joined = f"{current}\n\n{piece}" if current else piece
        if len(joined) <= limit:
            current = joined
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


@dataclass(frozen=True)
class RelayPlan:
    """What a relay will do, and its worst-case token cost, before it runs."""

    mode: str
    instances: int
    context_tokens: int
    notes_tokens: int
    answer_tokens: int
    chunk_tokens: int
    chunks: tuple[str, ...]
    #: Calls the relay will make: one per part, plus the answer (plus,
    #: in map mode, the folding of the digests, which is estimated).
    requests: int
    #: Upper bounds: every hop is counted at full notes and full parts.
    input_tokens: int
    output_tokens: int
    #: What one model with a big enough context would read and write.
    direct_input_tokens: int
    direct_output_tokens: int
    #: The largest single call, which is what has to fit the context.
    largest_call_tokens: int

    @property
    def single(self) -> bool:
        return self.requests == 1

    @property
    def overhead_factor(self) -> float:
        """Relay tokens over one-big-read tokens: the price of the context."""
        direct = self.direct_input_tokens + self.direct_output_tokens
        return round((self.input_tokens + self.output_tokens) / direct, 3) if direct else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "instances": self.instances,
            "context_tokens": self.context_tokens,
            "notes_tokens": self.notes_tokens,
            "answer_tokens": self.answer_tokens,
            "chunk_tokens": self.chunk_tokens,
            "chunks": len(self.chunks),
            "requests": self.requests,
            "single": self.single,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "direct_input_tokens": self.direct_input_tokens,
            "direct_output_tokens": self.direct_output_tokens,
            "largest_call_tokens": self.largest_call_tokens,
            "overhead_factor": self.overhead_factor,
        }


def _usable(context_tokens: int) -> int:
    return int(context_tokens * (1 - MARGIN))


def plan(
    document: str,
    question: str,
    *,
    context_tokens: int,
    instances: int = 1,
    mode: str = CHAIN,
    notes_tokens: int | None = None,
    answer_tokens: int = 512,
) -> RelayPlan:
    """Cut *document* into parts that fit *context_tokens* alongside the
    question, the instructions and the notes, and count the cost.

    *notes_tokens* is how long the carried notes may grow (default: a
    sixth of the context). Raises :class:`ContextTooSmall` when nothing
    of the document would fit.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
    if instances < 1:
        raise ValueError("a relay needs at least one instance")
    if context_tokens < 1:
        raise ValueError("context_tokens must be at least 1")
    notes = notes_tokens if notes_tokens is not None else max(64, context_tokens // 6)
    usable = _usable(context_tokens)
    q = estimate_tokens(question)
    doc = estimate_tokens(document)
    direct_in = OVERHEAD_TOKENS + q + doc

    if direct_in + answer_tokens <= usable:
        chunks = (document,) if document else ()
        return RelayPlan(mode, instances, context_tokens, notes, answer_tokens, doc, chunks, 1,
                         direct_in, answer_tokens, direct_in, answer_tokens, direct_in + answer_tokens)

    # A reading hop holds the instructions, the question, the notes it is
    # given and the notes it writes, and the part. The answer hop must fit too.
    carried = notes if mode == CHAIN else 0
    chunk_tokens = usable - OVERHEAD_TOKENS - q - carried - notes
    answer_call = OVERHEAD_TOKENS + q + notes + answer_tokens
    if chunk_tokens < 32 or answer_call > usable:
        raise ContextTooSmall(
            f"a {context_tokens}-token context cannot hold the instructions "
            f"(~{OVERHEAD_TOKENS}), the question (~{q}), {notes} tokens of notes in and out "
            f"and a {answer_tokens}-token answer with room for any of the document. "
            f"Use a model with a larger context, a shorter question, or smaller "
            f"notes_tokens/answer_tokens."
        )
    chunks = tuple(split(document, chunk_tokens))
    n = len(chunks)
    reads_in = sum(OVERHEAD_TOKENS + q + estimate_tokens(c) for c in chunks)
    if mode == CHAIN:
        # The first hop has no notes to read; every later one reads a full set.
        reads_in += carried * max(0, n - 1)
        requests = n + 1
    else:
        # Digests short enough together are answered from directly;
        # otherwise they are folded together in series, a fold hop
        # carrying notes like a chain hop does.
        fold_tokens = chunk_tokens - notes
        if n * notes <= usable - OVERHEAD_TOKENS - q - answer_tokens:
            folds = 0
        elif fold_tokens < 32:
            raise ContextTooSmall(
                f"a {context_tokens}-token context leaves no room to fold {n} digests of "
                f"{notes} tokens together. Use chain mode, or smaller notes_tokens."
            )
        else:
            folds = -(-n * notes // fold_tokens)
        reads_in += folds * (OVERHEAD_TOKENS + q + notes + fold_tokens)
        requests = n + folds + 1
    reads_out = notes * (requests - 1)
    largest = max(OVERHEAD_TOKENS + q + carried + estimate_tokens(c) + notes for c in chunks)
    return RelayPlan(
        mode, instances, context_tokens, notes, answer_tokens, chunk_tokens, chunks, requests,
        reads_in + OVERHEAD_TOKENS + q + notes, reads_out + answer_tokens,
        direct_in, answer_tokens, max(largest, answer_call),
    )


# -- running -----------------------------------------------------------------


@dataclass(frozen=True)
class Hop:
    """One call: which instance made it, what it was for, what it cost."""

    index: int
    instance: int
    kind: str           # "direct", "read", "digest", "fold" or "answer"
    part: int | None    # which part of the document (0-based), when it read one
    input_tokens: int
    output_tokens: int
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "instance": self.instance, "kind": self.kind, "part": self.part,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "seconds": round(self.seconds, 3),
        }


@dataclass
class RelayResult:
    answer: str
    plan: RelayPlan
    hops: list[Hop] = field(default_factory=list)
    notes: str = ""

    @property
    def input_tokens(self) -> int:
        return sum(h.input_tokens for h in self.hops)

    @property
    def output_tokens(self) -> int:
        return sum(h.output_tokens for h in self.hops)

    @property
    def requests(self) -> int:
        return len(self.hops)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "plan": self.plan.to_dict(),
            "hops": [h.to_dict() for h in self.hops],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "requests": self.requests,
        }


class Relay:
    """Several copies of one small-context model, reading as one.

    *on_hop* is called after every call with its :class:`Hop` -- the T1
    API meters each one there, so a relay that fails half way has still
    billed the half that ran.
    """

    def __init__(
        self,
        instances: Sequence[Instance],
        *,
        context_tokens: int,
        mode: str = CHAIN,
        notes_tokens: int | None = None,
        answer_tokens: int = 512,
        on_hop: Callable[[Hop], None] | None = None,
    ) -> None:
        if not instances:
            raise ValueError("a relay needs at least one instance")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
        self.instances = list(instances)
        self.context_tokens = context_tokens
        self.mode = mode
        self.notes_tokens = notes_tokens
        self.answer_tokens = answer_tokens
        self.on_hop = on_hop
        self._lock = threading.Lock()

    def plan(self, document: str, question: str) -> RelayPlan:
        return plan(document, question, context_tokens=self.context_tokens,
                    instances=len(self.instances), mode=self.mode,
                    notes_tokens=self.notes_tokens, answer_tokens=self.answer_tokens)

    def run(self, document: str, question: str) -> RelayResult:
        planned = self.plan(document, question)
        result = RelayResult("", planned)
        if planned.single:
            user = f"{document}\n\n{question}" if document else question
            result.answer = self._call(result, 0, "direct", None, [{"role": "user", "content": user}],
                                       planned.answer_tokens)
            return result
        if self.mode == CHAIN:
            notes = self._chain(result, planned.chunks, question, planned.notes_tokens)
        else:
            notes = self._map(result, planned, question)
        result.notes = notes
        result.answer = self._answer(result, notes, question)
        return result

    # -- the two modes ---------------------------------------------------

    def _chain(self, result: RelayResult, chunks: Sequence[str], question: str,
               notes_tokens: int) -> str:
        notes = ""
        total = len(chunks)
        for part, chunk in enumerate(chunks):
            user = (f"Question: {question}\n\nNotes so far:\n{notes or '(none yet)'}\n\n"
                    f"Part {part + 1} of {total}:\n{chunk}")
            notes = self._clip(self._call(
                result, len(result.hops) % len(self.instances), "read", part,
                [{"role": "system", "content": READ_PROMPT}, {"role": "user", "content": user}],
                notes_tokens,
            ), notes_tokens)
        return notes

    def _map(self, result: RelayResult, planned: RelayPlan, question: str) -> str:
        chunks, n = planned.chunks, len(self.instances)
        digests: list[str] = [""] * len(chunks)

        def worker(slot: int) -> None:
            # Each instance reads its own parts one after another: one copy
            # of a model answers one request at a time.
            for part in range(slot, len(chunks), n):
                user = f"Question: {question}\n\nPart {part + 1} of {len(chunks)}:\n{chunks[part]}"
                digests[part] = self._clip(self._call(
                    result, slot, "digest", part,
                    [{"role": "system", "content": DIGEST_PROMPT}, {"role": "user", "content": user}],
                    planned.notes_tokens,
                ), planned.notes_tokens)

        # One thread per instance, so every copy is reading at once.
        failures: list[BaseException] = []

        def guarded(slot: int) -> None:
            try:
                worker(slot)
            except BaseException as exc:  # noqa: BLE001 -- re-raised below, in the caller
                failures.append(exc)

        threads = [threading.Thread(target=guarded, args=(slot,), name=f"relay_{slot}", daemon=True)
                   for slot in range(min(n, len(chunks)))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            raise failures[0]

        joined = "\n\n".join(f"[Part {i + 1}] {d}" for i, d in enumerate(digests))
        room = (_usable(self.context_tokens) - OVERHEAD_TOKENS - estimate_tokens(question)
                - planned.answer_tokens)
        if estimate_tokens(joined) <= room:
            return joined
        # Too long to answer from at once: fold the digests together in
        # series. A fold hop carries the notes so far, as a chain hop does.
        folds = split(joined, planned.chunk_tokens - planned.notes_tokens)
        notes = ""
        for chunk in folds:
            user = (f"Question: {question}\n\nNotes so far:\n{notes or '(none yet)'}\n\n"
                    f"More notes to merge in:\n{chunk}")
            notes = self._clip(self._call(
                result, len(result.hops) % n, "fold", None,
                [{"role": "system", "content": READ_PROMPT}, {"role": "user", "content": user}],
                planned.notes_tokens,
            ), planned.notes_tokens)
        return notes

    def _answer(self, result: RelayResult, notes: str, question: str) -> str:
        user = f"Notes:\n{notes or '(nothing relevant was found)'}\n\nQuestion: {question}"
        return self._call(
            result, len(result.hops) % len(self.instances), "answer", None,
            [{"role": "system", "content": ANSWER_PROMPT}, {"role": "user", "content": user}],
            result.plan.answer_tokens,
        )

    # -- one call ----------------------------------------------------------

    def _call(self, result: RelayResult, slot: int, kind: str, part: int | None,
              messages: list[dict[str, Any]], max_tokens: int) -> str:
        started = time.monotonic()
        reply = self.instances[slot](messages, max_tokens)
        # Map-mode digests finish in whatever order they finish; the index
        # is the order they finished in.
        with self._lock:
            hop = Hop(len(result.hops), slot, kind, part, int(reply.input_tokens),
                      int(reply.output_tokens), time.monotonic() - started)
            result.hops.append(hop)
            if self.on_hop is not None:
                self.on_hop(hop)
        return reply.text.strip()

    @staticmethod
    def _clip(text: str, tokens: int) -> str:
        """Notes no longer than planned, whatever the model wrote: an
        over-long reply must not push the next hop past the context."""
        limit = tokens * CHARS_PER_TOKEN
        return text if len(text) <= limit else text[:limit]
