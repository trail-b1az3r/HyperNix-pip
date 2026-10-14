"""Hypercompress: fitting a system prompt into a model that cannot hold it.

A HyperLink system prompt is built from several places — the server's
default, the person's bio and own prompt, the chat's prompt and the
memory block — and nothing stops it outgrowing a small model. When the
prompt a turn would send is larger than the loaded model's context, the
model either refuses the request or the backend silently cuts the front
off, which is the part that says who the model is.

So before such a turn the system prompt is compressed to a share of the
context, chosen per person with ``/chat/hypercompress/size?s?=1|2``:

====  ==========================
size  the system prompt gets
====  ==========================
1     a third of the context (the default)
2     a fifth of the context
====  ==========================

The compression is extractive and needs no model — the model is the
thing that cannot read the whole prompt. Exact repeats go first, then
sentences are ranked, instructions ("never", "always", "you are", "do
not") and the start of each section highest, and the best are kept in
their original order until the budget is spent. Every section keeps at
least its first sentence, so a memory block or a chat's own prompt is
shortened rather than lost.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "DEFAULT_SIZE",
    "SIZES",
    "Compressed",
    "compress",
    "estimate_tokens",
    "maybe_compress",
    "parse_size",
    "target_tokens",
]

#: ``s`` -> the share of the context the system prompt may take.
SIZES = {1: 3, 2: 5}
DEFAULT_SIZE = 1
#: Where a person's choice is kept, in their preferences' metadata.
METADATA_KEY = "hypercompress_size"
#: Below this a prompt is not worth compressing, whatever the context.
FLOOR_TOKENS = 32

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[*`-])")
_INSTRUCTION = re.compile(
    r"\b(you are|you're|never|always|must|do not|don't|should|only|avoid|"
    r"important|required|refuse|use the|when the|if the|answer|reply)\b",
    re.IGNORECASE,
)


def estimate_tokens(text: str) -> int:
    """~4 characters a token, as everywhere else in HyperLink."""
    return (len(text) + 3) // 4 if text else 0


def parse_size(raw: object) -> int:
    """``1`` or ``2``; anything else is a ValueError naming both."""
    try:
        value = int(str(raw).strip().strip("[]"))
    except (TypeError, ValueError):
        raise ValueError("s is 1 (a third of the context) or 2 (a fifth)") from None
    if value not in SIZES:
        raise ValueError("s is 1 (a third of the context) or 2 (a fifth)")
    return value


def target_tokens(context_length: int, size: int = DEFAULT_SIZE) -> int:
    return max(FLOOR_TOKENS, int(context_length) // SIZES.get(size, SIZES[DEFAULT_SIZE]))


@dataclass(frozen=True)
class Compressed:
    text: str
    before_tokens: int
    after_tokens: int
    target_tokens: int
    size: int
    applied: bool

    def to_dict(self) -> dict:
        return {
            "applied": self.applied, "size": self.size, "share": f"1/{SIZES[self.size]}",
            "before_tokens": self.before_tokens, "after_tokens": self.after_tokens,
            "target_tokens": self.target_tokens,
        }


def _sections(text: str) -> list[list[str]]:
    """Paragraphs, each a list of sentences (lines of a list stay whole)."""
    sections: list[list[str]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        sentences: list[str] = []
        for line in block.splitlines():
            line = " ".join(line.split())
            if not line:
                continue
            if re.match(r"^([-*•]|\d+[.)]|#+)\s", line):
                sentences.append(line)
            else:
                sentences.extend(s for s in _SENTENCE.split(line) if s)
        if sentences:
            sections.append(sentences)
    return sections


def _score(sentence: str, index: int) -> float:
    score = 0.0
    if index == 0:
        score += 3.0  # what the section is about
    score += 2.0 * len(_INSTRUCTION.findall(sentence))
    if sentence.startswith("#"):
        score += 2.5
    if re.match(r"^([-*•]|\d+[.)])\s", sentence):
        score += 0.5
    score -= index * 0.15  # earlier is usually more important
    score -= max(0, len(sentence) - 240) / 240  # very long sentences cost a lot
    return score


def compress(text: str, budget_tokens: int) -> str:
    """*text* cut to about *budget_tokens*, keeping what matters most."""
    if estimate_tokens(text) <= budget_tokens:
        return text
    seen: set[str] = set()
    sections: list[list[str]] = []
    for sentences in _sections(text):
        kept = []
        for sentence in sentences:
            key = sentence.lower().strip(" .")
            if key in seen:
                continue
            seen.add(key)
            kept.append(sentence)
        if kept:
            sections.append(kept)

    def render(chosen: set[tuple[int, int]]) -> str:
        out = []
        for s, sentences in enumerate(sections):
            picked = [sentences[i] for i in range(len(sentences)) if (s, i) in chosen]
            if picked:
                joiner = "\n" if any(re.match(r"^([-*•]|\d+[.)]|#+)\s", p) for p in picked) else " "
                out.append(joiner.join(picked))
        return "\n\n".join(out)

    budget_chars = budget_tokens * 4

    def later(s: int) -> float:
        return s / max(1, len(sections) - 1)

    # Every section's first sentence first, so nothing is lost outright.
    chosen = {(s, 0) for s in range(len(sections))}
    while len(render(chosen)) > budget_chars and len(chosen) > 1:
        # Even the openings do not fit: drop the lowest-ranked openings
        # from the middle, keeping the first and the last section.
        middle = [(s, 0) for s in range(1, len(sections) - 1) if (s, 0) in chosen]
        if not middle:
            break
        chosen.discard(min(middle, key=lambda k: _score(sections[k[0]][0], 0) + later(k[0])))
    # Later sections are the person's own (their prompt, the chat's,
    # their memories) and win where they differ from the default, so
    # they are worth a little more.
    ranked = sorted(
        ((s, i) for s, sentences in enumerate(sections) for i in range(1, len(sentences))),
        key=lambda k: _score(sections[k[0]][k[1]], k[1]) + later(k[0]), reverse=True,
    )
    for key in ranked:
        trial = chosen | {key}
        if len(render(trial)) <= budget_chars:
            chosen = trial
    result = render(chosen)
    if len(result) > budget_chars:  # one enormous sentence: cut it at a word
        result = result[:budget_chars].rsplit(" ", 1)[0].rstrip() + " …"
    return result


def maybe_compress(system_prompt: str, *, context_length: int, other_tokens: int = 0,
                   size: int = DEFAULT_SIZE) -> Compressed:
    """Compress *system_prompt* when the whole prompt will not fit.

    *other_tokens* is the rest of what the turn sends (the conversation).
    Nothing changes when the context is unknown (0) or the prompt fits.
    """
    size = size if size in SIZES else DEFAULT_SIZE
    before = estimate_tokens(system_prompt)
    target = target_tokens(context_length, size) if context_length else 0
    if (not context_length or before <= FLOOR_TOKENS
            or before + other_tokens <= context_length or before <= target):
        return Compressed(system_prompt, before, before, target, size, False)
    text = compress(system_prompt, target)
    return Compressed(text, before, estimate_tokens(text), target, size, True)
