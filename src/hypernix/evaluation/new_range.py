"""new_range — entry-level rubric labeler.

A ``Range`` is a callable ``(prompt, response) -> "GOOD" | "BAD"`` you
can drop straight into :func:`hypernix.mediocre_fridge.collect_responses_from`
as ``label_rule=...``.  ``new_range`` is the lightest of the three: a
small bag of zero-dependency heuristics (length, refusal patterns,
math-prompt digit check, emptiness) wired together with first-fail
semantics — any failing rule yields ``BAD``, otherwise ``GOOD``.

Pick this when you want a rubric that runs in microseconds and you
don't have a teacher model handy.  Iterate to :class:`hypernix.old_range.OldRange`
when you need scored output and explainability, and to
:class:`hypernix.industrial_range.IndustrialRange` when you have a
bigger model that can act as the judge.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

LABEL_GOOD = "GOOD"
LABEL_BAD = "BAD"

#: Substrings that mark a refusal / "I can't help with that" response.
REFUSAL_PATTERNS: tuple[str, ...] = (
    "i cannot", "i can't", "i'm sorry", "i am sorry",
    "as an ai", "as a language model",
    "i'm not able", "i am not able",
    "i won't", "i will not",
    # 0.72.8: how current models decline.
    "i'm unable to", "i am unable to", "i can't help with", "i can't assist",
    "i'm not going to", "i must decline", "i'd rather not", "i apologize, but",
    "unfortunately, i can't", "i don't feel comfortable",
)

#: Curly apostrophes and quotes, as most models write them, read as
#: straight ones: "I can’t" is a refusal too.
_STRAIGHTEN = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'})

#: Trigger words that suggest a numeric answer is expected.
MATH_TRIGGER_RE = re.compile(
    r"\b("
    r"sum|add|plus|subtract|minus|multiply|times|divide|"
    r"square|cube|root|"
    r"how\s+many|how\s+much|"
    r"what\s+is\s+\d|"
    r"\d+\s*[+\-*/x×÷]\s*\d+"
    r")\b",
    re.IGNORECASE,
)

#: Any digit anywhere in the response.
DIGIT_RE = re.compile(r"\d")


# ---------------------------------------------------------------------------
# Individual rules — each is a callable (prompt, response) -> bool.
# True means "this response failed the rule and should be BAD".
# ---------------------------------------------------------------------------

def is_empty(prompt: str, response: str) -> bool:
    return not response.strip()


def is_too_short(prompt: str, response: str, *, min_chars: int = 1) -> bool:
    return len(response.strip()) < min_chars


def is_too_long(prompt: str, response: str, *, max_chars: int = 4096) -> bool:
    return len(response) > max_chars


def is_refusal(prompt: str, response: str) -> bool:
    low = response.translate(_STRAIGHTEN).lower()
    return any(p in low for p in REFUSAL_PATTERNS)


def math_lacks_digit(prompt: str, response: str) -> bool:
    """True if the prompt looks numeric and the response has no digits."""
    if not MATH_TRIGGER_RE.search(prompt):
        return False
    return DIGIT_RE.search(response) is None


def is_repetition(prompt: str, response: str, *, min_run: int = 8) -> bool:
    """True if the same character or short token repeats `min_run` times."""
    if len(response) < min_run:
        return False
    # Same-char run:
    for i in range(len(response) - min_run + 1):
        if len(set(response[i : i + min_run])) == 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Range
# ---------------------------------------------------------------------------

Rule = Callable[[str, str], bool]


@dataclass
class NewRange:
    """First-fail rubric labeler.

    Each rule in ``rules`` is consulted in order.  If any returns True
    the response is labelled ``BAD``; otherwise ``GOOD``.

    A "rule fired" log is exposed as :attr:`last_failed_rule` for cheap
    inspection during dataset triage.
    """

    rules: list[Rule] = field(default_factory=list)
    last_failed_rule: str | None = None

    def __post_init__(self) -> None:
        if not self.rules:
            self.rules = list(default_rules())

    def label(self, prompt: str, response: str) -> str:
        for rule in self.rules:
            if rule(prompt, response):
                self.last_failed_rule = getattr(rule, "__name__", repr(rule))
                return LABEL_BAD
        self.last_failed_rule = None
        return LABEL_GOOD

    # Make instances usable as ``label_rule=range`` directly.
    def __call__(self, prompt: str, response: str) -> str:
        return self.label(prompt, response)


def default_rules() -> Iterable[Rule]:
    """The out-of-the-box rule set used when no explicit ``rules`` is given."""
    return (is_empty, is_refusal, math_lacks_digit, is_repetition)


def new_range(rules: Iterable[Rule] | None = None) -> NewRange:
    """Construct a :class:`NewRange` with the given rules (or the defaults)."""
    return NewRange(list(rules) if rules is not None else list(default_rules()))
