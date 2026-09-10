"""mediocre_fridge — dataset generation for judge / evaluator training.

The "mediocre" name is a nod to what it produces: intentionally mixed
quality. Given a pool of prompts and a way to generate responses, this
module builds a small judge-training corpus of ``(prompt, response,
label)`` triples where ``label`` is ``GOOD`` or ``BAD`` — the exact
signal a pairwise or pointwise reward model needs.

Two generation modes:

* :func:`synthesize_judge_corpus` — zero-dependency: mangles known-good
  reference responses into deliberately-bad variants (truncate, shuffle
  lines, drop words) so the judge has a contrast set without needing a
  second model at hand.
* :func:`collect_responses_from` — if you *do* have an oven handy,
  sample real responses from it and tag each with a heuristic label.

Output format is plain text, one example per line::

    <JUDGE_PROMPT>QUESTION<JUDGE_RESPONSE>ANSWER<JUDGE_LABEL>GOOD

The delimiter tokens are plain ASCII strings (not tokenizer
special-tokens) so any byte/BPE tokenizer can consume them without
modification.
"""
from __future__ import annotations

from hypernix.system.deprecation import deprecated_module

# Announced before the heavy imports below, so the notice reaches the
# operator immediately rather than after torch has finished loading.
deprecated_module(
    "hypernix.data.mediocre_fridge",
    instead="hypernix.models.neo_oven",
    since="0.71.5a2",
)

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

JUDGE_PROMPT = "<JUDGE_PROMPT>"
JUDGE_RESPONSE = "<JUDGE_RESPONSE>"
JUDGE_LABEL = "<JUDGE_LABEL>"
LABEL_GOOD = "GOOD"
LABEL_BAD = "BAD"


@dataclass(frozen=True)
class JudgeExample:
    prompt: str
    response: str
    label: str  # "GOOD" or "BAD"

    def format(self) -> str:
        return (
            f"{JUDGE_PROMPT}{self.prompt}"
            f"{JUDGE_RESPONSE}{self.response}"
            f"{JUDGE_LABEL}{self.label}"
        )


# A tiny seed corpus so the fridge is useful out of the box. Each entry
# is a ``(prompt, good_response)`` pair; the :func:`synthesize_judge_corpus`
# routine derives bad counterparts by mangling the good one.
_SEED: list[tuple[str, str]] = [
    ("What is 2 + 2?", "4"),
    ("Capital of France?", "Paris"),
    ("Name a primary color.", "Red"),
    ("What gas do plants absorb?", "Carbon dioxide"),
    ("Largest planet in our solar system?", "Jupiter"),
    ("Who wrote Hamlet?", "William Shakespeare"),
    ("Speed of light in m/s (approx)?", "299792458"),
    ("What is H2O commonly called?", "Water"),
    ("5 * 6 = ?", "30"),
    ("Smallest prime number?", "2"),
    ("Sum of angles in a triangle (deg)?", "180"),
    ("What is the boiling point of water at sea level in Celsius?", "100"),
    ("Language used by Guido van Rossum's first release?", "Python"),
    ("How many continents are there?", "7"),
    ("What year did WW2 end?", "1945"),
    ("Protein-building cell structure?", "Ribosome"),
]


def _mangle(response: str, rng: random.Random) -> str:
    """Produce a plausibly-bad variant of ``response``."""
    choice = rng.choice(["truncate", "shuffle_chars", "wrong", "empty", "repeat"])
    if choice == "truncate" and len(response) > 1:
        return response[: max(1, len(response) // 2)]
    if choice == "shuffle_chars" and len(response) > 2:
        chars = list(response)
        rng.shuffle(chars)
        return "".join(chars)
    if choice == "wrong":
        return rng.choice(["I don't know.", "42", "blue", "maybe later"])
    if choice == "empty":
        return ""
    # "repeat"
    return (response + " ") * 3


def synthesize_judge_corpus(
    n: int,
    out_path: Path | str,
    *,
    seed: int | None = 0,
    good_ratio: float = 0.5,
) -> Path:
    """Write ``n`` judge examples to ``out_path`` as newline-separated text.

    Examples are drawn by cycling through :data:`_SEED`; roughly
    ``good_ratio`` of them keep the reference answer and get the
    ``GOOD`` label, while the rest are mangled and get ``BAD``.
    """
    rng = random.Random(seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for i in range(n):
        prompt, good = _SEED[i % len(_SEED)]
        if rng.random() < good_ratio:
            ex = JudgeExample(prompt, good, LABEL_GOOD)
        else:
            ex = JudgeExample(prompt, _mangle(good, rng), LABEL_BAD)
        lines.append(ex.format())

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def collect_responses_from(
    oven,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    label_rule=None,
) -> list[JudgeExample]:
    """Sample responses from ``oven.complete(...)`` and wrap them as JudgeExamples.

    ``label_rule`` is an optional ``(prompt, response) -> "GOOD" | "BAD"``
    callback. When omitted, everything is labelled ``GOOD`` — useful
    for collecting a teacher corpus that you'll pair with mangled negatives
    later via :func:`synthesize_judge_corpus`-style mangling.
    """
    examples: list[JudgeExample] = []
    for prompt in prompts:
        response = oven.complete(prompt, max_new_tokens=max_new_tokens,
                                  temperature=temperature, stop=())
        label = label_rule(prompt, response) if label_rule else LABEL_GOOD
        examples.append(JudgeExample(prompt, response, label))
    return examples


def write_examples(examples: Sequence[JudgeExample], out_path: Path | str) -> Path:
    """Serialize a sequence of JudgeExamples to a plain-text file."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(e.format() for e in examples) + "\n", encoding="utf-8")
    return out
