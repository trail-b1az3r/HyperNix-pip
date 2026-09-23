"""Best-of-n sampling end to end, with no model needed.

    python examples/dilute/best_of_n.py

The generator here is a stand-in so the script runs anywhere and the
output is the same every time. Swap it for a real one and nothing else
changes:

    from hypernix.dilute.backends import resolve_generator
    generate = resolve_generator("", server="http://127.0.0.1:8000")

What the script is showing is the part people get wrong. The same
prompts are run twice: once with an evaluator that scores every sample
the same, and once with one that can tell them apart. Both finish, both
write a file, both report the trace count asked for. Only one of them
produced traces worth training on, and the difference is visible in the
margin and nowhere else.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

from hypernix.dilute import DiluteConfig, dilute, jit_distil
from hypernix.dilute.evaluators import FunctionEvaluator

PROMPTS = [
    "What is 2 + 2?",
    "Name a prime number under 10.",
    "What colour is the sky at noon?",
    "How many sides does a hexagon have?",
    "What is the capital of France?",
    "Spell 'necessary'.",
]

ANSWERS = {
    "What is 2 + 2?": "4",
    "Name a prime number under 10.": "7",
    "What colour is the sky at noon?": "blue",
    "How many sides does a hexagon have?": "6",
    "What is the capital of France?": "Paris",
    "Spell 'necessary'.": "necessary",
}


def fake_model(prompt: str, *, temperature: float = 0.8, **_) -> str:
    """A model that is right more often when it is cold.

    Which is the reason the ladder exists: the cold rungs carry the
    correct answer most of the time, and the warm ones are there for
    the occasions when the cold one is confidently wrong.
    """
    # Seeded from the text, not from `hash()`: str hashing is salted
    # per process, so a `hash()` seed makes this script print something
    # different every run and the numbers below unciteable.
    rng = random.Random(f"{prompt}|{temperature:.3f}")
    if rng.random() > 0.15 + temperature * 0.55:
        return ANSWERS[prompt]
    return f"I think the answer might be something else entirely ({temperature:.1f})"


def is_correct(prompt: str, completion: str) -> float:
    """A checkable answer, so the judge is a function and not a model."""
    return 1.0 if ANSWERS[prompt].lower() in completion.lower() else 0.0


def main() -> None:
    config = DiluteConfig(samples_per_prompt=6, target_traces=6, seed=1)

    print("=== an evaluator that cannot tell the samples apart ===")
    flat = dilute(PROMPTS, fake_model, lambda p, t: 1.0, config)
    print(flat.summary())
    flat_correct = sum(
        1 for t in flat.traces if is_correct(t.prompt, t.chosen.text)
    )
    print(f"{flat_correct}/{len(flat.traces)} chosen answers are right")
    print(
        "Six traces, same count as below, and a mean margin of zero across "
        "the whole run — which means the evaluator never separated anything "
        "and every winner was a coin toss. This file is no better than one "
        "sample per prompt, and the trace count does not say so.\n"
    )

    print("=== an evaluator that can ===")
    sharp = dilute(PROMPTS, fake_model, FunctionEvaluator(is_correct), config)
    print(sharp.summary())
    correct = sum(1 for t in sharp.traces if is_correct(t.prompt, t.chosen.text))
    print(f"{correct}/{len(sharp.traces)} chosen answers are right")
    print(
        "Chosen temperatures: "
        + ", ".join(f"{t.chosen.temperature:g}" for t in sharp.traces)
    )
    print(
        "Note they are not all the coldest rung. If they were, the ladder "
        "would be doing nothing and this would be greedy decoding with "
        "extra steps.\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "traces.jsonl"
        sharp.write_jsonl(out)
        print(f"=== written to {out.name} ===")
        print(out.read_text().splitlines()[0])
        print("\nThe first line is the run itself — including why it stopped, "
              "which the line count cannot tell you.\n")

    print("=== the streaming form ===")
    for trace in jit_distil(
        PROMPTS, fake_model, FunctionEvaluator(is_correct),
        DiluteConfig(samples_per_prompt=6, target_traces=3, seed=1),
    ):
        print(f"  kept: {trace.prompt[:34]:36} "
              f"score {trace.chosen.score:.2f}  margin {trace.margin:+.2f}")
    print(
        "\nEach one arrives as it is made, so training can start on the "
        "first traces while the rest are still being generated — and a run "
        "killed halfway leaves half a file rather than none."
    )
    print(
        "\nThose margins are zero for the opposite reason to the first run: "
        "the score is 1.00 and more than one sample reached it, so there "
        "was nothing to separate at the top. Margin is the gap to the best "
        "*rejected* sample, which is the question worth asking — a zero "
        "margin on one trace means the prompt was easy, a zero mean margin "
        "across a run means the evaluator is blind. `hypernix dilute "
        "inspect` tells those apart by counting how many."
    )


if __name__ == "__main__":
    main()
