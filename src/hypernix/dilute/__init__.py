"""hypernix.dilute — many answers from one model, then keep the good ones.

The problem this is for: you have one model and you want a training set
of good answers. The obvious approach — ask once per prompt — gives you
the model's average, which is exactly the thing you were trying to
improve on. Asking *several* times at different temperatures gives a
spread, and the best of a spread is meaningfully better than the middle
of one.

That is the whole idea. It is usually called best-of-n sampling or
rejection sampling, and when the kept answers go back into training it
is self-distillation. Dilution, because you are drawing the same model
out thinner and picking what settles.

How it works
------------
For each prompt, generate `samples_per_prompt` completions across a
temperature ladder — cold ones for correctness, warm ones for the
occasional better idea a cold one never reaches. Score each with an
**evaluator**, which can be a second model, the same model asked to
judge, or a plain Python function when the task has a checkable answer.
Keep the winners. Repeat with new prompts until `target_traces` are
collected or `max_attempts` is spent.

Three things that are easy to get wrong
---------------------------------------
**A temperature ladder, not one temperature.** Sampling six times at
0.8 gives six draws from one distribution. The ladder is the point:
:data:`DEFAULT_LADDER` spans cold to warm so the set contains both
"most likely" and "worth a look".

**Ties must not always go to the first.** An evaluator that scores in
whole numbers ties constantly, and `max()` returns the earliest — which
is the coldest sample, every time, so the warm end of the ladder never
contributes anything and the whole exercise collapses to greedy
decoding. :func:`pick_best` breaks ties by a seeded shuffle.

**A budget that is spent is not a failure.** Running out of attempts
with fewer traces than asked for is an ordinary outcome, and the result
says so rather than raising — a pipeline that dies at hour six with
nothing written is worse than one that hands back what it has.

JIT
---
:func:`jit_distil` is the streaming form: generate, score, keep, and
hand each trace to a sink as it is made, so training can start on the
first traces while the rest are still being generated. Same machinery,
no list held in memory, and a stop condition the caller controls.
"""
from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "DEFAULT_LADDER",
    "Trace",
    "Candidate",
    "DiluteConfig",
    "DiluteResult",
    "Evaluator",
    "LengthEvaluator",
    "ModelEvaluator",
    "FunctionEvaluator",
    "temperature_ladder",
    "pick_best",
    "dilute",
    "jit_distil",
]

_LAZY = {"core", "evaluators"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    for module_name in ("core", "evaluators"):
        module = importlib.import_module(f".{module_name}", __name__)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
