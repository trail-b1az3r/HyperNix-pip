"""dilute.core — the ladder, the pick, and the loop around them."""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_LADDER",
    "Trace",
    "Candidate",
    "DiluteConfig",
    "DiluteResult",
    "Generator",
    "temperature_ladder",
    "pick_best",
    "dilute",
    "jit_distil",
]

#: Cold to warm. The cold end is where a correct answer usually is; the
#: warm end is where a *better* one occasionally is, and a set sampled
#: entirely at one temperature contains neither reliably.
DEFAULT_LADDER: tuple[float, ...] = (0.1, 0.35, 0.6, 0.8, 1.0, 1.2)

#: Bounds on how many samples per prompt make sense. Below four there is
#: not enough spread for "best of" to mean anything; above six the
#: returns fall off fast and the cost does not.
MIN_SAMPLES = 4
MAX_SAMPLES = 6


class Generator(Protocol):
    """Anything that turns a prompt and a temperature into text."""

    def __call__(self, prompt: str, *, temperature: float, **kwargs) -> str:
        ...


@dataclass
class Candidate:
    """One sample, its temperature, and what the evaluator made of it."""

    text: str
    temperature: float
    score: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trace:
    """A prompt and the answer that won, with the losers kept.

    The rejected candidates are kept on purpose. A trace set where you
    cannot see what was rejected is one you cannot audit later, and
    "the evaluator preferred this" is only checkable against what it
    was choosing between.
    """

    prompt: str
    chosen: Candidate
    rejected: list[Candidate] = field(default_factory=list)
    attempt: int = 0

    @property
    def margin(self) -> float:
        """How much better the winner was than the runner-up.

        A margin of zero across a whole run means the evaluator is not
        discriminating and the traces are effectively random picks —
        which is worth knowing before training on them.
        """
        if not self.rejected:
            return 0.0
        return self.chosen.score - max(c.score for c in self.rejected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen.to_dict(),
            "rejected": [c.to_dict() for c in self.rejected],
            "attempt": self.attempt,
            "margin": round(self.margin, 6),
        }

    def to_training_pair(self) -> dict[str, str]:
        """Just the prompt and the winner, for a supervised set."""
        return {"prompt": self.prompt, "completion": self.chosen.text}


@dataclass
class DiluteConfig:
    """Everything a run needs. Every field has a working default."""

    #: How many completions per prompt. Clamped to [MIN_SAMPLES, MAX_SAMPLES].
    samples_per_prompt: int = 5
    #: How many traces to collect before stopping.
    target_traces: int = 100
    #: Prompts tried before giving up, however many traces that yielded.
    max_attempts: int = 500
    #: Temperatures to spread the samples across.
    ladder: tuple[float, ...] = DEFAULT_LADDER
    #: Keep a trace only if the winner scores at least this. None keeps
    #: every prompt's winner regardless.
    min_score: float | None = None
    #: Keep a trace only if the winner beat the runner-up by this much.
    #: Guards against an evaluator that cannot tell the samples apart.
    min_margin: float | None = None
    seed: int = 0
    #: Where to write, if anywhere. JSONL, one trace per line.
    output: str = ""

    def __post_init__(self) -> None:
        if self.target_traces < 1:
            raise ValueError("target_traces has to be at least 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts has to be at least 1")
        if not self.ladder:
            raise ValueError("the temperature ladder cannot be empty")
        if any(t < 0 for t in self.ladder):
            raise ValueError("a negative temperature is not a temperature")
        # Clamped rather than rejected: asking for 2 or 20 is a guess
        # about a knob, not an error, and the useful range is narrow.
        self.samples_per_prompt = max(
            MIN_SAMPLES, min(MAX_SAMPLES, self.samples_per_prompt)
        )


@dataclass
class DiluteResult:
    traces: list[Trace] = field(default_factory=list)
    attempts: int = 0
    generated: int = 0
    rejected_prompts: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    #: "target-reached" | "out-of-prompts" | "out-of-attempts"
    stopped_because: str = "target-reached"

    @property
    def duration(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    @property
    def reached_target(self) -> bool:
        return self.stopped_because == "target-reached"

    @property
    def mean_margin(self) -> float:
        """Average winning margin. Near zero means the evaluator is not
        discriminating and these are effectively random picks."""
        if not self.traces:
            return 0.0
        return sum(t.margin for t in self.traces) / len(self.traces)

    def summary(self) -> str:
        return (
            f"{len(self.traces)} trace(s) from {self.attempts} prompt(s), "
            f"{self.generated} completion(s) generated, "
            f"mean margin {self.mean_margin:.3f} "
            f"[{self.stopped_because}]"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "traces": len(self.traces),
            "attempts": self.attempts,
            "generated": self.generated,
            "rejected_prompts": self.rejected_prompts,
            "mean_margin": round(self.mean_margin, 6),
            "duration_seconds": round(self.duration, 2),
            "stopped_because": self.stopped_because,
            "reached_target": self.reached_target,
        }

    def write_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"_dilute": self.to_dict()}) + "\n")
            for trace in self.traces:
                handle.write(
                    json.dumps(trace.to_dict(), ensure_ascii=False) + "\n"
                )
        return target


# ---------------------------------------------------------------------------


def temperature_ladder(count: int, ladder: Sequence[float] = DEFAULT_LADDER) -> list[float]:
    """*count* temperatures spread across *ladder*.

    Spread rather than repeated: sampling six times at 0.8 is six draws
    from one distribution, which is not what "different temperatures"
    means and not where the benefit comes from.
    """
    if count < 1:
        raise ValueError("need at least one sample")
    if not ladder:
        raise ValueError("the ladder cannot be empty")
    if count <= len(ladder):
        # Evenly spaced picks, keeping both ends — the cold end is
        # where correctness lives and the warm end is the whole reason
        # for doing this.
        if count == 1:
            return [ladder[0]]
        step = (len(ladder) - 1) / (count - 1)
        return [ladder[round(index * step)] for index in range(count)]
    out = list(ladder)
    while len(out) < count:
        out.append(ladder[len(out) % len(ladder)])
    return out[:count]


def pick_best(
    candidates: Sequence[Candidate], *, seed: int = 0
) -> tuple[Candidate, list[Candidate]]:
    """The winner and the rest. Ties broken at random, not by order.

    `max()` returns the earliest maximum, and the earliest candidate is
    the coldest one. An evaluator that scores in whole numbers ties
    constantly, so ordering by position means the warm end of the ladder
    never wins anything and the run quietly degrades into greedy
    decoding — which is the thing best-of-n exists to beat.
    """
    if not candidates:
        raise ValueError("nothing to pick from")
    best = max(c.score for c in candidates)
    tied = [c for c in candidates if c.score == best]
    winner = random.Random(seed).choice(tied)
    rest = [c for c in candidates if c is not winner]
    return winner, rest


# ---------------------------------------------------------------------------


def _evaluate(evaluator, prompt: str, candidate: Candidate) -> Candidate:
    try:
        outcome = evaluator(prompt, candidate.text)
    except Exception as exc:  # noqa: BLE001 - a bad sample is not a dead run
        logger.warning("dilute: evaluator raised on a candidate: %s", exc)
        return Candidate(candidate.text, candidate.temperature, 0.0,
                         f"evaluator failed: {exc}")
    if isinstance(outcome, tuple):
        score, detail = outcome
    else:
        score, detail = outcome, ""
    return Candidate(candidate.text, candidate.temperature, float(score),
                     str(detail))


def _one_prompt(
    prompt: str,
    generate: Generator,
    evaluator,
    config: DiluteConfig,
    attempt: int,
) -> tuple[Trace | None, int]:
    """One prompt's worth of work. ``(trace or None, completions made)``."""
    temperatures = temperature_ladder(config.samples_per_prompt, config.ladder)
    candidates: list[Candidate] = []
    for temperature in temperatures:
        try:
            text = generate(prompt, temperature=temperature)
        except Exception as exc:  # noqa: BLE001
            # One failed generation must not lose the other five.
            logger.warning("dilute: generation failed at T=%.2f: %s",
                           temperature, exc)
            continue
        candidates.append(Candidate(str(text), temperature))

    if not candidates:
        return None, 0

    scored = [_evaluate(evaluator, prompt, c) for c in candidates]
    winner, rest = pick_best(scored, seed=config.seed + attempt)
    trace = Trace(prompt=prompt, chosen=winner, rejected=rest, attempt=attempt)

    if config.min_score is not None and winner.score < config.min_score:
        return None, len(candidates)
    if config.min_margin is not None and trace.margin < config.min_margin:
        return None, len(candidates)
    return trace, len(candidates)


def dilute(
    prompts: Iterable[str],
    generate: Generator,
    evaluator,
    config: DiluteConfig | None = None,
) -> DiluteResult:
    """Collect traces until there are enough, or the budget is spent.

    *generate* takes ``(prompt, temperature=...)`` and returns text.
    *evaluator* takes ``(prompt, completion)`` and returns a score, or
    ``(score, detail)``.
    """
    config = config or DiluteConfig()
    result = DiluteResult(started_at=time.time())

    iterator = iter(prompts)
    while len(result.traces) < config.target_traces:
        if result.attempts >= config.max_attempts:
            result.stopped_because = "out-of-attempts"
            break
        try:
            prompt = next(iterator)
        except StopIteration:
            result.stopped_because = "out-of-prompts"
            break

        result.attempts += 1
        trace, made = _one_prompt(
            prompt, generate, evaluator, config, result.attempts
        )
        result.generated += made
        if trace is None:
            result.rejected_prompts += 1
            continue
        result.traces.append(trace)

    result.finished_at = time.time()
    if len(result.traces) >= config.target_traces:
        result.stopped_because = "target-reached"

    if config.output:
        result.write_jsonl(config.output)
    return result


def jit_distil(
    prompts: Iterable[str],
    generate: Generator,
    evaluator,
    config: DiluteConfig | None = None,
    *,
    sink: Callable[[Trace], None] | None = None,
    stop: Callable[[DiluteResult], bool] | None = None,
) -> Iterator[Trace]:
    """The streaming form: yield each trace as it is made.

    Training can start on the first traces while the rest are still
    being generated, which is the difference between a pipeline that
    waits an hour for a file and one that is already learning.

    Nothing is accumulated — a caller that wants the list can build it,
    and a caller distilling a million traces should not be forced to
    hold them. *stop* is asked after each trace, so a caller can end on
    something this module has no opinion about, like a validation score
    that has stopped moving.
    """
    config = config or DiluteConfig()
    result = DiluteResult(started_at=time.time())
    iterator = iter(prompts)
    kept = 0

    while kept < config.target_traces:
        if result.attempts >= config.max_attempts:
            result.stopped_because = "out-of-attempts"
            break
        try:
            prompt = next(iterator)
        except StopIteration:
            result.stopped_because = "out-of-prompts"
            break

        result.attempts += 1
        trace, made = _one_prompt(
            prompt, generate, evaluator, config, result.attempts
        )
        result.generated += made
        if trace is None:
            result.rejected_prompts += 1
            continue

        kept += 1
        # Counted but not kept: the whole point is not holding them.
        result.traces = result.traces[:0]
        if sink is not None:
            sink(trace)
        yield trace

        if stop is not None:
            probe = DiluteResult(
                traces=[trace], attempts=result.attempts,
                generated=result.generated,
                rejected_prompts=result.rejected_prompts,
                started_at=result.started_at, finished_at=time.time(),
            )
            if stop(probe):
                return
