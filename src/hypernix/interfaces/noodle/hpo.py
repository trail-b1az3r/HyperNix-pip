"""noodle.hpo — hyperparameter search, driven by the swarm.

Search over a space of training configurations, evaluating each with a
caller-supplied objective. Two strategies, and the choice between them
is not a matter of taste:

**Random** beats grid search on almost every real problem, because most
hyperparameters do not matter and a grid spends its budget varying them
anyway. Bergstra and Bengio's result, and it holds up.

**Successive halving** beats random when trials can be stopped early:
run everything briefly, keep the best half, run those longer, repeat.
For LLM fine-tuning — where a bad learning rate is visible within a few
hundred steps and a full run is hours — it is usually the right answer.

The objective is the caller's
-----------------------------
This module proposes configurations and ranks results. It does not know
what "good" means; :class:`Trial` carries whatever score the objective
returned, and lower-is-better is the convention because the objective is
nearly always a loss. Building the evaluation in here would mean
guessing at somebody's training loop.
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["SearchSpace", "Trial", "HPOResult", "random_search", "successive_halving"]

#: Evaluates one configuration. Given the config and a budget (steps,
#: epochs — whatever the caller's unit is), returns a score where lower
#: is better.
Objective = Callable[[dict[str, Any], int], float]


@dataclass
class SearchSpace:
    """A space to sample from.

    ``log_uniform`` exists because learning rate is the parameter that
    matters most and it is the one people get wrong by sampling
    uniformly: uniform over [1e-6, 1e-3] puts 99.9% of its samples above
    1e-4, which is not a search, it is a very slow way to test one
    region.
    """

    uniform: dict[str, tuple[float, float]] = field(default_factory=dict)
    log_uniform: dict[str, tuple[float, float]] = field(default_factory=dict)
    choice: dict[str, Sequence[Any]] = field(default_factory=dict)
    integer: dict[str, tuple[int, int]] = field(default_factory=dict)
    fixed: dict[str, Any] = field(default_factory=dict)

    def sample(self, rng: random.Random) -> dict[str, Any]:
        config: dict[str, Any] = dict(self.fixed)
        for name, (low, high) in self.uniform.items():
            config[name] = rng.uniform(low, high)
        for name, (low, high) in self.log_uniform.items():
            if low <= 0 or high <= 0:
                raise ValueError(f"log_uniform bounds for {name!r} must be positive")
            config[name] = math.exp(rng.uniform(math.log(low), math.log(high)))
        for name, options in self.choice.items():
            config[name] = rng.choice(list(options))
        for name, (low, high) in self.integer.items():
            config[name] = rng.randint(low, high)
        return config

    def describe(self) -> dict[str, Any]:
        return {
            "uniform": {k: list(v) for k, v in self.uniform.items()},
            "log_uniform": {k: list(v) for k, v in self.log_uniform.items()},
            "choice": {k: list(v) for k, v in self.choice.items()},
            "integer": {k: list(v) for k, v in self.integer.items()},
            "fixed": dict(self.fixed),
        }

    @classmethod
    def default_finetune(cls) -> SearchSpace:
        """A sensible starting space for LLM fine-tuning.

        The ranges are the ones that matter in practice: learning rate
        log-sampled across three decades, warmup as a ratio rather than a
        step count (so it survives a change in dataset size), and the
        optimisers HyperNix actually ships.
        """
        return cls(
            log_uniform={"learning_rate": (1e-6, 1e-3), "weight_decay": (1e-4, 1e-1)},
            uniform={"warmup_ratio": (0.0, 0.15)},
            integer={"gradient_accumulation": (1, 16)},
            choice={
                "optimizer": ("PressureCookerV5", "PressureCookerV5Plus", "AdamW", "Lion"),
                "lr_scheduler": ("cosine", "linear", "constant_with_warmup"),
                "micro_batch": (1, 2, 4, 8),
            },
            fixed={"max_grad_norm": 1.0},
        )


@dataclass
class Trial:
    trial_id: str
    config: dict[str, Any]
    score: float = math.inf
    budget: int = 0
    seconds: float = 0.0
    failed: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "config": dict(self.config),
            "score": None if math.isinf(self.score) else self.score,
            "budget": self.budget,
            "seconds": round(self.seconds, 2),
            "failed": self.failed,
            "error": self.error,
        }


@dataclass
class HPOResult:
    trials: list[Trial] = field(default_factory=list)
    strategy: str = ""
    seconds: float = 0.0

    @property
    def best(self) -> Trial | None:
        finished = [t for t in self.trials if not t.failed]
        return min(finished, key=lambda t: t.score) if finished else None

    def ranked(self) -> list[Trial]:
        return sorted(
            (t for t in self.trials if not t.failed), key=lambda t: t.score
        )

    def to_dict(self) -> dict[str, Any]:
        best = self.best
        return {
            "strategy": self.strategy,
            "trial_count": len(self.trials),
            "failed_count": sum(1 for t in self.trials if t.failed),
            "seconds": round(self.seconds, 2),
            "best": best.to_dict() if best else None,
            "trials": [t.to_dict() for t in self.trials],
        }


def _evaluate(objective: Objective, trial: Trial, budget: int) -> Trial:
    started = time.monotonic()
    try:
        trial.score = float(objective(trial.config, budget))
        if math.isnan(trial.score):
            # NaN would sort unpredictably and could win a comparison.
            # A NaN loss is a failed trial, and saying so is the point.
            trial.failed = True
            trial.error = "objective returned NaN — treating the trial as failed"
            trial.score = math.inf
    except Exception as exc:  # noqa: BLE001 - one bad config must not end the search
        trial.failed = True
        trial.error = f"{type(exc).__name__}: {exc}"
        trial.score = math.inf
    trial.budget = budget
    trial.seconds = time.monotonic() - started
    return trial


def random_search(
    space: SearchSpace,
    objective: Objective,
    *,
    trials: int = 20,
    budget: int = 100,
    seed: int | None = None,
) -> HPOResult:
    """Sample *trials* configurations and evaluate each at *budget*."""
    rng = random.Random(seed)  # nosec B311 - seeded for reproducibility (shuffles, splits, jitter); nothing secret
    started = time.monotonic()
    result = HPOResult(strategy="random")
    for index in range(trials):
        trial = Trial(trial_id=f"r{index + 1}", config=space.sample(rng))
        result.trials.append(_evaluate(objective, trial, budget))
    result.seconds = time.monotonic() - started
    return result


def successive_halving(
    space: SearchSpace,
    objective: Objective,
    *,
    trials: int = 27,
    min_budget: int = 10,
    reduction: int = 3,
    seed: int | None = None,
) -> HPOResult:
    """Run everything briefly, keep the best fraction, run those longer.

    Each rung keeps ``1/reduction`` of the survivors and multiplies the
    budget by ``reduction``, so total compute is roughly constant per
    rung — the property that makes this cheap. Failed trials are dropped
    at the rung boundary rather than carried: promoting a configuration
    that crashed would waste the next rung's entire budget on it.
    """
    if reduction < 2:
        raise ValueError("reduction must be at least 2")
    rng = random.Random(seed)  # nosec B311 - seeded for reproducibility (shuffles, splits, jitter); nothing secret
    started = time.monotonic()
    result = HPOResult(strategy="successive_halving")

    survivors = [
        Trial(trial_id=f"s{i + 1}", config=space.sample(rng)) for i in range(trials)
    ]
    budget = min_budget
    rung = 0
    while survivors:
        rung += 1
        logger.info(
            "noodle.hpo: rung %d — %d trial(s) at budget %d", rung, len(survivors), budget
        )
        for trial in survivors:
            _evaluate(objective, trial, budget)
            if trial not in result.trials:
                result.trials.append(trial)
        alive = [t for t in survivors if not t.failed]
        if len(alive) <= 1:
            break
        keep = max(1, len(alive) // reduction)
        survivors = sorted(alive, key=lambda t: t.score)[:keep]
        budget *= reduction
        if keep == 1:
            # One survivor at the top budget is the answer; another rung
            # would just re-evaluate it more expensively.
            _evaluate(objective, survivors[0], budget)
            break

    result.seconds = time.monotonic() - started
    return result
