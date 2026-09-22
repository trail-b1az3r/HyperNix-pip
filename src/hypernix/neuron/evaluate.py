"""neuron.evaluate — does it work when you run it?

Deliberately not a method on the trainers. A trainer that reports its
own score reports the number it was optimising, and for everything in
this package that number is the wrong one: cross-entropy on held-out
demonstrations says how well the network copies an expert, not whether
the policy reaches the goal. Those come apart constantly — see
:mod:`hypernix.neuron.imitation` for why — so they are produced by
different code and reported as different fields.

What is measured
----------------
**Return**, mean and spread. The spread matters more than people
expect: a policy that solves the task eight times in ten and falls over
twice has the same mean as one that is mediocre every time, and they
need completely different fixes.

**Success rate**, meaning episodes that ended by *terminating* rather
than by running out of clock. For a goal-reaching task that is the
number a person actually cares about.

**Episode length**, which is how you catch the policy that succeeds by
dithering until it stumbles into the goal.

Every evaluation is seeded and the seeds are reported, because "it got
0.82" is not a result anybody can check.
"""
from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .envs import Env, rollout

__all__ = ["EvalReport", "evaluate_policy", "compare"]


@dataclass
class EvalReport:
    """What running a policy actually produced."""

    episodes: int = 0
    mean_return: float = 0.0
    median_return: float = 0.0
    return_spread: float = 0.0
    worst_return: float = 0.0
    best_return: float = 0.0
    success_rate: float = 0.0
    mean_length: float = 0.0
    returns: list[float] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.episodes} episodes · return {self.mean_return:.3f} "
            f"± {self.return_spread:.3f} "
            f"(worst {self.worst_return:.3f}) · "
            f"success {self.success_rate:.0%} · "
            f"{self.mean_length:.1f} steps"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "episodes": self.episodes,
            "mean_return": self.mean_return,
            "median_return": self.median_return,
            "return_spread": self.return_spread,
            "worst_return": self.worst_return,
            "best_return": self.best_return,
            "success_rate": self.success_rate,
            "mean_length": self.mean_length,
            "seeds": list(self.seeds),
        }


def evaluate_policy(
    env: Env,
    policy: Callable,
    *,
    episodes: int = 20,
    seed: int = 0,
) -> EvalReport:
    """Run *policy* for *episodes* and report what happened.

    Seeds are ``seed``, ``seed + 1``, … and are recorded on the report
    so a surprising number can be reproduced exactly rather than argued
    about.
    """
    if episodes < 1:
        raise ValueError("evaluating zero episodes reports nothing")

    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    seeds: list[int] = []

    for index in range(episodes):
        episode_seed = seed + index
        seeds.append(episode_seed)
        _obs, actions, rewards, terminated = rollout(
            env, policy, seed=episode_seed
        )
        returns.append(float(sum(rewards)))
        lengths.append(len(actions))
        successes += int(terminated)

    return EvalReport(
        episodes=episodes,
        mean_return=float(np.mean(returns)),
        median_return=float(np.median(returns)),
        # Population stdev: this is the whole set of episodes that were
        # run, not a sample from a larger set of them.
        return_spread=float(statistics.pstdev(returns)) if len(returns) > 1 else 0.0,
        worst_return=float(min(returns)),
        best_return=float(max(returns)),
        success_rate=successes / episodes,
        mean_length=float(np.mean(lengths)),
        returns=returns,
        seeds=seeds,
    )


def compare(
    env: Env,
    policies: dict[str, Callable],
    *,
    episodes: int = 20,
    seed: int = 0,
) -> dict[str, EvalReport]:
    """Several policies on identical episodes.

    The same *seed* for every one of them, so they face the same starting
    states. Comparing policies across different episodes is the single
    most common way to conclude that a change helped when it did not —
    on these environments the spread between seeds is larger than most
    real improvements.
    """
    return {
        name: evaluate_policy(env, policy, episodes=episodes, seed=seed)
        for name, policy in policies.items()
    }
