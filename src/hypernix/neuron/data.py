"""neuron.data — what gets recorded, and what gets sampled from it.

Three containers, because the three training styles consume genuinely
different things and a single "dataset" type would end up as a bag with
half its fields unset.

:class:`Demonstrations` is (observation, action) pairs — what imitation
learns from, and what :func:`record_expert` produces by watching
something that already plays well.

:class:`Trajectory` is one episode, in order, with rewards. Policy
gradients need the order and the rewards; supervised learning needs
neither.

:class:`ReplayBuffer` is the off-policy store: individual transitions,
sampled at random, oldest dropped. Random sampling is not an
optimisation. Consecutive frames of a control problem are almost the
same frame, so training on them in order gives correlated gradients and
a network that tracks the last few seconds instead of learning the task.
"""
from __future__ import annotations

import random
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "Transition",
    "Trajectory",
    "Demonstrations",
    "ReplayBuffer",
    "record_expert",
    "discounted_returns",
]


@dataclass(frozen=True)
class Transition:
    """One step. The unit an off-policy learner samples."""

    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    #: True only when the episode *ended*, never when the clock ran out.
    #: Bootstrapping through a time limit is correct; bootstrapping
    #: through a terminal state inflates its value forever.
    terminated: bool


@dataclass
class Trajectory:
    """One episode, in order."""

    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: bool = False

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

    def transitions(self) -> list[Transition]:
        """Split into steps, pairing each observation with its successor.

        The last step has no recorded successor, so it reuses its own
        observation — which is only ever multiplied by ``1 - terminated``
        in a bootstrap, so it is never actually read on a terminal step.
        For a *truncated* episode it is read, and the last observation is
        the best estimate available; the alternative, dropping the step,
        throws away the only transition that saw the time limit.
        """
        out: list[Transition] = []
        for index, action in enumerate(self.actions):
            last = index == len(self.actions) - 1
            following = (
                self.observations[index + 1]
                if not last
                else self.observations[index]
            )
            out.append(
                Transition(
                    observation=self.observations[index],
                    action=action,
                    reward=self.rewards[index],
                    next_observation=following,
                    terminated=self.terminated and last,
                )
            )
        return out


@dataclass
class Demonstrations:
    """(observation, action) pairs. What behaviour cloning trains on."""

    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def add(self, observation, action: int) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.actions.append(int(action))

    def extend(self, other: Demonstrations) -> None:
        self.observations.extend(other.observations)
        self.actions.extend(other.actions)

    @classmethod
    def from_trajectories(cls, trajectories: Iterable[Trajectory]) -> Demonstrations:
        out = cls()
        for trajectory in trajectories:
            # strict: a trajectory whose observations and actions have
            # drifted out of step is a recording bug, and silently
            # training on the shorter of the two would hide it.
            for observation, action in zip(
                trajectory.observations, trajectory.actions, strict=True
            ):
                out.add(observation, action)
        return out

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.actions:
            raise ValueError(
                "no demonstrations to train on — record some first, with "
                "neuron.data.record_expert or by playing the task yourself"
            )
        x = torch.from_numpy(np.stack(self.observations).astype(np.float32))
        y = torch.from_numpy(np.asarray(self.actions, dtype=np.int64))
        return x, y

    def split(
        self, holdout: float = 0.2, *, seed: int = 0
    ) -> tuple[Demonstrations, Demonstrations]:
        """Train and held-out halves, shuffled.

        Shuffled before splitting because demonstrations arrive in
        episode order: taking the last 20% unshuffled would hold out
        whole episodes, and on a task where later episodes are easier
        that reads as a model that generalises when it does not.
        """
        count = len(self.actions)
        if count < 2:
            raise ValueError("need at least two demonstrations to split")
        order = list(range(count))
        random.Random(seed).shuffle(order)
        cut = max(1, int(count * (1.0 - holdout)))
        train, test = Demonstrations(), Demonstrations()
        for position, index in enumerate(order):
            target = train if position < cut else test
            target.add(self.observations[index], self.actions[index])
        return train, test

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            observations=np.stack(self.observations)
            if self.observations
            else np.zeros((0,), dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.int64),
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> Demonstrations:
        blob = np.load(Path(path), allow_pickle=False)
        out = cls()
        out.observations = [
            np.asarray(row, dtype=np.float32) for row in blob["observations"]
        ]
        out.actions = [int(a) for a in blob["actions"]]
        return out


class ReplayBuffer:
    """A fixed-size store of transitions, sampled at random.

    Random rather than in order: see the module docstring. Fixed-size
    because an unbounded buffer on a long run is a memory leak with a
    respectable name, and because very old transitions came from a
    policy so different that they describe a different problem.
    """

    def __init__(self, capacity: int = 10_000, *, seed: int = 0):
        if capacity < 1:
            raise ValueError("a replay buffer needs room for at least one step")
        self.capacity = capacity
        self._items: deque[Transition] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._items)

    def add(self, transition: Transition) -> None:
        self._items.append(transition)

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, size: int) -> list[Transition]:
        """*size* transitions, with replacement when the buffer is small.

        With replacement on purpose: refusing to sample until the buffer
        is full would mean the first thousand steps of every run train
        on nothing, and a caller who has to check `len()` before every
        sample will eventually forget.
        """
        if not self._items:
            raise ValueError("the replay buffer is empty; collect some steps first")
        if size <= 0:
            raise ValueError("sample size has to be positive")
        if size <= len(self._items):
            return self._random.sample(list(self._items), size)
        items = list(self._items)
        return [self._random.choice(items) for _ in range(size)]

    def batch(self, size: int) -> tuple[torch.Tensor, ...]:
        """A sample as stacked tensors, ready for a loss."""
        rows = self.sample(size)
        observations = torch.from_numpy(
            np.stack([r.observation for r in rows]).astype(np.float32)
        )
        actions = torch.from_numpy(
            np.asarray([r.action for r in rows], dtype=np.int64)
        )
        rewards = torch.from_numpy(
            np.asarray([r.reward for r in rows], dtype=np.float32)
        )
        following = torch.from_numpy(
            np.stack([r.next_observation for r in rows]).astype(np.float32)
        )
        terminated = torch.from_numpy(
            np.asarray([r.terminated for r in rows], dtype=np.float32)
        )
        return observations, actions, rewards, following, terminated


def record_expert(
    env,
    expert: Callable[[Any], int] | None = None,
    *,
    episodes: int = 20,
    seed: int = 0,
) -> tuple[Demonstrations, list[Trajectory]]:
    """Watch something that already plays well, and write it down.

    *expert* defaults to the environment's own ``expert_action`` when it
    has one, which is what makes the imitation tests in this package
    real rather than circular: the demonstrations are genuinely optimal,
    so "did cloning learn the expert" has a right answer.

    In anger the expert is a human with a gamepad, a scripted
    controller, or a slow planner you want to distil into something fast
    enough to run in the loop — the last of which is the honest reason
    most robotics people are here.
    """
    if expert is None:
        if not hasattr(env, "expert_action"):
            raise ValueError(
                "no expert given and this environment has no expert_action(); "
                "pass a callable that maps an observation to an action"
            )
        expert = lambda _obs: env.expert_action()  # noqa: E731

    demonstrations = Demonstrations()
    trajectories: list[Trajectory] = []
    for episode in range(episodes):
        observation, _info = env.reset(seed=seed + episode)
        trajectory = Trajectory()
        while True:
            action = int(expert(observation))
            trajectory.observations.append(
                np.asarray(observation, dtype=np.float32)
            )
            trajectory.actions.append(action)
            demonstrations.add(observation, action)
            observation, reward, terminated, truncated, _ = env.step(action)
            trajectory.rewards.append(float(reward))
            if terminated or truncated:
                trajectory.terminated = terminated
                break
        trajectories.append(trajectory)
    return demonstrations, trajectories


def discounted_returns(rewards: Sequence[float], gamma: float = 0.99) -> np.ndarray:
    """Reward-to-go for each step, discounted.

    Reward-to-go rather than the whole episode's total on every step:
    crediting a step with reward collected *before* it is pure variance,
    since nothing that step did could have influenced it.
    """
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in reversed(range(len(rewards))):
        running = rewards[index] + gamma * running
        out[index] = running
    return out
