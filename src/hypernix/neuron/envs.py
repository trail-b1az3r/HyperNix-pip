"""neuron.envs — the smallest thing a trainer can be pointed at.

Every reinforcement-learning implementation needs something to act in,
and the usual arrangement is that the library takes a Gymnasium
environment and the tests mock it. That produces an RL implementation
nobody has run end to end: a mocked environment returns whatever the
test author expected, so a policy-gradient sign error passes.

So this defines the interface and ships two real environments behind it.
Both are deterministic given a seed, both run on a CPU in milliseconds,
and both are *solvable* — a correct trainer reaches a known score, which
is a far stronger assertion than "loss went down".

The interface is Gymnasium's
------------------------------
``reset() -> (obs, info)`` and
``step(action) -> (obs, reward, terminated, truncated, info)``, with
``terminated`` (the task ended) separate from ``truncated`` (the clock
ran out). Those are different things and collapsing them into one
``done`` is a real bug in value estimation: bootstrapping through a time
limit is correct, bootstrapping through a terminal state is not.

Following that shape means a real Gymnasium environment, a robot
driver, or a screen-capture wrapper around a game satisfies this
protocol already, with no adapter and no dependency on Gymnasium here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Space",
    "Env",
    "GridWorld",
    "BalanceBeam",
    "rollout",
]


@dataclass(frozen=True)
class Space:
    """What an observation or action looks like.

    Deliberately not Gymnasium's ``Box``/``Discrete`` class tree: this
    package needs exactly two facts — the shape of a continuous vector,
    or the number of discrete choices — and depending on Gymnasium to
    express them would make it a hard dependency of a package that
    otherwise needs only torch.
    """

    #: Number of discrete actions, or 0 when this space is continuous.
    n: int = 0
    #: Shape of a continuous observation, or () when discrete.
    shape: tuple[int, ...] = ()

    @property
    def discrete(self) -> bool:
        return self.n > 0

    @property
    def size(self) -> int:
        """Flat element count — what a network's first layer needs."""
        if self.discrete:
            return self.n
        return int(math.prod(self.shape)) if self.shape else 0

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError("a discrete space cannot have a negative size")
        if self.n and self.shape:
            raise ValueError(
                "a space is discrete (n) or continuous (shape), not both"
            )
        if not self.n and not self.shape:
            raise ValueError("a space needs either n or shape")


@runtime_checkable
class Env(Protocol):
    """What the trainers require. Gymnasium environments satisfy it."""

    observation_space: Space
    action_space: Space

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        ...

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        ...


# ---------------------------------------------------------------------------
# GridWorld
# ---------------------------------------------------------------------------


@dataclass
class GridWorld:
    """Reach the goal on an ``size``x``size`` grid without hitting a wall.

    The smallest environment that still has the three properties that
    break naive implementations: a **sparse** reward (nothing until you
    arrive, so a trainer that only follows immediate reward learns
    nothing), **terminal states** that must not be bootstrapped through,
    and a **time limit** that must be.

    The optimal policy is known in closed form — walk the Manhattan path
    — so tests can assert that a trained policy is within one step of
    optimal rather than merely better than random.

    Observation is the agent's position and the goal's, normalised to
    [0, 1]: four floats. Actions are up/down/left/right.
    """

    size: int = 5
    max_steps: int = 50
    #: Cells that cannot be entered, as (row, col).
    walls: tuple[tuple[int, int], ...] = ()
    #: Reward for arriving. Everything else is the step cost.
    goal_reward: float = 1.0
    step_cost: float = -0.01
    wall_penalty: float = -0.05

    observation_space: Space = field(
        default_factory=lambda: Space(shape=(4,)), init=False
    )
    action_space: Space = field(default_factory=lambda: Space(n=4), init=False)

    _agent: tuple[int, int] = field(default=(0, 0), init=False)
    _goal: tuple[int, int] = field(default=(0, 0), init=False)
    _steps: int = field(default=0, init=False)
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), init=False
    )

    #: (row delta, col delta) per action, in action order.
    MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("a grid smaller than 2x2 has nowhere to go")

    # -- protocol ----------------------------------------------------

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        # Start and goal are placed apart so a reset cannot hand out a
        # zero-step episode, which would inflate every score silently.
        free = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in self.walls
        ]
        if len(free) < 2:
            raise ValueError("the walls leave nowhere to stand")
        first = int(self._rng.integers(len(free)))
        self._agent = free[first]
        while True:
            second = int(self._rng.integers(len(free)))
            if free[second] != self._agent:
                self._goal = free[second]
                break
        return self._observe(), {"optimal_steps": self.optimal_steps()}

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        index = int(action)
        if not 0 <= index < 4:
            raise ValueError(f"action {index} is not one of the 4 directions")

        self._steps += 1
        dr, dc = self.MOVES[index]
        row, col = self._agent[0] + dr, self._agent[1] + dc

        reward = self.step_cost
        if not (0 <= row < self.size and 0 <= col < self.size):
            reward += self.wall_penalty       # walked into the edge
        elif (row, col) in self.walls:
            reward += self.wall_penalty       # walked into a wall
        else:
            self._agent = (row, col)

        terminated = self._agent == self._goal
        if terminated:
            reward += self.goal_reward
        # Separate from `terminated` on purpose: see the module docstring.
        truncated = (not terminated) and self._steps >= self.max_steps
        return self._observe(), reward, terminated, truncated, {}

    # -- helpers -----------------------------------------------------

    def _observe(self) -> np.ndarray:
        scale = max(self.size - 1, 1)
        return np.array(
            [
                self._agent[0] / scale,
                self._agent[1] / scale,
                self._goal[0] / scale,
                self._goal[1] / scale,
            ],
            dtype=np.float32,
        )

    def optimal_steps(self) -> int:
        """Manhattan distance — exact when there are no walls between."""
        return abs(self._agent[0] - self._goal[0]) + abs(
            self._agent[1] - self._goal[1]
        )

    def expert_action(self) -> int:
        """The move an optimal player makes. Used to generate demos.

        Having the expert *inside* the environment is what lets the
        imitation tests be real: the demonstrations are genuinely
        optimal, so "did behaviour cloning learn the expert" is a
        question with a right answer rather than a trend.
        """
        dr = self._goal[0] - self._agent[0]
        dc = self._goal[1] - self._agent[1]
        # Close the larger gap first; ties go vertical. Deterministic,
        # because a stochastic expert makes cloning accuracy unmeasurable.
        if abs(dr) >= abs(dc) and dr != 0:
            return 0 if dr < 0 else 1
        if dc != 0:
            return 2 if dc < 0 else 3
        return 0


# ---------------------------------------------------------------------------
# BalanceBeam
# ---------------------------------------------------------------------------


@dataclass
class BalanceBeam:
    """Keep a mass balanced by pushing left or right. Continuous state.

    The counterpart to GridWorld: **dense** reward, no terminal goal to
    reach, and a continuous observation where the useful signal is a
    *velocity* the network has to use rather than a position it can
    memorise. Cart-pole's structure without the dependency.

    Physics is a damped point mass on a beam, integrated with explicit
    Euler at a fixed step. It is not a faithful simulation of anything
    and does not need to be — it needs to be a control problem with a
    stable optimum, which it is: the policy that pushes against the
    velocity keeps the mass near centre indefinitely.
    """

    max_steps: int = 200
    dt: float = 0.05
    force: float = 1.0
    damping: float = 0.1
    limit: float = 1.0

    observation_space: Space = field(
        default_factory=lambda: Space(shape=(2,)), init=False
    )
    action_space: Space = field(default_factory=lambda: Space(n=2), init=False)

    _position: float = field(default=0.0, init=False)
    _velocity: float = field(default=0.0, init=False)
    _steps: int = field(default=0, init=False)
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), init=False
    )

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        # Never exactly centred: a policy that does nothing would score
        # perfectly from a dead start, which hides a network that has
        # learned nothing at all.
        self._position = float(self._rng.uniform(-0.3, 0.3))
        self._velocity = float(self._rng.uniform(-0.1, 0.1))
        return self._observe(), {}

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        index = int(action)
        if index not in (0, 1):
            raise ValueError(f"action {index} is not left (0) or right (1)")

        self._steps += 1
        push = -self.force if index == 0 else self.force
        self._velocity += (push - self.damping * self._velocity) * self.dt
        self._position += self._velocity * self.dt

        # Falling off is terminal; the clock running out is not. Same
        # distinction as GridWorld, for the same reason.
        terminated = abs(self._position) >= self.limit
        truncated = (not terminated) and self._steps >= self.max_steps
        reward = 0.0 if terminated else 1.0 - abs(self._position) / self.limit
        return self._observe(), reward, terminated, truncated, {}

    def _observe(self) -> np.ndarray:
        return np.array([self._position, self._velocity], dtype=np.float32)

    def expert_action(self) -> int:
        """Push against the way it is going, or the way it is leaning."""
        signal = self._velocity if abs(self._velocity) > 1e-6 else self._position
        return 0 if signal > 0 else 1


# ---------------------------------------------------------------------------


def rollout(
    env: Env,
    policy,
    *,
    seed: int | None = None,
    max_steps: int | None = None,
) -> tuple[list[np.ndarray], list[int], list[float], bool]:
    """Run one episode. ``(observations, actions, rewards, terminated)``.

    The single place an episode is produced, so training, evaluation and
    demonstration recording cannot disagree about what an episode is —
    which is exactly the kind of drift that makes a training score and
    an evaluation score incomparable without anyone noticing.

    *policy* is any callable from observation to action, which covers a
    torch module wrapped by :func:`hypernix.neuron.nets.greedy`, a
    scripted expert, and ``lambda _: env.action_space.sample()`` alike.
    """
    observation, _info = env.reset(seed=seed)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminated = False
    steps = 0

    while True:
        action = int(policy(observation))
        observations.append(np.asarray(observation, dtype=np.float32))
        actions.append(action)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        steps += 1
        if terminated or truncated:
            break
        if max_steps is not None and steps >= max_steps:
            break

    return observations, actions, rewards, terminated
