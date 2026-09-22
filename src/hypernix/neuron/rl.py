"""neuron.rl — learning from outcomes, when there is nothing to copy.

Reach for this *second*. If you can demonstrate the task at all,
:mod:`hypernix.neuron.imitation` gets you a working policy in minutes
from data you can produce by hand; reinforcement learning takes orders
of magnitude more interaction and fails in ways that look like bugs.
The honest use for it is the case where no expert exists — where you
genuinely do not know the right action, only whether the outcome was
good.

Two algorithms, deliberately
----------------------------
:func:`reinforce` is the policy gradient in its plainest honest form:
run episodes, push up the log-probability of actions that preceded
better-than-average returns. On-policy, so every batch of experience is
used once and thrown away. Simple enough to read in full, which matters
when you are trying to work out whether the *algorithm* or the *reward*
is what is wrong.

:func:`dqn` learns an action-value function off-policy from a replay
buffer. Far more sample-efficient, and worth it once episodes get
expensive — which they always do the moment a real robot or a real game
is involved.

The three details that are not decoration
-----------------------------------------
**Reward-to-go, not episode total.** Crediting a step with reward
collected before it is pure variance; nothing that step did could have
caused it.

**A baseline.** Subtracting the batch mean return leaves the gradient
unbiased and cuts its variance enormously. Without it REINFORCE on a
task with all-positive rewards pushes *every* action up and learns
mostly nothing.

**`terminated`, never `terminated or truncated`.** A time limit is not
part of the world; the state after it still had value. Bootstrapping
through a terminal state, on the other hand, invents value that is not
there and the estimate diverges. This is the single most common bug in
hand-written DQN and it is invisible — training runs, loss falls, the
policy is just quietly wrong.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from . import nets
from .data import ReplayBuffer, Transition, discounted_returns
from .envs import Env

__all__ = ["TrainResult", "td_target", "reinforce", "dqn"]


def td_target(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """``r + gamma * V(s') * (1 - terminated)`` — the bootstrap, alone.

    A four-line function with its own tests because of what it gets
    wrong when it is inlined. Passing ``terminated or truncated`` here
    zeroes the bootstrap at a *time limit*, which throws away the value
    of a state that had plenty; passing neither bootstraps through a
    *terminal* state into its own successor, which is itself, and the
    estimate climbs forever.

    Both bugs train without error and both leave the loss looking
    healthy. Neither is reliably visible in a short run on a small
    environment — the integration test for DQN passes with the second
    one reintroduced — so the arithmetic is asserted directly rather
    than hoped for through a policy score.
    """
    if not (rewards.shape == next_values.shape == terminated.shape):
        raise ValueError(
            f"td_target needs matching shapes, got rewards {tuple(rewards.shape)}, "
            f"next_values {tuple(next_values.shape)}, "
            f"terminated {tuple(terminated.shape)}"
        )
    return rewards + gamma * next_values * (1.0 - terminated)


@dataclass
class TrainResult:
    """What a run did, per checkpoint, so a flat line is visible."""

    steps: int = 0
    episodes: int = 0
    best_return: float = -float("inf")
    final_return: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.episodes} episodes / {self.steps} steps · "
            f"best mean return {self.best_return:.3f} · "
            f"final {self.final_return:.3f}"
        )

    @property
    def improved(self) -> bool:
        """Did the score move between the first checkpoint and the best?

        A training loop that runs without error and does not move is the
        normal failure mode here, and it reads as success in any log
        that only prints losses.

        Read the caveat before asserting on this: a run that *solves the
        task before its first evaluation* reports False, because there
        was no improvement left to observe. That is not a failure, and
        it is exactly what DQN does on a small grid. For "is it any
        good", ask :meth:`reached`, which compares against a score you
        name instead of against the run's own past.
        """
        if len(self.history) < 2:
            return False
        return self.best_return > self.history[0]["mean_return"]

    def reached(self, target: float) -> bool:
        """Did the policy ever score at least *target*?

        The assertion worth making when the task has a known optimum,
        and the one the tests in this package use: "within a whisker of
        the expert" is checkable, where "went up" is not when the run
        started at the ceiling.
        """
        return self.best_return >= target

    @property
    def best_success_rate(self) -> float:
        """The best success rate at any checkpoint, or 0.0 if unevaluated."""
        if not self.history:
            return 0.0
        return max(float(row.get("success_rate", 0.0)) for row in self.history)


def _returns_batch(
    trajectories: list[tuple[list[np.ndarray], list[int], list[float]]],
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    returns: list[float] = []
    for episode_observations, episode_actions, episode_rewards in trajectories:
        observations.extend(episode_observations)
        actions.extend(episode_actions)
        returns.extend(discounted_returns(episode_rewards, gamma).tolist())

    x = torch.from_numpy(np.stack(observations).astype(np.float32))
    a = torch.from_numpy(np.asarray(actions, dtype=np.int64))
    g = torch.from_numpy(np.asarray(returns, dtype=np.float32))
    return x, a, g


def reinforce(
    env: Env,
    *,
    episodes: int = 600,
    batch_episodes: int = 10,
    gamma: float = 0.99,
    learning_rate: float = 3e-3,
    hidden: tuple[int, ...] = (64, 64),
    seed: int = 0,
    evaluate_every: int = 10,
    evaluate_episodes: int = 10,
    model: nn.Module | None = None,
) -> tuple[nn.Module, TrainResult]:
    """Policy gradient. Returns the policy network and the run.

    Collects *batch_episodes* episodes, computes reward-to-go, subtracts
    the batch mean as a baseline, and takes one step. On-policy: the
    experience is used once.
    """
    from .evaluate import evaluate_policy

    torch.manual_seed(seed)
    if model is None:
        model, _spec = nets.for_env(env, hidden=hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)

    result = TrainResult()
    batch: list[tuple[list[np.ndarray], list[int], list[float]]] = []

    for episode_index in range(episodes):
        observation, _info = env.reset(seed=seed + episode_index)
        episode_observations: list[np.ndarray] = []
        episode_actions: list[int] = []
        episode_rewards: list[float] = []

        while True:
            state = torch.from_numpy(
                np.asarray(observation, dtype=np.float32)
            ).unsqueeze(0)
            with torch.no_grad():
                probabilities = torch.softmax(model(state), dim=-1)[0]
            # Sampled, not greedy: a greedy on-policy gradient has no
            # exploration at all and locks onto whatever the random
            # initialisation preferred.
            action = int(torch.multinomial(probabilities, 1, generator=generator).item())

            episode_observations.append(np.asarray(observation, dtype=np.float32))
            episode_actions.append(action)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_rewards.append(float(reward))
            result.steps += 1
            if terminated or truncated:
                break

        batch.append((episode_observations, episode_actions, episode_rewards))
        result.episodes += 1

        if len(batch) >= batch_episodes:
            x, a, g = _returns_batch(batch, gamma)
            # The baseline. Standardising rather than only centring also
            # keeps the step size sane across tasks whose rewards differ
            # by orders of magnitude.
            advantage = g - g.mean()
            spread = advantage.std()
            if float(spread) > 1e-6:
                advantage = advantage / spread

            model.train()
            log_probabilities = torch.log_softmax(model(x), dim=-1)
            chosen = log_probabilities.gather(1, a.unsqueeze(1)).squeeze(1)
            loss = -(chosen * advantage).mean()

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            # Policy gradients produce occasional enormous steps from a
            # single lucky episode; without this one of them undoes the
            # whole run.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            batch.clear()

        if evaluate_every and result.episodes % evaluate_every == 0:
            report = evaluate_policy(
                env,
                nets.greedy(model),
                episodes=evaluate_episodes,
                seed=seed + 100_000,
            )
            result.history.append(
                {
                    "episode": float(result.episodes),
                    "steps": float(result.steps),
                    "mean_return": report.mean_return,
                    "success_rate": report.success_rate,
                }
            )
            result.best_return = max(result.best_return, report.mean_return)
            result.final_return = report.mean_return

    return model, result


def dqn(
    env: Env,
    *,
    steps: int = 20_000,
    gamma: float = 0.99,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    buffer_size: int = 10_000,
    warmup: int = 500,
    target_sync: int = 250,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 5_000,
    hidden: tuple[int, ...] = (64, 64),
    seed: int = 0,
    evaluate_every: int = 2_000,
    evaluate_episodes: int = 10,
    model: nn.Module | None = None,
) -> tuple[nn.Module, TrainResult]:
    """Deep Q-learning with a replay buffer and a target network.

    The target network is not optional. Without it the regression target
    moves every time the weights move, and the value estimate chases
    itself — the loss looks fine and the policy never settles.
    """
    from .evaluate import evaluate_policy

    torch.manual_seed(seed)
    if model is None:
        model, _spec = nets.for_env(env, hidden=hidden)
    target = copy.deepcopy(model)
    target.eval()
    for parameter in target.parameters():
        parameter.requires_grad_(False)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    buffer = ReplayBuffer(buffer_size, seed=seed)
    rng = np.random.default_rng(seed)

    result = TrainResult()
    actions_available = env.action_space.n
    observation, _info = env.reset(seed=seed)
    episode_index = 0

    for step in range(1, steps + 1):
        fraction = min(1.0, step / max(epsilon_decay_steps, 1))
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)

        if rng.random() < epsilon:
            action = int(rng.integers(actions_available))
        else:
            with torch.no_grad():
                state = torch.from_numpy(
                    np.asarray(observation, dtype=np.float32)
                ).unsqueeze(0)
                action = int(model(state).argmax(dim=-1).item())

        following, reward, terminated, truncated, _ = env.step(action)
        buffer.add(
            Transition(
                observation=np.asarray(observation, dtype=np.float32),
                action=action,
                reward=float(reward),
                next_observation=np.asarray(following, dtype=np.float32),
                # `terminated` alone. See the module docstring.
                terminated=bool(terminated),
            )
        )
        observation = following
        result.steps = step

        if terminated or truncated:
            episode_index += 1
            result.episodes = episode_index
            observation, _info = env.reset(seed=seed + episode_index)

        if len(buffer) >= warmup:
            states, chosen, rewards, followings, terminals = buffer.batch(batch_size)
            model.train()
            predicted = model(states).gather(1, chosen.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                best_next = target(followings).max(dim=1).values
                wanted = td_target(rewards, best_next, terminals, gamma)
            # Huber rather than MSE: a single bad bootstrap in early
            # training produces an enormous squared error and a step
            # that wrecks the network.
            loss = nn.functional.smooth_l1_loss(predicted, wanted)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimiser.step()

        if step % target_sync == 0:
            target.load_state_dict(model.state_dict())

        if evaluate_every and step % evaluate_every == 0:
            report = evaluate_policy(
                env,
                nets.greedy(model),
                episodes=evaluate_episodes,
                seed=seed + 100_000,
            )
            result.history.append(
                {
                    "episode": float(result.episodes),
                    "steps": float(step),
                    "epsilon": epsilon,
                    "mean_return": report.mean_return,
                    "success_rate": report.success_rate,
                }
            )
            result.best_return = max(result.best_return, report.mean_return)
            result.final_return = report.mean_return

    return model, result
