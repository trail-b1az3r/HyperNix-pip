"""neuron.imitation — learn by copying somebody who can already do it.

The fastest route to a working policy, and badly under-used because
reinforcement learning is the thing people have heard of. If you can
play the game, drive the arm, or write a slow planner that gets the
right answer, you can produce demonstrations — and cloning them takes
minutes on a CPU where RL takes hours on a GPU and often does not
converge at all.

The catch, which this module measures rather than hides
--------------------------------------------------------
Behaviour cloning learns the expert's *states*. The moment the policy
makes a small mistake it is somewhere the expert never was, its next
action is a guess, and the error compounds — the classic result is a
policy with 97% action accuracy that drives into a wall within ten
steps.

That is why :func:`clone` returns both numbers: accuracy on held-out
demonstrations, and mean return from *actually running the policy*. If
the first is high and the second is poor, you have compounding error and
no amount of further cloning fixes it. :func:`dagger` does: it runs the
current policy, asks the expert what it *should* have done in the states
the policy actually reached, and adds those to the data. That is the
whole idea, and it is the difference between imitation that works and
imitation that demos well.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from . import nets, supervised
from .data import Demonstrations, record_expert
from .envs import Env

__all__ = ["CloneResult", "clone", "dagger"]


@dataclass
class CloneResult:
    """Both numbers. See the module docstring for why one is not enough."""

    action_accuracy: float = 0.0
    mean_return: float = 0.0
    success_rate: float = 0.0
    demonstrations: int = 0
    fit: supervised.FitResult | None = None
    rounds: list[dict[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.demonstrations} demonstrations · "
            f"action accuracy {self.action_accuracy:.3f} · "
            f"mean return {self.mean_return:.3f} · "
            f"success {self.success_rate:.0%}"
        )

    @property
    def looks_like_compounding_error(self) -> bool:
        """High accuracy, poor return — the failure this module exists for.

        A heuristic and labelled as one: it is the shape that should send
        somebody to :func:`dagger` rather than to more epochs.
        """
        return self.action_accuracy >= 0.9 and self.success_rate < 0.5


def clone(
    env: Env,
    demonstrations: Demonstrations,
    *,
    hidden: tuple[int, ...] = (64, 64),
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    holdout: float = 0.2,
    evaluate_episodes: int = 20,
    seed: int = 0,
    model: nn.Module | None = None,
) -> tuple[nn.Module, CloneResult]:
    """Behaviour cloning: supervised learning on (observation, action).

    Returns the policy network and both measurements.
    """
    from .evaluate import evaluate_policy  # local: avoids a cycle

    if len(demonstrations) < 4:
        raise ValueError(
            f"{len(demonstrations)} demonstrations is not enough to split and "
            f"train on; record more with neuron.data.record_expert"
        )

    if model is None:
        model, _spec = nets.for_env(env, hidden=hidden)

    train, held_out = demonstrations.split(holdout, seed=seed)
    train_x, train_y = train.tensors()
    test_x, test_y = held_out.tensors()

    fit_result = supervised.fit(
        model,
        train_x,
        train_y,
        validation=(test_x, test_y),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )

    report = evaluate_policy(
        env, nets.greedy(model), episodes=evaluate_episodes, seed=seed + 5000
    )
    return model, CloneResult(
        action_accuracy=supervised.accuracy(model, test_x, test_y),
        mean_return=report.mean_return,
        success_rate=report.success_rate,
        demonstrations=len(demonstrations),
        fit=fit_result,
    )


def dagger(
    env: Env,
    *,
    expert: Callable[[Any], int] | None = None,
    rounds: int = 5,
    episodes_per_round: int = 10,
    seed: int = 0,
    hidden: tuple[int, ...] = (64, 64),
    epochs: int = 60,
    evaluate_episodes: int = 20,
    initial: Demonstrations | None = None,
) -> tuple[nn.Module, CloneResult]:
    """DAgger — ask the expert about the states the *policy* reaches.

    Plain cloning only ever sees states the expert visited, so the first
    mistake puts the policy somewhere it has no data for. Each round
    here runs the current policy, labels every state it actually reached
    with what the expert would have done there, and adds those to the
    training set. The dataset grows towards the policy's own state
    distribution, which is the one it has to be right about.

    Needs a *queryable* expert — one you can ask "what here?", not a
    fixed recording. A scripted controller or a slow planner qualifies;
    twenty replays from last week do not.
    """
    from .evaluate import evaluate_policy

    if expert is None:
        if not hasattr(env, "expert_action"):
            raise ValueError(
                "DAgger needs an expert it can query at new states; pass one, "
                "or use clone() with a fixed recording instead"
            )
        expert = lambda _obs: env.expert_action()  # noqa: E731

    collected = Demonstrations()
    if initial is not None:
        collected.extend(initial)
    else:
        seed_demos, _ = record_expert(
            env, expert, episodes=episodes_per_round, seed=seed
        )
        collected.extend(seed_demos)

    model, _spec = nets.for_env(env, hidden=hidden)
    result = CloneResult()

    for round_index in range(rounds):
        train, held_out = collected.split(0.2, seed=seed + round_index)
        train_x, train_y = train.tensors()
        test_x, test_y = held_out.tensors()
        fit_result = supervised.fit(
            model,
            train_x,
            train_y,
            validation=(test_x, test_y),
            epochs=epochs,
            seed=seed + round_index,
        )

        report = evaluate_policy(
            env,
            nets.greedy(model),
            episodes=evaluate_episodes,
            seed=seed + 9000 + round_index,
        )
        result.rounds.append(
            {
                "round": float(round_index + 1),
                "demonstrations": float(len(collected)),
                "action_accuracy": supervised.accuracy(model, test_x, test_y),
                "mean_return": report.mean_return,
                "success_rate": report.success_rate,
            }
        )
        result.action_accuracy = result.rounds[-1]["action_accuracy"]
        result.mean_return = report.mean_return
        result.success_rate = report.success_rate
        result.demonstrations = len(collected)
        result.fit = fit_result

        if round_index == rounds - 1:
            break

        # The aggregation step. Run the *policy*, label what it saw.
        policy = nets.greedy(model)
        for episode in range(episodes_per_round):
            observation, _info = env.reset(seed=seed + 20_000 + round_index * 100 + episode)
            while True:
                # The label is the expert's answer *at this state*, while
                # the action taken is the policy's. Using the expert's
                # action to step would only re-record expert states,
                # which is the thing DAgger exists to stop doing.
                collected.add(observation, int(expert(observation)))
                action = int(policy(observation))
                observation, _reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

    return model, result
