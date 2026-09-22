#!/usr/bin/env python3
"""Teach a policy to play, by playing it yourself first.

The shortest route to a working game bot, and the one people skip
because reinforcement learning is the thing they have heard of. If you
can play the game, you can produce demonstrations — and cloning them
takes a minute on a CPU where RL takes hours on a GPU and often does
not converge.

Run it:

    python examples/neuron/game_automation.py

What it shows, in order:

1. Recording an expert. Here the "expert" is a scripted player so the
   example runs anywhere; in anger it is you with a gamepad, or a slow
   planner you want to distil into something fast enough for the frame
   budget.
2. Cloning it, and getting *two* numbers back — action accuracy on
   held-out demonstrations, and mean return from actually running the
   policy. Those come apart constantly.
3. What to do when they disagree, which is the whole point.

Pointing this at a real game means replacing GridWorld with anything
that has `reset`/`step` and the two spaces — a screen-capture wrapper,
a Gymnasium env, an emulator bridge. Nothing below knows what the
environment is.
"""
from __future__ import annotations

from hypernix.neuron import nets
from hypernix.neuron.data import record_expert
from hypernix.neuron.envs import GridWorld
from hypernix.neuron.evaluate import compare
from hypernix.neuron.imitation import clone, dagger


def main() -> int:
    env = GridWorld(size=5, max_steps=50)

    print("1. Recording 40 episodes of expert play")
    demonstrations, trajectories = record_expert(env, episodes=40, seed=0)
    expert_return = sum(t.total_reward for t in trajectories) / len(trajectories)
    print(f"   {len(demonstrations)} (observation, action) pairs")
    print(f"   the expert scores {expert_return:.3f} per episode\n")

    print("2. Cloning it")
    policy, result = clone(env, demonstrations, epochs=200, seed=0)
    print(f"   {result.summary()}")
    print(f"   {nets.count_parameters(policy):,} parameters\n")

    # The check worth making. High accuracy with a poor return means
    # the policy copies the expert on states the expert visited and is
    # lost the moment it strays -- compounding error, and more epochs
    # will not fix it.
    if result.looks_like_compounding_error:
        print("3. High accuracy, poor return: compounding error.")
        print("   Running DAgger, which asks the expert about the states")
        print("   the *policy* reaches rather than the ones it visited.\n")
        policy, result = dagger(env, rounds=4, episodes_per_round=10, seed=0)
        for row in result.rounds:
            print(f"   round {int(row['round'])}: "
                  f"{int(row['demonstrations'])} demos, "
                  f"return {row['mean_return']:.3f}, "
                  f"success {row['success_rate']:.0%}")
        print()
    else:
        print("3. Accuracy and return agree — no compounding error here.\n")

    print("4. Against the expert, on identical episodes")
    reports = compare(
        env,
        {
            "cloned": nets.greedy(policy),
            "expert": lambda _obs: env.expert_action(),
            "random": lambda _obs: 0,
        },
        episodes=30,
        seed=9000,
    )
    for name, report in reports.items():
        print(f"   {name:8} {report.summary()}")

    print("\n5. Saving it")
    path = nets.save(policy, nets.for_env(env)[1], "game_policy.pt")
    print(f"   {path} — carries its own architecture, so nets.load()")
    print("   reopens it without this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
