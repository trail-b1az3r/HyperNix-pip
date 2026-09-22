"""``hypernix neuron`` — train a small policy from the command line.

Four verbs matching the four ways to train, plus ``demo`` for recording
what an expert does. Every one of them takes ``--env`` and writes a
checkpoint that :func:`hypernix.neuron.nets.load` can reopen without the
script that made it.

The environments here are the built-in ones. Pointing this at a real
task means importing the package and passing your own environment —
anything with ``reset``/``step`` and the two spaces, which a Gymnasium
environment already is. The CLI exists so the algorithms can be tried,
compared and sanity-checked on something known-solvable before anybody
wires them to a robot.
"""
from __future__ import annotations

import argparse
import json
import sys

__all__ = ["main", "cli_main", "build_parser"]

_EPILOG = """\
Examples:

  hypernix neuron demo --env grid --episodes 50 -o demos.npz
  hypernix neuron clone --env grid --demos demos.npz -o policy.pt
  hypernix neuron dagger --env grid --rounds 5 -o policy.pt
  hypernix neuron rl --env beam --algorithm reinforce -o policy.pt
  hypernix neuron eval --env grid --policy policy.pt

Reach for `clone` before `rl`. If you can demonstrate the task, cloning
gets a working policy in minutes from data you can produce by hand;
reinforcement learning needs orders of magnitude more interaction and
fails in ways that look like bugs.
"""


def _env(name: str, max_steps: int | None = None):
    from .envs import BalanceBeam, GridWorld

    if name == "grid":
        return GridWorld(size=5, max_steps=max_steps or 50)
    if name == "beam":
        return BalanceBeam(max_steps=max_steps or 200)
    raise SystemExit(
        f"unknown environment {name!r}; the built-in ones are 'grid' and "
        f"'beam'. For your own, import hypernix.neuron and pass it directly."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypernix neuron",
        description="Train small networks that act: imitation, RL, supervised.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    def common(action: argparse.ArgumentParser) -> None:
        action.add_argument("--env", default="grid", help="grid or beam")
        action.add_argument("--max-steps", type=int, default=None)
        action.add_argument("--seed", type=int, default=0)
        action.add_argument("--json", action="store_true",
                            help="print the report as JSON")

    record = sub.add_parser("demo", help="record an expert's play")
    common(record)
    record.add_argument("--episodes", type=int, default=50)
    record.add_argument("-o", "--output", default="demos.npz")

    cloning = sub.add_parser("clone", help="behaviour cloning from demonstrations")
    common(cloning)
    cloning.add_argument("--demos", default="",
                         help="a .npz from `demo` (default: record some now)")
    cloning.add_argument("--episodes", type=int, default=50,
                         help="episodes to record when --demos is not given")
    cloning.add_argument("--epochs", type=int, default=100)
    cloning.add_argument("-o", "--output", default="policy.pt")

    aggregated = sub.add_parser("dagger", help="cloning that fixes its own mistakes")
    common(aggregated)
    aggregated.add_argument("--rounds", type=int, default=5)
    aggregated.add_argument("--episodes", type=int, default=10,
                            help="episodes per round")
    aggregated.add_argument("-o", "--output", default="policy.pt")

    learning = sub.add_parser("rl", help="reinforcement learning")
    common(learning)
    learning.add_argument("--algorithm", default="dqn", choices=("dqn", "reinforce"))
    learning.add_argument("--steps", type=int, default=20_000,
                          help="dqn only: environment steps")
    learning.add_argument("--episodes", type=int, default=600,
                          help="reinforce only: episodes")
    learning.add_argument("-o", "--output", default="policy.pt")

    scoring = sub.add_parser("eval", help="run a saved policy and report")
    common(scoring)
    scoring.add_argument("--policy", required=True)
    scoring.add_argument("--episodes", type=int, default=50)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    from . import nets
    from .data import Demonstrations, record_expert
    from .evaluate import evaluate_policy

    env = _env(args.env, args.max_steps)

    if args.command == "demo":
        demonstrations, trajectories = record_expert(
            env, episodes=args.episodes, seed=args.seed
        )
        path = demonstrations.save(args.output)
        mean = sum(t.total_reward for t in trajectories) / len(trajectories)
        report = {
            "demonstrations": len(demonstrations),
            "episodes": len(trajectories),
            "expert_mean_return": round(mean, 4),
            "path": str(path),
        }
        _print(report, args.json,
               f"{len(demonstrations)} demonstrations from {len(trajectories)} "
               f"episodes (expert return {mean:.3f}) -> {path}")
        return 0

    if args.command == "clone":
        from .imitation import clone

        if args.demos:
            demonstrations = Demonstrations.load(args.demos)
        else:
            demonstrations, _ = record_expert(
                env, episodes=args.episodes, seed=args.seed
            )
        model, result = clone(
            env, demonstrations, epochs=args.epochs, seed=args.seed
        )
        path = nets.save(model, nets.for_env(env)[1], args.output)
        report = {
            "action_accuracy": round(result.action_accuracy, 4),
            "mean_return": round(result.mean_return, 4),
            "success_rate": result.success_rate,
            "demonstrations": result.demonstrations,
            "compounding_error_suspected": result.looks_like_compounding_error,
            "path": str(path),
        }
        text = f"{result.summary()} -> {path}"
        if result.looks_like_compounding_error:
            text += (
                "\n\nHigh action accuracy and poor return: the policy copies "
                "the expert on states the expert visited and is lost the "
                "moment it strays. More epochs will not fix that. Try "
                "`hypernix neuron dagger`."
            )
        _print(report, args.json, text)
        return 0

    if args.command == "dagger":
        from .imitation import dagger

        model, result = dagger(
            env,
            rounds=args.rounds,
            episodes_per_round=args.episodes,
            seed=args.seed,
        )
        path = nets.save(model, nets.for_env(env)[1], args.output)
        report = {
            "rounds": result.rounds,
            "action_accuracy": round(result.action_accuracy, 4),
            "mean_return": round(result.mean_return, 4),
            "success_rate": result.success_rate,
            "path": str(path),
        }
        lines = [
            f"  round {int(r['round'])}: {int(r['demonstrations'])} demos, "
            f"return {r['mean_return']:.3f}, success {r['success_rate']:.0%}"
            for r in result.rounds
        ]
        _print(report, args.json,
               "\n".join(lines) + f"\n\n{result.summary()} -> {path}")
        return 0

    if args.command == "rl":
        from .rl import dqn, reinforce

        if args.algorithm == "dqn":
            model, result = dqn(env, steps=args.steps, seed=args.seed)
        else:
            model, result = reinforce(env, episodes=args.episodes, seed=args.seed)
        path = nets.save(model, nets.for_env(env)[1], args.output)
        report = {
            "algorithm": args.algorithm,
            "steps": result.steps,
            "episodes": result.episodes,
            "best_return": round(result.best_return, 4),
            "final_return": round(result.final_return, 4),
            "best_success_rate": result.best_success_rate,
            "history": result.history,
            "path": str(path),
        }
        _print(report, args.json, f"{result.summary()} -> {path}")
        return 0

    if args.command == "eval":
        try:
            model, _spec = nets.load(args.policy)
        except (OSError, ValueError) as exc:
            print(f"hypernix neuron eval: {exc}", file=sys.stderr)
            return 1
        report_obj = evaluate_policy(
            env, nets.greedy(model), episodes=args.episodes, seed=args.seed
        )
        _print(report_obj.to_dict(), args.json, report_obj.summary())
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


def _print(payload: dict, as_json: bool, text: str) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(text)


def cli_main(argv: list[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":
    sys.exit(main())
