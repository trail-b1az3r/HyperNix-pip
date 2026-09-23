"""``hypernix dilute`` — best-of-n sampling from the command line.

Two verbs. ``run`` collects a fixed number of traces and writes them at
the end; ``jit`` streams them to the file as they are made, so a run
that is killed at hour three leaves three hours of traces behind rather
than nothing. Prefer ``jit`` for anything long.

``inspect`` reads a trace file back and says whether it is worth
training on — chiefly by the winning margin, because a trace set whose
margins are all zero is a set of random picks wearing an evaluator's
name, and that is not visible from the line count.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main", "cli_main", "build_parser"]

_EPILOG = """\
Examples:

  hypernix dilute run  --prompts prompts.txt -o traces.jsonl --traces 200
  hypernix dilute jit  --prompts prompts.txt -o traces.jsonl --judge-model qwen3
  hypernix dilute inspect traces.jsonl

The evaluator is the part that decides what you get. `--judge` uses a
model, which is the honest default; `--length` is a smoke test and will
happily teach a model to ramble if you train on what it picks.
"""


def _prompts(path: str) -> list[str]:
    """One prompt per line, or a JSON/JSONL file with a `prompt` field."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise SystemExit(f"no prompt file at {source}")
    text = source.read_text(encoding="utf-8")
    if source.suffix == ".json":
        loaded = json.loads(text)
        rows = loaded if isinstance(loaded, list) else loaded.get("prompts", [])
        return [r if isinstance(r, str) else str(r.get("prompt", "")) for r in rows]
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                out.append(str(json.loads(line).get("prompt", "")))
                continue
            except json.JSONDecodeError:
                pass
        out.append(line)
    return [p for p in out if p]


def _config(args) -> "object":
    from .core import DiluteConfig

    return DiluteConfig(
        samples_per_prompt=args.samples,
        target_traces=args.traces,
        max_attempts=args.tries,
        min_score=args.min_score,
        min_margin=args.min_margin,
        seed=args.seed,
        output="",          # written by the verb, so jit can stream
    )


def _evaluator(args, generate):
    from .evaluators import LengthEvaluator, ModelEvaluator

    if args.length:
        return LengthEvaluator(target=args.length)
    if args.judge_model or args.judge_server:
        from .backends import resolve_generator

        judge = resolve_generator(
            args.judge_model, server=args.judge_server, token=args.token,
            max_tokens=256,
        )
    else:
        # The model judging its own samples. Not as odd as it sounds:
        # picking the better of two answers is an easier task than
        # writing one, and it needs no second model in memory.
        judge = generate
    return ModelEvaluator(judge, rubric=args.rubric)


def _generator(args):
    from .backends import GeneratorError, resolve_generator

    try:
        return resolve_generator(
            args.model, server=args.server, token=args.token,
            max_tokens=args.max_tokens,
        )
    except GeneratorError as exc:
        raise SystemExit(str(exc))


def _run(args) -> int:
    from .core import dilute

    prompts = _prompts(args.prompts)
    if not prompts:
        raise SystemExit(f"{args.prompts} has no prompts in it")
    generate = _generator(args)
    evaluator = _evaluator(args, generate)
    config = _config(args)

    try:
        result = dilute(prompts, generate, evaluator, config)
    finally:
        close = getattr(generate, "close", None)
        if callable(close):
            close()

    if args.output:
        path = result.write_jsonl(args.output)
        print(f"wrote {path}")
    print(result.summary())
    if not result.reached_target:
        # Not an error: a spent budget is an outcome, and the file is
        # still worth having. Said plainly so nobody trains on half a
        # set thinking it is whole.
        print(
            f"note: asked for {config.target_traces}, got {len(result.traces)}"
            f" ({result.stopped_because})"
        )
    return 0


def _jit(args) -> int:
    from .core import DiluteResult, jit_distil

    prompts = _prompts(args.prompts)
    if not prompts:
        raise SystemExit(f"{args.prompts} has no prompts in it")
    generate = _generator(args)
    evaluator = _evaluator(args, generate)
    config = _config(args)

    target = Path(args.output).expanduser() if args.output else None
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
    handle = target.open("w", encoding="utf-8") if target else None
    kept = 0
    margins: list[float] = []

    def sink(trace) -> None:
        nonlocal kept
        kept += 1
        margins.append(trace.margin)
        if handle is not None:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
            # Flushed per trace on purpose: the reason to use jit is
            # that a killed run keeps what it made, and a buffer holding
            # the last few thousand traces gives that away.
            handle.flush()
        if not args.quiet:
            print(f"[{kept}/{config.target_traces}] margin {trace.margin:+.3f}",
                  file=sys.stderr)

    try:
        for _ in jit_distil(prompts, generate, evaluator, config, sink=sink):
            pass
    except KeyboardInterrupt:
        print("\nstopped; traces written so far are complete", file=sys.stderr)
    finally:
        if handle is not None:
            handle.close()
        close = getattr(generate, "close", None)
        if callable(close):
            close()

    mean = sum(margins) / len(margins) if margins else 0.0
    print(f"{kept} trace(s), mean margin {mean:.3f}"
          + (f", written to {target}" if target else ""))
    return 0


def _inspect(args) -> int:
    path = Path(args.file).expanduser()
    if not path.is_file():
        raise SystemExit(f"no trace file at {path}")

    count = 0
    margins: list[float] = []
    scores: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "_dilute" in row:
            continue
        count += 1
        margins.append(float(row.get("margin", 0.0)))
        scores.append(float((row.get("chosen") or {}).get("score", 0.0)))

    if not count:
        print("no traces in that file")
        return 1

    mean_margin = sum(margins) / count
    zero = sum(1 for m in margins if m == 0.0)
    print(f"{count} trace(s)")
    print(f"mean chosen score  {sum(scores) / count:.3f}")
    print(f"mean margin        {mean_margin:.3f}")
    print(f"zero-margin        {zero} ({100 * zero / count:.0f}%)")
    if mean_margin <= 0.01:
        print(
            "\nThe evaluator is not separating the samples. These are "
            "effectively random picks from the ladder, and training on "
            "them will not beat training on one sample per prompt. Use a "
            "stronger judge, or --min-margin to drop the ties."
        )
    elif zero > count // 2:
        print(
            "\nOver half of these were ties broken at random. Worth a "
            "sharper rubric or a finer scale before training."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypernix dilute",
        description="Generate several answers per prompt, keep the best.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def shared(p: argparse.ArgumentParser) -> None:
        p.add_argument("--prompts", required=True,
                       help="file of prompts, one per line (or .json/.jsonl)")
        p.add_argument("-o", "--output", default="",
                       help="where to write the traces (JSONL)")
        p.add_argument("--model", default="",
                       help="GGUF name or path; default is the first in "
                            "~/.hypernix/models")
        p.add_argument("--server", default="",
                       help="T1 server URL; preferred over loading a GGUF")
        p.add_argument("--token", default="", help="bearer token for --server")
        p.add_argument("--samples", type=int, default=5,
                       help="completions per prompt (4-6)")
        p.add_argument("--traces", type=int, default=100,
                       help="how many traces to collect")
        p.add_argument("--tries", type=int, default=500,
                       help="prompts to try before giving up")
        p.add_argument("--max-tokens", type=int, default=512)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--min-score", type=float, default=None,
                       help="drop a trace whose winner scores below this")
        p.add_argument("--min-margin", type=float, default=None,
                       help="drop a trace the evaluator could not separate")
        p.add_argument("--judge-model", default="",
                       help="a second model to score with; default is the "
                            "generating model judging its own samples")
        p.add_argument("--judge-server", default="")
        p.add_argument("--rubric", default="",
                       help="what the judge should reward")
        p.add_argument("--length", type=int, default=0,
                       help="score by closeness to this length instead of "
                            "using a model. A smoke test, not a judge.")

    run = sub.add_parser("run", help="collect traces, write at the end")
    shared(run)
    run.set_defaults(func=_run)

    jit = sub.add_parser("jit", help="stream traces to the file as they are made")
    shared(jit)
    jit.add_argument("-q", "--quiet", action="store_true")
    jit.set_defaults(func=_jit)

    inspect = sub.add_parser("inspect", help="is this trace file worth training on")
    inspect.add_argument("file")
    inspect.set_defaults(func=_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "rubric", ""):
        from .evaluators import DEFAULT_RUBRIC

        args.rubric = DEFAULT_RUBRIC
    return int(args.func(args) or 0)


def cli_main(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
