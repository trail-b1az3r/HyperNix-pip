"""scriptgen.cli — ``hnx scriptgen``.

Opens the GUI by default and falls back to the command line when there
is no display. That fallback is the point: the machine with the GPU is
frequently a headless box, and a script builder that only works where
there is a window is a script builder you use on the wrong machine and
then scp from.

Since 0.72.8rc1 the fallback, in a terminal, is a wizard: five
questions (the model, the data, the hardware, the goal, where to write
it) and a script that runs. ``hnx scriptgen train.py`` writes one
without asking; ``--recipe`` starts from a ready-made setup; ``--set``
suggests the setting a typo meant; ``--explain`` says what one does.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .app import FormModel, launch
from .params import ALL_PARAMS, GROUPS, search, suggest
from .recipes import RECIPES, get_recipe
from .templates import CONFIG_FORMATS
from .widgets import tk_available

__all__ = ["main", "cli_main", "wizard"]


def _parse_set(pairs: list[str]) -> dict[str, Any]:
    """``--set learning_rate=3e-5`` into a values dict."""
    values: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set needs name=value, got {pair!r}")
        name, _, raw = pair.partition("=")
        name = name.strip().replace("-", "_")
        if name not in ALL_PARAMS:
            close = suggest(name)
            raise SystemExit(
                f"Unknown parameter {name!r}"
                + (f". Did you mean: {', '.join(close)}?" if close else "")
                + "  (hnx scriptgen --list-params lists them all)"
            )
        param = ALL_PARAMS[name]
        raw = raw.strip()
        if param.kind.value == "range":
            values[name] = tuple(part.strip() for part in raw.split(","))
        else:
            values[name] = raw
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hnx scriptgen",
        description="Build a HyperNix training script that runs: in a window, in a terminal "
                    "wizard, or straight from flags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  hnx scriptgen                                   the window (or the wizard, headless)\n"
            "  hnx scriptgen --wizard                          five questions, then a script\n"
            "  hnx scriptgen train.py                          defaults, written to train.py\n"
            "  hnx scriptgen train.py --recipe quick-test --set dataset=data.jsonl\n"
            "  hnx scriptgen train.py --set learning_rate=3e-5 --set epochs=2\n"
            "  hnx scriptgen --inject existing.py              update its config block\n"
            "  hnx scriptgen --config-only --format yaml       just the values\n"
            "  hnx scriptgen --list-recipes                    the starting points\n"
            "  hnx scriptgen --list-params lora                settings that mention LoRA\n"
            "  hnx scriptgen --explain warmup_ratio            what one setting does\n"
        ),
    )
    parser.add_argument("output_path", nargs="?", metavar="OUTPUT",
                        help="write the script here, without the GUI (same as -o)")
    parser.add_argument("--cli", action="store_true", help="skip the GUI")
    parser.add_argument("--wizard", action="store_true",
                        help="answer five questions in the terminal")
    parser.add_argument("-o", "--output", help="write the script here")
    parser.add_argument("--recipe", metavar="NAME",
                        help="start from a recipe (see --list-recipes)")
    parser.add_argument("--inject", metavar="SCRIPT",
                        help="inject a config block into an existing script")
    parser.add_argument("--config-only", action="store_true",
                        help="emit only the config, not a whole script")
    parser.add_argument("--format", choices=CONFIG_FORMATS, default="python",
                        help="with --config-only (default: python)")
    parser.add_argument("--set", action="append", dest="sets", metavar="NAME=VALUE",
                        help="set a parameter (repeatable); a range takes LOW,HIGH")
    parser.add_argument("--preset", help="load a preset .json first")
    parser.add_argument("--changes", action="store_true",
                        help="print what differs from the starting point, then exit")
    parser.add_argument("--check", action="store_true",
                        help="validate and print problems, then exit")
    parser.add_argument("--list-params", nargs="?", const="", metavar="QUERY",
                        help="list the parameters (or those matching QUERY), then exit")
    parser.add_argument("--list-recipes", action="store_true", help="list the recipes, then exit")
    parser.add_argument("--explain", metavar="NAME", help="explain one parameter, then exit")
    args = parser.parse_args(argv)

    if args.list_params is not None:
        return _list_params(args.list_params)
    if args.list_recipes:
        return _list_recipes()
    if args.explain:
        return _explain(args.explain)
    if args.output_path and args.output:
        parser.error("give the output once: as OUTPUT or as -o")
    output = args.output or args.output_path

    model = FormModel()
    if args.recipe:
        try:
            model.apply_recipe(get_recipe(args.recipe).key)
        except KeyError as exc:
            print(f"scriptgen: {exc.args[0]}", file=sys.stderr)
            return 2
    if args.preset:
        try:
            rejected = model.load_json(Path(args.preset).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"scriptgen: {exc}", file=sys.stderr)
            return 1
        for message in rejected:
            print(f"scriptgen: skipped — {message}", file=sys.stderr)

    problems = []
    for name, raw in _parse_set(args.sets or []).items():
        ok, message = model.set(name, raw)
        if not ok:
            problems.append(message)
    if problems:
        for message in problems:
            print(f"scriptgen: {message}", file=sys.stderr)
        return 1

    if args.changes:
        changed = model.changed_from_defaults()
        start = f"the {model.recipe} recipe" if model.recipe else "the defaults"
        print(f"{len(changed)} setting(s) differ from {start}" + (":" if changed else "."))
        for name, (old, new) in changed.items():
            print(f"  {name:26} {old!r} -> {new!r}")
        return 0

    if args.inject:
        model.mode = "inject"
        model.target_path = Path(args.inject)
    elif args.config_only:
        model.mode = "config"
        model.config_format = args.format

    if args.wizard:
        return wizard(model, output=output)

    wants_gui = not (args.cli or output or args.inject or args.config_only or args.check)
    if wants_gui:
        ok, reason = tk_available()
        if ok:
            return launch(model.values, recipe=model.recipe)
        # Not an error: a headless machine is the normal case here.
        print(f"scriptgen: no window available — {reason}", file=sys.stderr)
        if sys.stdin.isatty() and sys.stdout.isatty():
            print("scriptgen: asking in the terminal instead.\n", file=sys.stderr)
            return wizard(model)
        print("scriptgen: falling back to command-line generation.\n", file=sys.stderr)

    errors, warnings = model.validate()
    for warning in warnings:
        print(f"scriptgen: note — {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"scriptgen: {error}", file=sys.stderr)
        return 1
    if args.check:
        print(f"scriptgen: no problems ({len(warnings)} note(s))", file=sys.stderr)
        return 0

    existing = ""
    if args.inject:
        target = Path(args.inject)
        if not target.exists():
            print(f"scriptgen: {args.inject} does not exist", file=sys.stderr)
            return 1
        existing = target.read_text(encoding="utf-8", errors="replace")

    text = model.preview(existing)
    destination = output or (args.inject if args.inject else "")
    if destination:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text(text, encoding="utf-8")
        print(f"scriptgen: wrote {destination} ({len(text.splitlines())} lines)")
        if model.mode == "generate":
            print(f"scriptgen: run it with  python {destination}")
    else:
        print(text)
    return 0


# ---------------------------------------------------------------------------
# The wizard
# ---------------------------------------------------------------------------

_HARDWARE = (
    ("An NVIDIA GPU (RTX, A-series, …)", {"device": "cuda", "precision": "auto", "pascal_autotune": False}),
    ("A GTX 10-series or other Pascal card", {"device": "cuda", "precision": "fp16_fp32_compute",
                                              "pascal_autotune": True, "optimizer": "Agedcookerv5"}),
    ("Apple silicon", {"device": "mps", "precision": "fp32", "pascal_autotune": False}),
    ("The CPU only", {"device": "cpu", "precision": "fp32", "pascal_autotune": False}),
)

_GOALS = (
    ("Fine-tune it on my text", "hypernix-finetune"),
    ("LoRA: low memory, for a bigger model", "lora-low-vram"),
    ("Continued pretraining on raw text", "continued-pretraining"),
    ("A quick smoke test (20 steps)", "quick-test"),
    ("Find a good learning rate first", "lr-sweep"),
)


def _choose(ask: Callable[[str], str], say: Callable[[str], None], question: str,
            options: list[str], default: int = 1) -> int:
    say(question)
    for number, text in enumerate(options, start=1):
        say(f"  {number}) {text}")
    while True:
        answer = ask(f"Choose 1-{len(options)} [{default}]: ").strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer)
        say(f"  Type a number from 1 to {len(options)}.")


def wizard(model: FormModel | None = None, *, output: str | None = None,
           ask: Callable[[str], str] = input, say: Callable[[str], None] = print) -> int:
    """Five questions in the terminal, then a script. *ask* and *say* are
    ``input`` and ``print`` unless a test supplies its own."""
    model = model or FormModel()
    say("hnx scriptgen — a training script in five questions. Enter takes the default.\n")
    try:
        goal = _choose(ask, say, "1. What do you want to do?", [g[0] for g in _GOALS])
        model.apply_recipe(_GOALS[goal - 1][1])

        say("")
        default_model = str(model.get("model_id"))
        answer = ask(f"2. Which model? A Hugging Face repo id or a folder [{default_model}]: ").strip()
        if answer:
            model.set("model_id", answer)

        say("")
        while True:
            answer = ask("3. Your data: a .jsonl / .json / .txt file or a Hugging Face dataset "
                         "(Enter to fill in later): ").strip()
            if not answer or "/" in answer or Path(answer).expanduser().exists():
                break
            say(f"  {answer} is not a file here, and a Hub dataset is named owner/name.")
        if answer:
            model.set("dataset", answer)

        say("")
        hardware = _choose(ask, say, "4. What will it train on?", [h[0] for h in _HARDWARE])
        for name, value in _HARDWARE[hardware - 1][1].items():
            if model.recipe == "quick-test" and name == "device":
                continue  # a smoke test stays on the CPU whatever the box has
            model.set(name, value)

        say("")
        target = output or ask("5. Write it to [train.py]: ").strip() or "train.py"
    except (EOFError, KeyboardInterrupt):
        say("\nscriptgen: stopped; nothing written.")
        return 130

    errors, warnings = model.validate()
    say("")
    for warning in warnings:
        say(f"  note: {warning}")
    if errors:
        for error in errors:
            say(f"  problem: {error}")
        return 1
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.preview(), encoding="utf-8")
    summary = model.summary()
    say(f"Wrote {path}: {model.get('model_id')} on {summary['device']}, {summary['optimizer']}, "
        f"effective batch {summary['effective_batch']}"
        + (", LoRA" if summary["lora"] else "") + ".")
    data = "" if model.get("dataset") else " --dataset YOUR_DATA.jsonl"
    say(f"Run it:  python {path}{data}")
    say("Change anything later with --name value, e.g.  python "
        f"{path} --learning_rate 1e-4")
    return 0


# ---------------------------------------------------------------------------
# Listings
# ---------------------------------------------------------------------------

def _bounds(param: Any) -> str:
    if param.minimum is not None or param.maximum is not None:
        return f"[{param.minimum} … {param.maximum}]"
    if param.choices:
        return "{" + ", ".join(c[0] for c in param.choices) + "}"
    return ""


def _list_params(query: str = "") -> int:
    wanted = {p.name for p in search(query)} if query else None
    shown = 0
    for group in GROUPS:
        params = [p for p in group.params if wanted is None or p.name in wanted]
        if not params:
            continue
        print(f"\n{group.title}  —  {group.description}")
        for param in params:
            flag = " (advanced)" if param.advanced else ""
            print(f"  {param.name:26} {str(param.default):<30} {_bounds(param)}{flag}")
            if param.hint:
                print(f"  {'':26} {param.hint}")
            shown += 1
    if query and not shown:
        close = suggest(query)
        print(f"No setting mentions {query!r}." + (f" Did you mean: {', '.join(close)}?" if close else ""))
        return 1
    return 0


def _list_recipes() -> int:
    for recipe in RECIPES:
        print(f"  {recipe.key:24} {recipe.title}")
        print(f"  {'':24} {recipe.description}")
    print("\nUse one:  hnx scriptgen train.py --recipe NAME")
    return 0


def _explain(name: str) -> int:
    key = name.strip().replace("-", "_")
    param = ALL_PARAMS.get(key)
    if param is None:
        close = suggest(key)
        print(f"Unknown parameter {name!r}" + (f". Did you mean: {', '.join(close)}?" if close else ""),
              file=sys.stderr)
        return 1
    group = next(g for g in GROUPS if param in g.params)
    print(f"{param.label}  ({param.name}, in {group.title})")
    print(f"  type     {param.kind.value}")
    print(f"  default  {param.default!r}")
    if _bounds(param):
        print(f"  range    {_bounds(param)}")
    if param.depends_on:
        print(f"  applies  when {param.depends_on.replace('=', ' is ').replace('|', ' or ')}")
    if param.hint:
        print(f"\n  {param.hint}")
    for key_, text in param.choices:
        print(f"    {key_:22} {text}")
    print(f"\n  hnx scriptgen train.py --set {param.name}=VALUE")
    return 0


def cli_main() -> None:
    raise SystemExit(main())
