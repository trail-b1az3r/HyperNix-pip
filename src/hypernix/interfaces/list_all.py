"""``hnx list-all`` — every command HyperNix installs, in one place (0.72.8a4).

    hnx list-all                 # everything, grouped
    hnx list-all quant           # only what mentions "quant"
    hnx list-all --section t1    # one group: hnx, extra, aliases, t1, other
    hnx list-all --json          # for a script

Four groups: ``hnx``'s own subcommands (the short menu, then ``hnx
extra``, then the other spellings it takes), ``hnx-t1`` / ``hypernix-t1``,
and every other console script the package puts on PATH. The last is
read from the installed package's entry points, or from pyproject.toml in
a checkout, so a script added there shows up here without an edit -- and
tests/interfaces/test_list_all.py fails if the T1 table falls behind the
bash dispatcher it describes.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["Command", "T1_COMMANDS", "collect", "cli_main"]


@dataclass
class Command:
    name: str
    section: str
    what: str = ""
    also: list[str] = field(default_factory=list)
    run: str = ""

    def matches(self, needle: str) -> bool:
        text = " ".join([self.name, self.what, *self.also]).lower()
        return all(word in text for word in needle.lower().split())


#: `hypernix-t1` (also `hnx-t1`): name -> (aliases, description). Kept in
#: step with bin/hypernix-t1's dispatcher by the tests.
T1_COMMANDS: dict[str, tuple[tuple[str, ...], str]] = {
    "setup": (("install",), "the interactive installer (b1): server, access, models and features, then start it"),
    "start": ((), "start the server in the background"),
    "stop": ((), "ask it to stop, and wait"),
    "kill": (("force-stop",), "force it to stop, losing in-flight requests"),
    "restart": ((), "stop (or kill) then start"),
    "upgrade": (("update",), "upgrade, then restart: -S venv (the default) the server's Python, -S env its "
                             ".env, -S all both"),
    "status": ((), "is it running, where, what version, and where HyperLink on the web is"),
    "logs": ((), "the last lines of its log, or follow it with -f"),
    "create": ((), "set up a new server: the installer from a checkout, else a minimal local config"),
    "configure": (("config", "settings"), "edit the configuration; `config web` for the web search settings"),
    "test": ((), "health, status, and a real end-to-end probe"),
    "key": (("keys",), "gkey against this server's key store"),
    "index": ((), "write the model registry from the .gguf files on disk"),
    "sync": ((), "mirror ~/.hypernix/models into the server's models folder"),
    "built-in-runner": (("runner", "builtin-runner"), "serve a model from the server's own llama.cpp"),
    "chat": ((), "send the served model a message and print the reply"),
    "launch-script": ((), "run a script or command so it survives an ssh disconnect"),
    "override": ((), "point LM Studio's models folder at HyperNix's (and back)"),
    "training": ((), "what training is doing on this machine: pause, resume, stop"),
    "autostart": ((), "start at login with a systemd user service"),
    "remove": (("uninstall",), "stop, disable and delete the configuration (keeps keys)"),
    "start-foreground": ((), "run in the foreground (what the systemd unit runs)"),
    "version": (("--version",), "the package and T1 API versions"),
    "help": (("-h", "--help"), "the T1 manager's own help"),
}

#: One line for the console scripts whose module docstring would say it
#: worse. Anything not here is described from its module's docstring.
_OTHER: dict[str, str] = {
    "hypernix": "the main CLI (same as hnx)",
    "hypernix-quantize": "the main CLI, under its original name",
    "hnx": "the main CLI",
    "hypernix-t1": "manage a T1 API server (also hnx-t1)",
    "hnx-t1": "manage a T1 API server: the short name for hypernix-t1",
    "waiter": "the T1 API's client: a TUI and CLI for any T1 server",
    "gkey": "keys and access: Gatekeeper and Keymaster",
    "hyped": "a local AI you can talk to, with nothing to set up",
    "hyped-agent": "the autonomous agent TUI",
    "hyped-pro": "the chat client with tools, on OpenTUI",
    "hyped-plus": "the earlier Node.js hyped TUI (hyped-pro is the OpenTUI one)",
    "hyped-pro-gui": "hyped-pro's desktop GUI (Qt6, GTK4 fallback)",
    "tvtop": "the live training dashboard (cctvtop, in C++)",
    "tvtop-max": "the run's dashboard on OpenTUI, with its modules, model and logs",
    "tvtoppro": "tvtop++'s numbers with btop++'s presentation and themes",
    "tvtop-old": "tvtop++, the btop++-style training dashboard before cctvtop",
    "tvtop-older": "tv, the original btop-style training dashboard",
    "hyprslug": "quantise a GGUF without llama.cpp, sub-bit types included",
    "hyprslug-headers": "self-describing GGUF headers and the sub-bit server",
    "hnx-imatrix": "measure an importance matrix for quantising, or convert one",
    "dflash1": "derive a standalone draft model for speculative decoding",
    "dflash2": "put a draft model inside the model it drafts for",
    "hnx-bundle": "several quants of one model in one file",
    "multilama": "one interface over several llama.cpp builds",
    "steamroller": "the descending llama.cpp quantiser",
    "noodle": "run a swarm of agents from the command line",
    "hnx-map": "a steampunk schematic TUI of the model you are training",
    "hnx-c4": "explode a model; measure, reap and sow its layers and weights",
    "hnx-scriptgen": "build a training script (GUI, or headless)",
    "t1-accounts": "manage a T1 server's web accounts, on the machine they live on",
    "hypernix-sync": "mirror ~/.hypernix/models into the T1 server's folder",
    "eth": "ethanol: turn the GPU clock up (overclocking; read its warning first)",
    "ups": "watch for power or weather trouble and checkpoint training before it hits",
}


def _cli_tables():
    from . import cli

    return cli._MAIN_MENU, cli._EXTRA_MENU, set(cli._SUBCOMMANDS)


#: Subcommands that are another spelling of one in a menu.
_SPELLINGS = {"fllq": "fetch-llama-quantize", "fiz": "fizzle", "camouflage": "camo",
              "protect": "prot", "fuse-box": "fusebox"}


def _hnx_commands() -> list[Command]:
    main, extra, subcommands = _cli_tables()
    out = [Command(name, "hnx", what.replace("\\", ""), run=f"hnx {name}") for name, what in main]
    out += [Command(name, "extra", what.replace("\\", ""), run=f"hnx {name}") for name, what in extra]
    listed = {c.name for c in out}
    # Only spellings that run. cli._COMMAND_ALIASES ("dl", "quantise", ...)
    # are suggestions printed after a typo, so they are not listed.
    for name in sorted(subcommands - listed):
        target = _SPELLINGS.get(name, "")
        out.append(Command(name, "aliases", f"same as `hnx {target}`" if target else "", run=f"hnx {name}"))
    return out


def _t1_commands() -> list[Command]:
    return [Command(name, "t1", what, list(also), run=f"hnx-t1 {name}")
            for name, (also, what) in T1_COMMANDS.items()]


def _pyproject() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "pyproject.toml"
        if candidate.is_file() and (parent / "src" / "hypernix").is_dir():
            return candidate
    return None


def _scripts() -> tuple[dict[str, str], list[str]]:
    """``{console script: "module:function"}`` and the plain script files."""
    scripts: dict[str, str] = {}
    try:
        from importlib.metadata import distribution

        dist = distribution("hypernix")
        for ep in dist.entry_points:
            if ep.group == "console_scripts":
                scripts[ep.name] = ep.value
    except Exception:  # noqa: BLE001 - not installed: read the checkout instead
        pass
    files: list[str] = []
    project = _pyproject()
    if project is not None:
        try:
            import tomllib

            data = tomllib.loads(project.read_text(encoding="utf-8"))
            for name, target in (data.get("project", {}).get("scripts") or {}).items():
                scripts.setdefault(name, target)
            files = [Path(f).name for f in data.get("tool", {}).get("setuptools", {}).get("script-files", [])]
        except (OSError, ValueError):
            pass
    if not files:
        files = ["hypernix-t1", "hnx-t1"]
    return scripts, files


def _docstring_line(target: str) -> str:
    """The first line of a module's docstring, read without importing it."""
    module = target.split(":", 1)[0]
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return ""
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return ""
    try:
        doc = ast.get_docstring(ast.parse(Path(spec.origin).read_text(encoding="utf-8"))) or ""
    except (OSError, SyntaxError, ValueError):
        return ""
    first = doc.strip().splitlines()[0] if doc.strip() else ""
    # "hypernix.x.y — what it does" -> "what it does"
    first = re.sub(r"^[`\w.\-]+(?:\s+\([^)]*\))?\s+[—-]+\s+", "", first).rstrip(".")
    return first[:110]


def _other_commands() -> list[Command]:
    scripts, files = _scripts()
    by_target: dict[str, list[str]] = {}
    for name, target in scripts.items():
        by_target.setdefault(target, []).append(name)
    out = []
    main_cli = {"hypernix", "hnx", "hypernix-quantize"}
    for target, names in by_target.items():
        names = sorted(names, key=lambda n: (n not in _OTHER, len(n), n))
        first, also = names[0], names[1:]
        if first in main_cli:
            continue        # it is the hnx above
        what = next((_OTHER[n] for n in names if n in _OTHER), "") or _docstring_line(target)
        out.append(Command(first, "other", what, also, run=first))
    for name in files:
        if name in ("hypernix-t1", "hnx-t1"):
            continue        # the T1 group
        out.append(Command(name, "other", _OTHER.get(name, ""), run=name))
    out.sort(key=lambda c: c.name)
    return out


SECTIONS = {
    "hnx": "hnx (also hypernix) — the main menu",
    "extra": "hnx extra — more commands",
    "aliases": "hnx — other spellings it takes",
    "t1": "hnx-t1 (also hypernix-t1) — the T1 API server",
    "other": "every other HyperNix command",
}


def collect() -> list[Command]:
    return [*_hnx_commands(), *_t1_commands(), *_other_commands()]


def _print(rows: list[Command]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        Console = None  # noqa: N806
    for section, title in SECTIONS.items():
        part = [r for r in rows if r.section == section]
        if not part:
            continue
        if Console is None:
            print(f"\n{title}")
            for r in part:
                also = f" (also {', '.join(r.also)})" if r.also else ""
                print(f"  {r.run:<34} {r.what}{also}")
            continue
        table = Table(title=title, title_justify="left", title_style="bold cyan",
                      header_style="bold magenta", border_style="cyan", expand=False)
        table.add_column("Command", style="green", no_wrap=True)
        table.add_column("What it does")
        table.add_column("Also", style="dim")
        for r in part:
            # Text, not markup: "[quantize]" in a description is words.
            table.add_row(Text(r.run), Text(r.what), Text(", ".join(r.also)))
        Console().print(table)
    print(f"\n{len(rows)} command(s). `<command> --help` for a command's flags.")


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hnx list-all",
                                     description="Every HyperNix command: hnx, hnx extra, hnx-t1, and the rest.")
    parser.add_argument("search", nargs="*", help="only commands whose name or description has these words")
    parser.add_argument("--section", choices=list(SECTIONS), help="one group only")
    parser.add_argument("--json", action="store_true", help="machine-readable")
    args = parser.parse_args(argv)
    rows = collect()
    if args.section:
        rows = [r for r in rows if r.section == args.section]
    if args.search:
        rows = [r for r in rows if r.matches(" ".join(args.search))]
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
        return 0
    if not rows:
        print("nothing matches. `hnx list-all` with no words lists everything.", file=sys.stderr)
        return 1
    _print(rows)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
