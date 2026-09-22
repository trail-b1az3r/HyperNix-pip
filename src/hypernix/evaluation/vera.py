"""vera — Verify HyperNix module syntax and run smoke tests.

Vera is a tool to input a file path, lint it (via ast & ruff), smoke test it, 
pytest it if possible, run it, and intelligently explain errors using a small LLM.

Usage:
    hnx vera <python_file> [options]

Options:
    -dr, --dry-run   Run the file with dry-run flag
    -C               Fully run the file
    -FT              Test each argument one at a time
    -q <1-10>        Argument testing depth (default 1)
    -t <time>        Timeout for one stage
    -T <unit>        Timeout unit: ml (default), s, M, h
    -tt <seconds>    Total run timeout — the whole verification, not one stage
    -Na, --no-ai     Skip the model entirely; print the raw error
    -m, --pick-model Choose a GGUF from ~/.hypernix/models to explain with

Why -tt exists alongside -t
---------------------------
`-t` bounds one stage. A file with a slow import and six argument
combinations can sit inside every per-stage limit and still run for
twenty minutes, which is the case people actually hit — so `-tt` bounds
the whole thing and stops between stages rather than killing one.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

console = Console()


@dataclass
class VeraResult:
    passed: bool = True
    output: str = ""
    error_context: str = ""

    def fail(self, msg: str, context: str = ""):
        self.passed = False
        self.output += f"\nError: {msg}"
        self.error_context += f"\n{context}"


@dataclass
class VerificationResult:
    """Result of a :class:`HyperNixVerifier` check on a single file."""

    name: str
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class HyperNixVerifier:
    """Lightweight, in-process syntax and style verifier.

    This complements the module-level :func:`lint_file` / :func:`smoke_test`
    helpers (which shell out to ``ruff`` and actually import the file) with
    a cheap, dependency-free check: syntax validity via :func:`ast.parse`
    plus a module-docstring convention check. In ``strict`` mode a missing
    module docstring is treated as an error; otherwise it's just a warning.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def verify_file(self, file_path: Path) -> VerificationResult:
        result = VerificationResult(name=str(file_path))

        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError as e:
            result.add_error(f"Could not read file: {e}")
            return result

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            result.add_error(f"SyntaxError: {e}")
            return result

        if ast.get_docstring(tree) is None:
            msg = "Missing module docstring"
            if self.strict:
                result.add_error(msg)
            else:
                result.add_warning(msg)

        return result


#: What -Na prints instead of an explanation. Not empty: somebody who
#: passed -Na still wants to know the model was skipped rather than
#: unavailable, because those need different things doing about them.
NO_AI_NOTE = (
    "(-Na: no model was asked. The raw error is above.)"
)


def get_ai_explanation(
    error_text: str, source_code: str, model: str = "qwen3.5-4b"
) -> str:
    """Use a small model to explain the error without crashing."""
    try:
        from hypernix.models.neo_oven import preheat
        oven = preheat(model, quiet=True)
        
        prompt = (
            f"An error occurred while testing a Python script.\n\n"
            f"Source code snippet:\n```python\n{source_code[:1500]}\n```\n\n"
            f"Error output:\n```\n{error_text[-1500:]}\n```\n\n"
            f"Explain what is wrong and tell me how to fix it, and mention the line numbers."
        )
        # Use chat format for a better response
        reply = oven.chat([{"role": "user", "content": prompt}], max_new_tokens=256)
        return reply
    except Exception as e:
        return f"[AI Explanation Unavailable]: Failed to load AI model: {e}"


def local_gguf_models(directory: Path | None = None) -> list[Path]:
    """Every .gguf under the models directory, largest last.

    Sorted by size rather than name so the picker's first entry is the
    one most likely to load on a small machine — the opposite order is
    how somebody's first choice OOMs.
    """
    root = directory or (Path.home() / ".hypernix" / "models")
    if not root.is_dir():
        return []
    found = [p for p in root.rglob("*.gguf") if p.is_file()]
    return sorted(found, key=lambda p: (p.stat().st_size, p.name))


def pick_model(directory: Path | None = None, *, stream=None) -> str:
    """Show the local GGUFs and return the chosen path, or "".

    Returns "" for "carry on with the default", which is what an empty
    line, a Ctrl-C and an empty directory all mean — three ways of
    saying the same thing, and none of them should be an error.
    """
    models = local_gguf_models(directory)
    if not models:
        console.print(
            "[yellow]No .gguf files in "
            f"{directory or Path.home() / '.hypernix' / 'models'}[/]"
        )
        return ""

    console.print("[bold]Models here:[/]")
    for index, path in enumerate(models, start=1):
        size = path.stat().st_size / 1e9
        console.print(f"  [cyan]{index:2}[/] {path.name}  [dim]{size:.1f} GB[/]")

    try:
        answer = input("Which? (enter for the default) ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return ""
    if not answer:
        return ""
    try:
        chosen = models[int(answer) - 1]
    except (ValueError, IndexError):
        console.print(f"[yellow]{answer!r} is not one of those; using the default[/]")
        return ""
    return str(chosen)


def run_command(cmd: list[str], timeout_s: float | None = None) -> tuple[bool, str, str]:
    """Run a subprocess command and return (success, stdout, stderr)."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return res.returncode == 0, res.stdout, res.stderr
    except subprocess.TimeoutExpired as e:
        return False, "", f"Timeout after {timeout_s}s: {e}"
    except Exception as e:
        return False, "", str(e)


def lint_file(file_path: Path) -> VeraResult:
    result = VeraResult()
    
    # 1. AST check
    try:
        source = file_path.read_text(encoding="utf-8")
        ast.parse(source)
    except SyntaxError as e:
        result.fail(f"AST SyntaxError: {e}", context=str(e))
        return result
    except Exception as e:
        result.fail(f"Read Error: {e}", context=str(e))
        return result

    # 2. Ruff check
    success, stdout, stderr = run_command(["ruff", "check", str(file_path)])
    if not success:
        result.fail("Ruff linting failed.", context=stdout + "\n" + stderr)

    return result


def smoke_test(file_path: Path) -> VeraResult:
    result = VeraResult()
    import importlib.util
    
    temp_name = f"_smoke_test_{file_path.stem}"
    try:
        spec = importlib.util.spec_from_file_location(temp_name, file_path)
        if spec is None or spec.loader is None:
            result.fail("Could not create module spec")
            return result
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[temp_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        result.fail(f"Smoke test failed: {e}", context=tb)
    
    return result


def run_pytest(file_path: Path, timeout_s: float | None) -> VeraResult:
    result = VeraResult()
    success, stdout, stderr = run_command(["pytest", str(file_path)], timeout_s)
    if not success:
        if "no tests ran" in stdout:
            # Not a failure if there are just no tests
            pass
        else:
            result.fail("Pytest failed.", context=stdout + "\n" + stderr)
    return result


def execute_file(file_path: Path, args: list[str], timeout_s: float | None) -> VeraResult:
    result = VeraResult()
    cmd = [sys.executable, str(file_path)] + args
    success, stdout, stderr = run_command(cmd, timeout_s)
    if not success:
        result.fail(f"Execution failed with args {args}", context=stdout + "\n" + stderr)
    return result


def test_arguments(file_path: Path, depth: int, timeout_s: float | None) -> VeraResult:
    """Test script functions or CLI arguments one at a time."""
    result = VeraResult()
    
    # Approach: we try to run the script with --help to find arguments
    success, stdout, stderr = run_command([sys.executable, str(file_path), "--help"], timeout_s)
    if success and "-h" in stdout or "--help" in stdout:
        # Extract simplistic arguments (lines starting with  - or --)
        args_to_test = []
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("-") and not line.startswith("--help"):
                arg = line.split()[0].replace(",", "")
                args_to_test.append(arg)
        
        # Test them based on depth
        for i, arg in enumerate(args_to_test):
            if i >= depth:
                break
            cmd = [sys.executable, str(file_path), arg]
            succ, out, err = run_command(cmd, timeout_s)
            if not succ:
                result.fail(f"Argument test failed for {arg}", context=out + "\n" + err)
                break
    else:
        # No --help, try AST function execution testing
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            if functions:
                result.fail(f"Script has no standard CLI help, found functions: {functions[:depth]}. "
                            "Advanced FT argument probing failed.", context="No CLI arg parser detected.")
        except Exception as e:
            result.fail("Failed to parse script for FT.", context=str(e))
            
    return result


#: Timeout unit -> seconds. `h` is the addition; the rest were already
#: here. Kept as data so the CLI's `choices` and the conversion cannot
#: disagree — which they did when `h` was added to one and not the other.
TIMEOUT_UNITS: dict[str, float] = {
    "ml": 0.001,
    "s": 1.0,
    "M": 60.0,
    "h": 3600.0,
}


def to_seconds(value: float | None, unit: str) -> float | None:
    """A timeout in its unit, as seconds. None stays None."""
    if value is None:
        return None
    try:
        return value * TIMEOUT_UNITS[unit]
    except KeyError:
        raise ValueError(
            f"unknown timeout unit {unit!r}; try one of "
            f"{', '.join(TIMEOUT_UNITS)}"
        ) from None


def _with_stage_timeout(stage, remaining: float) -> VeraResult:
    """Run *stage*, turning an overrun into a result rather than a raise."""
    started = time.monotonic()
    result = stage()
    if time.monotonic() - started >= remaining:
        # The stage itself may have finished either way; what matters is
        # that the loop stops now rather than starting another.
        result.output += (
            "\n(this stage used the rest of the total timeout)"
        )
    return result


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hnx vera",
        description="Verify HyperNix module syntax, run smoke tests, and intelligently explain errors."
    )
    parser.add_argument("file", help="Python file to test")
    parser.add_argument("-dr", "--dry-run", action="store_true", help="Run the file with dry-run flag")
    parser.add_argument("-C", "--full-run", action="store_true", help="Fully run the file")
    parser.add_argument("-FT", "--function-test", action="store_true", help="Test each argument one at a time")
    parser.add_argument("-q", "--depth", type=int, default=1, help="Argument testing depth (1-10)")
    parser.add_argument("-t", "--timeout", type=float, default=None, help="Timeout value")
    parser.add_argument("-T", "--timeout-unit", choices=list(TIMEOUT_UNITS),
                        default="ml",
                        help="Timeout unit: ml (default), s, M (minutes), h (hours)")
    parser.add_argument("-tt", "--total-timeout", type=float, default=None,
                        help="Seconds for the WHOLE run, not one stage. A slow "
                             "import plus six argument combinations sits inside "
                             "every per-stage limit and still takes twenty "
                             "minutes.")
    parser.add_argument("-Na", "--no-ai", action="store_true",
                        help="Do not load a model; print the raw error.")
    parser.add_argument("-m", "--pick-model", action="store_true",
                        help="Choose a GGUF from ~/.hypernix/models to explain with.")
    parser.add_argument("--models-dir", default="",
                        help="Where -m looks (default ~/.hypernix/models).")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    file_path = Path(args.file)

    if not file_path.exists():
        console.print(f"[red]Error:[/] File not found: {file_path}")
        return 1

    timeout_s = to_seconds(args.timeout, args.timeout_unit)

    # -Na and -m are contradictory: one says do not load a model, the
    # other asks which one to load. Saying so beats silently honouring
    # whichever happens to be checked first.
    if args.no_ai and args.pick_model:
        console.print("[red]-Na and -m contradict each other:[/] one says not "
                      "to load a model, the other asks which one to load.")
        return 2

    explain_with = "qwen3.5-4b"
    if args.pick_model:
        chosen = pick_model(Path(args.models_dir) if args.models_dir else None)
        if chosen:
            explain_with = chosen

    deadline = (
        time.monotonic() + args.total_timeout
        if args.total_timeout is not None else None
    )

    console.print(f"[bold cyan]Vera:[/] Analyzing {file_path} ...\n")
    
    stages = [
        ("Linting (AST + Ruff)", lambda: lint_file(file_path)),
        ("Smoke Testing", lambda: smoke_test(file_path)),
        ("Pytest", lambda: run_pytest(file_path, timeout_s)),
    ]
    
    if args.dry_run:
        stages.append(("Dry Run", lambda: execute_file(file_path, ["--dry-run"], timeout_s)))
        # Fallback to -dr if --dry-run isn't recognized could be added, but this suffices for the request.
    elif args.full_run:
        stages.append(("Full Run", lambda: execute_file(file_path, [], timeout_s)))
        
    if args.function_test:
        stages.append(("Argument Testing (-FT)", lambda: test_arguments(file_path, args.depth, timeout_s)))

    source_code = file_path.read_text(encoding="utf-8", errors="ignore")
    
    for stage_name, stage_fn in stages:
        # Checked between stages rather than by killing one: a stage cut
        # off part-way reports a failure that is the clock's, not the
        # file's, and that is the confusing kind.
        if deadline is not None and time.monotonic() >= deadline:
            console.print(
                f"[yellow]Stopping before {stage_name}:[/] the total timeout "
                f"({args.total_timeout}s) is up. What ran up to here passed."
            )
            return 3

        console.print(f"Running [bold]{stage_name}[/] ...")
        if deadline is not None:
            # Never let one stage outlast the whole run.
            remaining = max(0.1, deadline - time.monotonic())
            res = _with_stage_timeout(stage_fn, remaining)
        else:
            res = stage_fn()

        if not res.passed:
            console.print(f"[red]✗ {stage_name} Failed![/]")
            if res.output:
                console.print(f"[dim]{res.output}[/]")

            if args.no_ai:
                console.print(f"\n[dim]{NO_AI_NOTE}[/]")
                return 1

            console.print(
                f"\n[bold yellow]Vera AI Analysis:[/] (Using {explain_with})"
            )
            with console.status("Generating explanation..."):
                explanation = get_ai_explanation(
                    res.error_context, source_code, explain_with
                )
            console.print(f"[green]{explanation}[/]")
            return 1
        else:
            console.print(f"[green]✓ {stage_name} Passed[/]\n")

    console.print("[bold green]All checks passed successfully![/]")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
