"""hypernix wiki / hnx wiki — HyperNix CLI wiki and documentation browser.

v0.70.5: Auto-generating documentation browser that reads docstrings
and type hints directly from the source code, so it never goes stale.

This module is not a standalone entry point — it's reached through the
`wiki` subcommand of the main `hypernix`/`hnx` CLI (see cli.py:_run_wiki).

Usage:
    hnx wiki                          # Show table of contents + all modules
    hnx wiki <module>                 # Show docs for a specific module
    hnx wiki -q <module>              # Quick mode: stream docs section by section
    hnx wiki <module> -b              # Open wiki in browser
    hnx wiki --version                # Show version range (0.31.1 to current)
    hnx wiki --search <keyword>       # Search across all modules
"""
from __future__ import annotations

import ast
import pkgutil
import sys
import textwrap
import webbrowser
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

# HyperNix version range
VERSION_START = "0.31.1"


def _get_hypernix_root() -> Path:
    """Return the root directory of the hypernix package."""
    import hypernix
    return Path(hypernix.__file__).parent


def _module_files() -> dict[str, Path]:
    """Map every hypernix module's flat name to its file on disk.

    Modules live in category subpackages (``hypernix/timing/timer.py``) but
    are documented — and importable — under the flat name they have always
    had, so that's what the wiki keys on. The package root is scanned too,
    for anything that hasn't been categorised.
    """
    import hypernix

    root = _get_hypernix_root()
    found: dict[str, Path] = {}
    for directory in [root, *(root / cat for cat in sorted(hypernix.MODULE_CATEGORIES))]:
        for _finder, name, ispkg in pkgutil.iter_modules([str(directory)]):
            if name.startswith("_") or ispkg:
                continue
            found[name] = directory / f"{name}.py"
    return found


def _get_all_modules() -> list[str]:
    """Discover all hypernix modules by scanning the package."""
    return sorted(_module_files())


def _module_category(module_name: str) -> str:
    """Category a module belongs to, or ``""`` for uncategorised ones."""
    import hypernix

    return hypernix.CATEGORY_OF.get(module_name, "")


def _get_module_doc(module_name: str) -> dict[str, Any] | None:
    """Extract documentation from a module's source code."""
    file_path = _module_files().get(module_name)
    if file_path is None or not file_path.exists():
        return None

    source = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    doc = {
        "name": module_name,
        "category": _module_category(module_name),
        "docstring": ast.get_docstring(tree) or "",
        "classes": [],
        "functions": [],
        "version_added": None,
    }

    # Extract version from docstring
    for line in doc["docstring"].splitlines():
        if "v0." in line or "v" in line and "." in line:
            import re
            match = re.search(r'v(\d+\.\d+(?:\.\d+)?)', line)
            if match:
                doc["version_added"] = match.group(1)
                break

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            class_doc = {
                "name": node.name,
                "docstring": ast.get_docstring(node) or "",
                "bases": [ast.unparse(base) for base in node.bases],
                "methods": [],
            }
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    if item.name.startswith("_") and item.name not in ("__init__", "__call__", "__post_init__"):
                        continue
                    method_doc = {
                        "name": item.name,
                        "docstring": ast.get_docstring(item) or "",
                        "signature": _get_signature_from_ast(item),
                    }
                    class_doc["methods"].append(method_doc)
            doc["classes"].append(class_doc)

        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name.startswith("_"):
                continue
            func_doc = {
                "name": node.name,
                "docstring": ast.get_docstring(node) or "",
                "signature": _get_signature_from_ast(node),
            }
            doc["functions"].append(func_doc)

    return doc


def _get_signature_from_ast(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a human-readable signature from an AST function node."""
    args = []
    defaults_start = len(node.args.args) - len(node.args.defaults)
    for i, arg in enumerate(node.args.args):
        name = arg.arg
        annotation = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        default = ""
        if i >= defaults_start:
            default = f"={ast.unparse(node.args.defaults[i - defaults_start])}"
        args.append(f"{name}{annotation}{default}")

    if node.args.vararg:
        name = node.args.vararg.arg
        args.append(f"*{name}")

    kw_defaults_start = len(node.args.kwonlyargs) - len(node.args.kw_defaults)
    for i, arg in enumerate(node.args.kwonlyargs):
        name = arg.arg
        annotation = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        default = ""
        if i >= kw_defaults_start and node.args.kw_defaults[i] is not None:
            default = f"={ast.unparse(node.args.kw_defaults[i])}"
        args.append(f"{name}{annotation}{default}")

    if node.args.kwarg:
        name = node.args.kwarg.arg
        args.append(f"**{name}")

    return f"({', '.join(args)})"


def _format_module_doc(doc: dict[str, Any], console: Console, quick: bool = False) -> Iterator[str]:
    """Yield formatted sections of module documentation."""
    header = Text()
    header.append(f"📖 {doc['name']}\n", style="bold bright_blue")
    if doc.get('category'):
        header.append(f"   hypernix.{doc['category']}.{doc['name']}\n", style="magenta")
    if doc['version_added']:
        header.append(f"   Added in v{doc['version_added']}\n", style="dim")
    yield header

    if doc['docstring']:
        md = Markdown(doc['docstring'])
        yield md
        yield "\n"

    if quick:
        yield "─" * 60 + "\n"

    if doc['classes']:
        yield Text("\n🏗️  Classes\n", style="bold yellow")
        for cls in doc['classes']:
            cls_header = Text(f"\n  class {cls['name']}", style="bold cyan")
            if cls['bases']:
                cls_header.append(f"({', '.join(cls['bases'])})", style="dim")
            yield cls_header
            yield "\n"

            if cls['docstring']:
                for line in textwrap.wrap(cls['docstring'], width=76, initial_indent="    ", subsequent_indent="    "):
                    yield line + "\n"

            for method in cls['methods']:
                sig = f"    def {method['name']}{method['signature']}"
                yield Text(sig, style="green") + "\n"
                if method['docstring'] and not quick:
                    for line in textwrap.wrap(method['docstring'], width=72, initial_indent="        ", subsequent_indent="        "):
                        yield line + "\n"

            if quick:
                yield "\n"

    if doc['functions']:
        yield Text("\n⚙️  Functions\n", style="bold yellow")
        for func in doc['functions']:
            sig = f"  def {func['name']}{func['signature']}"
            yield Text(sig, style="green") + "\n"
            if func['docstring'] and not quick:
                for line in textwrap.wrap(func['docstring'], width=72, initial_indent="      ", subsequent_indent="      "):
                    yield line + "\n"

        if quick:
            yield "\n"


def _show_toc(console: Console) -> None:
    """Display the table of contents with all modules."""
    modules = _get_all_modules()

    table = Table(title="HyperNix Wiki — Table of Contents", show_lines=True)
    table.add_column("Module", style="cyan", no_wrap=True)
    table.add_column("Category", style="magenta", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Version", style="dim", justify="right")

    for mod_name in modules:
        doc = _get_module_doc(mod_name)
        if doc:
            desc = doc['docstring'].splitlines()[0] if doc['docstring'] else ""
            if len(desc) > 60:
                desc = desc[:57] + "..."
            ver = doc['version_added'] or ""
            table.add_row(mod_name, doc['category'], desc, ver)
        else:
            table.add_row(mod_name, _module_category(mod_name), "", "")

    console.print(table)
    console.print(f"\n[dim]Total: {len(modules)} modules | Version range: {VERSION_START} → latest[/]")
    console.print("[dim]Tip: Use 'hnx wiki <module>' for detailed docs, 'hnx wiki -q <module>' for quick mode[/]")


def _open_in_browser(module_name: str | None) -> None:
    """Open the wiki page for a module in the browser."""
    base_url = "https://github.com/trail-b1az3r/HyperNix-pip/wiki"
    if module_name:
        page_map = {
            "pressure_cooker_v6": "Pressure-Cooker-V6",
            "pressure_cooker_v6v": "Pressure-Cooker-V6",
            "pressure_cooker_v5": "Pressure-Cooker-V5",
            "pressure_cooker_v4": "Pressure-Cooker-V4",
            "pressure_cooker_v3": "Pressure-Cooker-V3",
            "old_oven": "Ovens",
            "old_fridge": "Fridges",
            "freezer": "Freezer",
            "smoke_alarm": "Alarms",
            "pans": "Pans",
            "microwave": "Microwave",
            "tv": "TV",
            "tvtop_plus_plus": "TV",
            "countertop": "Countertop",
            "bell": "Bell",
            "flour": "Flour",
            "stml": "STML",
            "vram": "VRAM",
            "abbicus": "Abbicus",
            "workshop": "Workshop",
            "scavenger": "Scavenger",
        }
        page = page_map.get(module_name, module_name.replace("_", "-").title())
        url = f"{base_url}/{page}"
    else:
        url = base_url

    console = Console()
    console.print(f"[dim]Opening {url}...[/]")
    webbrowser.open(url)


def _search_modules(keyword: str, console: Console | None) -> None:
    """Search across all modules for a keyword."""
    modules = _get_all_modules()
    matches = []

    for mod_name in modules:
        doc = _get_module_doc(mod_name)
        if not doc:
            continue

        score = 0
        locations = []

        if keyword.lower() in mod_name.lower():
            score += 10
            locations.append("module name")

        if keyword.lower() in doc['docstring'].lower():
            score += 5
            locations.append("module doc")

        for cls in doc['classes']:
            if keyword.lower() in cls['name'].lower():
                score += 8
                locations.append(f"class:{cls['name']}")
            for method in cls['methods']:
                if keyword.lower() in method['name'].lower():
                    score += 3
                    locations.append(f"method:{cls['name']}.{method['name']}")

        for func in doc['functions']:
            if keyword.lower() in func['name'].lower():
                score += 5
                locations.append(f"func:{func['name']}")

        if score > 0:
            matches.append((score, mod_name, locations))

    matches.sort(key=lambda x: -x[0])

    if not matches:
        if console:
            console.print(f"[yellow]No matches found for '{keyword}'[/]")
        return

    table = Table(title=f"Search Results: '{keyword}'")
    table.add_column("Module", style="cyan")
    table.add_column("Score", style="yellow", justify="right")
    table.add_column("Matches", style="dim")

    for score, mod_name, locations in matches[:20]:
        table.add_row(mod_name, str(score), ", ".join(locations[:3]))

    if console:
        console.print(table)


def cli_main(argv: list[str] | None = None) -> int:
    """Entry point for the `hypernix wiki` / `hnx wiki` subcommand."""
    args = list(argv if argv is not None else sys.argv[1:])
    console = Console()

    quick = False
    browser = False
    version_flag = False
    search_keyword: str | None = None

    if "-q" in args:
        quick = True
        args.remove("-q")
    if "--quick" in args:
        quick = True
        args.remove("--quick")
    if "-b" in args:
        browser = True
        args.remove("-b")
    if "--browser" in args:
        browser = True
        args.remove("--browser")
    if "--version" in args:
        version_flag = True
        args.remove("--version")
    if "-v" in args:
        version_flag = True
        args.remove("-v")
    if "--search" in args:
        idx = args.index("--search")
        if idx + 1 < len(args):
            search_keyword = args[idx + 1]
            del args[idx:idx + 2]
    if "-s" in args:
        idx = args.index("-s")
        if idx + 1 < len(args):
            search_keyword = args[idx + 1]
            del args[idx:idx + 2]

    # Note: only an *explicit* --help/-h prints this usage block. Bare
    # `hnx wiki` (no args at all) falls through to the table-of-contents
    # branch below, matching what this help text itself advertises.
    if "--help" in args or "-h" in args:
        console.print("""
[bold]hypernix wiki / hnx wiki[/] — HyperNix Wiki & Documentation Browser

[bold]Usage:[/]
  hnx wiki                          Show table of contents and all modules
  hnx wiki <module>                 Show detailed documentation for a module
  hnx wiki -q <module>              Quick mode: stream docs section by section
  hnx wiki <module> -b              Open module wiki in browser
  hnx wiki --version                Show version range
  hnx wiki --search <keyword>       Search across all modules

[bold]Examples:[/]
  hnx wiki                          # TOC with all modules
  hnx wiki pressure_cooker_v5       # Full docs for Pressure Cooker V5
  hnx wiki -q freezer               # Quick stream docs for Freezer
  hnx wiki workshop -b              # Open Workshop wiki in browser
  hnx wiki --search QAT             # Find all QAT-related modules

[dim]The wiki auto-updates by reading docstrings from source code.[/]
        """)
        return 0

    if version_flag:
        import hypernix
        console.print("[bold]HyperNix Wiki[/]")
        console.print(f"  Version range: [cyan]{VERSION_START}[/] → [cyan]latest[/]")
        console.print(f"  Package: [cyan]{hypernix.__file__}[/]")
        return 0

    if search_keyword:
        _search_modules(search_keyword, console)
        return 0

    if not args or args[0] == "":
        _show_toc(console)
        return 0

    module_name = args[0]

    if browser:
        _open_in_browser(module_name)
        return 0

    doc = _get_module_doc(module_name)
    if doc is None:
        console.print(f"[red]Module '{module_name}' not found.[/]")
        console.print("[dim]Run 'hnx wiki' to see all available modules.[/]")
        return 1

    if quick:
        console.print(f"[dim]Quick mode: streaming docs for '{module_name}'...[/]\n")
        for section in _format_module_doc(doc, console, quick=True):
            console.print(section)
            import time
            time.sleep(0.1)
    else:
        for section in _format_module_doc(doc, console, quick=False):
            console.print(section)

    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
