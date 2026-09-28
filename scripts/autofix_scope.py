#!/usr/bin/env python3
"""Shared plumbing for the ``autofix-*`` scripts.

Three jobs, all of which more than one caller needs:

**Which tests belong to a category.** ``hypernix.MODULE_CATEGORIES`` says
which modules are, say, timing modules; this walks ``tests/`` and works
out which individual test functions actually exercise them, by resolving
each test file's imports and looking for uses of the resulting names. The
answer is a list of pytest node ids, so a caller can run *only* those
tests instead of the whole suite. ``.github/workflows/ci.yml`` uses the
same list, which is why this lives in a module rather than inside one
script.

**Which time constants a category's tests use.** Derived from the modules
themselves — any float field named like a duration — so ``autofix-F``
doesn't carry a hardcoded list that drifts from the code.

**What kind of failure a CI log describes.** Lint, imports, timing tests,
or something that needs a human. The router (``scripts/autofix``) and
``autofix-F`` both classify, and they must agree.

Usage:
    python3 scripts/autofix_scope.py --list              # node ids, one per line
    python3 scripts/autofix_scope.py --list --category data
    python3 scripts/autofix_scope.py --time-kwargs
    python3 scripts/autofix_scope.py --classify ci-log.txt
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
TESTS = REPO_ROOT / "tests"

DEFAULT_CATEGORY = "timing"

# Names that look like a wall-clock knob. Used to *filter* real dataclass
# fields, not as a standalone list, so a renamed field can't leave a stale
# entry behind.
_TIME_NAME_RE = re.compile(
    r"(^|_)(duration|interval|seconds|timeout|period|delay|cooldown|every)($|_)"
)


# ---------------------------------------------------------------------------
# Category -> modules
# ---------------------------------------------------------------------------

def category_modules(category: str = DEFAULT_CATEGORY) -> list[str]:
    """Modules in ``category``, straight from the package's own table."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import hypernix

    try:
        return list(hypernix.MODULE_CATEGORIES[category])
    except KeyError:
        raise SystemExit(
            f"unknown category {category!r}; known: "
            f"{', '.join(sorted(hypernix.MODULE_CATEGORIES))}"
        ) from None


def time_kwargs(category: str = DEFAULT_CATEGORY) -> set[str]:
    """Keyword arguments in ``category``'s API that hold a number of seconds.

    Found by importing each module and inspecting its dataclass fields, so
    ``IntervalTimer.interval_seconds`` is discovered rather than listed.
    Modules that can't be imported here are skipped, not fatal.
    """
    import dataclasses
    import importlib
    import inspect

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    found: set[str] = set()
    for module_name in category_modules(category):
        try:
            module = importlib.import_module(f"hypernix.{module_name}")
        except Exception:  # noqa: BLE001 - an optional dep shouldn't break discovery
            continue
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if not dataclasses.is_dataclass(obj):
                continue
            for field in dataclasses.fields(obj):
                if field.type in ("float", "int", float, int) or isinstance(
                    field.default, (int, float),
                ):
                    if _TIME_NAME_RE.search(field.name):
                        found.add(field.name)
    return found


# ---------------------------------------------------------------------------
# Category -> test node ids
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestRef:
    """One test function, with enough context to re-run it on its own."""

    path: Path
    node_id: str
    modules: frozenset[str]

    def __str__(self) -> str:
        return self.node_id


def _known_modules() -> set[str]:
    """Every hypernix module name, from the package's own layout table."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    import hypernix

    return set(hypernix.CATEGORY_OF)


def _binding_targets(tree: ast.Module | ast.AST) -> dict[str, str]:
    """Map local names to the hypernix module they came from.

    Covers the spellings tests actually use::

        from hypernix import timer            -> {"timer": "timer"}
        from hypernix.timing import timer     -> {"timer": "timer"}
        from hypernix.timer import EggTimer   -> {"EggTimer": "timer"}
        from hypernix import timer as t       -> {"t": "timer"}
        import hypernix.timing.timer as clock -> {"clock": "timer"}
    """
    known = _known_modules()
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module != "hypernix" and not module.startswith("hypernix."):
                continue
            # The last component names a module when it is one; otherwise it
            # is the package (`hypernix`, or a category like `hypernix.timing`)
            # and the imported names are the modules.
            owner = module.split(".")[-1]
            for alias in node.names:
                local = alias.asname or alias.name
                if alias.name in known:
                    bindings[local] = alias.name       # importing the module
                elif owner in known:
                    bindings[local] = owner            # importing one of its symbols
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("hypernix."):
                    continue
                leaf = alias.name.split(".")[-1]
                if leaf in known:
                    bindings[alias.asname or alias.name] = leaf
    return bindings


def _used_modules(node: ast.AST, bindings: dict[str, str]) -> set[str]:
    """Which hypernix modules a chunk of AST touches, via ``bindings``."""
    used: set[str] = set()
    local = dict(bindings)
    local.update(_binding_targets(node))  # imports inside the test body
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in local:
            used.add(local[child.id])
        elif isinstance(child, ast.Attribute):
            base = child
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in local:
                used.add(local[base.id])
    return used


def _iter_test_functions(tree: ast.Module):
    """Yield ``(node_id_suffix, function_node)`` for every test in a file."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name.startswith("test"):
                yield node.name, node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    if item.name.startswith("test"):
                        yield f"{node.name}::{item.name}", item


def _node_path(path: Path) -> str:
    """The path part of a node id: repo-relative when it can be."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()  # a --tests-dir outside the repo


def discover_tests(
    category: str = DEFAULT_CATEGORY,
    tests_dir: Path | None = None,
) -> list[TestRef]:
    """Every test function that exercises a module in ``category``."""
    tests_dir = Path(tests_dir).resolve() if tests_dir else TESTS
    wanted = set(category_modules(category))
    refs: list[TestRef] = []

    for path in sorted(tests_dir.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        bindings = _binding_targets(tree)
        if not (set(bindings.values()) & wanted) and not any(
            m in path.stem for m in wanted
        ):
            continue
        for suffix, func in _iter_test_functions(tree):
            hit = _used_modules(func, bindings) & wanted
            if hit:
                refs.append(TestRef(path, f"{_node_path(path)}::{suffix}", frozenset(hit)))
    return refs


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

LINT_MARKERS = (
    re.compile(r"^[A-Z]+\d+ .*", re.MULTILINE),          # ruff diagnostic code
    re.compile(r"ruff check", re.MULTILINE),
    re.compile(r"\bFound \d+ errors?\b"),
)
IMPORT_MARKERS = (
    re.compile(r"\bModuleNotFoundError\b"),
    re.compile(r"\bImportError\b"),
    re.compile(r"\bSyntaxError\b"),
    re.compile(r"\bIndentationError\b"),
    re.compile(r"errors during collection"),
    re.compile(r"^ERROR tests/", re.MULTILINE),
)
TEST_FAILURE_RE = re.compile(r"^FAILED (\S+)", re.MULTILINE)


def failing_node_ids(log: str) -> list[str]:
    """Node ids pytest reported as FAILED, in the order they appeared."""
    seen: dict[str, None] = {}
    for match in TEST_FAILURE_RE.finditer(log):
        seen.setdefault(match.group(1).split(" - ")[0], None)
    return list(seen)


def classify(log: str, category: str = DEFAULT_CATEGORY) -> str:
    """One of ``imports`` / ``lint`` / ``<category>`` / ``tests`` / ``clean``.

    Import and syntax errors come first on purpose: nothing else can be
    diagnosed while the tree doesn't import, and they are ``autofix-E``'s
    job, not ``autofix-B``'s or ``autofix-F``'s.
    """
    if any(marker.search(log) for marker in IMPORT_MARKERS):
        return "imports"
    if any(marker.search(log) for marker in LINT_MARKERS):
        return "lint"

    failures = failing_node_ids(log)
    if not failures:
        return "clean"
    in_category = {ref.node_id for ref in discover_tests(category)}
    if any(_matches(node_id, in_category) for node_id in failures):
        return category
    return "tests"


def node_key(node_id: str) -> str:
    """A canonical form for comparing node ids.

    pytest reports paths relative to its rootdir, which for a ``--tests-dir``
    outside the repo comes back as ``../../tmp/...`` while discovery produced
    an absolute path. Resolving the path part — and dropping any
    ``[parametrize-id]`` — makes the two comparable.
    """
    path, _, rest = node_id.partition("::")
    rest = "::".join(part.split("[", 1)[0] for part in rest.split("::"))
    return f"{(REPO_ROOT / path).resolve().as_posix()}::{rest}"


def _matches(node_id: str, known: set[str]) -> bool:
    """True when ``node_id`` names one of ``known``, ignoring any [params]."""
    if node_id in known or node_id.split("[", 1)[0] in known:
        return True
    return node_key(node_id) in {node_key(k) for k in known}


SCRIPT_FOR = {
    "imports": "autofix-E",
    "lint": "autofix-B",
    DEFAULT_CATEGORY: "autofix-F",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--tests-dir", type=Path, default=None,
                        help="scan this directory instead of tests/")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="print matching node ids")
    group.add_argument("--time-kwargs", action="store_true", help="print time-valued kwargs")
    group.add_argument("--classify", metavar="LOGFILE", help="classify a CI log ('-' for stdin)")
    group.add_argument(
        "--validate", metavar="NODE_IDS",
        help=(
            "check a whitespace-separated list of node ids against the ones "
            "discovered for --category and print them back; fails if any is "
            "not in the category. CI passes an attacker-controllable commit "
            "trailer through here before handing it to pytest."
        ),
    )
    args = parser.parse_args(argv)

    if args.list:
        for ref in discover_tests(args.category, args.tests_dir):
            print(ref.node_id)
        return 0

    if args.validate is not None:
        allowed = {ref.node_id for ref in discover_tests(args.category, args.tests_dir)}
        requested = args.validate.split()
        if not requested:
            print(f"no node ids given for category {args.category!r}", file=sys.stderr)
            return 1
        unknown = [n for n in requested if not _matches(n, allowed)]
        if unknown:
            print(
                f"not {args.category} tests: {', '.join(unknown)}",
                file=sys.stderr,
            )
            return 1
        for node_id in requested:
            print(node_id)
        return 0

    if args.time_kwargs:
        for name in sorted(time_kwargs(args.category)):
            print(name)
        return 0

    log = sys.stdin.read() if args.classify == "-" else Path(args.classify).read_text(
        encoding="utf-8", errors="replace",
    )
    print(classify(log, args.category))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
