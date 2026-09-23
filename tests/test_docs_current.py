"""The documentation describes the code that is here (0.72.6).

Each check reads the docs as a reader would and asks the code whether
what they say exists: a link goes somewhere, an anchor is a heading on
that page, a subcommand is one the parser takes, a setting is one the
server reads, a module is one that imports. None of them judge the
prose; they catch the page that still points at something renamed.
"""
from __future__ import annotations

import collections
import functools
import importlib
import importlib.util
import re
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "wiki"

#: Pages that are history: they name what existed then, on purpose.
HISTORY = {"Changelog.md", "Release-Timeline.md"}

DOCS = [p for p in [ROOT / "README.md", ROOT / "ios" / "README.md",
                    ROOT / "docs" / "README.md", ROOT / "docs" / "API-AUTOMATION.md",
                    *sorted(WIKI.glob("*.md"))] if p.exists()]
CURRENT = [p for p in DOCS if p.name not in HISTORY]


def _ids(paths):
    return [str(p.relative_to(ROOT)) for p in paths]


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading).replace("`", "")
    text = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    seen: collections.Counter[str] = collections.Counter()
    out = set()
    for heading in re.findall(r"^#{1,6}\s+(.*?)\s*#*\s*$", text, re.M):
        base = _slug(heading)
        out.add(base if seen[base] == 0 else f"{base}-{seen[base]}")
        seen[base] += 1
    return out


def _links(path: Path):
    text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
    for match in re.finditer(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", text):
        target = match.group(1)
        if not target.startswith(("http://", "https://", "mailto:")):
            yield target


def _resolve(page: Path, target: str) -> Path | None:
    path = target.partition("#")[0]
    if not path:
        return page
    candidate = page.parent / path
    if not candidate.exists() and (page.parent / f"{path}.md").exists():
        candidate = page.parent / f"{path}.md"   # GitHub wiki style
    return candidate if candidate.exists() else None


@pytest.mark.parametrize("page", DOCS, ids=_ids(DOCS))
def test_every_relative_link_goes_somewhere(page):
    broken = [t for t in _links(page) if _resolve(page, t) is None]
    assert not broken, f"{page.name} links to files that do not exist: {broken}"


@pytest.mark.parametrize("page", DOCS, ids=_ids(DOCS))
def test_every_anchor_is_a_heading_on_its_page(page):
    broken = []
    for target in _links(page):
        fragment = target.partition("#")[2]
        found = _resolve(page, target)
        if fragment and found is not None and found.suffix == ".md":
            if fragment.lower() not in _anchors(found):
                broken.append(target)
    assert not broken, f"{page.name} links to headings that do not exist: {broken}"


def test_the_cli_page_covers_every_hypernix_subcommand():
    from hypernix.interfaces.cli import _SUBCOMMANDS

    page = (WIKI / "CLI.md").read_text(encoding="utf-8")
    missing = sorted(s for s in _SUBCOMMANDS
                     if not re.search(rf"(?:hypernix|hnx) {re.escape(s)}\b|`{re.escape(s)}`", page))
    assert not missing, f"wiki/CLI.md never mentions: {missing}"


def test_every_t1_setting_is_in_the_example_env():
    config = (ROOT / "src" / "hypernix" / "t1api" / "config.py").read_text(encoding="utf-8")
    example = (ROOT / "examples" / "t1api" / ".env.example").read_text(encoding="utf-8")
    settings = set(re.findall(r"\"(T1_[A-Z0-9_]+)\"", config))
    assert len(settings) > 40, "the settings could not be read from config.py"
    missing = sorted(s for s in settings if not re.search(rf"\b{s}\b", example))
    assert not missing, f"examples/t1api/.env.example does not document: {missing}"


def test_every_waiter_serv_letter_is_in_the_flag_table():
    from hypernix.waiter.servargs import FLAG_LETTERS, VALUE_LETTERS

    page = (WIKI / "Waiter-TUI.md").read_text(encoding="utf-8")
    table = page.split("## `serv` flags", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| `(-[A-Za-z]{1,2})\b", table, re.M)
    missing = sorted(f"-{letter}" for letter in {**FLAG_LETTERS, **VALUE_LETTERS}
                     if f"-{letter}" not in rows)
    assert not missing, f"wiki/Waiter-TUI.md's serv table has no row for: {missing}"


def _optional_missing(exc: BaseException) -> bool:
    """An optional dependency (torch, fastapi, ...) is not installed here,
    which says nothing about whether the doc is right.

    The missing module can be a cause rather than the error itself:
    ``hypernix.t1api.create_app`` re-raises fastapi's absence as a
    plain ImportError with install advice, chained ``from`` it.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        name = getattr(exc, "name", None) or ""
        if isinstance(exc, ModuleNotFoundError) and name and not name.startswith("hypernix"):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


@functools.cache   # a failed import is retried every time otherwise
def _importable(dotted: str) -> bool:
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        module = ".".join(parts[:i])
        try:
            spec = importlib.util.find_spec(module)
        except ImportError as exc:
            if _optional_missing(exc):
                return True
            spec = None
        except ValueError:
            spec = None
        if spec is None:
            continue
        if i == len(parts):
            return True
        try:
            obj = importlib.import_module(module)
            for attribute in parts[i:]:
                obj = getattr(obj, attribute)
        except Exception as exc:  # noqa: BLE001 - any other failure means the doc is wrong
            return _optional_missing(exc)
        return True
    return False


#: Names that look like modules and are not: files the docs tell you to
#: write, and an extension that is only there when you build it.
NOT_MODULES = {"hypernix.pt", "hypernix.db", "hypernix.cctvtop_ext"}


@pytest.mark.parametrize("page", CURRENT, ids=_ids(CURRENT))
def test_every_module_the_docs_name_imports(page):
    text = page.read_text(encoding="utf-8")
    named = {m.rstrip(".") for m in re.findall(r"\bhypernix(?:\.[a-z_][a-z0-9_]*)+", text)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # the deprecated aliases warn on import
        missing = sorted(m for m in named - NOT_MODULES if not _importable(m))
    assert not missing, f"{page.name} names modules that do not import: {missing}"
