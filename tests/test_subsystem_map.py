"""`wiki/Home.md`'s subsystem map has to describe the tree that exists.

A map is only worth reading if its boxes can be looked up, and this one
had drifted a long way: it showed the training pipeline as of about
0.70 and nothing since — no T1 API, no HyperLink, no Studio, no
quantisation stack, no `gather`, no `fuse box`. It also carried a
`new_oven` box beside `old_oven`, and `new_oven` is a *function* in
`models.old_oven`, not a module.

So every dotted name the map prints is resolved here, and every
directory it points at is checked. The map is allowed to be
incomplete -- it is a map -- but it is not allowed to be wrong.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HOME = REPO_ROOT / "wiki" / "Home.md"
SRC = REPO_ROOT / "src"


def map_section() -> str:
    text = HOME.read_text(encoding="utf-8")
    start = text.index("## The subsystem map")
    end = text.index("## Design principles")
    return text[start:end]


SECTION = map_section()

# `data.{cutting_board, pans}` is brace-expansion shorthand, the way the
# map writes a row of siblings. Expand it before resolving anything.
_BRACE = re.compile(r"\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)\.\{([^}]*)\}")
# A dotted module path: `quant.runtime_bridge`, `system.gpus`.
_DOTTED = re.compile(r"\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)\b")

# Words that look like module paths but are not: prose, filenames, and
# the four console-script names the map lists as things people run.
_NOT_MODULES = {
    "model.gguf", "models.json", "models.jsonl", "pyproject.toml",
    "setup.cfg", "wiki.home", "home.md", "changelog.md", "roadmap.md",
    "ggml.c",
}


def dotted_names() -> set[str]:
    text = SECTION
    for whole, prefix, inner in [(m.group(0), *m.groups()) for m in _BRACE.finditer(text)]:
        expanded = " ".join(
            f"{prefix}.{part.strip()}" for part in inner.split(",") if part.strip()
        )
        text = text.replace(whole, expanded)
    found = {name for name in _DOTTED.findall(text) if name not in _NOT_MODULES}
    # Drop anything ending in a file extension -- those are paths, and
    # the directory test below covers them.
    return {n for n in found if not re.search(r"\.(md|py|cpp|h|c|cu|toml|cfg|sh|json|gguf)$", n)}


def resolves(dotted: str) -> bool:
    """Is this a module, a package, or a top-level export?

    Checked as files rather than imported: importing every subsystem
    pulls in torch, Qt and a FastAPI app, which is a slow and fragile
    way to ask whether a path exists.

    The map writes some names fully qualified (`hypernix.data.gather`)
    and some as the top-level shortcut they really are
    (`hypernix.preheat`, which is a function, not a module). Both have
    to resolve, or the check is only testing one spelling.
    """
    if dotted.startswith("hypernix."):
        dotted = dotted[len("hypernix."):]
        if "." not in dotted:
            init = (SRC / "hypernix" / "__init__.py").read_text(encoding="utf-8")
            return bool(re.search(rf"""["']{re.escape(dotted)}["']""", init))
    parts = dotted.split(".")
    base = SRC / "hypernix"
    for part in parts[:-1]:
        base = base / part
        if not base.is_dir():
            return False
    leaf = base / parts[-1]
    return leaf.is_dir() or leaf.with_suffix(".py").is_file()


class TestEveryNameOnTheMapIsReal:
    def test_the_section_still_exists(self):
        assert "## The subsystem map" in HOME.read_text(encoding="utf-8")

    def test_it_found_something_to_check(self):
        """A regex that matches nothing would make every test below pass."""
        names = dotted_names()
        assert len(names) > 40, f"only found {len(names)}: {sorted(names)}"

    def test_every_dotted_name_resolves(self):
        missing = sorted(n for n in dotted_names() if not resolves(n))
        assert not missing, f"the map names modules that do not exist: {missing}"

    @pytest.mark.parametrize(
        "directory",
        ["desktop/src", "native/ggml-hnx", "ios", "bin", "src/hypernix/t1api"],
    )
    def test_the_directories_it_points_at_exist(self, directory):
        assert (REPO_ROOT / directory).is_dir()

    @pytest.mark.parametrize(
        "path",
        [
            "desktop/src/ModelCatalogue.cpp",
            "desktop/src/LocalEngine.cpp",
            "desktop/src/LocalSession.cpp",
            "desktop/src/StudioBridge.cpp",
            "desktop/src/HyperLinkClient.cpp",
            "desktop/src/ToolPolicy.cpp",
            "desktop/src/ToolRunner.cpp",
            "bin/hypernix-t1",
        ],
    )
    def test_the_files_it_names_exist(self, path):
        assert (REPO_ROOT / path).is_file()


class TestTheThingsItGotWrongBefore:
    """Each of these was on the old map, and each was false."""

    def test_new_oven_is_a_function_not_a_module(self):
        assert not (SRC / "hypernix" / "models" / "new_oven.py").exists()
        source = (SRC / "hypernix" / "models" / "old_oven.py").read_text(encoding="utf-8")
        assert re.search(r"^def new_oven\(", source, re.M), "new_oven moved"
        assert "models.new_oven" not in SECTION, "the map calls new_oven a module again"

    def test_code_oven_is_named_where_it_lives(self):
        source = (SRC / "hypernix" / "models" / "old_oven.py").read_text(encoding="utf-8")
        assert "class CodeOven" in source
        assert "CodeOven" in SECTION

    def test_neo_oven_is_shown_as_the_successor(self):
        """Its own docstring says so; the map used to show three peers."""
        doc = (SRC / "hypernix" / "models" / "neo_oven.py").read_text(encoding="utf-8")[:1200]
        assert "successor" in doc
        assert "successor" in SECTION

    def test_the_top_level_shortcuts_are_described_correctly(self):
        """`hypernix.preheat` resolves to old_oven, `NeoOven` to neo_oven.

        The map says this because the two disagree with neo_oven's
        docstring, which claims the top-level shortcuts return a NeoOven.
        If the lazy map is ever changed to match, this test says so.
        """
        init = (SRC / "hypernix" / "__init__.py").read_text(encoding="utf-8")
        assert "'preheat': ('models.old_oven', 'preheat')" in init
        assert "'NeoOven': ('models.neo_oven', 'NeoOven')" in init
        assert "resolve" in SECTION and "old_oven" in SECTION

    @pytest.mark.parametrize(
        "subsystem",
        ["t1api", "hyperlink", "quant", "monitoring", "security", "waiter",
         "t1sdk", "bridge", "scriptgen", "audio", "chat", "evaluation"],
    )
    def test_the_subsystems_the_old_map_omitted_are_on_it(self, subsystem):
        assert (SRC / "hypernix" / subsystem).is_dir(), "moved or renamed"
        # A *module* inside it, not the bare word. Two weaker forms
        # passed while the map said nothing: `CodeOven.chat` satisfied a
        # plain substring search for `chat`, and so did the prose "the
        # chat TUI" once dots were excluded. Naming a module is the only
        # form that cannot be satisfied by accident -- and every name it
        # matches has already been resolved against the tree above.
        named = {n for n in dotted_names() if n.split(".")[0] == subsystem}
        assert named, f"the map names no module inside {subsystem}"


class TestItRenders:
    def test_the_code_fences_are_balanced(self):
        """An unbalanced fence swallows the rest of the page."""
        assert SECTION.count("\n```") % 2 == 0

    def test_no_line_is_absurdly_wide(self):
        """The wiki does not wrap inside a fence; it scrolls sideways."""
        too_wide = [ln for ln in SECTION.splitlines() if len(ln) > 78]
        assert not too_wide, f"lines over 78 columns: {too_wide}"
