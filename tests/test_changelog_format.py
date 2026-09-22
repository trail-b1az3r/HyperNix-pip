"""The changelog, against the format `Changelog-guide.md` describes.

The guide is the spec and it has a rule about itself worth reading
before touching this file:

    Corrections to previous changelog entries should be made explicitly
    rather than silently rewriting history.

So this does not demand that a thousand lines of existing entries be
restyled. It checks the guide's *invariants* over the whole file — the
legend is complete, categories are ones the guide names, no category is
empty — and holds the newest entry to the full format, which is the one
somebody is about to copy.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "wiki" / "Changelog.md"
GUIDE = ROOT / "wiki" / "Changelog-guide.md"

#: The categories the guide names. Anything else is a typo or an
#: invention, and both make the file harder to scan.
CATEGORIES = {
    "Added", "Changed", "Fixed", "Performance", "Security",
    "Deprecated", "Removed", "Documentation", "Tests", "Known Issues",
}


@pytest.fixture(scope="module")
def text() -> str:
    return CHANGELOG.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def _entries(text: str) -> list[tuple[str, str]]:
    """(header, body) per version entry, newest first."""
    parts = re.split(r"^## (?=\d)", text, flags=re.M)[1:]
    return [(part.splitlines()[0].strip(), part) for part in parts]


class TestTheGuideAndTheLegendAgree:
    def test_every_legend_symbol_in_the_guide_is_in_the_changelog(self, text, guide):
        """A symbol the guide defines and the changelog's own legend
        omits is a symbol a reader meets with no way to look it up."""
        legend = text.split("## Legend", 1)[1].split("\n## ", 1)[0]
        missing = [
            symbol
            for symbol in ("🧪", "✨", "🐛", "🛡️", "📚", "🔧", "✂️", "🔁",
                           "𖢥", "꩜", "❗", "❌", "๋࣭⭑", "𖥔")
            if symbol in guide and symbol not in legend
        ]
        assert not missing, f"legend is missing {missing}"

    def test_the_guide_still_lists_the_categories_this_test_enforces(self, guide):
        """If the guide grows a category, this test should fail rather
        than quietly enforce a stale list."""
        found = set(re.findall(r"^### (\w[\w ]*)$", guide, re.M))
        assert CATEGORIES <= found, sorted(CATEGORIES - found)


class TestEveryEntry:
    def test_there_is_at_least_one(self, text):
        assert _entries(text)

    def test_no_category_heading_is_invented(self, text):
        """Only over entries that use the guide's category style at all.

        The older narrative entries head each section with a sentence
        and a trailing symbol — `### The embedding table is the file 𖢥`
        — which is their own shape, not a typo, and the guide's rule
        against rewriting history says to leave them. So an entry
        counts as category-style only if at least one of its headings
        is a category the guide names; within such an entry, every
        other heading has to be one too, because a mix is how `Fixes`
        ends up beside `Fixed`.
        """
        bad: list[str] = []
        for header, body in _entries(text):
            headings = re.findall(r"^### (.+)$", body, re.M)
            if not any(h.strip() in CATEGORIES for h in headings):
                continue                       # narrative entry, left alone
            for heading in headings:
                if heading.strip() not in CATEGORIES:
                    bad.append(f"{header}: {heading.strip()}")
        assert not bad, f"headings the guide does not name: {bad}"

    def test_no_category_is_empty(self, text):
        """"Do not create empty categories" — a heading with nothing
        under it reads as work that was lost rather than absent."""
        empty: list[str] = []
        for header, body in _entries(text):
            blocks = re.split(r"^### ", body, flags=re.M)[1:]
            for block in blocks:
                name = block.splitlines()[0].strip()
                if name not in CATEGORIES:
                    continue
                rest = "\n".join(block.splitlines()[1:]).strip()
                if not rest:
                    empty.append(f"{header}: {name}")
        assert not empty, f"empty categories: {empty}"


class TestTheNewestEntry:
    """Held to the whole format. It is the one somebody copies."""

    @pytest.fixture
    def newest(self, text) -> tuple[str, str]:
        entries = _entries(text)
        assert entries, "no version entries at all"
        return entries[0]

    def test_the_header_carries_a_version_and_a_date(self, newest):
        header, _body = newest
        assert re.match(r"^\d+\.\d+[\w.]* — \d{4}-\d{2}-\d{2}$", header), header

    def test_it_uses_the_guide_categories(self, newest):
        _header, body = newest
        used = set(re.findall(r"^### ([\w ]+)$", body, re.M))
        assert used, "no categories at all"
        assert used <= CATEGORIES, sorted(used - CATEGORIES)

    def test_every_bullet_under_a_category_carries_a_legend_symbol(self, newest):
        """The symbol is how the file is skimmed. A bullet without one
        is invisible to anyone looking for, say, every major bug fix.

        The format cannot be checked naively, and two attempts proved
        it. Per-line flagged the second line of every wrapped bullet.
        Per-blank-line-paragraph caught nothing, because the guide's
        own example puts bullets on consecutive lines — so one
        "paragraph" is the whole category and any symbol in it passes.

        What does work: a new bullet only ever follows a *finished
        sentence*. So a line whose predecessor ended in `.`, `:` or
        `!` and which itself starts a new thought must carry a symbol;
        everything else is a wrap.
        """
        _header, body = newest
        symbols = "🧪✨🐛🛡️📚🔧✂️🔁𖢥꩜❗❌๋࣭⭑𖥔"
        bare: list[str] = []

        for block in re.split(r"^### ", body, flags=re.M)[1:]:
            lines = block.splitlines()
            category = lines[0].strip()
            if category not in CATEGORIES:
                continue

            previous_finished = True          # the first line starts a bullet
            for line in lines[1:]:
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", "⸻")):
                    previous_finished = True
                    continue
                starts_a_bullet = previous_finished
                has_symbol = any(symbol in stripped for symbol in symbols)
                if starts_a_bullet and not has_symbol:
                    bare.append(f"{category}: {stripped[:60]}")
                previous_finished = stripped.endswith((".", ":", "!"))

        assert not bare, f"bullets with no legend symbol: {bare}"

    def test_it_is_the_version_the_package_reports_or_the_next_one(self, newest):
        """A changelog whose newest entry is three releases behind is a
        changelog nobody is updating."""
        import tomllib

        header, _body = newest
        entry_version = header.split(" — ")[0]
        current = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["version"]

        base = lambda v: re.sub(r"\.post\d+$", "", v)  # noqa: E731
        assert base(entry_version) == base(current), (
            f"changelog newest is {entry_version}, package is {current}"
        )
