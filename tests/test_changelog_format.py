"""Tests for the canonical changelog format defined by wiki/Changelog-guide.md."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "wiki" / "Changelog.md"
GUIDE = ROOT / "wiki" / "Changelog-guide.md"

CATEGORIES = (
    "Beta / Dev → Release Summary",
    "Breaking Changes",
    "Added",
    "Changed",
    "API Changes",
    "Architecture",
    "CLI and UX",
    "Performance",
    "Compatibility",
    "Security",
    "Fixed",
    "Dependencies and Packaging",
    "Data, Checkpoints, and Migrations",
    "Deprecated",
    "Removed",
    "Documentation",
    "Site Changes",
    "Tests",
    "Known Issues",
    "Temporary Workarounds",
)
CATEGORY_SET = set(CATEGORIES)
SUMMARY = "Beta / Dev → Release Summary"
SYMBOLS = (
    "๋࣭⭑", "✨", "𖥔", "𖢥", "🐛", "🛡️", "🔁", "🔧", "⚡", "🔒", "⚠️",
    "🧪", "📚", "🛜", "🔌", "🔗", "❌", "✂️", "꩜", "❗", "🩹", "♻️", "📦",
)


@pytest.fixture(scope="module")
def text() -> str:
    return CHANGELOG.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def guide() -> str:
    return GUIDE.read_text(encoding="utf-8")


def _entries(text: str) -> list[tuple[str, str]]:
    all_headings = list(re.finditer(r"^##\s+(.+?)\s*$", text, flags=re.M))
    entries: list[tuple[str, str]] = []
    for index, match in enumerate(all_headings):
        header = match.group(1).strip()
        if not re.match(r"^v?\d+\.\d+\.\d+", header):
            continue
        next_index = index + 1
        end = all_headings[next_index].start() if next_index < len(all_headings) else len(text)
        entries.append((header, text[match.end():end]))
    return entries


def _release_kind(version: str) -> str:
    v = version.strip().lower().lstrip("v")
    base = re.match(r"^\d+\.\d+\.\d+", v)
    tail = v[base.end():] if base else v
    compact = re.sub(r"[._ -]+", "", tail)
    if re.search(r"(?:b\d+|beta\d*)$", compact):
        return "beta"
    if re.search(r"(?:a\d+|alpha\d*)$", compact):
        return "alpha"
    if re.search(r"(?:rc\d+|releasecandidate\d*)$", compact):
        return "rc"
    if re.search(r"(?:dev\d*|development)$", compact):
        return "dev"
    return "stable" if not tail.strip() else "other"


class TestGuideAgreement:
    def test_all_categories_are_present_in_the_guide(self, guide):
        missing = [category for category in CATEGORIES if f"`{category}`" not in guide]
        assert not missing, missing

    def test_summary_heading_is_documented(self, guide):
        assert f"### {SUMMARY}" in guide

    def test_every_guide_symbol_is_in_the_changelog_legend(self, text, guide):
        legend = text.split("## Legend", 1)[1].split("\n## ", 1)[0]
        missing = [symbol for symbol in SYMBOLS if symbol in guide and symbol not in legend]
        assert not missing, missing


class TestEveryReleaseEntry:
    def test_there_are_release_entries(self, text):
        assert _entries(text)

    def test_release_labels_are_unique(self, text):
        entries = _entries(text)
        labels = [header.split(" — ", 1)[0].strip() for header, _body in entries]
        assert len(labels) == len(set(labels)), "duplicate release headings remain"

    def test_all_level3_headings_are_canonical_categories(self, text):
        bad: list[str] = []
        for header, body in _entries(text):
            for heading in re.findall(r"^###\s+(.+)$", body, re.M):
                if heading.strip() not in CATEGORY_SET:
                    bad.append(f"{header}: {heading.strip()}")
        assert not bad, bad

    def test_no_category_is_empty(self, text):
        empty: list[str] = []
        for header, body in _entries(text):
            blocks = re.split(r"^###\s+", body, flags=re.M)[1:]
            for block in blocks:
                name = block.splitlines()[0].strip()
                if name not in CATEGORY_SET:
                    continue
                content = "\n".join(block.splitlines()[1:]).strip()
                assert content, f"empty category: {header}: {name}"

    def test_category_order_follows_the_guide(self, text):
        positions = {name: index for index, name in enumerate(CATEGORIES)}
        errors: list[str] = []
        for header, body in _entries(text):
            used = [h.strip() for h in re.findall(r"^###\s+(.+)$", body, re.M)]
            indexes = [positions[x] for x in used]
            if indexes != sorted(indexes):
                errors.append(f"{header}: {used}")
        assert not errors, errors

    def test_every_top_level_change_line_has_a_symbol(self, text):
        bad: list[str] = []
        for header, body in _entries(text):
            current_category = None
            for line in body.splitlines():
                if line.startswith("### "):
                    current_category = line[4:].strip()
                    continue
                if not current_category or line.startswith("#### "):
                    continue
                if current_category == SUMMARY and not line.startswith(("  ", "\t")):
                    continue
                if not line.strip():
                    continue
                if line.startswith(" ") or line.startswith("\t"):
                    continue
                if not any(line.startswith(symbol + " ") for symbol in SYMBOLS):
                    bad.append(f"{header} / {current_category}: {line[:100]}")
        assert not bad, bad[:50]

    def test_stable_releases_following_prereleases_have_a_summary(self, text):
        entries = _entries(text)
        problems: list[str] = []
        for index, (header, body) in enumerate(entries):
            label = header.split(" — ", 1)[0].strip().lstrip("v")
            if _release_kind(label) != "stable" or not re.match(r"^\d+\.\d+\.\d+$", label):
                continue
            prerelease_seen = False
            for older_header, _older_body in entries[index + 1:]:
                older_label = older_header.split(" — ", 1)[0].strip().lstrip("v")
                if not older_label.startswith(label):
                    continue
                if _release_kind(older_label) in {"alpha", "beta", "rc", "dev"}:
                    prerelease_seen = True
                    break
            if prerelease_seen and f"### {SUMMARY}" not in body:
                problems.append(header)
        assert not problems, problems


class TestNewestEntry:
    @pytest.fixture
    def newest(self, text):
        entries = _entries(text)
        assert entries
        return entries[0]

    def test_newest_header_has_version_and_date(self, newest):
        header, _body = newest
        assert re.match(r"^\d+\.\d+\.\d+[^—]* — \d{4}-\d{2}-\d{2}$", header), header

    def test_newest_matches_package_version(self, newest):
        header, _body = newest
        package_version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
        entry_version = header.split(" — ", 1)[0].strip().lstrip("v")
        assert entry_version == package_version
