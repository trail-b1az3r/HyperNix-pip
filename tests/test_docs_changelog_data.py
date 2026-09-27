"""The Pages site's changelog data follows wiki/Changelog-guide.md.

scripts/generate_docs_data.py turns wiki/Changelog.md into
docs/public/v1/changelog.json, which the site's Home and Stats pages
read. These hold it to the guide's patch heading, its bullets and their
nested details, and its legend.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "wiki" / "Changelog-guide.md"


def _load():
    spec = importlib.util.spec_from_file_location("generate_docs_data", ROOT / "scripts" / "generate_docs_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load()

ENTRY = """\
### Added

๋࣭⭑ Added HyperLink on the web, served by the T1 API on port
  37965.
  - It listens on 127.0.0.1 and this machine's Tailscale addresses
    only.
  - It is dark.
𖥔 Added `model.safetensors` loading to `brewer_adapter`.

### Security

🔒 Fixed robots.txt letting a server-side fetch reach the LAN.
"""


class TestThePatchHeading:
    def test_it_is_read_as_the_guide_writes_it(self):
        label, title, date = gen._parse_changelog_heading(
            "0.72.6.post1 — patch 1 - HyperLink tool calling fixed")
        assert (label, title, date) == ("0.72.6.post1", "patch 1 - HyperLink tool calling fixed", "")
        found = gen._PATCH_TITLE.match(title)
        assert found and found.group(1) == "1" and found.group(2) == "HyperLink tool calling fixed"

    def test_a_dated_heading_is_unchanged(self):
        assert gen._parse_changelog_heading("0.72.6 — 2026-09-24") == ("0.72.6", "", "2026-09-24")

    def test_the_newest_entry_carries_its_patch_and_headline(self, monkeypatch, tmp_path):
        changelog = tmp_path / "Changelog.md"
        changelog.write_text(
            "# Changelog\n\n## 0.72.6.post1 — patch 1 - HyperLink tool calling fixed\n\n"
            + ENTRY + "\n## 0.72.6 — 2026-09-24\n\n### Added\n\n✨ Added a thing.\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(gen, "CHANGELOG_PATH", changelog)
        newest, dated = gen.parse_changelog()["entries"]
        assert newest["patch"] == 1 and newest["headline"] == "HyperLink tool calling fixed"
        assert newest["summary"] == "HyperLink tool calling fixed"  # the headline is the summary
        assert newest["date"] == "" and newest["kind"] == "post"
        assert dated["patch"] is None and dated["date"] == "2026-09-24"
        assert dated["summary"] == "Added a thing."


class TestTheChanges:
    def test_details_are_kept_apart_from_the_change(self):
        changes = gen._changelog_changes(ENTRY)
        assert [c["text"] for c in changes] == [
            "Added HyperLink on the web, served by the T1 API on port 37965.",
            "Added model.safetensors loading to brewer_adapter.",
            "Fixed robots.txt letting a server-side fetch reach the LAN.",
        ]
        assert changes[0]["details"] == [
            "It listens on 127.0.0.1 and this machine's Tailscale addresses only.",
            "It is dark.",
        ]
        assert [(c["symbol"], c["label"], c["category"]) for c in changes] == [
            ("๋࣭⭑", "major", "Added"), ("𖥔", "minor", "Added"), ("🔒", "security", "Security"),
        ]

    def test_highlights_are_the_change_sentences(self):
        assert gen._changelog_bullet_highlights(ENTRY)[0] == (
            "Added HyperLink on the web, served by the T1 API on port 37965.")

    def test_an_older_entry_keeps_its_wrapped_lines(self):
        body = "### Fixed\n\n𖢥 Fixed a thing that\n  wrapped onto a second line.\n"
        assert gen._changelog_bullet_highlights(body) == ["Fixed a thing that wrapped onto a second line."]

    @pytest.mark.parametrize("raw, clean", [
        ("`brewer_adapter` and T1_WEB_PORT", "brewer_adapter and T1_WEB_PORT"),
        ("an _emphasis_ and __bold__ and **strong**", "an emphasis and bold and strong"),
    ])
    def test_names_keep_their_underscores(self, raw, clean):
        assert gen.clean_changelog_text(raw) == clean


class TestTheLegend:
    def test_every_guide_symbol_has_a_label(self):
        legend = GUIDE.read_text(encoding="utf-8").split("## Legend", 1)[1].split("\n## ", 1)[0]
        symbols = re.findall(r"^\| `([^`]+)` \|", legend, re.M)
        assert symbols
        assert set(symbols) == set(gen.CHANGELOG_SYMBOLS) == set(gen.CHANGELOG_SYMBOL_LABELS)


class TestThePublishedData:
    def test_the_site_data_has_the_newest_entry(self):
        """docs/public/v1/changelog.json is regenerated with the changelog."""
        published = json.loads((ROOT / "docs" / "public" / "v1" / "changelog.json").read_text(encoding="utf-8"))
        assert published["entries"][0]["version"] == gen.parse_changelog()["entries"][0]["version"]

    def test_the_site_shows_the_patch_and_the_labels(self):
        home = (ROOT / "docs" / "src" / "pages" / "Home.tsx").read_text(encoding="utf-8")
        stats = (ROOT / "docs" / "src" / "pages" / "Stats.tsx").read_text(encoding="utf-8")
        assert "changelogEntries[0].patch" in home and "c.label" in home
        assert "r.patch" in stats and "Patch releases (.postN)" in stats
