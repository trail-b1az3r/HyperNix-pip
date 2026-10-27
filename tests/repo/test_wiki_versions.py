"""The wiki's "current version" lines name the current versions.

They named the previous release: Home.md and T1-API.md said T1 was
current "in 0.72.8a3" after 0.72.8a4 was out, because the release bump
rewrote six files and these were not among them. The bump runs
scripts/sync_wiki_versions.py now; this fails when the two disagree.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _sync():
    spec = importlib.util.spec_from_file_location(
        "sync_wiki_versions", ROOT / "scripts" / "sync_wiki_versions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_wiki_names_the_current_versions():
    stale = _sync().sync(check=True)
    assert not stale, (
        f"{', '.join(stale)} name an old version; run "
        f"`python scripts/sync_wiki_versions.py`"
    )


def test_it_reads_the_same_t1_version_the_package_reports():
    from hypernix.t1api.version import T1_VERSION_LONG, T1_VERSION_SHORT

    assert _sync().t1_versions(ROOT) == (T1_VERSION_SHORT, T1_VERSION_LONG)


def test_it_rewrites_every_line_it_keeps(tmp_path):
    sync = _sync()
    for page in sync.PAGES:
        target = tmp_path / page
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / page).read_text(encoding="utf-8"), encoding="utf-8")
    for part in ("src/hypernix/__init__.py", "src/hypernix/t1api/version.py"):
        (tmp_path / part).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / part).write_text((ROOT / part).read_text(encoding="utf-8"), encoding="utf-8")

    assert sorted(sync.sync(version="9.9.9a1", root=tmp_path)) == sorted(sync.PAGES)
    home = (tmp_path / "wiki/Home.md").read_text(encoding="utf-8")
    api = (tmp_path / "wiki/T1-API.md").read_text(encoding="utf-8")
    assert "in 9.9.9a1." in home
    assert "in\n`9.9.9a1`" in api and '"version": "9.9.9a1"' in api
    assert sync.sync(version="9.9.9a1", root=tmp_path) == []  # idempotent


def test_the_release_timeline_has_one_section_per_release():
    """A re-run of a release appended a second "v0.72.8a4: No commits
    found" section; the workflow now replaces a version's section."""
    text = (ROOT / "wiki" / "Release-Timeline.md").read_text(encoding="utf-8")
    headers = re.findall(r"^## Release v(\S+) \(", text, re.M)
    repeated = sorted({h for h in headers if headers.count(h) > 1})
    assert not repeated, f"sections repeated: {repeated}"
    workflow = (ROOT / ".github" / "workflows" / "public-release.yml").read_text(encoding="utf-8")
    step = workflow[workflow.index("- name: Update Release Timeline"):]
    step = step[:step.index("- name: Upload build artifacts")]
    assert "if header in existing:" in step
