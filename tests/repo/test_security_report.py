"""The site's security page: scripts/security_report.py, the Pages build
that runs it, and the page's hidden way in.

The report is what the repository's own scanners find, written for
``#/security``. A scanner that is missing or fails must be *said* on the
page rather than silently leave a gap, and the docs build must not stop
because a scanner did.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location("security_report", ROOT / "scripts" / "security_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_report_runs_offline_and_says_what_it_skipped(tmp_path):
    sr = _module()
    out = tmp_path / "security.json"
    assert sr.main(["--out", str(out), "--skip", "bandit,semgrep,osv"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    tools = {t["name"]: t for t in data["tools"]}
    assert tools["hypernix-t1-audit"]["status"] == "ok"
    assert all(tools[name]["status"] == "skipped" for name in ("bandit", "semgrep", "osv"))
    assert set(data["totals"]) == {"critical", "high", "medium", "low", "tests"}
    counted = sum(data["totals"][s] for s in sr.SEVERITIES) + data["totals"]["tests"]
    assert counted == len(data["findings"])
    for finding in data["findings"]:
        assert finding["severity"] in sr.SEVERITIES
        assert not finding["path"].startswith(("/", "./"))
        assert finding["test"] == bool(sr.TEST_PATH.search(finding["path"]))


def test_a_missing_scanner_is_skipped_and_a_broken_one_is_failed(monkeypatch):
    sr = _module()
    monkeypatch.setattr(sr.shutil, "which", lambda name: None)

    def broken():
        raise ValueError("the scanner fell over")

    monkeypatch.setitem(sr.SCANNERS, "osv", ("OSV (dependencies)", broken))
    data = sr.report({"hypernix-t1-audit"})
    tools = {t["name"]: t for t in data["tools"]}
    assert tools["bandit"]["status"] == "skipped" and "not installed" in tools["bandit"]["detail"]
    assert tools["osv"]["status"] == "failed" and "fell over" in tools["osv"]["detail"]


def test_dependency_advisories_are_mapped(monkeypatch):
    sr = _module()
    monkeypatch.setattr(sr, "_floors", lambda: [("PyPI", "examplepkg", "1.0", "pyproject.toml:12")])
    monkeypatch.setattr(sr, "_post", lambda url, body: {"results": [{"vulns": [{"id": "GHSA-test"}]}]})
    monkeypatch.setattr(sr, "_get", lambda url: {
        "summary": "Something bad", "database_specific": {"severity": "MODERATE"},
        "affected": [{"package": {"name": "examplepkg"},
                      "ranges": [{"events": [{"introduced": "0"}, {"fixed": "1.2"}]}]}]})
    found, _ = sr.osv()
    assert found == [sr._finding("osv", "GHSA-test", "medium", "pyproject.toml", 12,
                                 "examplepkg 1.0 (PyPI): Something bad",
                                 package="examplepkg", version="1.0", fixed="1.2")]


def test_the_pages_build_writes_it_and_does_not_depend_on_it():
    workflow = (ROOT / ".github" / "workflows" / "deploy-docs.yml").read_text(encoding="utf-8")
    scan = workflow.index("python scripts/security_report.py --out docs/public/v1/security.json")
    assert scan < workflow.index("npm run build")
    step = workflow[workflow.rindex("- name:", 0, scan):scan]
    assert "continue-on-error: true" in step
    # Generated per build, never committed.
    assert "docs/public/v1/security.json" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_the_page_is_not_in_the_menu_but_the_footer_has_a_way_in():
    app = (ROOT / "docs" / "src" / "App.tsx").read_text(encoding="utf-8")
    pages = re.search(r"const PAGES = \[([^\]]*)\]", app).group(1)
    names = re.findall(r"'([^']+)'", pages)
    assert "security" in names and names.index("security") >= 6       # the menu shows PAGES.slice(0, 6)
    assert "<SecurityPage />" in app
    footer = (ROOT / "docs" / "src" / "components" / "Footer.tsx").read_text(encoding="utf-8")
    assert "setPage('security')" in footer and 'aria-label="Security scan"' in footer
    css = (ROOT / "docs" / "src" / "index.css").read_text(encoding="utf-8")
    assert re.search(r"\.hidden-door\s*\{\s*opacity:\s*0", css)
    assert ".hidden-door:focus-visible" in css                        # reachable with Tab
