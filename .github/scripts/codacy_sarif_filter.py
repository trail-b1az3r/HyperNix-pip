#!/usr/bin/env python3
"""Keep Codacy's SARIF to what belongs in the Security tab, and within GitHub's limits.

Codacy runs every linter the project has enabled and reports all of it:
on this repository about 105,000 results, 37,000 of them pylint and
32,000 more from pylint's Python 3 build. GitHub refuses a SARIF run of
more than 25,000 results outright, so an unfiltered upload never lands,
and the alerts from the last one that did (before the analysis started
crashing) could never be marked fixed.

What is dropped, and why:

* **Tools that report no defects:** duplication, metrics, and the
  Markdown and CSS style checkers. Their findings are formatting.
* **Pylint and prospector conventions and refactor hints** (C, R and I
  message classes: line length, naming, docstrings, "too many
  branches"). They are style, and ruff already enforces the ones this
  project follows.
* **Anything under tests/ or build/** for the Python analysers: a test's
  assert is how it checks a result.

Everything else is kept: every security tool (Bandit, flawfinder,
shellcheck, hadolint, semgrep and so on), cppcheck, and pylint's errors
and warnings. A run is capped at MAX_RESULTS, most severe first, and the
job summary says what was kept and dropped per tool, so nothing is
removed silently.

    python codacy_sarif_filter.py results.sarif
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

MAX_RESULTS = 20_000
STYLE_ONLY_TOOLS = {"duplication", "metrics", "remark-lint", "markdownlint", "stylelint",
                    "csslint", "tailor"}
PYLINT_TOOLS = {"pylint", "pylintpython3", "prospector"}
PY_ANALYSERS = PYLINT_TOOLS | {"bandit"}
SKIP_DIRS = ("tests/", "build/")
LEVEL_RANK = {"error": 0, "warning": 1, "note": 2, "none": 3}
# Pylint message ids as Codacy writes them: PyLint_C0301, PyLintPython3_R0913, ...
PYLINT_ID = re.compile(r"(?:^|_)([CRIEWF])\d{4}\b")


def _tool(run: dict) -> str:
    return str(run.get("tool", {}).get("driver", {}).get("name", "")).strip().lower()


def _path(result: dict) -> str:
    try:
        uri = result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    except (KeyError, IndexError, TypeError):
        return ""
    return str(uri).replace("file://", "").lstrip("./")


def _keep(tool: str, result: dict) -> bool:
    if tool in STYLE_ONLY_TOOLS:
        return False
    path = _path(result)
    if tool in PY_ANALYSERS and any(f"/{d}" in f"/{path}" for d in SKIP_DIRS):
        return False
    if tool in PYLINT_TOOLS:
        match = PYLINT_ID.search(str(result.get("ruleId", "")))
        if match and match.group(1) in "CRI":
            return False
    return True


def filter_sarif(data: dict) -> list[tuple[str, int, int]]:
    report = []
    for run in data.get("runs", []):
        tool = _tool(run)
        results = run.get("results") or []
        kept = [r for r in results if _keep(tool, r)]
        kept.sort(key=lambda r: LEVEL_RANK.get(str(r.get("level", "warning")), 1))
        run["results"] = kept[:MAX_RESULTS]
        report.append((tool or "?", len(results), len(run["results"])))
    return report


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "results.sarif"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    report = filter_sarif(data)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    totals = Counter()
    lines = ["| Tool | Found | Uploaded |", "|---|---:|---:|"]
    for tool, found, kept in sorted(report, key=lambda r: -r[1]):
        totals["found"] += found
        totals["kept"] += kept
        lines.append(f"| {tool} | {found} | {kept} |")
    lines.append(f"| **all** | **{totals['found']}** | **{totals['kept']}** |")
    text = "### Codacy results\n\n" + "\n".join(lines) + "\n"
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
