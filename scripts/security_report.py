#!/usr/bin/env python3
"""What this repository's own security scanners find, for the docs site.

Runs the scanners the security workflow runs and writes one JSON file the
site's security page reads (``#/security``)::

    python scripts/security_report.py                       # docs/public/v1/security.json
    python scripts/security_report.py --out report.json --skip semgrep

* **hypernix-t1-audit** -- the T1 API audit, taken from
  ``.github/workflows/hypernix-security-scan.yml`` so there is one copy.
* **Bandit** and **Semgrep** -- when installed (the Pages build installs
  both), with the workflow's rule sets.
* **OSV** -- every dependency floor in ``pyproject.toml`` and every
  package pinned in the npm and Bun lockfiles, against osv.dev.

A scanner that is missing or fails is reported as skipped or failed, with
the reason, and the rest still run: the page says what was *not* looked
at as plainly as what was found. Nothing here exits non-zero for a
finding -- this is a report, not a gate (the security workflow is the
gate).

Secrets are never copied into the report: a finding carries the rule,
the place and the scanner's description, not the matched text.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "public" / "v1" / "security.json"
WORKFLOW = ROOT / ".github" / "workflows" / "hypernix-security-scan.yml"
SEVERITIES = ("critical", "high", "medium", "low")
#: What is a test, an example or a fixture: reported, but apart.
TEST_PATH = re.compile(r"(^|/)(tests?|testing|examples?|fixtures|benchmarks)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$")
EXCLUDE = ".git,.venv,venv,env,node_modules,build,dist,site-packages,.tox"
OSV_BATCH = "https://api.osv.dev/v1/querybatch"
OSV_VULN = "https://api.osv.dev/v1/vulns/"


def _bucket(score: float | None) -> str:
    """GitHub's buckets for a security-severity score."""
    if score is None:
        return "medium"
    return "critical" if score >= 9 else "high" if score >= 7 else "medium" if score >= 4 else "low"


def _finding(tool: str, rule: str, severity: str, path: str, line: int, message: str, **extra: Any) -> dict:
    path = path.replace("\\", "/").lstrip("./") if path else ""
    return {"tool": tool, "rule": rule, "severity": severity if severity in SEVERITIES else "medium",
            "path": path, "line": int(line or 0), "message": " ".join(str(message).split())[:400],
            "test": bool(TEST_PATH.search(path)), **extra}


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)


def _version(argv: list[str]) -> str:
    try:
        out = _run(argv, 60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (out.stdout or out.stderr).strip().splitlines()[0][:80] if (out.stdout or out.stderr) else ""


# ---------------------------------------------------------------------------
# The scanners
# ---------------------------------------------------------------------------

def t1_audit() -> tuple[list[dict], str]:
    text = WORKFLOW.read_text(encoding="utf-8")
    try:
        body = text.split("<<'HNX_T1_AUDIT_EOF'\n", 1)[1].split("\n          HNX_T1_AUDIT_EOF", 1)[0]
    except IndexError:
        raise RuntimeError("the audit is no longer in hypernix-security-scan.yml") from None
    with tempfile.TemporaryDirectory(prefix="hnx-audit-") as scratch:
        script = Path(scratch) / "t1_audit.py"
        script.write_text(re.sub(r"(?m)^ {10}", "", body), encoding="utf-8")
        sarif = Path(scratch) / "out.sarif"
        done = _run([sys.executable, "-I", str(script), str(ROOT), str(sarif)], 600)
        if not sarif.is_file():
            raise RuntimeError((done.stderr or done.stdout).strip()[-300:] or "no output")
        run = json.loads(sarif.read_text(encoding="utf-8"))["runs"][0]
    rules = {r["id"]: r for r in run["tool"]["driver"].get("rules", [])}
    out = []
    for result in run.get("results", []):
        rule = rules.get(result["ruleId"], {})
        score = float((rule.get("properties") or {}).get("security-severity", "5.5"))
        where = result["locations"][0]["physicalLocation"]
        out.append(_finding("hypernix-t1-audit", result["ruleId"], _bucket(score),
                            where["artifactLocation"]["uri"], where.get("region", {}).get("startLine", 0),
                            result["message"]["text"],
                            title=(rule.get("shortDescription") or {}).get("text", "")))
    return out, run["tool"]["driver"].get("version", "")


def bandit() -> tuple[list[dict], str]:
    exe = shutil.which("bandit")
    if not exe:
        raise LookupError("not installed (pip install 'bandit[toml]')")
    with tempfile.TemporaryDirectory(prefix="hnx-bandit-") as scratch:
        report = Path(scratch) / "bandit.json"
        config = ["-c", "pyproject.toml"] if "[tool.bandit" in (ROOT / "pyproject.toml").read_text() else []
        _run([exe, "-r", ".", *config, "-x", EXCLUDE, "-f", "json", "-o", str(report), "-q", "--exit-zero"], 1800)
        if not report.is_file():
            raise RuntimeError("bandit wrote no report")
        data = json.loads(report.read_text(encoding="utf-8"))
    out = []
    for issue in data.get("results", []):
        severity = {"HIGH": "high", "MEDIUM": "medium", "LOW": "low"}.get(issue.get("issue_severity", ""), "medium")
        out.append(_finding("bandit", issue.get("test_id", ""), severity, issue.get("filename", ""),
                            issue.get("line_number", 0), issue.get("issue_text", ""),
                            title=issue.get("test_name", ""), confidence=str(issue.get("issue_confidence", "")).lower()))
    return out, _version([exe, "--version"])


def semgrep() -> tuple[list[dict], str]:
    exe = shutil.which("semgrep")
    if not exe:
        raise LookupError("not installed (pip install semgrep)")
    configs = ["p/python", "p/security-audit", "p/owasp-top-ten", "p/secrets"]
    argv = [exe, "scan", "--metrics=off", "--quiet", "--json"]
    for config in configs:
        argv += ["--config", config]
    for skip in ("node_modules", ".venv", "venv", "build", "dist"):
        argv += ["--exclude", skip]
    done = _run(argv + ["."], 3000)
    try:
        data = json.loads(done.stdout or "{}")
    except ValueError:
        raise RuntimeError((done.stderr or "unreadable output").strip()[-300:]) from None
    out = []
    for result in data.get("results", []):
        extra = result.get("extra") or {}
        severity = {"ERROR": "high", "WARNING": "medium", "INFO": "low"}.get(extra.get("severity", ""), "medium")
        rule = result.get("check_id", "")
        out.append(_finding("semgrep", rule.rsplit(".", 1)[-1], severity, result.get("path", ""),
                            (result.get("start") or {}).get("line", 0), extra.get("message", ""), full_rule=rule))
    return out, _version([exe, "--version"])


def _floors() -> list[tuple[str, str, str, str]]:
    """``(ecosystem, package, version, where)`` for every dependency floor and lock pin."""
    found: list[tuple[str, str, str, str]] = []
    pyproject = ROOT / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    lines = pyproject.read_text(encoding="utf-8").splitlines()
    specs = [(s, "") for s in project.get("dependencies", [])]
    for extra, deps in (project.get("optional-dependencies") or {}).items():
        specs += [(s, extra) for s in deps]
    for spec, _extra in specs:
        m = re.match(r"\s*([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(.*)", spec)
        if not m or m.group(1).lower() == "hypernix":
            continue
        low = re.search(r"(>=|==|~=)\s*([0-9][0-9A-Za-z.\-+]*)", m.group(3).split(";")[0])
        if low:
            line = next((n for n, text in enumerate(lines, 1) if f'"{spec}"' in text), 0)
            found.append(("PyPI", m.group(1).lower(), low.group(2), f"pyproject.toml:{line}"))
    for lock in sorted(ROOT.rglob("package-lock.json")):
        if "node_modules" in lock.parts:
            continue
        data = json.loads(lock.read_text(encoding="utf-8"))
        for key, info in (data.get("packages") or {}).items():
            if key and info.get("version") and "node_modules/" in key:
                found.append(("npm", key.rsplit("node_modules/", 1)[-1], info["version"],
                              f"{lock.relative_to(ROOT).as_posix()}:0"))
    for lock in sorted(ROOT.rglob("bun.lock")):
        if "node_modules" in lock.parts:
            continue
        for m in re.finditer(r'\[\s*"((?:@[^/"]+/)?[^@"]+)@([^"]+)"', lock.read_text(encoding="utf-8")):
            found.append(("npm", m.group(1), m.group(2), f"{lock.relative_to(ROOT).as_posix()}:0"))
    return sorted(set(found))


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310  # nosec B310 - a fixed https URL
        return json.load(response)


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310  # nosec B310 - a fixed https osv.dev URL
        return json.load(response)


def osv() -> tuple[list[dict], str]:
    floors = _floors()
    queries = [{"package": {"name": name, "ecosystem": eco}, "version": version} for eco, name, version, _ in floors]
    results: list[dict] = []
    for start in range(0, len(queries), 500):
        results += _post(OSV_BATCH, {"queries": queries[start:start + 500]}).get("results", [])
    hits = [(floor, [v["id"] for v in r.get("vulns", [])]) for floor, r in zip(floors, results, strict=True) if r.get("vulns")]
    ids = sorted({vid for _, vids in hits for vid in vids})
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        details = dict(zip(ids, pool.map(lambda vid: _get(OSV_VULN + vid), ids), strict=True))
    out = []
    for (eco, name, version, where), vids in hits:
        path, _, line = where.partition(":")
        for vid in vids:
            vuln = details.get(vid, {})
            level = str((vuln.get("database_specific") or {}).get("severity", "")).lower()
            severity = {"moderate": "medium"}.get(level, level) if level in (*SEVERITIES, "moderate") else "medium"
            fixed = sorted({ev["fixed"] for aff in vuln.get("affected", [])
                            if (aff.get("package") or {}).get("name", "").lower() == name.lower()
                            for rng in aff.get("ranges", []) for ev in rng.get("events", []) if "fixed" in ev})
            out.append(_finding("osv", vid, severity, path, int(line or 0),
                                f"{name} {version} ({eco}): {vuln.get('summary') or vuln.get('details', '')[:200]}",
                                package=name, version=version, fixed=fixed[-1] if fixed else ""))
    return out, f"osv.dev, {len(floors)} pins checked"


SCANNERS = {
    "hypernix-t1-audit": ("HyperNix T1 audit", t1_audit),
    "bandit": ("Bandit", bandit),
    "semgrep": ("Semgrep", semgrep),
    "osv": ("OSV (dependencies)", osv),
}


def report(skip: set[str] = frozenset()) -> dict:
    tools, findings = [], []
    for name, (label, scan) in SCANNERS.items():
        entry = {"name": name, "label": label, "status": "ok", "detail": "", "version": "", "count": 0}
        if name in skip:
            entry.update(status="skipped", detail="skipped for this build")
        else:
            try:
                found, version = scan()
                entry.update(version=version, count=len(found))
                findings += found
            except LookupError as exc:
                entry.update(status="skipped", detail=str(exc))
            except Exception as exc:  # noqa: BLE001 - one scanner failing is reported, not fatal
                entry.update(status="failed", detail=f"{type(exc).__name__}: {exc}"[:300])
        tools.append(entry)
    order = {s: n for n, s in enumerate(SEVERITIES)}
    findings.sort(key=lambda f: (order[f["severity"]], f["test"], f["tool"], f["path"], f["line"]))
    totals = {s: sum(1 for f in findings if f["severity"] == s and not f["test"]) for s in SEVERITIES}
    totals["tests"] = sum(1 for f in findings if f["test"])
    return {
        "schema": 1,
        "generated": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "commit": os.environ.get("GITHUB_SHA") or _git("rev-parse", "HEAD"),
        "repository": os.environ.get("GITHUB_REPOSITORY", "trail-b1az3r/HyperNix-pip"),
        "tools": tools,
        "totals": totals,
        "findings": findings,
    }


def _git(*args: str) -> str:
    try:
        return _run(["git", *args], 30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--skip", default="", help="comma-separated scanners: " + ", ".join(SCANNERS))
    args = parser.parse_args(argv)
    data = report({s.strip() for s in args.skip.split(",") if s.strip()})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    for tool in data["tools"]:
        print(f"{tool['label']}: {tool['status']}, {tool['count']} finding(s){' -- ' + tool['detail'] if tool['detail'] else ''}")
    print(f"{out}: " + ", ".join(f"{data['totals'][s]} {s}" for s in SEVERITIES) + f", {data['totals']['tests']} in tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
