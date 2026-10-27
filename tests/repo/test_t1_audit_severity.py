"""The T1 audit in hypernix-security-scan.yml, run against a made-up tree.

Three things it got wrong, each a "high" alert that was not one:

* A fake secret in a test was reported at level "note" -- and GitHub filed
  it as high anyway, because an alert's severity is the *rule's*
  security-severity, not the result's level. 59 of the 60 high alerts.
* ``sed -n "s/^T1_TOKEN_SECRET=//p"`` reads a setting; its ``//p`` was
  taken for a 3-character secret.
* ``T1_BOOTSTRAP_KEY=1`` is an on/off switch, not a key.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "hypernix-security-scan.yml"
# hypernix-t1-audit: ignore-file  (the made-up secrets below are the point)


def _audit(tmp_path: Path) -> Path:
    text = WORKFLOW.read_text(encoding="utf-8")
    body = text.split("<<'HNX_T1_AUDIT_EOF'\n", 1)[1].split("\n          HNX_T1_AUDIT_EOF", 1)[0]
    script = tmp_path / "t1_audit.py"
    script.write_text(re.sub(r"(?m)^ {10}", "", body), encoding="utf-8")
    return script


def _run(tmp_path: Path, files: dict[str, str]) -> tuple[list[dict], dict[str, float]]:
    tree = tmp_path / "tree"
    for name, content in files.items():
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text(content, encoding="utf-8")
    out = tmp_path / "out.sarif"
    subprocess.run([sys.executable, str(_audit(tmp_path)), str(tree), str(out)],
                   check=True, capture_output=True, text=True)
    run = json.loads(out.read_text(encoding="utf-8"))["runs"][0]
    severity = {r["id"]: float(r["properties"]["security-severity"]) for r in run["tool"]["driver"]["rules"]}
    return run["results"], severity


SECRET = 'config = T1APIConfig(token_secret="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")\n'


def test_a_secret_in_a_test_is_a_low_alert_and_in_code_a_high_one(tmp_path):
    results, severity = _run(tmp_path, {"tests/test_x.py": SECRET, "src/app/settings.py": SECRET})
    by_path = {r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]: r for r in results}
    in_test, in_code = by_path["tests/test_x.py"], by_path["src/app/settings.py"]
    assert in_test["ruleId"] == "HNX-T1-001.example" and in_test["level"] == "note"
    assert severity[in_test["ruleId"]] < 4.0      # GitHub: low
    assert in_code["ruleId"] == "HNX-T1-001" and in_code["level"] == "error"
    assert severity[in_code["ruleId"]] >= 7.0     # GitHub: high


def test_reading_a_setting_with_sed_is_not_a_secret(tmp_path):
    results, _ = _run(tmp_path, {"scripts/run.sh":
                                 'T1_TOKEN_SECRET="$(sed -n "s/^T1_TOKEN_SECRET=//p" "$DIR/.env" | tail -1)"\n'})
    assert results == []


def test_the_bootstrap_switch_is_not_a_secret(tmp_path):
    results, _ = _run(tmp_path, {"deploy/server.env": "T1_BOOTSTRAP_KEY=1\nT1_BOOTSTRAP_ADMIN_KEY=bbbbbbbbbbbbbbbbbbbb\n"})
    flagged = [r["message"]["text"] for r in results]
    assert not any("T1_BOOTSTRAP_KEY " in m for m in flagged)
    assert any("T1_BOOTSTRAP_ADMIN_KEY" in m for m in flagged)
