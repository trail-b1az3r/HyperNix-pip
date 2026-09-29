"""The Codacy SARIF filter: security findings kept, formatting dropped, limits kept."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "codacy_sarif_filter", ROOT / ".github" / "scripts" / "codacy_sarif_filter.py")
flt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flt)


def _result(rule, path, level="warning"):
    return {"ruleId": rule, "level": level, "message": {"text": rule},
            "locations": [{"physicalLocation": {"artifactLocation": {"uri": path}}}]}


def _run(tool, results):
    return {"tool": {"driver": {"name": tool}}, "results": results}


def test_security_findings_stay_and_formatting_goes(tmp_path):
    sarif = {"version": "2.1.0", "runs": [
        _run("Bandit", [_result("B608", "src/hypernix/t1api/db.py"),
                        _result("B101", "tests/t1api/test_db.py")]),
        _run("pylint", [_result("PyLint_C0301", "src/a.py"), _result("PyLint_R0913", "src/a.py"),
                        _result("PyLint_W0703", "src/a.py"), _result("PyLint_E1101", "src/a.py"),
                        _result("PyLint_W0612", "tests/x.py")]),
        _run("duplication", [_result("dup", "src/a.py")]),
        _run("remark-lint", [_result("md", "wiki/Home.md")]),
        _run("flawfinder", [_result("FF1", "native/ggml-hnx/ggml-hnx.c", "error")]),
    ]}
    path = tmp_path / "results.sarif"
    path.write_text(json.dumps(sarif))
    assert flt.main(["x", str(path)]) == 0
    runs = {r["tool"]["driver"]["name"]: [x["ruleId"] for x in r["results"]]
            for r in json.loads(path.read_text())["runs"]}
    assert runs["Bandit"] == ["B608"]
    assert sorted(runs["pylint"]) == ["PyLint_E1101", "PyLint_W0703"]
    assert runs["duplication"] == [] and runs["remark-lint"] == []
    assert runs["flawfinder"] == ["FF1"]


def test_a_run_is_capped_most_severe_first(monkeypatch):
    monkeypatch.setattr(flt, "MAX_RESULTS", 2)
    data = {"runs": [_run("cppcheck", [_result("a", "x.c", "note"), _result("b", "x.c", "error"),
                                        _result("c", "x.c", "warning")])]}
    report = flt.filter_sarif(data)
    assert [r["ruleId"] for r in data["runs"][0]["results"]] == ["b", "c"]
    assert report == [("cppcheck", 3, 2)]


def test_the_workflow_sets_the_charset_and_filters():
    text = (ROOT / ".github" / "workflows" / "codacy.yml").read_text()
    assert "-Dfile.encoding=UTF-8" in text
    assert "codacy_sarif_filter.py results.sarif" in text
    assert text.index("codacy_sarif_filter.py results") < text.index("uses: github/codeql-action/upload-sarif")
