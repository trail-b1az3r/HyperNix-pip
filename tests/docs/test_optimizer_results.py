"""The V7 page's measured tables are the committed benchmark results."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results" / "optimizers-0.73.0a2"


def test_the_page_matches_the_results():
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "render_optimizer_results.py"),
                          "--check"], capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stderr


def test_every_result_folder_is_described_and_complete():
    readme = (RESULTS / "README.md").read_text(encoding="utf-8")
    folders = sorted(p for p in RESULTS.iterdir() if p.is_dir())
    assert {p.name for p in folders} >= {"main", "high-lr", "ablations", "in-band", "band",
                                         "band-edge"}
    for folder in folders:
        assert f"`{folder.name}/`" in readme, folder.name
        for suffix in ("json", "csv", "md"):
            assert (folder / f"optimizer-bench.{suffix}").exists(), (folder.name, suffix)
        assert (RESULTS / f"{folder.name}.log").exists(), folder.name
        record = json.loads((folder / "optimizer-bench.json").read_text(encoding="utf-8"))
        assert record["hypernix_version"].startswith("0.73.0a2")
        assert record["machine"]["device"] == "cpu"
