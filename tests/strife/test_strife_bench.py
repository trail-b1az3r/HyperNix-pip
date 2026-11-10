"""benchmarks/strife_bench.py runs and writes an honest record."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from hypernix.strife.fixtures import make_model

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "strife_bench.py"


def _bench():
    spec = importlib.util.spec_from_file_location("strife_bench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_record_and_target(tmp_path, capsys):
    model = make_model(tmp_path / "m.gguf", arch="qwen2", context=512)
    out = tmp_path / "r.json"
    bench = _bench()
    assert bench.main([str(model["path"]), "-p", "16", "-n", "4", "-r", "2", "--no-llama",
                       "--target", "0.001", "--json", str(out)]) == 0
    record = json.loads(out.read_text())
    strife = record["results"][0]
    assert strife["runtime"] == "strife" and strife["prefill_tps"] > 0
    assert len(strife["decoded_tokens"]) == 2 and record["target_met"]
    assert len(record["sha256"]) == 64
    # An impossible target is reported unmet, and the exit code says so.
    assert bench.main([str(model["path"]), "-p", "8", "-n", "2", "-r", "1", "--no-llama",
                       "--no-hash", "--target", "1e12"]) == 1
    assert "UNMET" in capsys.readouterr().out


def test_llama_comparison_is_skipped_with_a_reason_when_there_is_no_build(tmp_path, monkeypatch):
    bench = _bench()
    monkeypatch.setattr(bench, "_llama_bench_binary", lambda: None)
    row = bench.bench_llama(tmp_path / "x.gguf", prompt_tokens=8, gen_tokens=2, repeat=1, threads=0)
    assert row["skipped"]
