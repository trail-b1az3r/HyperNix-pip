"""``hnx strife benchmark`` runs, checks quality, and writes an honest record."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from hypernix.strife import bench
from hypernix.strife.cli import cli_main
from hypernix.strife.fixtures import make_model

REPO = Path(__file__).resolve().parents[2]


def test_record_csv_report_and_target(tmp_path, capsys):
    model = make_model(tmp_path / "m.gguf", arch="qwen2", context=512)
    out = tmp_path / "out"
    assert cli_main(["benchmark", str(model["path"]), "-p", "16", "-n", "6", "-r", "2",
                     "--warmup", "1", "--no-llama", "--target", "0.001",
                     "--out-dir", str(out)]) == 0
    record = json.loads((out / "strife-bench.json").read_text())
    strife = record["results"][0]
    assert strife["runtime"].startswith("strife (") and strife["prefill_tps"] > 0
    assert strife["first_token_seconds"] > 0 and strife["token_latency_ms"]["samples"] > 0
    assert strife["cold"]["prefill_tps"] > 0            # reported apart from the warm runs
    assert strife["memory"]["peak_rss_bytes"] > 0
    assert len(strife["decoded_tokens"]) == 2 and record["target_met"]
    assert strife["quality"]["deterministic"] and record["quality"]["ok"]
    assert len(record["sha256"]) == 64 and record["hypernix_version"]
    assert record["warmup"] == 1 and record["sampling"]["temperature"] == 0.8
    rows = list(csv.DictReader((out / "strife-bench.csv").open()))
    assert rows[0]["runtime"] == strife["runtime"] and float(rows[0]["decode_tps"]) > 0
    report = (out / "strife-bench.md").read_text()
    assert "Cold run" in report and "measured in this run" in report
    # An impossible target is reported unmet, and the exit code says so.
    assert bench.main([str(model["path"]), "-p", "8", "-n", "2", "-r", "1", "--no-llama",
                       "--no-hash", "--target", "1e12"]) == 1
    assert "UNMET" in capsys.readouterr().out


def test_a_broken_quality_check_fails_the_run(tmp_path, monkeypatch, capsys):
    model = make_model(tmp_path / "m.gguf", arch="qwen2", context=512)
    real = bench._one_run
    calls = {"n": 0}

    def flaky(engine, ids, gen_tokens, sampling):
        result = real(engine, ids, gen_tokens, sampling)
        if sampling.temperature == 0:
            calls["n"] += 1
            if calls["n"] == 2:
                result["tokens"] = [*result["tokens"][:-1], -1]   # the repeat differs
        return result

    monkeypatch.setattr(bench, "_one_run", flaky)
    assert bench.main([str(model["path"]), "-p", "8", "-n", "4", "-r", "1", "--no-llama",
                       "--no-hash"]) == 3
    assert "quality check FAILED" in capsys.readouterr().err


def test_llama_comparison_is_skipped_with_a_reason_when_there_is_no_build(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "_llama_bench_binary", lambda: None)
    row = bench.bench_llama(tmp_path / "x.gguf", prompt_tokens=8, gen_tokens=2, repeat=1, threads=0)
    assert row["skipped"]


def test_a_missing_model_is_an_error_not_a_download(tmp_path, capsys):
    assert bench.main([str(tmp_path / "nope.gguf"), "--no-llama"]) == 2
    assert "strife benchmark:" in capsys.readouterr().err


def test_the_script_in_benchmarks_is_the_same_program(tmp_path):
    model = make_model(tmp_path / "m.gguf", arch="qwen2", context=256)
    done = subprocess.run([sys.executable, str(REPO / "benchmarks" / "strife_bench.py"),
                           str(model["path"]), "-p", "8", "-n", "2", "-r", "1", "--no-llama",
                           "--no-hash"], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert "decode tok/s" in done.stdout
