"""``hnx optimizer benchmark``: a short synthetic run, and what it records."""
from __future__ import annotations

import csv
import json

import pytest
import torch

from hypernix.optimizers import benchmark
from hypernix.optimizers.optimizer_framework_v2 import OptimizerFrameworkV2

FAST = dict(workloads=("mlp",), lrs=(1e-3, 1e-2), steps=12, optimizer_steps=4, warmup=1)


@pytest.fixture(scope="module")
def record():
    out = benchmark.run_benchmark(optimizers=("adamw", "v6", "v6v", "v7", "v7q"), **FAST)
    out["measured"] = benchmark.measured_table(out)
    return out


def test_default_set_is_the_one_the_comparison_names():
    assert benchmark.DEFAULT_OPTIMIZERS == ("adamw", "v5", "v5plus", "v5s", "v6", "v6v", "v7",
                                            "v7mini", "v7q", "v7s", "v7m")
    assert {"small-transformer", "medium-transformer", "mlp", "embedding-heavy",
            "mixed-1d-2d"} <= set(benchmark.WORKLOADS)


def test_record_keeps_optimizer_only_apart_from_the_full_step(record):
    json.dumps(record)
    entry = record["workloads"]["mlp"]
    assert entry["parameters"] > 0
    for key in ("adamw", "v6", "v7", "v7q"):
        got = entry["optimizers"][key]
        assert got["available"] is True
        only = got["optimizer_only"]
        assert only["ms_per_step"] > 0 and only["state_bytes"] > 0
        assert only["parameters"] == entry["parameters"]
        assert len(got["runs"]) == 2
        for run in got["runs"]:
            assert run["ms_per_step"] >= run["optimizer_ms_per_step"] > 0
            assert 0 < run["optimizer_share"] <= 1
            assert run["steps"] == 12 and run["loss_at_step"]["step"] == 6
            assert run["peak_device_bytes"] is None        # a CPU run: not measured
    assert entry["optimizers"]["adamw"]["optimizer_only"]["state_bytes_per_parameter"] == pytest.approx(8.0, abs=0.01)
    assert entry["optimizers"]["v7q"]["optimizer_only"]["state_bytes_per_parameter"] < 4.5


def test_what_cannot_run_is_not_available_rather_than_a_number(record):
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    v6v = record["workloads"]["mlp"]["optimizers"]["v6v"]
    assert v6v == {"available": False, "reason": v6v["reason"]} and "CUDA" in v6v["reason"]
    assert "v6v" not in record["measured"]
    assert "N/A" in benchmark.render_report(record)


def test_unmeasured_things_are_named(record):
    assert "gpu_utilization" in record["not_measured"]
    assert record["settings"]["seed"] == 0 and record["machine"]["device"] == "cpu"


def test_quality_is_reproducible():
    runs = [benchmark.bench_training("v7", benchmark.WORKLOADS["mlp"], steps=10, seed=3,
                                     device="cpu", lr=1e-2) for _ in range(2)]
    assert runs[0]["final_loss"] == runs[1]["final_loss"]
    assert runs[0]["validation_loss"] == runs[1]["validation_loss"]


def test_measured_table_feeds_the_framework(record):
    measured = record["measured"]
    assert {"adamw", "v6", "v7", "v7q"} == set(measured)
    rows = {r["key"]: r for r in OptimizerFrameworkV2.table(measured)}
    assert rows["v7"]["measured"]["ms_per_step"] == measured["v7"]["ms_per_step"]
    chosen = OptimizerFrameworkV2.auto_select(objective="throughput", measured=measured)
    faster = "v7q" if measured["v7q"]["ms_per_step"] < measured["v6"]["ms_per_step"] else "v6"
    assert chosen.key == faster


def test_outputs(tmp_path, record):
    benchmark.write_csv(record, tmp_path / "b.csv")
    with open(tmp_path / "b.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["optimizer"] for r in rows] == ["adamw", "v6", "v6v", "v7", "v7q"]
    report = benchmark.render_report(record)
    assert "## mlp" in report and "measured in this run" in report


def test_command_line(tmp_path, capsys):
    from hypernix.interfaces.cli import main

    code = main(["optimizer", "benchmark", "--optimizers", "adamw,v7", "--workloads", "mlp",
                 "--lrs", "1e-2", "--steps", "8", "--optimizer-steps", "2", "--warmup", "1",
                 "--out-dir", str(tmp_path)])
    assert code == 0
    assert "adamw" in capsys.readouterr().out
    saved = json.loads((tmp_path / "optimizer-bench.json").read_text())
    assert set(saved["measured"]) == {"adamw", "v7"}
    assert (tmp_path / "optimizer-bench.csv").exists()
    assert (tmp_path / "optimizer-bench.md").exists()
    assert main(["optimizer", "list", "--measured", str(tmp_path / "optimizer-bench.json")]) == 0
    assert "measured:" in capsys.readouterr().out
    assert main(["optimizer", "select", "--params", "100000000", "--verified"]) == 0
    assert "v7mini" in capsys.readouterr().out
    assert main(["optimizer", "select", "--params", "100000000"]) == 0
    assert capsys.readouterr().out.startswith("v7:")
    assert main(["optimizer", "benchmark", "--workloads", "nope"]) == 2
    assert main(["optimizer", "frobnicate"]) == 2
