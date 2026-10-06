#!/usr/bin/env python3
"""hyperNix-pip's benchmark suite, run the same way on Python 3.12-3.15.

    python benchmarks/bench.py                   # this interpreter, table to stdout
    python benchmarks/bench.py --json out.json   # and the raw numbers
    python benchmarks/bench.py --repeat 15 --only startup

Every measurement is the median of --repeat runs (and its spread), taken
in fresh subprocesses where startup is what is measured, so one run's
caches never flatter the next. What is measured:

startup     a bare interpreter; ``import hypernix``; the CLIs' ``--help``
            (hnx, hyprslug, gkey, the runner CLI) -- cold is the first
            run after the bytecode cache is cleared, warm the median after
imports     the expensive entry modules: the T1 API app, HyperLink's
            runner, the converter, hyped
init        T1 API construction (config, keys, routes) and one request
            through it; configuration loading; key store creation
data        serialisation (JSON of a chat transcript, the runner history
            file); image conversion (PNG -> WebP) when Pillow is there
native      the ggml-hnx sub-bit decoder through its Python twin
profiling   the deterministic profiler's overhead on a fixed workload
py315       on 3.15 only: PEP 810 deferral on vs filtered off for each
            module that declares __lazy_modules__, and PEP 798 vs the
            legacy flatten

Numbers are wall-clock on whatever machine runs this. Compare runs from
one machine; never across machines.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PY = sys.executable


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["T1_CONFIG_DIR"] = tempfile.mkdtemp(prefix="hnx-bench-cfg-")
    env["HYPERNIX_DEPRECATION_WARNINGS"] = "0"
    env["NO_COLOR"] = "1"
    return env


def _clear_bytecode() -> None:
    for cache in SRC.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def _time_subprocess(argv: list[str], env: dict[str, str]) -> float:
    start = time.perf_counter()
    done = subprocess.run(argv, env=env, capture_output=True, check=False)  # noqa: S603
    elapsed = time.perf_counter() - start
    if done.returncode not in (0, 2):  # argparse --help exits 0; usage errors 2
        raise RuntimeError(f"{argv} exited {done.returncode}: {done.stderr.decode()[-400:]}")
    return elapsed


def _stats(samples: list[float]) -> dict[str, float]:
    return {"median_ms": round(statistics.median(samples) * 1000, 2),
            "min_ms": round(min(samples) * 1000, 2),
            "stdev_ms": round(statistics.pstdev(samples) * 1000, 2),
            "runs": len(samples)}


def _in_process(fn: Callable[[], object], repeat: int, warmup: int = 1) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return _stats(samples)


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

STARTUP = {
    "python -c pass": ["-c", "pass"],
    "import hypernix": ["-c", "import hypernix"],
    "hnx --help": ["-m", "hypernix", "--help"],
    "hyprslug --help": ["-m", "hypernix.quant.hyprslug_cli", "--help"],
    "gkey --help": ["-m", "hypernix.security.gkey_cli", "--help"],
    "runner CLI --help": ["-m", "hypernix.t1api.runner_cli", "--help"],
}

IMPORTS = {
    "import hypernix.t1api.app": "hypernix.t1api.app",
    "import hypernix.hyperlink.managed": "hypernix.hyperlink.managed",
    "import hypernix.interfaces.hyped": "hypernix.interfaces.hyped",
    "import hypernix.quant.hyprslug": "hypernix.quant.hyprslug",
    "import hypernix.quant.convert": "hypernix.quant.convert",
}


def bench_startup(repeat: int) -> dict:
    env = _env()
    out = {}
    for name, args in STARTUP.items():
        _clear_bytecode()
        cold = _time_subprocess([PY, *args], env)
        warm = [_time_subprocess([PY, *args], env) for _ in range(repeat)]
        out[name] = {"cold_ms": round(cold * 1000, 2), **_stats(warm)}
    return out


def bench_imports(repeat: int) -> dict:
    env = _env()
    out = {}
    for name, module in IMPORTS.items():
        try:
            samples = [_time_subprocess([PY, "-c", f"import {module}"], env) for _ in range(repeat)]
        except RuntimeError as exc:
            out[name] = {"skipped": str(exc).splitlines()[-1][:160]}
            continue
        out[name] = _stats(samples)
    return out


def bench_init(repeat: int) -> dict:
    out: dict = {}
    os.environ.setdefault("T1_CONFIG_DIR", tempfile.mkdtemp(prefix="hnx-bench-cfg-"))
    try:
        from fastapi.testclient import TestClient

        from hypernix.security.gatekeeper import Gatekeeper
        from hypernix.security.keymaster import Keymaster, KeyScope, KeyType
        from hypernix.t1api.app import create_app
        from hypernix.t1api.config import T1APIConfig
    except ImportError as exc:
        return {"skipped": f"t1api extra not installed ({exc.name})"}

    def config() -> T1APIConfig:
        work = Path(tempfile.mkdtemp(prefix="hnx-bench-"))
        return T1APIConfig(token_secret="benchmark-secret-value-long-enough",
                           db_path=str(work / "t1.db"), module_storage_dir=str(work / "m"),
                           hyperlink_files_dir=str(work / "f"), default_plan="free",
                           web_enabled=False)

    out["T1APIConfig()"] = _in_process(config, repeat)

    def keystore():
        work = Path(tempfile.mkdtemp(prefix="hnx-bench-km-"))
        km = Keymaster(store_dir=work, auto_rotate=False)
        km.create(key_type=KeyType.USER, scopes={KeyScope.READ})
        return km

    out["Keymaster + one key"] = _in_process(keystore, repeat)

    def app():
        import contextlib
        import io

        work = Path(tempfile.mkdtemp(prefix="hnx-bench-app-"))
        km = Keymaster(store_dir=work / "km", auto_rotate=False)
        gk = Gatekeeper(keymaster=km, data_dir=work / "gk", log_to_file=False)
        # A fresh server prints its one-time bootstrap key; not a result.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return create_app(config=config(), keymaster=km, gatekeeper=gk), km

    out["create_app()"] = _in_process(app, repeat)

    application, km = app()
    client = TestClient(application, client=("127.0.0.1", 5000))
    key = km.create(key_type=KeyType.USER, scopes={KeyScope.READ}).key
    headers = {"Authorization": f"Bearer {key}"}
    out["GET /health"] = _in_process(lambda: client.get("/health"), repeat * 10)
    out["authenticated GET /hyperlink/models"] = _in_process(
        lambda: client.get("/hyperlink/models", headers=headers), repeat * 5)
    return out


def bench_data(repeat: int) -> dict:
    out: dict = {}
    transcript = {"messages": [{"role": "user" if i % 2 else "assistant",
                                "content": "word " * 60 + str(i)} for i in range(400)]}
    out["json dumps (400-message chat)"] = _in_process(lambda: json.dumps(transcript), repeat * 10)
    blob = json.dumps(transcript)
    out["json loads (400-message chat)"] = _in_process(lambda: json.loads(blob), repeat * 10)

    from hypernix.hyperlink import runner_history as rh

    work = Path(tempfile.mkdtemp(prefix="hnx-bench-rh-")) / "h.json"
    for i in range(rh.MAX_RECORDS):
        rh.record(rh.LoadRecord(model_id=f"m{i % 7}", backend="cuda"), work)
    out["runner history read + choose (200 loads)"] = _in_process(
        lambda: rh.choose(rh.read(work)), repeat * 5)

    try:
        import io

        from PIL import Image, ImageDraw

        from hypernix.hyperlink import imagecodec
    except ImportError:
        out["image PNG -> WebP (640x480)"] = {"skipped": "Pillow not installed"}
    else:
        image = Image.new("RGB", (640, 480), "white")
        draw = ImageDraw.Draw(image)
        for x in range(0, 640, 16):
            draw.line([(x, 0), (640 - x, 480)], fill=(x % 255, 80, 160))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        png = buffer.getvalue()
        out["image PNG -> WebP (640x480)"] = _in_process(lambda: imagecodec.compress(png, "a.png"), repeat)
    return out


def bench_native(repeat: int) -> dict:
    try:
        from hypernix.quant.subbit import BLOCK_SIZE, dequantize_array, quantize_tensor
    except ImportError as exc:
        return {"skipped": str(exc)}
    import random

    rng = random.Random(0)
    values = [rng.gauss(0, 0.08) for _ in range(BLOCK_SIZE * 64)]
    packed = bytes(quantize_tensor(values, "sign_scale_l"))
    return {
        "sub-bit quantise (64 blocks)": _in_process(lambda: quantize_tensor(values, "sign_scale_l"), repeat),
        "sub-bit dequantise (64 blocks)": _in_process(lambda: dequantize_array(packed, "sign_scale_l"), repeat),
    }


def _workload() -> int:
    total = 0
    for i in range(20000):
        total += len(str(i)) * (i % 7)
    return total


def bench_profiling(repeat: int) -> dict:
    from hypernix import _profiling

    plain = _in_process(_workload, repeat)
    traced = _in_process(lambda: _profiling.profile_call(_workload), repeat)
    overhead = round(traced["median_ms"] / plain["median_ms"], 2) if plain["median_ms"] else None
    return {"workload": plain, f"workload under {_profiling.BACKEND}": traced,
            "overhead (x)": {"value": overhead}}


def bench_py315(repeat: int) -> dict:
    if sys.version_info < (3, 15):
        return {"skipped": "3.15 only"}
    out: dict = {}
    env = _env()
    off = "import sys\nsys.set_lazy_imports_filter(lambda *a: False)\n"
    for module in ("hypernix.quant.llamaquants", "hypernix.models.download"):
        on = [_time_subprocess([PY, "-c", f"import {module}"], env) for _ in range(repeat)]
        filtered = [_time_subprocess([PY, "-c", off + f"import {module}"], env) for _ in range(repeat)]
        out[f"import {module}: PEP 810 on"] = _stats(on)
        out[f"import {module}: PEP 810 off"] = _stats(filtered)
    from hypernix._compat import _iter_legacy, _iter_py315

    rows = [list(range(i % 9)) for i in range(5000)]
    out["flatten 5000 rows: PEP 798"] = _in_process(lambda: _iter_py315.flatten(rows), repeat * 5)
    out["flatten 5000 rows: itertools.chain"] = _in_process(lambda: _iter_legacy.flatten(rows), repeat * 5)
    maps = [{str(i % 50): i} for i in range(5000)]
    out["merge 5000 dicts: PEP 798"] = _in_process(lambda: _iter_py315.merge(maps), repeat * 5)
    out["merge 5000 dicts: update loop"] = _in_process(lambda: _iter_legacy.merge(maps), repeat * 5)
    return out


GROUPS: dict[str, Callable[[int], dict]] = {
    "startup": bench_startup, "imports": bench_imports, "init": bench_init,
    "data": bench_data, "native": bench_native, "profiling": bench_profiling, "py315": bench_py315,
}


def _table(results: dict) -> str:
    lines = [f"Python {results['python']} on {results['platform']}", ""]
    for group, rows in results["groups"].items():
        lines.append(f"[{group}]")
        if "skipped" in rows:
            lines.append(f"  skipped: {rows['skipped']}")
            continue
        for name, row in rows.items():
            if "skipped" in row:
                lines.append(f"  {name:<52} skipped: {row['skipped']}")
            elif "value" in row:
                lines.append(f"  {name:<52} {row['value']}")
            else:
                cold = f"  cold {row['cold_ms']:>8.1f}" if "cold_ms" in row else ""
                lines.append(f"  {name:<52} {row['median_ms']:>9.2f} ms  ±{row['stdev_ms']:.2f}{cold}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--only", action="append", choices=sorted(GROUPS))
    parser.add_argument("--json", type=Path, help="write the raw results here")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(SRC))

    results = {"python": platform.python_version(), "implementation": platform.python_implementation(),
               "platform": f"{platform.system()} {platform.machine()}", "repeat": args.repeat,
               "groups": {}}
    for name in args.only or list(GROUPS):
        results["groups"][name] = GROUPS[name](args.repeat)
    print(_table(results))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
