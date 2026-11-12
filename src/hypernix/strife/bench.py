"""strife.bench — Strife's benchmark: one GGUF, measured, next to llama.cpp.

    hnx strife benchmark model.gguf
    hnx strife benchmark model.gguf -p 512 -n 128 -r 5 --out-dir results/
    hnx strife benchmark model.gguf --target 8        # decode tok/s to meet; exit 1 if unmet
    hnx strife benchmark model.gguf --numpy           # also without the native kernels
    hnx strife benchmark model.gguf --gpu-layers -1   # as many blocks on the GPU as fit

(``python benchmarks/strife_bench.py`` is the same program.) It takes a
local file or a model already on this machine, and never downloads one.

What is measured. A **cold** run comes first and is reported on its own:
it pays for page-ins and first-touch decoding, and is not mixed into the
rest. Then ``--warmup`` runs are discarded, and ``--repeat`` runs are
measured, each in a fresh session so none reuses another's prefix:

load        seconds to open the model
prefill     prompt tokens per second (``-p`` tokens)
first token seconds from the request to the first generated token
decode      generated tokens per second, and the time between tokens as a
            distribution (p50, p90, p99, max)
memory      peak resident memory of this process, the decode cache, and
            device memory when layers are offloaded

Every figure is the median of the repeats with its spread. Everything
here is measured; nothing is estimated.

Quality. Speed from a broken model is not a result, so each run also
generates greedily from a fixed prompt twice and checks the two agree, and
with ``--numpy`` checks that the native kernels and numpy produce the same
tokens. A failed check is in the record and fails the run.

Comparison. When a llama.cpp build is found (the search the runner uses),
its ``llama-bench`` runs on the same file with the same ``-p``/``-n``,
threads and GPU layers. When none is found the row says so; nothing is
filled in.

``--target`` states a decode speed to meet. Below it, the record says
**unmet** and the exit code is 1.

Output: a table, and with ``--json``/``--csv``/``--report`` (or
``--out-dir`` for all three) a machine-readable record, a one-row-per-
runtime CSV, and a Markdown report. The record names the file by SHA-256
and carries the HyperNix version, the machine, and every setting, so two
records can be compared honestly. Compare runs from one machine; never
across machines.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

#: The fixed prompt of the quality check, as token-id offsets into the
#: middle of the vocabulary (the same for every runtime path).
_QUALITY_PROMPT = (3, 141, 59, 26, 53, 58, 97, 93, 23, 84, 62, 64)


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 3) if values else 0.0


def _spread(values: list[float]) -> float:
    return round(max(values) - min(values), 3) if len(values) > 1 else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 3)


def _peak_rss_bytes() -> int:
    """Peak resident memory of this process, as the OS counts it."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:                      # Windows
        try:
            import psutil

            info = psutil.Process().memory_info()
            return int(getattr(info, "peak_wset", info.rss))
        except Exception:  # noqa: BLE001 - a missing figure is reported as 0
            return 0


def _one_run(engine: Any, ids: list[int], gen_tokens: int, sampling: Any) -> dict[str, Any]:
    """One request in a fresh session, with the time of every token."""
    session = engine.new_session()
    started = time.perf_counter()
    stamps: list[float] = []
    final: dict[str, Any] = {}
    try:
        for event in engine.stream(prompt_ids=ids, session=session, max_tokens=gen_tokens,
                                   sampling=sampling):
            if event["type"] == "token":
                stamps.append(time.perf_counter())
            elif event["type"] == "done":
                final = event
    finally:
        engine.close_session(session)
    stats = final.get("stats", {})
    return {
        "prefill_tps": float(stats.get("prefill_tokens_per_second") or 0),
        "decode_tps": float(stats.get("decode_tokens_per_second") or 0),
        "first_token_seconds": (stamps[0] - started) if stamps else 0.0,
        "gaps_ms": [(b - a) * 1000 for a, b in zip(stamps, stamps[1:], strict=False)],
        "tokens": list(final.get("tokens", [])),
        "stop_reason": final.get("stop_reason", ""),
    }


def bench_strife(path: Path, *, prompt_tokens: int, gen_tokens: int, repeat: int,
                 ctx: int, threads: int, seed: int, kernels: str | None = None,
                 warmup: int = 0, gpu_layers: int = 0, gpu_device: str | None = None,
                 temperature: float = 0.8) -> dict[str, Any]:
    import numpy as np

    from hypernix.strife import Engine, SamplerConfig

    started = time.perf_counter()
    engine = Engine(path, n_ctx=ctx or max(prompt_tokens + gen_tokens + 8, 256),
                    threads=threads or None, kernels=kernels, gpu_layers=gpu_layers,
                    gpu_device=gpu_device)
    load_s = time.perf_counter() - started
    try:
        vocab = engine.report.n_vocab
        rng = np.random.default_rng(seed)
        # Ordinary token ids from the middle of the vocabulary: the speed of
        # a forward pass does not depend on which ids, and special tokens at
        # either end are kept out.
        low, high = min(256, vocab // 4), max(min(256, vocab // 4) + 1, vocab - 256)

        def prompt() -> list[int]:
            return rng.integers(low, high, size=prompt_tokens).tolist()

        def sampling(i: int) -> Any:
            return SamplerConfig(temperature=temperature, seed=seed + i)

        cold = _one_run(engine, prompt(), gen_tokens, sampling(10_000))
        for i in range(max(0, warmup)):
            _one_run(engine, prompt(), gen_tokens, sampling(20_000 + i))
        runs = [_one_run(engine, prompt(), gen_tokens, sampling(i)) for i in range(max(1, repeat))]

        # Quality: greedy from a fixed prompt, twice. The two must agree.
        fixed = [low + (offset % max(1, high - low)) for offset in _QUALITY_PROMPT]
        greedy = SamplerConfig(temperature=0)
        first = _one_run(engine, fixed, min(16, gen_tokens), greedy)["tokens"]
        second = _one_run(engine, fixed, min(16, gen_tokens), greedy)["tokens"]

        prefill = [r["prefill_tps"] for r in runs]
        decode = [r["decode_tps"] for r in runs if r["decode_tps"]]
        ttft = [r["first_token_seconds"] for r in runs if r["first_token_seconds"]]
        gaps = [g for r in runs for g in r["gaps_ms"]]
        status = engine.status()
        placement = {k: v for k, v in (status.get("placement") or {}).items() if k != "hardware"}
        device = (status.get("weights") or {}).get("device") or {}
        return {
            "runtime": ("strife (native)" if engine.weights.native_matmul else "strife (numpy)")
                       + (" +gpu" if placement.get("gpu_layers_used") else ""),
            "kernels": status.get("kernels", {}).get("note", ""),
            "backend": placement.get("backend", ""),
            "threads": engine.threads,
            "gpu_layers_requested": placement.get("gpu_layers_requested", 0),
            "gpu_layers_used": placement.get("gpu_layers_used", 0),
            "load_seconds": round(load_s, 3),
            "cold": {"prefill_tps": round(cold["prefill_tps"], 3),
                     "decode_tps": round(cold["decode_tps"], 3),
                     "first_token_seconds": round(cold["first_token_seconds"], 4)},
            "prefill_tps": _median(prefill), "prefill_spread": _spread(prefill),
            "decode_tps": _median(decode), "decode_spread": _spread(decode),
            "first_token_seconds": _median(ttft), "first_token_spread": _spread(ttft),
            "token_latency_ms": {"p50": _percentile(gaps, 0.50), "p90": _percentile(gaps, 0.90),
                                 "p99": _percentile(gaps, 0.99),
                                 "max": round(max(gaps), 3) if gaps else 0.0,
                                 "samples": len(gaps)},
            "decoded_tokens": [len(r["tokens"]) for r in runs],
            "stop_reasons": sorted({r["stop_reason"] for r in runs}),
            "quality": {"deterministic": bool(first) and first == second, "tokens": first},
            "architecture": engine.report.architecture,
            "quant_types": dict(engine.report.types),
            "context": engine.n_ctx,
            "weights_bytes": engine.report.weight_bytes,
            "memory": {"peak_rss_bytes": _peak_rss_bytes(),
                       "decode_cache_peak_bytes": (status.get("weights") or {}).get(
                           "peak_cached_bytes", 0),
                       "mapped_bytes": (status.get("weights") or {}).get("mapped_bytes", 0),
                       "device_bytes": device.get("bytes", 0),
                       "device_peak_bytes": device.get("peak_bytes", 0)},
            "placement": placement,
        }
    finally:
        engine.close()


def _llama_bench_binary() -> Path | None:
    try:
        from hypernix.quant.runtime_bridge import BridgeError, find_build

        build = find_build()
    except (ImportError, BridgeError):
        return None
    for name in ("llama-bench", "llama-bench.exe"):
        candidate = build.bin_dir / name
        if candidate.is_file():
            return candidate
    found = shutil.which("llama-bench")
    return Path(found) if found else None


def bench_llama(path: Path, *, prompt_tokens: int, gen_tokens: int, repeat: int,
                threads: int, gpu_layers: int = 0) -> dict[str, Any]:
    binary = _llama_bench_binary()
    if binary is None:
        return {"runtime": "llama.cpp", "skipped": "no llama.cpp build with llama-bench was found"}
    argv = [str(binary), "-m", str(path), "-p", str(prompt_tokens), "-n", str(gen_tokens),
            "-r", str(max(1, repeat)), "-ngl", str(max(0, gpu_layers) if gpu_layers >= 0 else 999),
            "-o", "json"]
    if threads:
        argv += ["-t", str(threads)]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=3600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"runtime": "llama.cpp", "skipped": f"llama-bench failed to run: {exc}"}
    if done.returncode != 0:
        return {"runtime": "llama.cpp",
                "skipped": f"llama-bench exited {done.returncode}: {done.stderr[-400:]}"}
    try:
        rows = json.loads(done.stdout)
    except ValueError:
        return {"runtime": "llama.cpp", "skipped": "llama-bench printed no JSON"}
    out: dict[str, Any] = {"runtime": "llama.cpp", "binary": str(binary),
                           "build": rows[0].get("build_commit", "") if rows else ""}
    for row in rows:
        if row.get("n_prompt") and not row.get("n_gen"):
            out["prefill_tps"] = round(float(row.get("avg_ts") or 0), 3)
        elif row.get("n_gen") and not row.get("n_prompt"):
            out["decode_tps"] = round(float(row.get("avg_ts") or 0), 3)
    out["note"] = f"llama-bench with -ngl {argv[argv.index('-ngl') + 1]}"
    return out


_CSV_COLUMNS = ("runtime", "backend", "threads", "gpu_layers_used", "load_seconds",
                "prefill_tps", "prefill_spread", "decode_tps", "decode_spread",
                "first_token_seconds", "latency_p50_ms", "latency_p90_ms", "latency_p99_ms",
                "cold_prefill_tps", "cold_decode_tps", "peak_rss_bytes", "device_peak_bytes",
                "quality_deterministic", "skipped")


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    latency, cold, memory = row.get("token_latency_ms", {}), row.get("cold", {}), row.get("memory", {})
    flat = {**{k: row.get(k, "") for k in _CSV_COLUMNS},
            "latency_p50_ms": latency.get("p50", ""), "latency_p90_ms": latency.get("p90", ""),
            "latency_p99_ms": latency.get("p99", ""),
            "cold_prefill_tps": cold.get("prefill_tps", ""),
            "cold_decode_tps": cold.get("decode_tps", ""),
            "peak_rss_bytes": memory.get("peak_rss_bytes", ""),
            "device_peak_bytes": memory.get("device_peak_bytes", ""),
            "quality_deterministic": row.get("quality", {}).get("deterministic", "")}
    return flat


def write_csv(record: dict[str, Any], path: Path) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in record["results"]:
            writer.writerow(_csv_row(row))


def render_report(record: dict[str, Any]) -> str:
    """The record as Markdown, for a person."""
    def gib(n: int) -> str:
        return f"{n / 2**30:.2f} GiB" if n >= 2**30 else f"{n / 2**20:.0f} MiB"

    machine = record["machine"]
    lines = [f"# Strife benchmark: {Path(record['model']).name}", "",
             f"- When: {record['when']}",
             f"- HyperNix: {record['hypernix_version']}",
             f"- Model: `{record['model']}` ({gib(record['file_bytes'])})"
             + (f", SHA-256 `{record['sha256']}`" if record["sha256"] else ", not hashed"),
             f"- Machine: {machine['os']}, {machine['cpu']}, {machine['cpu_count']} threads, "
             f"{gib(machine['ram_bytes'])} RAM, Python {machine['python']}",
             f"- GPUs: {', '.join(g['name'] for g in machine['gpus']) or 'none found'}",
             f"- Workload: {record['prompt_tokens']} prompt tokens, {record['gen_tokens']} "
             f"generated, temperature {record['sampling']['temperature']}, seed "
             f"{record['sampling']['seed']}; 1 cold run, {record['warmup']} warm-up, "
             f"median of {record['repeat']}", "",
             "| Runtime | Prefill tok/s | Decode tok/s | First token s | Token p50 / p90 / p99 ms "
             "| Load s | Peak RSS |", "|---|---|---|---|---|---|---|"]
    for row in record["results"]:
        if row.get("skipped"):
            lines.append(f"| {row['runtime']} | skipped: {row['skipped']} | | | | | |")
            continue
        latency, memory = row.get("token_latency_ms", {}), row.get("memory", {})
        lines.append(
            f"| {row['runtime']} | {row.get('prefill_tps', 0):.2f} | {row.get('decode_tps', 0):.2f} "
            f"| {row.get('first_token_seconds', 0) or 0:.3f} | "
            + (f"{latency['p50']:.1f} / {latency['p90']:.1f} / {latency['p99']:.1f}" if latency else "")
            + f" | {row.get('load_seconds', 0) or 0:.2f} | "
            + (gib(memory["peak_rss_bytes"]) if memory.get("peak_rss_bytes") else "") + " |")
    lines.append("")
    for row in record["results"]:
        if row.get("skipped") or "cold" not in row:
            continue
        cold, place = row["cold"], row.get("placement", {})
        lines += [f"## {row['runtime']}", "",
                  f"- Backend: {row.get('backend', '')}; kernels: {row.get('kernels', '')}",
                  f"- Layers: {place.get('cpu_layers', '?')} on the CPU, "
                  f"{place.get('gpu_layers_used', 0)} on a GPU "
                  f"(asked for {place.get('gpu_layers_requested', 0)})",
                  f"- Cold run: prefill {cold['prefill_tps']:.2f} tok/s, decode "
                  f"{cold['decode_tps']:.2f} tok/s, first token {cold['first_token_seconds']:.3f} s",
                  f"- Quality: greedy output repeats exactly: "
                  f"{'yes' if row['quality']['deterministic'] else 'NO'}"]
        lines += [f"- Fallback: {reason}" for reason in place.get("fallback_reasons", [])]
        lines.append("")
    quality = record.get("quality", {})
    if "native_matches_numpy" in quality:
        lines.append(f"Native kernels and numpy produce the same greedy tokens: "
                     f"{'yes' if quality['native_matches_numpy'] else 'NO'}.")
    if "target_decode_tps" in record:
        lines.append(f"Target {record['target_decode_tps']:g} decode tok/s: "
                     f"**{'met' if record['target_met'] else 'unmet'}**.")
    lines += ["", "Every figure above was measured in this run. Compare runs from one machine only."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hnx strife benchmark",
                                     description="Measure Strife on one local GGUF, next to "
                                     "llama.cpp when a build is found. Never downloads a model.")
    parser.add_argument("model", help="a .gguf file, or the id of a model on this machine")
    parser.add_argument("-p", "--prompt-tokens", type=int, default=512)
    parser.add_argument("-n", "--gen-tokens", type=int, default=128)
    parser.add_argument("-r", "--repeat", type=int, default=3, help="measured runs (median)")
    parser.add_argument("--warmup", type=int, default=0,
                        help="runs discarded after the cold run and before the measured ones")
    parser.add_argument("-c", "--ctx", type=int, default=0)
    parser.add_argument("-t", "--threads", type=int, default=0)
    parser.add_argument("--gpu-layers", type=int, default=0,
                        help="blocks on the GPU (-1: as many as fit)")
    parser.add_argument("--gpu-device", default=None)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=float, default=None,
                        help="decode tokens per second to meet; below it is reported unmet")
    parser.add_argument("--no-llama", action="store_true", help="skip the llama.cpp comparison")
    parser.add_argument("--numpy", action="store_true",
                        help="also run with the native kernels off, to compare")
    parser.add_argument("--no-hash", action="store_true",
                        help="skip the file's SHA-256 (slow on a large model)")
    parser.add_argument("--json", metavar="OUT", default=None, help="write the record here")
    parser.add_argument("--csv", metavar="OUT", default=None, help="one row per runtime")
    parser.add_argument("--report", metavar="OUT", default=None, help="a Markdown report")
    parser.add_argument("--out-dir", metavar="DIR", default=None,
                        help="write strife-bench.json, .csv and .md there")
    args = parser.parse_args(argv)
    if args.prompt_tokens < 1 or args.gen_tokens < 1 or args.repeat < 1 or args.warmup < 0:
        parser.error("-p, -n and -r must be at least 1, and --warmup not negative")

    from hypernix import __version__
    from hypernix.strife.errors import StrifeError
    from hypernix.strife.inspect import file_sha256
    from hypernix.strife.models import resolve_model
    from hypernix.strife.placement import detect

    try:
        path = Path(resolve_model(args.model))
    except StrifeError as exc:
        print(f"strife benchmark: {exc} [{exc.code}]", file=sys.stderr)
        return 2
    hw = detect()
    record: dict[str, Any] = {
        "hypernix_version": __version__,
        "model": str(path), "file_bytes": path.stat().st_size,
        "sha256": "" if args.no_hash else file_sha256(path),
        "machine": {"os": hw["os"], "machine": hw["machine"], "cpu": hw["cpu"],
                    "cpu_count": hw["cpu_count"], "python": hw["python"],
                    "ram_bytes": hw["ram"]["total_bytes"],
                    "ram_available_bytes": hw["ram"]["available_bytes"],
                    "gpus": hw["gpus"], "support": hw["support"]},
        "prompt_tokens": args.prompt_tokens, "gen_tokens": args.gen_tokens,
        "repeat": args.repeat, "warmup": args.warmup,
        "sampling": {"temperature": args.temp, "seed": args.seed},
        "threads_requested": args.threads, "gpu_layers_requested": args.gpu_layers,
        "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [], "quality": {},
    }
    common = {"prompt_tokens": args.prompt_tokens, "gen_tokens": args.gen_tokens,
              "repeat": args.repeat, "ctx": args.ctx, "threads": args.threads, "seed": args.seed,
              "warmup": args.warmup, "temperature": args.temp}
    print(f"Strife: {path.name}, {args.prompt_tokens} prompt + {args.gen_tokens} generated, "
          f"1 cold run then the median of {args.repeat}...", file=sys.stderr)
    try:
        strife = bench_strife(path, gpu_layers=args.gpu_layers, gpu_device=args.gpu_device,
                              **common)
    except StrifeError as exc:
        print(f"strife benchmark: {exc} [{exc.code}]", file=sys.stderr)
        if exc.remedy:
            print(f"  {exc.remedy}", file=sys.stderr)
        return 2
    record["results"].append(strife)
    quality_ok = strife["quality"]["deterministic"]
    if args.numpy and not strife["runtime"].startswith("strife (numpy)"):
        print("Strife again, without the native kernels...", file=sys.stderr)
        plain = bench_strife(path, kernels="off", **common)
        record["results"].append(plain)
        record["quality"]["native_matches_numpy"] = (
            strife["quality"]["tokens"] == plain["quality"]["tokens"])
        quality_ok = quality_ok and plain["quality"]["deterministic"] \
            and record["quality"]["native_matches_numpy"]
    record["quality"]["ok"] = quality_ok
    if not args.no_llama:
        print("llama.cpp: llama-bench on the same file...", file=sys.stderr)
        record["results"].append(bench_llama(
            path, prompt_tokens=args.prompt_tokens, gen_tokens=args.gen_tokens,
            repeat=args.repeat, threads=args.threads,
            gpu_layers=strife.get("gpu_layers_used", 0)))
    if args.target is not None:
        record["target_decode_tps"] = args.target
        record["target_met"] = strife["decode_tps"] >= args.target

    print(f"\n{'runtime':<20} {'prefill tok/s':>14} {'decode tok/s':>13} {'first tok s':>12} "
          f"{'p90 ms':>8} {'load s':>7}")
    for row in record["results"]:
        if row.get("skipped"):
            print(f"{row['runtime']:<20} skipped: {row['skipped']}")
            continue
        latency = row.get("token_latency_ms", {})
        print(f"{row['runtime']:<20} {row.get('prefill_tps', 0):>14.2f} "
              f"{row.get('decode_tps', 0):>13.2f} {row.get('first_token_seconds', 0) or 0:>12.3f} "
              f"{latency.get('p90', 0) or 0:>8.1f} {row.get('load_seconds', 0) or 0:>7.2f}")
    cold = strife["cold"]
    print(f"\ncold run: prefill {cold['prefill_tps']:.2f} tok/s, decode {cold['decode_tps']:.2f} "
          f"tok/s, first token {cold['first_token_seconds']:.3f} s")
    print(f"kernels: {strife['kernels']}; layers on a GPU: {strife['gpu_layers_used']} of "
          f"{strife['gpu_layers_requested']} asked for")
    for reason in strife["placement"].get("fallback_reasons", []):
        print(f"  fallback: {reason}")
    print("quality: greedy output repeats exactly: "
          + ("yes" if strife["quality"]["deterministic"] else "NO")
          + ("" if "native_matches_numpy" not in record["quality"] else
             "; native matches numpy: "
             + ("yes" if record["quality"]["native_matches_numpy"] else "NO")))
    if strife["decoded_tokens"] and min(strife["decoded_tokens"]) < args.gen_tokens:
        print(f"note: the model ended early in some runs (decoded {strife['decoded_tokens']}).")
    if args.target is not None:
        verdict = "met" if record["target_met"] else "UNMET"
        print(f"target {args.target:g} decode tok/s: {verdict} ({strife['decode_tps']:.2f})")

    outputs = {"json": args.json, "csv": args.csv, "report": args.report}
    if args.out_dir:
        folder = Path(args.out_dir)
        folder.mkdir(parents=True, exist_ok=True)
        outputs = {"json": outputs["json"] or str(folder / "strife-bench.json"),
                   "csv": outputs["csv"] or str(folder / "strife-bench.csv"),
                   "report": outputs["report"] or str(folder / "strife-bench.md")}
    if outputs["json"]:
        Path(outputs["json"]).write_text(json.dumps(record, indent=2, default=str) + "\n",
                                         encoding="utf-8")
    if outputs["csv"]:
        write_csv(record, Path(outputs["csv"]))
    if outputs["report"]:
        Path(outputs["report"]).write_text(render_report(record), encoding="utf-8")
    for kind, where in outputs.items():
        if where:
            print(f"{kind}: {where}", file=sys.stderr)
    if not quality_ok:
        print("strife benchmark: the quality check FAILED; the speeds above are not a result.",
              file=sys.stderr)
        return 3
    return 0 if record.get("target_met", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
