#!/usr/bin/env python3
"""Strife's benchmark: prefill and decode speed on one GGUF, next to llama.cpp.

    python benchmarks/strife_bench.py model.gguf
    python benchmarks/strife_bench.py qwen3.5-2b -p 512 -n 128 -r 3 --json out.json
    python benchmarks/strife_bench.py model.gguf --target 8      # decode tok/s to meet
    python benchmarks/strife_bench.py model.gguf --no-llama      # Strife only
    python benchmarks/strife_bench.py model.gguf --numpy         # also without the native kernels

What is measured, each the median of ``--repeat`` runs in a fresh session
(so no run reuses another's prefix):

prefill   ``-p`` prompt tokens, in tokens per second
decode    up to ``-n`` generated tokens, in tokens per second (a model that
          ends its answer early decodes fewer; the count is reported)

Strife runs with its native kernels when they are built (``hnx strife
kernels build``) and says which it used; ``--numpy`` adds a second run with
them off, so the two can be compared on the same file.

When a llama.cpp build is found (the same search the runner uses), its
``llama-bench`` is run on the same file with the same ``-p``/``-n``, and the
table shows both. ``--target`` states a decode speed to meet; a run below
it is reported as **unmet**, not hidden.

The record names the file by SHA-256, the machine, and Strife's placement
and cache figures, so two records can be compared honestly. Compare runs
from one machine; never across machines.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 3) if values else 0.0


def _spread(values: list[float]) -> float:
    return round(max(values) - min(values), 3) if len(values) > 1 else 0.0


def bench_strife(path: Path, *, prompt_tokens: int, gen_tokens: int, repeat: int,
                 ctx: int, threads: int, seed: int, kernels: str | None = None) -> dict[str, Any]:
    import numpy as np

    from hypernix.strife import Engine, SamplerConfig

    started = time.monotonic()
    engine = Engine(path, n_ctx=ctx or max(prompt_tokens + gen_tokens + 8, 256),
                    threads=threads or None, kernels=kernels)
    load_s = time.monotonic() - started
    try:
        vocab = engine.report.n_vocab
        rng = np.random.default_rng(seed)
        # Ordinary token ids from the middle of the vocabulary: the speed of
        # a forward pass does not depend on which ids, and special tokens at
        # either end are kept out.
        low, high = min(256, vocab // 4), max(min(256, vocab // 4) + 1, vocab - 256)
        prefill, decode, decoded = [], [], []
        for i in range(max(1, repeat)):
            ids = rng.integers(low, high, size=prompt_tokens).tolist()
            session = engine.new_session()
            try:
                final: dict[str, Any] = {}
                for event in engine.stream(prompt_ids=ids, session=session,
                                           max_tokens=gen_tokens,
                                           sampling=SamplerConfig(temperature=0.8, seed=seed + i)):
                    if event["type"] == "done":
                        final = event
            finally:
                engine.close_session(session)
            stats = final.get("stats", {})
            prefill.append(float(stats.get("prefill_tokens_per_second") or 0))
            if stats.get("decode_tokens_per_second"):
                decode.append(float(stats["decode_tokens_per_second"]))
            decoded.append(len(final.get("tokens", [])))
        status = engine.status()
        return {
            "runtime": "strife (native)" if engine.weights.native_matmul else "strife (numpy)",
            "kernels": status.get("kernels", {}).get("note", ""),
            "load_seconds": round(load_s, 3),
            "prefill_tps": _median(prefill), "prefill_spread": _spread(prefill),
            "decode_tps": _median(decode), "decode_spread": _spread(decode),
            "decoded_tokens": decoded,
            "architecture": engine.report.architecture,
            "weights_bytes": engine.report.weight_bytes,
            "placement": {k: v for k, v in (status.get("placement") or {}).items()
                          if k != "hardware"},
            "weights_cache": status.get("weights", {}),
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
                threads: int) -> dict[str, Any]:
    binary = _llama_bench_binary()
    if binary is None:
        return {"runtime": "llama.cpp", "skipped": "no llama.cpp build with llama-bench was found"}
    argv = [str(binary), "-m", str(path), "-p", str(prompt_tokens), "-n", str(gen_tokens),
            "-r", str(max(1, repeat)), "-ngl", "0", "-o", "json"]
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
    out["note"] = "llama.cpp on the CPU (-ngl 0), the same placement Strife alpha 1 uses"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", help="a .gguf file, or a model id Strife can resolve")
    parser.add_argument("-p", "--prompt-tokens", type=int, default=512)
    parser.add_argument("-n", "--gen-tokens", type=int, default=128)
    parser.add_argument("-r", "--repeat", type=int, default=3)
    parser.add_argument("-c", "--ctx", type=int, default=0)
    parser.add_argument("-t", "--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=float, default=None,
                        help="decode tokens per second to meet; below it is reported unmet")
    parser.add_argument("--no-llama", action="store_true", help="skip the llama.cpp comparison")
    parser.add_argument("--numpy", action="store_true",
                        help="also run with the native kernels off, to compare")
    parser.add_argument("--no-hash", action="store_true",
                        help="skip the file's SHA-256 (slow on a large model)")
    parser.add_argument("--json", metavar="OUT", default=None, help="write the record here")
    args = parser.parse_args(argv)
    if args.prompt_tokens < 1 or args.gen_tokens < 1 or args.repeat < 1:
        parser.error("-p, -n and -r must be at least 1")

    from hypernix.strife.inspect import file_sha256
    from hypernix.strife.models import resolve_model

    path = Path(resolve_model(args.model))
    record: dict[str, Any] = {
        "model": str(path), "file_bytes": path.stat().st_size,
        "sha256": "" if args.no_hash else file_sha256(path),
        "machine": {"os": f"{platform.system()} {platform.release()}",
                    "cpu": platform.processor() or platform.machine(),
                    "python": platform.python_version()},
        "prompt_tokens": args.prompt_tokens, "gen_tokens": args.gen_tokens,
        "repeat": args.repeat, "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [],
    }
    print(f"Strife: {path.name}, {args.prompt_tokens} prompt + {args.gen_tokens} generated, "
          f"median of {args.repeat}...", file=sys.stderr)
    strife = bench_strife(path, prompt_tokens=args.prompt_tokens, gen_tokens=args.gen_tokens,
                          repeat=args.repeat, ctx=args.ctx, threads=args.threads, seed=args.seed)
    record["results"].append(strife)
    if args.numpy and strife["runtime"] != "strife (numpy)":
        print("Strife again, without the native kernels...", file=sys.stderr)
        record["results"].append(bench_strife(
            path, prompt_tokens=args.prompt_tokens, gen_tokens=args.gen_tokens,
            repeat=args.repeat, ctx=args.ctx, threads=args.threads, seed=args.seed,
            kernels="off"))
    if not args.no_llama:
        print("llama.cpp: llama-bench on the same file...", file=sys.stderr)
        record["results"].append(bench_llama(path, prompt_tokens=args.prompt_tokens,
                                             gen_tokens=args.gen_tokens, repeat=args.repeat,
                                             threads=args.threads))
    if args.target is not None:
        record["target_decode_tps"] = args.target
        record["target_met"] = strife["decode_tps"] >= args.target

    print(f"\n{'runtime':<16} {'prefill tok/s':>14} {'decode tok/s':>13} {'load s':>8}")
    for row in record["results"]:
        if row.get("skipped"):
            print(f"{row['runtime']:<16} skipped: {row['skipped']}")
            continue
        load = f"{row['load_seconds']:>8.2f}" if "load_seconds" in row else ""
        print(f"{row['runtime']:<16} {row.get('prefill_tps', 0):>14.2f} "
              f"{row.get('decode_tps', 0):>13.2f} {load}")
    if strife.get("kernels"):
        print(f"\nkernels: {strife['kernels']}")
    if strife["decoded_tokens"] and min(strife["decoded_tokens"]) < args.gen_tokens:
        print(f"\nnote: the model ended early in some runs (decoded {strife['decoded_tokens']}).")
    if args.target is not None:
        verdict = "met" if record["target_met"] else "UNMET"
        print(f"\ntarget {args.target:g} decode tok/s: {verdict} ({strife['decode_tps']:.2f})")
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nrecord: {args.json}", file=sys.stderr)
    return 0 if record.get("target_met", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
