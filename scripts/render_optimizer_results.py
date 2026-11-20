#!/usr/bin/env python3
"""Render the V7 wiki page's measured tables from committed benchmark results.

    python scripts/render_optimizer_results.py                 # rewrite the page
    python scripts/render_optimizer_results.py --check         # fail if it is stale

Reads ``benchmarks/results/optimizers-0.73.0a2/`` (each folder is one
``hnx optimizer benchmark --out-dir`` run; the commands are in that
folder's README) and replaces everything between ``<!-- MEASURED -->`` and
``<!-- /MEASURED -->`` in ``wiki/Pressure-Cooker-V7.md``. Every number on
the page between those markers comes from those files; nothing is typed in
by hand. The written analysis lives outside the markers.

Standard library only: it does not import hypernix or torch.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "optimizers-0.73.0a2"
PAGE = ROOT / "wiki" / "Pressure-Cooker-V7.md"
START, END = "<!-- MEASURED -->", "<!-- /MEASURED -->"

NAMES = {"adamw": "AdamW", "v5": "V5", "v5plus": "V5+", "v5s": "V5S", "v6": "V6",
         "v6v": "V6V", "v7": "V7 Balanced", "v7mini": "V7Mini", "v7q": "V7Q", "v7s": "V7S",
         "v7m": "V7M"}


def _load(name: str) -> dict[str, Any] | None:
    path = RESULTS / name / "optimizer-bench.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _name(key: str) -> str:
    return NAMES.get(key, f"`{key}`")


def _f(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _setup(record: dict[str, Any], what: str) -> str:
    m, st = record["machine"], record["settings"]
    rates = ", ".join(f"{v:g}" for v in st["lrs"])
    return (f"{what}: HyperNix {record['hypernix_version']}, torch {record['torch']}, "
            f"{m['os'].split()[0]}, {m['threads']} CPU threads, {st['precision']}, seed "
            f"{st['seed']}, {st['steps']} training steps at each of {rates}; "
            f"{st['optimizer_steps']} optimizer-only steps after {st['warmup']} warm-up.")


def _averages(record: dict[str, Any]) -> list[str]:
    lines = ["| Optimizer | Step only, ms | Full step, ms | State bytes/param | "
             "Final training loss | Held-out loss | Workloads |",
             "|---|---|---|---|---|---|---|"]
    for key, got in record["measured"].items():
        lines.append(f"| {_name(key)} | {got['ms_per_step']:.2f} | {got['full_step_ms']:.2f} | "
                     f"{got['state_bytes_per_parameter']:.2f} | {_f(got.get('final_loss'))} | "
                     f"{_f(got.get('validation_loss'))} | {got['workloads']} |")
    missing = [k for k in record["settings"]["optimizers"] if k not in record["measured"]]
    for key in missing:
        reasons = {got.get("reason", "") for entry in record["workloads"].values()
                   for k, got in entry["optimizers"].items() if k == key}
        lines.append(f"| {_name(key)} | N/A: {'; '.join(sorted(reasons))} | | | | | |")
    return lines


def _workload(entry: dict[str, Any]) -> list[str]:
    lines = ["| Optimizer | Step only, ms (p10–p90) | State B/param | Best lr | Final loss | "
             "Held-out loss | Full step, ms | Optimizer share | Grad-norm peak/median |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, got in entry["optimizers"].items():
        if not got.get("available"):
            lines.append(f"| {_name(key)} | N/A | | | | | | | |")
            continue
        only, best = got["optimizer_only"], got.get("best")
        spread = f"{only['ms_per_step']:.2f} ({only['ms_per_step_p10']:.2f}–" \
                 f"{only['ms_per_step_p90']:.2f})"
        if best is None:
            lines.append(f"| {_name(key)} | {spread} | {only['state_bytes_per_parameter']:.2f} "
                         "| diverged at every rate | | | | | |")
            continue
        edge = " (edge)" if got.get("best_at_grid_edge") else ""
        lines.append(
            f"| {_name(key)} | {spread} | {only['state_bytes_per_parameter']:.2f} | "
            f"{best['lr']:g}{edge} | {best['final_loss']:.4f} | {best['validation_loss']:.4f} | "
            f"{best['ms_per_step']:.2f} | {100 * best['optimizer_share']:.0f}% | "
            f"{_f(best.get('grad_norm_peak_ratio'), 2)} |")
    return lines


def _high(main: dict[str, Any], high: dict[str, Any]) -> list[str]:
    """Best over both grids, for the optimizers the high-rate run repeated."""
    lo, hi = min(main["settings"]["lrs"]), max(high["settings"]["lrs"])
    keys = high["settings"]["optimizers"]
    lines = [f"Best final training loss over {lo:g} to {hi:g}, per workload "
             "(rate in brackets; `edge` means the best was still the largest rate tried):", "",
             "| Workload | " + " | ".join(_name(k) for k in keys) + " |",
             "|---|" + "---|" * len(keys)]
    for name, entry in high["workloads"].items():
        cells = []
        for key in keys:
            runs = list(main["workloads"].get(name, {}).get("optimizers", {})
                        .get(key, {}).get("runs", []))
            runs += entry["optimizers"].get(key, {}).get("runs", [])
            good = [r for r in runs if not r["diverged"] and r["final_loss"] is not None]
            if not good:
                cells.append("diverged")
                continue
            best = min(good, key=lambda r: r["final_loss"])
            edge = ", edge" if best["lr"] == hi else ""
            cells.append(f"{best['final_loss']:.3f} ({best['lr']:g}{edge})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    diverged = sorted({f"{_name(k)} at {r['lr']:g} on {n}"
                       for n, e in high["workloads"].items()
                       for k, g in e["optimizers"].items() for r in g.get("runs", [])
                       if r["diverged"]})
    if diverged:
        lines += ["", "Diverged: " + "; ".join(diverged) + "."]
    return lines


def _ablations(record: dict[str, Any]) -> list[str]:
    lines = ["| Configuration | Step only, ms | Full step, ms | State bytes/param | "
             "Final training loss | Held-out loss |",
             "|---|---|---|---|---|---|"]
    for key, got in record["measured"].items():
        label = next((g["label"] for e in record["workloads"].values()
                      for k, g in e["optimizers"].items() if k == key and g.get("label")), key)
        lines.append(f"| {label} | {got['ms_per_step']:.2f} | {got['full_step_ms']:.2f} | "
                     f"{got['state_bytes_per_parameter']:.2f} | {_f(got.get('final_loss'))} | "
                     f"{_f(got.get('validation_loss'))} |")
    return lines


def _in_band(record: dict[str, Any]) -> list[str]:
    entry = next(iter(record["workloads"].values()))
    rates = record["settings"]["lrs"]
    lines = ["| Optimizer | Step only, ms | Full step, ms | "
             + " | ".join(f"Final / held-out at {r:g}" for r in rates) + " |",
             "|---|---|---|" + "---|" * len(rates)]
    for key, got in entry["optimizers"].items():
        if not got.get("available"):
            lines.append(f"| {_name(key)} | N/A | | " + "| " * len(rates))
            continue
        cells = [f"{r['final_loss']:.3f} / {r['validation_loss']:.3f}" if not r["diverged"]
                 else "diverged" for r in got["runs"]]
        full = statistics.median(r["ms_per_step"] for r in got["runs"])
        lines.append(f"| {_name(key)} | {got['optimizer_only']['ms_per_step']:.1f} | "
                     f"{full:.1f} | " + " | ".join(cells) + " |")
    return lines


def _band(*records: dict[str, Any] | None) -> list[str]:
    sizes: dict[str, dict[str, Any]] = {}
    for record in records:
        if record and record.get("band"):
            sizes.update(record["band"]["sizes"])
    if not sizes:
        return ["Not measured."]
    names = list(next(iter(sizes.values()))["optimizers"])
    lines = ["| Parameters | In band | Auto-selects | "
             + " | ".join(f"{_name(n)}, ms" for n in names) + " |",
             "|---|---|---|" + "---|" * len(names)]
    reasons = []
    for entry in sorted(sizes.values(), key=lambda e: e["parameters"]):
        cells = []
        for name in names:
            got = entry["optimizers"][name]
            if got.get("available"):
                cells.append(f"{got['ms_per_step']:.1f}")
            else:
                cells.append("N/A")
                reasons.append(f"{entry['parameters']:,} {_name(name)}: {got.get('reason', '')}")
        lines.append(f"| {entry['parameters']:,} | {'yes' if entry['in_band'] else 'no'} | "
                     f"{_name(entry['auto_select'])} | " + " | ".join(cells) + " |")
    if reasons:
        lines += ["", "N/A: " + "; ".join(reasons) + "."]
    return lines


def render() -> str:
    main = _load("main")
    if main is None:
        raise SystemExit(f"no results in {RESULTS / 'main'}")
    out = ["### Measured", "",
           "Every figure in this section is read from "
           "`benchmarks/results/optimizers-0.73.0a2/` by "
           "`scripts/render_optimizer_results.py`; the commands that produced each file "
           "are in that folder's README. One machine, one seed, CPU only.", "",
           "#### Averaged over the six workloads", "",
           _setup(main, "Run"), "", *_averages(main), ""]
    high = _load("high-lr")
    if high:
        out += ["#### Rates above the grid", "",
                "V5, V5+, V5S, V6 and V7Q did best at the top of the main grid, so they "
                "were run again at higher rates.", "", _setup(high, "Run"), "",
                *_high(main, high), ""]
    out += ["#### Per workload", ""]
    for name, entry in main["workloads"].items():
        out += [f"**{name}** ({entry['parameters']:,} parameters)", "", *_workload(entry), ""]
    ablations = _load("ablations")
    if ablations:
        out += ["#### Ablations", "",
                "One mechanism changed at a time, averaged over "
                + ", ".join(ablations["settings"]["workloads"]) + ".", "",
                _setup(ablations, "Run"), "", *_ablations(ablations), ""]
    in_band = _load("in-band")
    if in_band:
        entry = next(iter(in_band["workloads"].values()))
        out += ["#### Inside V7Mini's band: training", "",
                f"`band-transformer`, {entry['parameters']:,} parameters. "
                + _setup(in_band, "Run"), "", *_in_band(in_band), ""]
    out += ["#### Around V7Mini's band: `step()` alone", "",
            "Tensors with a transformer's shapes and exactly the parameter count shown, "
            "random gradients; median ms per step. This is the optimizer's cost at that size "
            "and says nothing about training quality.", "",
            *_band(_load("band"), _load("band-edge")), ""]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the page does not match the results")
    args = parser.parse_args()
    page = PAGE.read_text(encoding="utf-8")
    if START not in page or END not in page:
        print(f"{PAGE} has no {START} ... {END} section", file=sys.stderr)
        return 2
    head, rest = page.split(START, 1)
    _old, tail = rest.split(END, 1)
    new = f"{head}{START}\n{render()}{END}{tail}"
    if args.check:
        if new != page:
            print(f"{PAGE.relative_to(ROOT)} is out of date: run "
                  "python scripts/render_optimizer_results.py", file=sys.stderr)
            return 1
        return 0
    PAGE.write_text(new, encoding="utf-8")
    print(f"wrote {PAGE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
