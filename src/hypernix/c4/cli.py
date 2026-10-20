"""``hnx c4`` — explode, measure, test, reap and sow from the command line.

::

    hnx c4 explode MODEL [--save DIR]                 what the model is made of
    hnx c4 stem MODEL [--test] [--calib FILE]         layer report: which layers are needed
    hnx c4 dial MODEL --calib FILE [--score taylor] [--unneeded 0.2] [--priority pdial]
    hnx c4 reap MODEL --calib FILE [--drop 2 | --keep 10] [--unneeded 0.2] [--ffn-keep 0.75] -o HARVEST
    hnx c4 reap --best A B C --calib FILE -o HARVEST  cherry-pick layers from same-shaped models
    hnx c4 sow HARVEST -o OUT [--format safetensors|pt|gguf] [--heal 50 --calib FILE]
    hnx c4 restore MODEL -o OUT --format pt           write a model back in another format

``--calib`` is a text file (blank lines separate passages) read with the
model's own tokenizer. Without it, tests run on random tokens and say so.
``--json`` prints the result as JSON. ``--device`` picks the GPU
(default: CUDA, then MPS, then CPU).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .model import C4Error

__all__ = ["cli_main"]


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", help="cuda, cuda:1, mps, cpu or auto (default)")
    p.add_argument("--calib", default=None, help="calibration text file")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--json", action="store_true", help="print JSON")


def _prober(model: Any, args: argparse.Namespace) -> Any:
    from .probe import Prober

    texts = None
    if args.calib:
        text = Path(args.calib).read_text(encoding="utf-8")
        texts = [t for t in text.split("\n\n") if t.strip()] or [text]
    return Prober.for_model(model, texts=texts, seq_len=args.seq_len)


def _print(args: argparse.Namespace, data: Any, human: str) -> None:
    print(json.dumps(data, indent=1, default=str) if args.json else human)


def _explode(args: argparse.Namespace) -> int:
    from .explode import explode

    ex = explode(args.model, device=args.device)
    if args.save:
        ex.save(args.save, weights=not args.manifest_only)
    s = ex.summary()
    lines = [f"{s['path']} ({s['format']}, on {s['device']})",
             f"  {s['mstems']} mstems, {s['stems']} stems, {s['row_pdials']:,} row Pdials, "
             f"{s['mdials']:,} mdials, {s['dials']:,} dials"]
    lines += [f"  layer {m.index}: {len(m.stems)} stems, {m.numel:,} parameters" for m in ex.mstems[:args.show]]
    _print(args, s, "\n".join(lines))
    return 0


def _stem(args: argparse.Namespace) -> int:
    from .explode import explode
    from .stem import layer_report, measure

    ex = explode(args.model, device=args.device)
    prober = _prober(ex.model, args) if (args.test or args.calib) else None
    report = layer_report(ex, prober, test=args.test, spectral=args.spectral)
    data: dict[str, Any] = {"layers": [v.to_dict() for v in report],
                            "calibration": prober.describe() if prober else None}
    if args.stems:
        data["stems"] = [measure(ex, n, spectral=args.spectral).to_dict() for n in ex.stems]
    lines = [f"{'layer':>5}  {'score':>9}  verdict             why"]
    for v in report:
        lines.append(f"{v.index:>5}  {v.score:>+9.4f}  {v.verdict:<18}  {'; '.join(v.reasons)}")
    if prober is not None and prober.calibration.synthetic:
        lines.append("(tested on random tokens: give --calib for a real answer)")
    _print(args, data, "\n".join(lines))
    return 0


def _dial(args: argparse.Namespace) -> int:
    from .dial import score, train_priority, unneeded
    from .explode import explode

    ex = explode(args.model, device=args.device)
    prober = _prober(ex.model, args) if args.score != "magnitude" else None
    scores = score(ex, prober, kind=args.score)
    data: dict[str, Any] = {"score": args.score, "top": scores.top(args.top),
                            "synthetic": scores.synthetic}
    lines = [f"highest {args.score} dials:"] + [f"  {n}[{', '.join(map(str, i))}]  {v:.4g}"
                                                 for n, i, v in data["top"]]
    if args.unneeded:
        cut = unneeded(scores, fraction=args.unneeded)
        data["unneeded"] = {"dials": cut.dials, "rows": {k: v[:50] for k, v in cut.rows.items()}}
        lines.append(f"unneeded: {cut.dials:,} dials; whole rows in {len(cut.rows)} stems")
    if args.priority:
        if not scores.grads:
            raise C4Error("--priority needs --score taylor or fisher")
        ranked = train_priority(ex, scores, granularity=args.priority, top=args.top)
        data["priority"] = [p.to_dict() for p in ranked]
        lines.append(f"train first ({args.priority}):")
        lines += [f"  {p.piece}  gain {p.gain:.3g}  ‖g‖/‖w‖ {p.relative:.3g}  {p.reason}" for p in ranked]
    _print(args, data, "\n".join(lines))
    return 0


def _reap(args: argparse.Namespace) -> int:
    from .dial import score, unneeded
    from .explode import explode
    from .reap import reap, reap_best
    from .stem import layer_report

    if args.best:
        exs = [explode(m, device=args.device) for m in args.best]
        probers = [_prober(e.model, args) for e in exs]
        harvest = reap_best(exs, probers=probers, layers=args.keep)
    else:
        if not args.model:
            raise C4Error("give a MODEL, or --best A B ... to cherry-pick across models")
        ex = explode(args.model, device=args.device)
        prober = _prober(ex.model, args)
        report = layer_report(ex, prober)
        scores = score(ex, prober, kind=args.score) if (args.unneeded or args.ffn_keep < 1.0) else None
        cut = unneeded(scores, fraction=args.unneeded) if (scores is not None and args.unneeded) else None
        harvest = reap(ex, report=report, keep_layers=args.keep, drop_layers=args.drop,
                       unneeded=cut, dial_scores=scores, ffn_keep=args.ffn_keep)
    harvest.save(args.output)
    _print(args, harvest.summary(), "\n".join([f"harvest written to {args.output}", *harvest.reasons]))
    return 0


def _sow(args: argparse.Namespace) -> int:
    from .reap import Harvest
    from .sow import heal, sow

    harvest = Harvest.load(args.harvest)
    report = sow(harvest, args.output, format=args.format, device=args.device)
    data: dict[str, Any] = {"sow": report.to_dict()}
    lines = [f"sown: {report.path}: {report.layers} layers, "
             f"{report.parameters_before:,} -> {report.parameters_after:,} parameters, "
             f"{report.zeroed_dials:,} dials zeroed"]
    if args.heal:
        from .model import load

        calibration = _prober(load(report.path, device=args.device), args).calibration
        healed = heal(report.path, calibration, steps=args.heal, optimizer=args.optimizer,
                      lr=args.lr, device=args.device)
        data["heal"] = healed
        lines.append(f"healed {args.heal} steps ({args.optimizer}): loss "
                     f"{healed['loss_before']:.4f} -> {healed['loss_after']:.4f}")
    _print(args, data, "\n".join(lines))
    return 0


def _restore(args: argparse.Namespace) -> int:
    from .model import load
    from .restore import write

    model = load(args.model, device=args.device)
    out = write(model, args.output, format=args.format)
    _print(args, {"path": str(out)}, f"written: {out}")
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hnx c4", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("explode", help="what the model is made of")
    p.add_argument("model")
    p.add_argument("--save", default=None, help="write the pieces (one file per mstem) to DIR")
    p.add_argument("--manifest-only", action="store_true")
    p.add_argument("--show", type=int, default=8)
    _common(p)

    p = sub.add_parser("stem", help="measure stems and test layers")
    p.add_argument("model")
    p.add_argument("--test", action="store_true", help="ablation-test every layer (runs the model)")
    p.add_argument("--spectral", action="store_true", help="effective and stable rank (SVD)")
    p.add_argument("--stems", action="store_true", help="every stem's statistics too")
    _common(p)

    p = sub.add_parser("dial", help="score parameters; unneeded ones; what to train")
    p.add_argument("model")
    p.add_argument("--score", default="taylor", choices=["magnitude", "taylor", "fisher", "wanda"])
    p.add_argument("--unneeded", type=float, default=0.0, help="fraction of dials to call unneeded")
    p.add_argument("--priority", choices=["stem", "pdial", "mstem"], default=None)
    p.add_argument("--top", type=int, default=10)
    _common(p)

    p = sub.add_parser("reap", help="choose layers, dials and channels for a new model")
    p.add_argument("model", nargs="?")
    p.add_argument("--best", nargs="+", default=None, help="cherry-pick layers across these models")
    p.add_argument("--keep", type=int, default=None, help="keep this many layers")
    p.add_argument("--drop", type=int, default=None, help="drop this many layers")
    p.add_argument("--unneeded", type=float, default=0.0)
    p.add_argument("--score", default="taylor", choices=["magnitude", "taylor", "fisher", "wanda"])
    p.add_argument("--ffn-keep", type=float, default=1.0, help="share of feed-forward channels to keep")
    p.add_argument("-o", "--output", required=True, help="harvest folder")
    _common(p)

    p = sub.add_parser("sow", help="build the new model from a harvest")
    p.add_argument("harvest")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--format", default=None, choices=["safetensors", "pt", "gguf"])
    p.add_argument("--heal", type=int, default=0, help="train this many steps afterwards")
    p.add_argument("--optimizer", default="hybrid", choices=["hybrid", "v5s", "adamw"])
    p.add_argument("--lr", type=float, default=1e-4)
    _common(p)

    p = sub.add_parser("restore", help="write a model back out, in another format")
    p.add_argument("model")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--format", default="safetensors", choices=["safetensors", "pt", "gguf"])
    _common(p)

    args = parser.parse_args(argv)
    handler = {"explode": _explode, "stem": _stem, "dial": _dial, "reap": _reap,
               "sow": _sow, "restore": _restore}[args.cmd]
    try:
        return handler(args)
    except (C4Error, FileNotFoundError) as exc:
        print(f"hnx c4: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
