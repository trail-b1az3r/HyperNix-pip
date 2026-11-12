"""``hnx strife`` — inspect, run, chat with and diagnose GGUF models on Strife.

::

    hnx strife inspect MODEL.gguf [--json]     # what it is, and whether Strife runs it
    hnx strife run MODEL.gguf -p "prompt"      # generate, streaming to the terminal
    hnx strife chat MODEL                      # the chat screen: chats, context, mouse
    hnx strife chat MODEL --plain              # line by line, for pipes and scripts
    hnx strife diag [MODEL.gguf]               # hardware, backend and placement
    hnx strife types                           # tensor types Strife can read
    hnx strife selftest                        # build a tiny model and run it

Errors print their code (``strife.<area>.<reason>``) and a remedy; with
``--json`` every command prints one JSON object (``run --json`` prints one
per event).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

__all__ = ["cli_main"]


def _apply_threads_early(argv: list[str]) -> None:
    """BLAS reads *_NUM_THREADS when numpy loads, so set them first."""
    for i, arg in enumerate(argv):
        value = None
        if arg in ("-t", "--threads") and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith("--threads="):
            value = arg.split("=", 1)[1]
        if value and value.isdigit():
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(name, value)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="machine-readable output")


def _load_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="a model id from the models folder, or a path to a .gguf")
    p.add_argument("-c", "--ctx", type=int, default=0, help="context length (default: the "
                   "model's, at most 4096)")
    p.add_argument("-t", "--threads", type=int, default=0, help="CPU threads")
    p.add_argument("--kernels", choices=("auto", "off"), default="auto",
                   help="native kernels: auto uses them when built (hnx strife kernels build), "
                   "off computes with numpy")
    p.add_argument("--cache-mb", type=int, default=-1,
                   help="RAM for decoded weights (default: 60%% of free RAM)")
    p.add_argument("--gpu-layers", type=int, default=0,
                   help="blocks whose matrices go on the GPU, through PyTorch (-1: as many "
                   "as fit). What was placed, and why not more, is reported")
    p.add_argument("--gpu-device", default=None, metavar="DEVICE",
                   help="cuda:N (default: the first CUDA device)")


def _sampling_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-n", "--max-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--min-p", type=float, default=0.05)
    p.add_argument("--repeat-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stop", action="append", default=[], help="stop string (repeatable)")
    p.add_argument("--timeout", type=float, default=None, help="seconds")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hnx strife",
                                     description="Strife: HyperNix's own GGUF runtime.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect", help="check a model without loading it")
    p.add_argument("model")
    p.add_argument("--tensors", action="store_true", help="list every tensor")
    _common(p)
    p = sub.add_parser("run", help="generate from a prompt")
    _load_args(p)
    _sampling_args(p)
    p.add_argument("-p", "--prompt", default=None)
    p.add_argument("--chat", action="store_true",
                   help="wrap the prompt in the model's chat template as a user message")
    p.add_argument("--system", default=None, help="system message (with --chat)")
    _common(p)
    p = sub.add_parser("chat", help="talk to a model: a screen with chats, context and the mouse")
    _load_args(p)
    _sampling_args(p)
    p.add_argument("--system", default=None)
    p.add_argument("--plain", action="store_true",
                   help="line-by-line chat instead of the screen (also when not on a terminal)")
    p.add_argument("--chats", default=None, help="where chats are kept "
                   "(default ~/.hypernix/strife/chats.json)")
    p = sub.add_parser("diag", help="hardware, backend selection and placement")
    p.add_argument("model", nargs="?")
    p.add_argument("-c", "--ctx", type=int, default=0)
    p.add_argument("--gpu-layers", type=int, default=0)
    p.add_argument("--gpu-device", default=None)
    _common(p)
    p = sub.add_parser("types", help="the tensor types Strife can read")
    _common(p)
    p = sub.add_parser("selftest", help="build a tiny model and generate with it")
    _common(p)
    p = sub.add_parser("build", help="register your own Strife build as a T1 runner",
                       description="Register a Strife you built -- a HyperNix checkout "
                       "(--source) or an interpreter with HyperNix installed (--python) -- so "
                       "the T1 runner can serve with it: POST /runner/load "
                       '{"backend": "strife", "strife_build": NAME}. The build is selftested '
                       "in its own interpreter first; nothing is recorded if it fails.")
    p.add_argument("name", nargs="?", help="what to call it (lower case)")
    p.add_argument("--source", default=None, help="a HyperNix source checkout")
    p.add_argument("--python", default=None, help="the interpreter to run it with")
    p.add_argument("--default", action="store_true",
                   help="use it whenever a strife load names no build")
    p.add_argument("--list", action="store_true", help="show the registered builds")
    p.add_argument("--remove", metavar="NAME", default=None, help="forget a build")
    _common(p)
    sub.add_parser("benchmark", help="measure a local model: prefill, decode, first token, "
                   "memory; beside llama.cpp when a build is found", add_help=False)
    p = sub.add_parser("kernels", help="build and check the native kernels for this machine",
                       description="Strife's native kernels: C code that decodes quantised "
                       "weights and multiplies by them without decoding them whole. `build` "
                       "compiles it for this CPU and checks every tensor type against the "
                       "Python decoders; Strife then uses it for the types that passed.")
    p.add_argument("action", choices=("build", "status", "verify", "remove"), nargs="?",
                   default="status")
    p.add_argument("--force", action="store_true", help="build again even if already built")
    _common(p)
    return parser


def cmd_kernels(args: argparse.Namespace) -> int:
    from . import kernels

    if args.action == "remove":
        count = kernels.remove()
        print(json.dumps({"removed": count}) if args.json else f"Removed {count} kernel build(s).")
        return 0
    if args.action == "verify":
        report = kernels.verify()
        if args.json:
            print(json.dumps(report))
        else:
            for name, entry in report["types"].items():
                print(f"  {name:<5} {'ok' if entry['ok'] else 'FAILED: ' + entry.get('error', '')}")
            print("Kernels " + ("verified." if report["ok"] else "FAILED verification for some types; "
                                "those types use numpy."))
        return 0 if report["ok"] else 1
    if args.action == "build":
        print("Compiling and verifying the kernels...", file=sys.stderr)
        built = kernels.build(force=args.force)
        info = {**built.describe(), "verification": built.manifest.get("verification", {})}
        if args.json:
            print(json.dumps(info))
        else:
            failed = [n for n, e in info["verification"].items() if not e.get("ok")]
            print(f"Built for this CPU ({info['simd']}, "
                  f"{info['openmp_threads'] or 'no'} OpenMP threads) with {info['compiler']}.")
            print(f"Verified: {', '.join(info['types'])}.")
            if failed:
                print(f"NOT verified (these use numpy): {', '.join(failed)}.")
            print(f"  {info['path']}")
        return 0
    info = kernels.status()
    if args.json:
        print(json.dumps(info))
    elif info["active"]:
        print(f"Native kernels: built ({info['simd']}, {info['openmp_threads'] or 'no'} OpenMP "
              f"threads), {len(info['types'])} types verified.")
        print(f"  {', '.join(info['types'])}")
        print(f"  {info['path']}")
    else:
        print(f"Native kernels: not in use: {info.get('reason', '')}")
        print(f"  compiler: {info['compiler'] or 'none found'}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from . import builds

    if args.remove:
        gone = builds.remove(args.remove)
        print(json.dumps({"removed": gone}) if args.json else
              (f"Removed {args.remove}." if gone else f"No build called {args.remove}."))
        return 0 if gone else 1
    if args.list or not args.name:
        found = builds.list_builds()
        chosen = builds.default_build()
        if args.json:
            print(json.dumps({"builds": [b.public() for b in found],
                              "default": chosen.name if chosen else builds.BUILTIN}))
            return 0
        print(f"  {builds.BUILTIN:<16} the Strife in this package"
              + ("  (default)" if chosen is None else ""))
        for b in found:
            mark = "  (default)" if chosen is not None and chosen.name == b.name else ""
            print(f"  {b.name:<16} {b.version or '?':<14} {b.source or b.python}{mark}")
        if not found:
            print("\nNo builds of your own yet: hnx strife build NAME --source CHECKOUT")
        return 0
    print(f"Selftesting {args.name} in its own interpreter...", file=sys.stderr)
    build = builds.register(args.name, source=args.source, python=args.python,
                            make_default=args.default)
    if args.json:
        print(json.dumps(build.public()))
    else:
        print(f"Registered {build.name}: HyperNix {build.version}, "
              f"selftest {build.selftest.get('tokens', 0)} tokens in "
              f"{build.selftest.get('seconds', 0)}s.")
        print(f'  T1: POST /runner/load {{"model_id": ..., "backend": "strife", '
              f'"strife_build": "{build.name}"}}')
    return 0


def _fail(exc: Exception, as_json: bool) -> int:
    from .errors import StrifeError

    if isinstance(exc, StrifeError):
        if as_json:
            print(json.dumps({"error": exc.to_dict()}))
        else:
            print(f"strife: {exc} [{exc.code}]", file=sys.stderr)
            if exc.remedy:
                print(f"  {exc.remedy}", file=sys.stderr)
            for problem in exc.details.get("problems", [])[1:6]:
                print(f"  also: {problem['message']}", file=sys.stderr)
        return 2
    raise exc


def _engine(args: argparse.Namespace) -> Any:
    from .engine import Engine
    from .models import resolve_model

    return Engine(resolve_model(args.model), n_ctx=args.ctx or None, threads=args.threads or None,
                  cache_bytes=None if args.cache_mb < 0 else args.cache_mb * 1024 * 1024,
                  gpu_layers=args.gpu_layers, gpu_device=getattr(args, "gpu_device", None),
                  kernels=None if getattr(args, "kernels", "auto") == "auto" else "off")


def _sampling(args: argparse.Namespace) -> Any:
    from .sampler import SamplerConfig

    return SamplerConfig(temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
                         min_p=args.min_p, repeat_penalty=args.repeat_penalty, seed=args.seed)


def _gib(n: int) -> str:
    return f"{n / 2**30:.2f} GiB" if n >= 2**30 else f"{n / 2**20:.1f} MiB"


def cmd_inspect(args: argparse.Namespace) -> int:
    from .inspect import read_gguf
    from .models import resolve_model
    from .quants import quant_name

    gguf, report = read_gguf(resolve_model(args.model))
    if args.json:
        out = report.to_dict()
        if args.tensors and gguf is not None:
            out["tensors"] = [{"name": t.name, "shape": list(t.shape),
                               "type": quant_name(t.ggml_type), "bytes": t.nbytes}
                              for t in gguf.tensors]
        print(json.dumps(out))
        return 0 if report.runnable else 2
    r = report
    print(f"{r.path}  ({_gib(r.file_bytes)}, GGUF v{r.gguf_version})")
    if r.architecture:
        print(f"  architecture  {r.architecture}  {r.name}")
        print(f"  shape         {r.n_layers} layers, {r.n_embd} wide, {r.n_heads} heads "
              f"({r.n_kv_heads} kv), head {r.head_dim}, ff {r.n_ff}, vocab {r.n_vocab}")
        print(f"  context       {r.context_length} trained, rope base {r.rope_freq_base:g}"
              + (f", linear scale {r.rope_scale:g}" if r.rope_scale != 1 else ""))
        print(f"  tokenizer     {r.tokenizer or '-'}  bos={r.bos_id} eos={r.eos_id}  "
              f"chat template: {'yes' if r.chat_template else 'no'}")
        print("  types         " + ", ".join(f"{k}×{v}" for k, v in sorted(r.types.items())))
        print(f"  weights       {_gib(r.weight_bytes)} in {r.tensor_count} tensors"
              + ("  (output tied to the embedding)" if r.tied_output else ""))
    if args.tensors and gguf is not None:
        for t in gguf.tensors:
            print(f"    {t.name:<40} {quant_name(t.ggml_type):<10} {list(t.shape)}")
    if r.runnable:
        print("  Strife: runs this model")
        return 0
    print("  Strife: cannot run this model")
    for p in r.problems:
        print(f"    - {p['message']} [{p['code']}]")
        if p["remedy"]:
            print(f"      {p['remedy']}")
    return 2


def cmd_run(args: argparse.Namespace) -> int:
    prompt = args.prompt if args.prompt is not None else (
        sys.stdin.read() if not sys.stdin.isatty() else None)
    if prompt is None:
        print("strife: give a prompt with -p, or on stdin", file=sys.stderr)
        return 2
    with _engine(args) as engine:
        kwargs: dict[str, Any] = {"max_tokens": args.max_tokens, "stop": args.stop,
                                  "sampling": _sampling(args), "timeout": args.timeout}
        if args.chat:
            messages = ([{"role": "system", "content": args.system}] if args.system else [])
            messages.append({"role": "user", "content": prompt})
            kwargs["messages"] = messages
        else:
            kwargs["prompt"] = prompt
        final: dict[str, Any] = {}
        try:
            for event in engine.stream(**kwargs):
                if args.json:
                    print(json.dumps(event if event["type"] == "done"
                                     else {"type": "token", "text": event["text"]}), flush=True)
                elif event["type"] == "token":
                    print(event["text"], end="", flush=True)
                if event["type"] == "done":
                    final = event
        except KeyboardInterrupt:
            print("\nstrife: interrupted", file=sys.stderr)
            return 130
        if not args.json:
            s = final["stats"]
            tps = s.get("decode_tokens_per_second")
            print(f"\n\n[{final['stop_reason']}] {s['generated_tokens']} tokens, "
                  f"prefill {s.get('prefill_tokens_per_second') or 0:.1f} tok/s, "
                  f"decode {tps or 0:.2f} tok/s, first token "
                  f"{s['time_to_first_token_seconds']:.2f} s", file=sys.stderr)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    if not args.plain and sys.stdin.isatty() and sys.stdout.isatty():
        return _chat_screen(args)
    return _chat_plain(args)


def _chat_screen(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .tui import ChatApp, ChatStore, run

    print(f"strife: loading {args.model}…", file=sys.stderr, flush=True)
    with _engine(args) as engine:
        store = ChatStore(Path(args.chats).expanduser() if args.chats else None)
        app = ChatApp(engine, store, max_tokens=args.max_tokens, sampling=_sampling(args))
        if args.system and not app.chat.messages:
            app.chat.messages.append({"role": "system", "content": args.system})
        return run(app)


def _chat_plain(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        session = engine.new_session()
        messages: list[dict[str, str]] = (
            [{"role": "system", "content": args.system}] if args.system else [])
        print(f"Strife chat with {engine.report.name or args.model}. Empty line or Ctrl-D "
              "to leave.", file=sys.stderr)
        while True:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                return 0
            if not line.strip():
                return 0
            messages.append({"role": "user", "content": line})
            reply = ""
            try:
                for event in engine.stream(messages=messages, session=session,
                                           max_tokens=args.max_tokens, stop=args.stop,
                                           sampling=_sampling(args), timeout=args.timeout):
                    if event["type"] == "token":
                        print(event["text"], end="", flush=True)
                    else:
                        reply = event["text"]
            except KeyboardInterrupt:
                print("\n(stopped)", file=sys.stderr)
            print()
            messages.append({"role": "assistant", "content": reply})


def cmd_diag(args: argparse.Namespace) -> int:
    from .placement import detect, plan

    hw = detect()
    from . import blasctl
    from . import kernels as kernels_module

    kern = kernels_module.status()
    out: dict[str, Any] = {"hardware": hw, "kernels": kern,
                           "backend": "cpu-native" if kern["active"] else "cpu-numpy",
                           "blas_threads_controllable": blasctl.controllable()}
    try:
        import numpy as np

        cfg: Any = (np.show_config(mode="dicts")
                    if "mode" in np.show_config.__code__.co_varnames else {})
        blas: Any = (cfg or {}).get("Build Dependencies", {}).get("blas", {})
        out["blas"] = {"name": blas.get("name"), "version": blas.get("version")}
    except Exception:
        out["blas"] = {}
    if args.model:
        from .inspect import inspect_model
        from .models import resolve_model

        report = inspect_model(resolve_model(args.model))
        out["model"] = {"runnable": report.runnable, "problems": report.problems}
        if report.runnable:
            gpu_info: dict[str, Any] = {}
            if args.gpu_layers:
                # What a load would place, worked out without loading anything.
                from . import gpu as gpu_module
                from .inspect import read_gguf

                choice = gpu_module.choose_device(args.gpu_device)
                gpu_info = {"reasons": list(choice.reasons), "device": choice.device,
                            "name": choice.name, "free_bytes": choice.free_bytes,
                            "total_bytes": choice.total_bytes}
                if choice.usable:
                    gguf, _ = read_gguf(report.path)
                    per_layer = gpu_module.layer_bytes(gguf, bytes_per_weight=2 if choice.half else 4)
                    layers, need = gpu_module.layers_that_fit(per_layer, args.gpu_layers,
                                                              choice.free_bytes)
                    gpu_info.update(layers_used=layers, bytes=need)
            out["placement"] = plan(report, None, n_ctx=args.ctx or min(report.context_length or 4096,
                                                                    4096),
                                    gpu_layers=args.gpu_layers, hardware=hw, gpu=gpu_info)
    if args.json:
        print(json.dumps(out, default=str))
        return 0
    print(f"Strife on {hw['os']} {hw['machine']}, Python {hw['python']}")
    print(f"  support   {hw['support']['tier']}: {hw['support']['note']}")
    print(f"  CPU       {hw['cpu_count']} threads ({hw['cpu']})")
    print(f"  RAM       {_gib(hw['ram']['total_bytes'])} total, "
          f"{_gib(hw['ram']['available_bytes'])} free")
    if hw["gpus"]:
        for g in hw["gpus"]:
            print(f"  GPU {g['index']}     {g['name']} ({_gib(g['vram_bytes'])}, "
                  f"compute {g['compute_capability'] or '?'})")
    else:
        print("  GPU       none found")
    blas = out.get("blas") or {}
    print(f"  backend   {out['backend']} (BLAS: {blas.get('name') or 'unknown'})")
    if kern["active"]:
        print(f"  kernels   native, {kern['simd']}, {kern['openmp_threads'] or 'no'} OpenMP threads; "
              f"verified: {', '.join(kern['types'])}")
        if not out["blas_threads_controllable"]:
            print("  kernels   decode only: BLAS threads cannot be limited here "
                  "(pip install threadpoolctl)")
    else:
        print(f"  kernels   not in use: {kern.get('reason', '')}")
    placement = out.get("placement")
    if placement:
        print(f"  placement {placement['cpu_layers']} of {placement['layers_total']} layers on "
              f"the CPU, {placement['gpu_layers_used']} on a GPU; weights mapped "
              f"{_gib(placement['weights_mapped_bytes'])}, KV cache "
              f"{_gib(placement['kv_cache_bytes_per_session'])} per session")
        for reason in placement["fallback_reasons"]:
            print(f"  fallback  {reason}")
    elif args.model:
        print("  model     not runnable: " + "; ".join(p["message"] for p in out["model"]["problems"][:3]))
    return 0


def cmd_types(args: argparse.Namespace) -> int:
    from .quants import supported_types

    types = supported_types()
    if args.json:
        print(json.dumps({str(k): v for k, v in types.items()}))
    else:
        for tid, name in types.items():
            print(f"  {tid:>4}  {name}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    import tempfile

    from .engine import Engine
    from .fixtures import make_model
    from .sampler import SamplerConfig

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for arch in ("llama", "qwen2", "qwen3"):
            path = os.path.join(tmp, f"{arch}.gguf")
            make_model(path, arch=arch)
            with Engine(path, n_ctx=128) as engine:
                a = engine.generate("hello world", max_tokens=8,
                                    sampling=SamplerConfig(temperature=0))
                b = engine.generate("hello world", max_tokens=8,
                                    sampling=SamplerConfig(temperature=0))
            ok = a.tokens == b.tokens and len(a.tokens) > 0
            results.append({"arch": arch, "ok": ok, "tokens": len(a.tokens),
                            "decode_tokens_per_second": a.stats["decode_tokens_per_second"]})
    passed = all(r["ok"] for r in results)
    if args.json:
        print(json.dumps({"ok": passed, "results": results}))
    else:
        for r in results:
            print(f"  {r['arch']:<7} {'ok' if r['ok'] else 'FAILED'}  {r['tokens']} tokens")
        print("Strife selftest " + ("passed" if passed else "FAILED"))
    return 0 if passed else 1


def cli_main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _apply_threads_early(argv)
    if argv and argv[0] in ("benchmark", "bench"):
        # Its own parser (the same one `benchmarks/strife_bench.py` runs).
        from .bench import main as bench_main

        return bench_main(argv[1:])
    args = _parser().parse_args(argv)
    handler = {"inspect": cmd_inspect, "run": cmd_run, "chat": cmd_chat, "diag": cmd_diag,
               "types": cmd_types, "selftest": cmd_selftest,
               "build": cmd_build, "kernels": cmd_kernels}[args.cmd]
    try:
        return handler(args)
    except Exception as exc:  # StrifeError is printed; anything else propagates
        return _fail(exc, getattr(args, "json", False))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
