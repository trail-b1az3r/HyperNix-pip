"""Command-line interface for the hypernix package.

Usage:
    hypernix <subcommand> [options]

Subcommands are script-friendly: each one wraps a single public function
from the library and returns a non-zero exit code on failure.

    all                 (default) download -> convert -> [quantize]
    download            fetch a HuggingFace snapshot
    convert             produce fp32 / fp16 GGUF from a snapshot
    quantize            run llama-quantize on an fp16/fp32 GGUF
    wakeup              train a wake word, then listen for it
    verify              read-check a GGUF and print its headers
    info                show package + GGUF header info
    upload              push files to a HuggingFace repo
    doctor              environment diagnostic
    path                put hypernix's console scripts on your PATH
    fetch-llama-quantize  pre-seed the llama-quantize cache
    train               training utilities (scratch / expand / loop)

Back-compat: invoking ``hypernix`` with flags and no subcommand runs
``hypernix all`` with those flags, so existing scripts keep working.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path


def _safe_print(text: str, **kwargs) -> None:
    """print() that degrades gracefully instead of crashing when the
    console encoding (e.g. Windows cp1252) can't represent the model's
    output. Falls back to replacing unencodable characters."""
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc), **kwargs)

DEFAULT_QUANTS: list[str] = ["fp32", "fp16"]

_ALIAS: dict[str, str] = {
    "fp32": "fp32",
    "f32": "fp32",
    "fp16": "fp16",
    "f16": "fp16",
    "q8": "q8_0",
    "q8_0": "q8_0",
    "q6": "q6_k",
    "q6_k": "q6_k",
    "q4km": "q4_k_m",
    "q4_k_m": "q4_k_m",
    # 0.72.3.post2: q4m is what people type for the mix. It is not a
    # block format of its own -- Q4_K is the format, _M is the policy
    # that widens attn_v/ffn_down and keeps the head at Q6_K.
    "q4m": "q4_k_m",
    "q4_m": "q4_k_m",
    "q5km": "q5_k_m",
    "q5_k_m": "q5_k_m",
}

_SUBCOMMANDS = {
    "all",
    "download",
    "convert",
    "quantize",
    "wakeup",
    "verify",
    "info",
    "upload",
    "doctor",
    "fetch-llama-quantize",
    "devices",
    "hyprslug-headers",
    "train",
    "generate",
    "oven",
    "chat",
    "brew",
    "pipeline",
    "assistant",
    "cli",
    "tvtop",
    "fizzle",
    "stml",
    "camo",
    "camouflage",
    "prot",
    "protect",
    "net",
    "wiki",
    "vera",
    "scavenger",
    "config",
    "gather",
    # v0.71.0
    "gkey",
    # v0.71.1
    "map",
    "websearch",
    "cctvtop",
    "fiz",
    # v0.71.5rc2
    "path",
}


def _print_usage() -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        
        console = Console()
        
        title = Text("HyperNix", style="bold cyan")
        title.append(" — download, convert, quantize, train HyperNix models", style="dim")
        
        table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
        table.add_column("Command")
        table.add_column("Description")
        
        table.add_row("[green]all[/]", "download -> convert -> \\[quantize] (classic pipeline)")
        table.add_row("[green]download[/]", "fetch a HuggingFace model snapshot to disk")
        table.add_row("[green]convert[/]", "produce fp32 / fp16 GGUF from a local snapshot")
        table.add_row("[green]quantize[/]", "run llama-quantize on an fp16/fp32 GGUF")
        table.add_row("[green]verify[/]", "read-check a GGUF and print its headers")
        table.add_row("[green]info[/]", "show package + GGUF header info")
        table.add_row("[green]upload[/]", "push files to a HuggingFace repo")
        table.add_row("[green]doctor[/]", "environment diagnostic (pass --fix to install missing deps)")
        table.add_row("[green]path[/]", "put hypernix's console scripts on your PATH")
        table.add_row("[green]fetch-llama-quantize[/]", "pre-seed the llama-quantize cache")
        table.add_row("[green]train[/]", "init / expand / run training utilities")
        table.add_row("[green]generate[/]", "sample text from a local HyperNix snapshot")
        table.add_row("[green]oven[/]", "code-generation wrapper (preheat + complete/fill)")
        table.add_row("[green]chat[/]", "interactive chat REPL with any HyperNix-family model")
        table.add_row("[green]brew[/]", "custom architecture builder & model training suite (brewer)")
        table.add_row("[green]pipeline[/]", "ASR -> LLM -> TTS pipeline")
        table.add_row("[green]assistant[/]", "Linux local AI assistant with voice commands")
        table.add_row("[green]cli[/]", "Interactive TUI/CLI menu for all operations")
        table.add_row("[green]stml[/]", "VRAM trained context length calculator")
        table.add_row("[green]fizzle[/]", "Fuzed Architecture module: fuse models and LoRAs (CLI: fiz)")
        table.add_row("[green]cctvtop[/]", "Live training dashboard TUI")
        table.add_row("[green]camo[/]", "RLHF/RLAF Camouflage scaffolding module")
        table.add_row("[green]net[/]", "Distributed network operations & Tailscale integration")
        table.add_row("[green]prot[/]", "Hardware health and monitor protection module")
        table.add_row("[green]wiki[/]", "HyperNix documentation wiki CLI")
        table.add_row("[green]vera[/]", "Vera assistant CLI")
        table.add_row("[green]scavenger[/]", "Scavenger tools")
        table.add_row("[green]gather[/]", "Crawl a site into a corpus (robots-aware)")
        table.add_row("[green]config[/]", "Configuration management")
        table.add_row("[green]gkey[/]", "API key & access management (Gatekeeper + Keymaster)")
        table.add_row("[green]map[/]", "Steampunk schematic TUI: dials/pipes/steam for model + training state")
        
        shortcuts = Text("Shortcuts:\n", style="bold yellow")
        shortcuts.append("  --auto-oven            download the default snapshot and run code completion\n", style="white")
        shortcuts.append("                         (equivalent to `hypernix oven --auto ...`).\n", style="dim")
        
        help_text = Text("\nRun `hypernix <subcommand> --help` for per-command flags.\nRun `hypernix all --help` for the classic pipeline flags.", style="italic dim")
        
        console.print(Panel.fit(title))
        console.print(table)
        console.print(shortcuts)
        console.print(help_text)
    except ImportError:
        print(
            "hypernix — download, convert, quantize, train HyperNix models\n\n"
            "usage: hypernix <subcommand> [options]\n\n"
            "Subcommands:\n"
            "  all                    download -> convert -> [quantize] (the classic pipeline)\n"
            "  download               fetch a HuggingFace model snapshot to disk\n"
            "  convert                produce fp32 / fp16 GGUF from a local snapshot\n"
            "  quantize               run llama-quantize on an fp16/fp32 GGUF\n"
            "  verify                 read-check a GGUF and print its headers\n"
            "  info                   show package + GGUF header info\n"
            "  upload                 push files to a HuggingFace repo\n"
            "  doctor                 environment diagnostic (pass --fix to install missing deps)\n"
            "  path                   put hypernix's console scripts on your PATH\n"
            "  fetch-llama-quantize   pre-seed the llama-quantize cache\n"
            "  train                  init / expand / run training utilities\n"
            "  generate               sample text from a local HyperNix snapshot\n"
            "  oven                   code-generation wrapper (preheat + complete/fill)\n"
            "  chat                   interactive chat REPL with any HyperNix-family model\n"
            "  brew                   custom architecture builder & training suite\n"
            "  pipeline               ASR -> LLM -> TTS pipeline (speech-to-speech or speech-to-text-to-speech)\n"
            "  assistant              Linux local AI assistant with voice commands (ASR + LLM + TTS)\n"
            "  cli                    Interactive TUI/CLI menu for all HyperNix operations\n"
            "  stml                   VRAM trained context length calculator\n"
            "  fizzle                 Fuzed Architecture module: fuse models and LoRAs\n"
            "  cctvtop                Live training dashboard TUI\n"
            "  camo                   RLHF/RLAF Camouflage scaffolding\n"
            "  net                    Distributed network & Tailscale integration\n"
            "  prot                   Hardware health & monitor protection\n"
            "  wiki                   HyperNix documentation wiki CLI\n"
            "  vera                   Vera assistant CLI\n"
            "  scavenger              Scavenger tools\n"
            "  gather                 crawl a site into a corpus (robots-aware)\n"
            "  config                 Configuration management\n"
            "  gkey                   API key & access management (Gatekeeper + Keymaster)\n"
            "  map                    Steampunk schematic TUI: dials/pipes/steam for model + training state\n\n"
            "Shortcuts:\n"
            "  --auto-oven            download the default snapshot and run code completion\n"
            "                         (equivalent to `hypernix oven --auto ...`).\n\n"
            "Run `hypernix <subcommand> --help` for per-command flags.\n"
            "Run `hypernix all --help` for the classic pipeline flags.\n"
        )


def _canonical(quant: str) -> str:
    key = quant.lower().replace("-", "_")
    if key not in _ALIAS:
        raise SystemExit(f"Unknown quant {quant!r}. Valid: {sorted(set(_ALIAS))}")
    return _ALIAS[key]


def _plan(quants: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for q in quants:
        c = _canonical(q)
        if c not in seen:
            seen.append(c)
    return seen


# ---------------------------------------------------------------------------
# `hypernix all` — the pipeline that used to be the only mode of operation.
# ---------------------------------------------------------------------------

def _build_all_parser(prog: str = "hypernix all") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Download the HyperNix PyTorch model and emit GGUF files.",
    )
    p.add_argument("--repo-id", default="ray0rf1re/hyper-nix.1")
    p.add_argument("--revision", default=None)
    p.add_argument("--model-dir", default=None, help="Reuse an existing local snapshot.")
    p.add_argument("--model-dir-name", default=None, help="Name for model cache directory under $HOME/.cache/hypernix/models.")
    p.add_argument("--output-dir", default="./hypernix-gguf")
    p.add_argument("--name", default="HyperNix")
    p.add_argument("--arch", default="hypernix")
    p.add_argument(
        "--quants", nargs="*", default=DEFAULT_QUANTS, metavar="QUANT",
        help=f"Valid: {sorted(set(_ALIAS))}. Default: {' '.join(DEFAULT_QUANTS)}",
    )
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--context-length", type=int, default=None)
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--llama-quantize", default=None)
    p.add_argument(
        "--no-auto-fetch", dest="auto_fetch", action="store_false", default=True,
        help="Disable auto-download of llama-quantize from GitHub releases.",
    )
    p.add_argument(
        "--auto", action="store_true", default=False,
        help="Unattended mode: configure everything automatically. Walks "
             "back through recent llama.cpp releases when the latest tag has "
             "no CPU-only asset, and falls back to `pip install "
             "llama-cpp-python` if GitHub fetching fails entirely.",
    )
    p.add_argument("--keep-intermediate", action="store_true")
    p.add_argument("--token", default=None)
    p.add_argument("--upload-to", default=None, metavar="REPO_ID")
    p.add_argument("--upload-private", action="store_true")
    return p


def _pick_source_for(q: str, produced: dict[str, Path]) -> Path:
    if "fp16" in produced:
        return produced["fp16"]
    if "fp32" in produced:
        return produced["fp32"]
    raise RuntimeError(f"Cannot produce {q!r}: need an fp16 or fp32 GGUF first.")


def _run_all(raw: list[str]) -> int:
    from hypernix.models.download import download_model, verify_snapshot
    from hypernix.quant.convert import convert_to_gguf
    from hypernix.quant.quantize import quantize_gguf
    from hypernix.timing.spinner import Spinner

    args = _build_all_parser().parse_args(raw)
    plan = _plan(args.quants)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
        if not model_dir.exists():
            print(f"--model-dir {model_dir} does not exist", file=sys.stderr)
            return 2
        with Spinner(f"Verifying snapshot at {model_dir}"):
            verify_snapshot(model_dir)
    else:
        with Spinner(f"Downloading {args.repo_id}", style="arc"):
            model_dir = download_model(repo_id=args.repo_id, revision=args.revision, token=args.token)
    print(f"[hypernix] model dir: {model_dir}", file=sys.stderr)

    base_name = args.repo_id.split("/")[-1].replace(".", "-")
    produced: dict[str, Path] = {}
    need_fp16 = any(q not in {"fp32", "fp16"} for q in plan) or "fp16" in plan
    need_fp32 = "fp32" in plan

    if need_fp32:
        out = output_dir / f"{base_name}-fp32.gguf"
        with Spinner("Converting to fp32 GGUF", style="grow"):
            convert_to_gguf(
                model_dir, out, dtype="fp32", arch_name=args.arch, name=args.name,
                n_head_hint=args.n_head, context_length=args.context_length,
            )
        produced["fp32"] = out
    if need_fp16:
        out = output_dir / f"{base_name}-fp16.gguf"
        with Spinner("Converting to fp16 GGUF", style="grow"):
            convert_to_gguf(
                model_dir, out, dtype="fp16", arch_name=args.arch, name=args.name,
                n_head_hint=args.n_head, context_length=args.context_length,
            )
        produced["fp16"] = out

    for q in plan:
        if q in {"fp32", "fp16"}:
            continue
        out = output_dir / f"{base_name}-{q}.gguf"
        with Spinner(f"Quantizing {q.upper()}", style="bar"):
            quantize_gguf(
                source_gguf=_pick_source_for(q, produced), output_gguf=out,
                quant_type=q, threads=args.threads,
                llama_quantize_bin=args.llama_quantize, auto_fetch=args.auto_fetch,
                auto=args.auto,
            )
        produced[q] = out

    if not args.keep_intermediate and "fp16" in produced and "fp16" not in plan:
        try:
            produced["fp16"].unlink()
            produced.pop("fp16")
        except OSError:
            pass

    print("[hypernix] done:", file=sys.stderr)
    for q, path in produced.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {q:<8} {size_mb:8.1f} MB  {path}", file=sys.stderr)

    if args.upload_to:
        from hypernix.quant.upload import upload_gguf
        with Spinner(f"Uploading to {args.upload_to}", style="arrows"):
            url = upload_gguf(
                files=list(produced.values()), repo_id=args.upload_to,
                token=args.token, private=args.upload_private,
            )
        print(f"[hypernix] uploaded: {url}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Individual subcommands (script-friendly)
# ---------------------------------------------------------------------------

def _run_download(raw: list[str]) -> int:
    from hypernix.models.download import download_model
    from hypernix.timing.spinner import Spinner

    p = argparse.ArgumentParser(prog="hypernix download")
    p.add_argument("--repo-id", default="ray0rf1re/hyper-nix.1")
    p.add_argument("--revision", default=None)
    p.add_argument("--local-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--token", default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--no-verify", dest="verify", action="store_false", default=True,
        help="Skip the post-download sanity check.",
    )
    ns = p.parse_args(raw)
    with Spinner(f"Downloading {ns.repo_id}", style="arc"):
        path = download_model(
            repo_id=ns.repo_id, revision=ns.revision,
            local_dir=ns.local_dir, cache_dir=ns.cache_dir, token=ns.token,
            quiet=ns.quiet, verify=ns.verify,
        )
    print(path)
    return 0


def _run_convert(raw: list[str]) -> int:
    from hypernix.quant.convert import convert_to_gguf
    from hypernix.timing.spinner import Spinner

    p = argparse.ArgumentParser(prog="hypernix convert")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", default="fp16", choices=["fp32", "f32", "fp16", "f16"])
    p.add_argument("--arch", default="hypernix")
    p.add_argument("--name", default="HyperNix")
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--context-length", type=int, default=None)
    ns = p.parse_args(raw)
    with Spinner(f"Converting to {ns.dtype.upper()} GGUF", style="grow"):
        out = convert_to_gguf(
            model_dir=ns.model_dir, output=ns.output, dtype=ns.dtype,
            arch_name=ns.arch, name=ns.name,
            n_head_hint=ns.n_head, context_length=ns.context_length,
        )
    print(out)
    return 0


def _run_quantize(raw: list[str]) -> int:
    from hypernix.quant.quantize import quantize_gguf
    from hypernix.timing.spinner import Spinner

    p = argparse.ArgumentParser(prog="hypernix quantize")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--type", dest="qtype", required=True,
                   help=f"Quant type. Valid: {sorted(set(_ALIAS))}")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--llama-quantize", default=None)
    p.add_argument("--no-auto-fetch", dest="auto_fetch", action="store_false", default=True)
    p.add_argument(
        "--auto", action="store_true", default=False,
        help="Unattended mode: walks back through recent llama.cpp "
             "releases and falls back to `pip install llama-cpp-python` "
             "if GitHub fetching fails.",
    )
    p.add_argument(
        "-hnx", "--hnx", dest="hnx", action="store_true",
        help="Quantise with hyprslug and never touch llama.cpp — no binary "
             "looked for, downloaded or built, and llama-cpp-python not "
             "needed. Covers the HyperNix sub-bit tiers (IQ0.9_L, IQ0.75_M, "
             "IQ0.5_XXXL), which llama-quantize cannot produce at all.",
    )
    p.add_argument("--imatrix", default=None,
                   help="Importance matrix as JSON (hyprslug tiers).")
    p.add_argument(
        "--quantize-embeddings", dest="quantize_embeddings",
        action=argparse.BooleanOptionalAction, default=None,
        help="Include token_embd in the quantisation. A sub-bit tier leaves "
             "it in float by default, because at half a bit the embedding "
             "table is the model — but on a 7B that untouched table is then "
             "most of the file, so a genuinely sub-1-bit result needs this.",
    )
    p.add_argument(
        "--quantize-output", dest="quantize_output",
        action=argparse.BooleanOptionalAction, default=None,
        help="Include the output head. Same default and the same trade.",
    )
    ns = p.parse_args(raw)
    with Spinner(f"Quantizing → {ns.qtype.upper()}", style="bar"):
        out = quantize_gguf(
            source_gguf=ns.source, output_gguf=ns.output, quant_type=ns.qtype,
            threads=ns.threads, llama_quantize_bin=ns.llama_quantize,
            auto_fetch=ns.auto_fetch, auto=ns.auto,
            backend="hnx" if ns.hnx else "auto",
            imatrix=ns.imatrix,
            quantize_embeddings=ns.quantize_embeddings,
            quantize_output=ns.quantize_output,
        )
    print(out)
    return 0


def _run_devices(raw: list[str]) -> int:
    """`hypernix devices` — what can run a sub-bit model here, and why not.

    Worth its own command because the interesting answer is usually a
    refusal. "CUDA is not available" and "CUDA is available and this
    wheel has no kernels for your card" are different problems with
    different fixes, and the second one is invisible until the first
    kernel launch.
    """
    from hypernix.models.hnxdevice import describe, probe, select

    p = argparse.ArgumentParser(
        prog="hypernix devices",
        description="List the accelerators this runtime can and cannot use.",
    )
    p.add_argument("--json", dest="as_json", action="store_true")
    ns = p.parse_args(raw)

    found = probe()
    if ns.as_json:
        chosen = select("auto")
        print(json.dumps({
            "devices": [d.to_dict() for d in found],
            "auto": chosen.name,
        }, indent=2))
        return 0
    print(describe(found))
    return 0


def _run_verify(raw: list[str]) -> int:
    """Read-validate a GGUF file by parsing its header with the `gguf` library."""
    p = argparse.ArgumentParser(prog="hypernix verify")
    p.add_argument("gguf", help="Path to a .gguf file")
    p.add_argument("--tensors", action="store_true", help="Also list tensors.")
    ns = p.parse_args(raw)

    from gguf import GGUFReader  # type: ignore

    path = Path(ns.gguf)
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 2
    try:
        reader = GGUFReader(str(path), "r")
    except Exception as exc:  # noqa: BLE001
        print(f"[hypernix verify] FAILED to parse {path}: {exc}", file=sys.stderr)
        return 1

    # `GGUFReader.version` existed in older gguf; newer versions expose it
    # as the `GGUF.version` field instead. Try both, then fall back to
    # reading the raw magic bytes.
    gguf_version: int | str = getattr(reader, "version", None) or "?"
    if gguf_version == "?":
        field = reader.fields.get("GGUF.version")
        if field is not None and field.parts:
            try:
                gguf_version = int(field.parts[-1][0])
            except (IndexError, TypeError, ValueError):
                pass

    print(f"[hypernix verify] {path}")
    print(f"  version: {gguf_version}")
    print(f"  tensors: {len(reader.tensors)}   fields: {len(reader.fields)}")
    for name in sorted(reader.fields):
        f = reader.fields[name]
        try:
            value = f.parts[-1]
            summary = repr(value)[:80]
        except Exception:  # noqa: BLE001
            summary = "<binary>"
        print(f"    {name:<40} {summary}")
    if ns.tensors:
        for t in reader.tensors:
            print(f"    tensor {t.name}  shape={tuple(t.shape)}  type={t.tensor_type.name}")
    return 0


def _run_info(raw: list[str]) -> int:
    p = argparse.ArgumentParser(prog="hypernix info")
    p.add_argument("--gguf", default=None, help="Optional .gguf path to summarize.")
    ns = p.parse_args(raw)
    from hypernix import __version__

    print(f"hypernix {__version__}")
    print(f"python   {sys.version.split()[0]}")
    try:
        import torch
        print(f"torch    {torch.__version__}")
    except Exception:
        pass
    if ns.gguf:
        return _run_verify([ns.gguf])
    return 0


def _run_upload(raw: list[str]) -> int:
    from hypernix.quant.upload import upload_gguf

    p = argparse.ArgumentParser(prog="hypernix upload")
    p.add_argument("--repo-id", default="ray0rf1re/HyperNix.1-gguf")
    p.add_argument("--token", default=None)
    p.add_argument("--private", action="store_true")
    p.add_argument("--commit-message", default="Add HyperNix GGUF quantizations")
    p.add_argument("files", nargs="+")
    ns = p.parse_args(raw)
    url = upload_gguf(
        files=ns.files, repo_id=ns.repo_id, token=ns.token,
        private=ns.private, commit_message=ns.commit_message,
    )
    print(url)
    return 0


def _run_fetch_llama_quantize(raw: list[str]) -> int:
    from hypernix.quant.fetcher import cache_dir, cached_binary, fetch_llama_quantize
    from hypernix.quant.quantize import _find_llama_quantize  # noqa: PLC2701

    p = argparse.ArgumentParser(prog="hypernix fetch-llama-quantize")
    p.add_argument("--force", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--auto", action="store_true", default=False,
        help="Also try `pip install llama-cpp-python` if GitHub fetching fails.",
    )
    p.add_argument(
        "--search-releases", type=int, default=10,
        help="How many recent llama.cpp releases to probe for a matching "
             "CPU asset (newest first). Default: 10.",
    )
    ns = p.parse_args(raw)
    existing = cached_binary()
    if existing and not ns.force:
        print(f"[hypernix] already cached: {existing}", file=sys.stderr)
        return 0
    if ns.auto:
        # Route through the resolver so the PyPI fallback is engaged when
        # the GitHub fetch legitimately fails.
        path = _find_llama_quantize(auto_fetch=True, auto=True, quiet=ns.quiet)
    else:
        path = str(fetch_llama_quantize(
            force=ns.force, quiet=ns.quiet, search_releases=ns.search_releases,
        ))
    print(f"[hypernix] {path}", file=sys.stderr)
    print(f"[hypernix] cache dir: {cache_dir()}", file=sys.stderr)
    return 0


def _run_train(raw: list[str]) -> int:
    """`hypernix train {init,expand,run}` training utilities."""
    from hypernix.training.train import HyperNixConfig, expand_checkpoint, init_from_scratch, train

    p = argparse.ArgumentParser(prog="hypernix train")
    sub = p.add_subparsers(dest="action", required=True)

    p_init = sub.add_parser("init", help="Initialize a fresh HyperNix snapshot.")
    p_init.add_argument("--out-dir", required=True)
    p_init.add_argument("--tokenizer-source", default=None,
                        help="Existing snapshot to copy tokenizer files from.")
    p_init.add_argument("--vocab-size", type=int, default=32000)
    p_init.add_argument("--hidden-size", type=int, default=1024)
    p_init.add_argument("--intermediate-size", type=int, default=4096)
    p_init.add_argument("--num-hidden-layers", type=int, default=16)
    p_init.add_argument("--num-attention-heads", type=int, default=16)
    p_init.add_argument("--num-key-value-heads", type=int, default=None)
    p_init.add_argument("--max-position-embeddings", type=int, default=2048)
    p_init.add_argument("--rope-theta", type=float, default=10000.0)
    p_init.add_argument("--tie-word-embeddings", action="store_true")
    p_init.add_argument("--seed", type=int, default=None,
                        help="Seed torch RNG before init for reproducibility.")

    p_exp = sub.add_parser("expand", help="Warm-start a bigger model from a smaller one.")
    p_exp.add_argument("--src-dir", required=True)
    p_exp.add_argument("--dst-dir", required=True)
    p_exp.add_argument("--hidden-size", type=int, default=None)
    p_exp.add_argument("--intermediate-size", type=int, default=None)
    p_exp.add_argument("--num-hidden-layers", type=int, default=None)
    p_exp.add_argument("--num-attention-heads", type=int, default=None)
    p_exp.add_argument("--vocab-size", type=int, default=None)
    p_exp.add_argument("--init-std", type=float, default=0.02)
    p_exp.add_argument("--seed", type=int, default=None,
                       help="Seed torch RNG before expansion for reproducibility.")

    p_run = sub.add_parser("run", help="Run a minimal causal-LM training loop.")
    p_run.add_argument("--model-dir", required=True)
    p_run.add_argument("--dataset", required=True, help="Path to a raw-text file.")
    p_run.add_argument("--out-dir", required=True)
    p_run.add_argument("--steps", type=int, default=1000)
    p_run.add_argument("--batch-size", type=int, default=2)
    p_run.add_argument("--context-length", type=int, default=512)
    p_run.add_argument("--lr", type=float, default=3e-4)
    p_run.add_argument("--weight-decay", type=float, default=0.1)
    p_run.add_argument("--grad-clip", type=float, default=1.0)
    p_run.add_argument("--device", default=None)
    p_run.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p_run.add_argument("--log-every", type=int, default=10)
    p_run.add_argument("--save-every", type=int, default=500)
    p_run.add_argument("--seed", type=int, default=None,
                       help="Seed torch RNG before training for reproducibility.")
    p_run.add_argument("--use-abbicus", action="store_true")
    p_run.add_argument("--use-turbo-abbicus", action="store_true")
    p_run.add_argument("--use-stml", action="store_true")
    p_run.add_argument("--untrained-max-context", type=int, default=8192)
    p_run.add_argument("--segment-length", type=int, default=512)
    p_run.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Recompute activations in backward instead of storing them: "
             "~30%% more compute for most of the activation memory.",
    )
    p_run.add_argument(
        "--checkpoint-every", type=int, default=1,
        help="Checkpoint every Nth block (1 = all, 2 = half the saving "
             "for half the extra compute).",
    )
    p_run.add_argument(
        "--fuse-optimizer", action="store_true",
        help="Step and free each gradient as it is produced, instead of "
             "holding a full copy of them. Needs --grad-clip 0.",
    )
    p_run.add_argument(
        "--tune-allocator", action="store_true",
        help="Set expandable_segments on the CUDA allocator so a long run "
             "fragments less.",
    )

    ns = p.parse_args(raw)
    if ns.action == "init":
        cfg = HyperNixConfig(
            vocab_size=ns.vocab_size, hidden_size=ns.hidden_size,
            intermediate_size=ns.intermediate_size,
            num_hidden_layers=ns.num_hidden_layers,
            num_attention_heads=ns.num_attention_heads,
            num_key_value_heads=ns.num_key_value_heads,
            max_position_embeddings=ns.max_position_embeddings,
            rope_theta=ns.rope_theta,
            tie_word_embeddings=ns.tie_word_embeddings,
        )
        out = init_from_scratch(
            ns.out_dir, cfg, tokenizer_source=ns.tokenizer_source, seed=ns.seed,
        )
    elif ns.action == "expand":
        out = expand_checkpoint(
            ns.src_dir, ns.dst_dir,
            hidden_size=ns.hidden_size, intermediate_size=ns.intermediate_size,
            num_hidden_layers=ns.num_hidden_layers,
            num_attention_heads=ns.num_attention_heads,
            vocab_size=ns.vocab_size, init_std=ns.init_std, seed=ns.seed,
        )
    else:  # run
        out = train(
            ns.model_dir, ns.dataset, ns.out_dir,
            steps=ns.steps, batch_size=ns.batch_size,
            context_length=ns.context_length, lr=ns.lr,
            weight_decay=ns.weight_decay, grad_clip=ns.grad_clip,
            device=ns.device, dtype=ns.dtype,
            log_every=ns.log_every, save_every=ns.save_every, seed=ns.seed,
            use_abbicus=ns.use_abbicus,
            use_turbo_abbicus=ns.use_turbo_abbicus,
            use_stml=ns.use_stml,
            untrained_max_context=ns.untrained_max_context,
            segment_length=ns.segment_length,
            gradient_checkpointing=ns.gradient_checkpointing,
            checkpoint_every=ns.checkpoint_every,
            fuse_optimizer=ns.fuse_optimizer,
            tune_allocator=ns.tune_allocator,
        )
    print(out)
    return 0


def _run_oven(raw: list[str]) -> int:
    """`hypernix oven` — code-generation wrapper around HyperNix.

    Equivalent (roughly) to::

        oven = hypernix.neo_oven.preheat(repo_id, local_dir=..., device=...)
        print(oven.complete(prompt))
        # or oven.fill(prefix, suffix) when --fill-prefix is given
        oven.save_pt(save_pt_path)  # if --save-pt
    """
    from hypernix.models.neo_oven import preheat

    p = argparse.ArgumentParser(
        prog="hypernix oven",
        description="Preheat a HyperNix snapshot and bake code out of it.",
    )
    p.add_argument("--repo-id", default="ray0rf1re/hyper-nix.1")
    p.add_argument("--revision", default=None)
    p.add_argument("--model-dir", default=None,
                   help="An existing local snapshot, or a .gguf file, instead "
                        "of downloading.")
    p.add_argument("--token", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--quiet", action="store_true")

    p.add_argument("--prompt", default=None,
                   help="Prompt to complete. Mutually exclusive with --fill-prefix.")
    p.add_argument("--fill-prefix", default=None,
                   help="FIM prefix; requires --fill-suffix.")
    p.add_argument("--fill-suffix", default=None,
                   help="FIM suffix; requires --fill-prefix.")

    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--save-pt", default=None,
                   help="Also write a self-contained torch.load-able bundle to this path.")
    p.add_argument("--auto", action="store_true", default=False,
                   help="Unattended mode alias (same defaults; reserved for future use).")

    ns = p.parse_args(raw)

    has_fill = ns.fill_prefix is not None or ns.fill_suffix is not None
    if has_fill and (ns.fill_prefix is None or ns.fill_suffix is None):
        p.error("--fill-prefix and --fill-suffix must be used together")
    if has_fill and ns.prompt is not None:
        p.error("--prompt cannot be combined with --fill-prefix/--fill-suffix")

    oven = preheat(
        repo_id=ns.repo_id, revision=ns.revision, local_dir=ns.model_dir,
        token=ns.token, device=ns.device, dtype=ns.dtype, quiet=ns.quiet,
    )

    if has_fill:
        text = oven.fill(
            prefix=ns.fill_prefix, suffix=ns.fill_suffix,
            max_new_tokens=ns.max_new_tokens,
            temperature=ns.temperature, top_k=ns.top_k, top_p=ns.top_p,
            seed=ns.seed,
        )
    elif ns.prompt is not None:
        text = oven.complete(
            prompt=ns.prompt,
            max_new_tokens=ns.max_new_tokens,
            temperature=ns.temperature, top_k=ns.top_k, top_p=ns.top_p,
            seed=ns.seed,
        )
    else:
        text = None

    if text is not None:
        print(text)

    if ns.save_pt:
        out = oven.save_pt(ns.save_pt)
        print(f"[hypernix] wrote {out}", file=sys.stderr)
    return 0


_SIZE_SUFFIXES = {"": 1, "b": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}


def _parse_size(text: str) -> int:
    """``"2G"`` -> bytes. Plain digits are already bytes.

    Written out because ``--cache-bytes 2000000000`` is a number nobody
    types correctly, and a silently mis-parsed budget is a memory limit
    that does not hold.
    """
    cleaned = str(text).strip().lower().removesuffix("ib").removesuffix("b")
    if not cleaned:
        raise argparse.ArgumentTypeError("expected a size like 512M or 2G")
    suffix = cleaned[-1] if cleaned[-1] in _SIZE_SUFFIXES else ""
    number = cleaned[:-1] if suffix else cleaned
    try:
        value = float(number)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a size. Try 512M, 2G, or a plain byte count."
        ) from None
    if value < 0:
        raise argparse.ArgumentTypeError("a size cannot be negative")
    return int(value * _SIZE_SUFFIXES[suffix])


_DEVICE_HELP = (
    "Where to run a sub-bit model: auto, cpu, cuda, cuda:1, mps, xpu. "
    "`auto` takes the first that works and falls back to CPU; a named one "
    "that cannot run says why and what to install rather than silently "
    "using the CPU. `hypernix devices` lists them. Vulkan goes through "
    "llama.cpp, not this runtime — see `hypernix devices`."
)

#: Shared by `generate` and `chat`, because a flag that means one thing
#: on one command and is missing on the other is worse than no flag.
_CACHE_BYTES_HELP = (
    "Memory to spend keeping sub-bit weights decoded (e.g. 512M, 2G). "
    "The IQ0.x tiers hold their weights packed and unpack inside every "
    "matmul, which costs about 4x float32 in time; this buys some of it "
    "back, largest tensors first. Ignored for models llama.cpp runs."
)


def _run_chat(raw: list[str]) -> int:
    """`hypernix chat` — chat REPL against any HyperNix-family model.

    Accepts short names from :data:`hypernix.download.KNOWN_MODELS`
    (``nano-nano``, ``nano-mini``, ``nano-nano-927``, ``hyper-nix``) or a
    full HF repo id. ``--message`` runs a single turn and exits (useful
    for scripting); without it, drops into an interactive REPL.
    """
    from hypernix.models.neo_oven import preheat

    p = argparse.ArgumentParser(
        prog="hypernix chat",
        description="Chat with any HyperNix-family model.",
    )
    p.add_argument("--repo-id", default="ray0rf1re/hyper-nix.1",
                   help="Short name or full HF repo id.")
    p.add_argument("--revision", default=None)
    p.add_argument("--model-dir", default=None,
                   help="Reuse an existing local snapshot instead of downloading.")
    p.add_argument("--token", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--system", default=None,
                   help="Optional system prompt prepended to every turn.")
    p.add_argument("--message", default=None,
                   help="Single user message. If set, runs one turn and exits.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--cache-bytes", dest="cache_bytes", type=_parse_size,
                   default=0, help=_CACHE_BYTES_HELP)
    p.add_argument("--hnx-device", dest="hnx_device", default="auto",
                   help=_DEVICE_HELP)

    ns = p.parse_args(raw)

    # A .gguf given as --model-dir is a model, not a snapshot directory.
    # Held open across the whole session rather than reloaded per turn: a
    # GGUF load is seconds to minutes, and paying it on every message
    # would make the REPL unusable.
    from hypernix.models.ggufrun import GGUFRunError, is_gguf, load_gguf

    gguf_model = None
    if ns.model_dir and is_gguf(ns.model_dir):
        try:
            gguf_model = load_gguf(
                ns.model_dir, quiet=ns.quiet, cache_bytes=ns.cache_bytes,
                device=ns.hnx_device,
            )
        except GGUFRunError as exc:
            print(f"hypernix chat: {exc}", file=sys.stderr)
            return 1
        oven = None
    else:
        oven = preheat(
            repo_id=ns.repo_id, revision=ns.revision, local_dir=ns.model_dir,
            token=ns.token, device=ns.device, dtype=ns.dtype, quiet=ns.quiet,
        )

    history: list[dict[str, str]] = []
    if ns.system:
        history.append({"role": "system", "content": ns.system})

    def turn(user_msg: str) -> str:
        history.append({"role": "user", "content": user_msg})
        if gguf_model is not None:
            reply = gguf_model.chat(
                [m for m in history if m["role"] != "system"],
                system=ns.system,
                max_tokens=ns.max_new_tokens,
                temperature=ns.temperature,
            )
        else:
            reply = oven.chat(
                list(history),
                max_new_tokens=ns.max_new_tokens,
                temperature=ns.temperature, top_k=ns.top_k, top_p=ns.top_p,
                seed=ns.seed,
            )
        history.append({"role": "assistant", "content": reply})
        return reply

    if ns.message is not None:
        _safe_print(turn(ns.message))
        return 0

    print("[hypernix chat] Ctrl-D or empty line to exit.", file=sys.stderr)
    while True:
        try:
            line = input("you> ")
        except EOFError:
            print(file=sys.stderr)
            return 0
        if not line.strip():
            return 0
        print(f"assistant> {turn(line)}")


def _run_generate(raw: list[str]) -> int:
    """`hypernix generate` — sample text from a local HyperNix snapshot."""
    from hypernix.models.generate import generate_text

    p = argparse.ArgumentParser(
        prog="hypernix generate",
        description="Sample text from a local HyperNix snapshot directory.",
    )
    p.add_argument("--model-dir", required=True,
                   help="A HF-style snapshot (config.json + safetensors), or a "
                        ".gguf file.")
    p.add_argument("--prompt", default="",
                   help="Prompt to condition on. Empty => start from BOS.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--cache-bytes", dest="cache_bytes", type=_parse_size,
                   default=0, help=_CACHE_BYTES_HELP)
    p.add_argument("--hnx-device", dest="hnx_device", default="auto",
                   help=_DEVICE_HELP)
    ns = p.parse_args(raw)

    # A .gguf is not a snapshot directory, and generate_text would fail
    # looking for config.json. GGUF is the format this package spends most
    # of its time producing, so `generate` reading one is the least
    # surprising behaviour available.
    from hypernix.models.ggufrun import GGUFRunError, generate_with_gguf, is_gguf

    if is_gguf(ns.model_dir):
        try:
            print(generate_with_gguf(
                ns.model_dir, ns.prompt,
                max_new_tokens=ns.max_new_tokens,
                temperature=ns.temperature,
                cache_bytes=ns.cache_bytes,
                device=ns.hnx_device,
            ))
        except GGUFRunError as exc:
            print(f"hypernix generate: {exc}", file=sys.stderr)
            return 1
        return 0

    text = generate_text(
        model_dir=ns.model_dir, prompt=ns.prompt,
        max_new_tokens=ns.max_new_tokens, temperature=ns.temperature,
        top_k=ns.top_k, top_p=ns.top_p, seed=ns.seed,
        device=ns.device, dtype=ns.dtype,
    )
    _safe_print(text)
    return 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    # No args / top-level --help / --version -> print the subcommand menu.
    if not raw or raw[0] in ("-h", "--help"):
        try:
            from hypernix.timing.spinner import anime_print
            anime_print("HyperNix", style="banner", delay=0.03)
        except Exception:
            pass
        _print_usage()
        return 0
    if raw[0] in ("-V", "--version"):
        from hypernix import __version__
        try:
            from hypernix.timing.spinner import anime_print
            anime_print(f"hypernix {__version__}", style="typewriter", delay=0.04)
        except Exception:
            print(f"hypernix {__version__}")
        return 0

    # Top-level --auto-oven shortcut: translate to `oven --auto ...` so users
    # can run a one-liner `hypernix --auto-oven --prompt "def fib(n):"` and
    # get a working PyTorch model + a completion with zero extra ceremony.
    if raw[0] == "--auto-oven":
        return _run_oven(["--auto", *raw[1:]])

    # First arg isn't a subcommand -> print help instead of falling back to 'all'
    if raw[0] not in _SUBCOMMANDS:
        _print_usage()
        return 1

    cmd, rest = raw[0], raw[1:]
    if cmd == "all":
        return _run_all(rest)
    if cmd == "download":
        return _run_download(rest)
    if cmd == "convert":
        return _run_convert(rest)
    if cmd == "quantize":
        return _run_quantize(rest)
    if cmd == "wakeup":
        from hypernix.audio.wakeup_cli import main as _wakeup_main

        return _wakeup_main(rest)
    if cmd == "gather":
        from hypernix.data.gather_cli import main as _gather_main

        return _gather_main(rest)
    if cmd == "verify":
        return _run_verify(rest)
    if cmd == "info":
        return _run_info(rest)
    if cmd == "upload":
        return _run_upload(rest)
    if cmd == "doctor":
        from hypernix.system.doctor import run
        dp = argparse.ArgumentParser(prog="hypernix doctor")
        dp.add_argument("--fix", action="store_true",
                        help="Install missing runtime dependencies via pip.")
        ns = dp.parse_args(rest)
        return run(fix=ns.fix)
    if cmd == "path":
        from hypernix.system.pathfix import cli_main as path_main
        return path_main(rest)
    if cmd == "fetch-llama-quantize":
        return _run_fetch_llama_quantize(rest)
    if cmd == "devices":
        return _run_devices(rest)
    if cmd == "hyprslug-headers":
        from hypernix.quant.hyprslug_headers_cli import main as _headers_main

        return _headers_main(rest)
    if cmd == "train":
        return _run_train(rest)
    if cmd == "generate":
        return _run_generate(rest)
    if cmd == "oven":
        return _run_oven(rest)
    if cmd == "chat":
        return _run_chat(rest)
    if cmd == "brew":
        return _run_brew(rest)
    if cmd == "pipeline":
        return _run_pipeline(rest)
    if cmd == "assistant":
        return _run_assistant(rest)
    if cmd in ("fizzle", "fiz"):
        return _run_fizzle(rest)
    if cmd in ("camo", "camouflage"):
        return _run_camo(rest)
    if cmd == "tvtop":
        return _run_tvtop(rest)
    if cmd == "cctvtop":
        return _run_cctvtop(rest)
    if cmd == "cli":
        return _run_cli(rest)
    if cmd == "stml":
        return _run_stml(rest)
    if cmd in ("prot", "protect"):
        return _run_protect(rest)
    if cmd == "net":
        return _run_net(rest)
    if cmd == "wiki":
        return _run_wiki(rest)
    if cmd == "vera":
        return _run_vera(rest)
    if cmd == "scavenger":
        return _run_scavenger(rest)
    if cmd == "config":
        return _run_config(rest)
    if cmd == "gkey":
        return _run_gkey(rest)
    if cmd == "map":
        return _run_map(rest)
    if cmd == "websearch":
        return _run_websearch(rest)
    _print_usage()
    return 1

def _run_websearch(raw: list[str]) -> int:
    """`hypernix websearch` — Non-API web searching module."""
    from hypernix.interfaces.websearch import cli_main as websearch_main
    return websearch_main(raw)

def _run_protect(raw: list[str]) -> int:
    """`hypernix protect` / `prot` — Hardware health and monitor protection."""
    from hypernix.system.protect import cli_main as protect_main
    return protect_main(raw)

def _run_net(raw: list[str]) -> int:
    """`hypernix net` — Advanced tailscale & distributed network manager."""
    from hypernix.system.net import cli_main as net_main
    return net_main(raw)

def _run_wiki(raw: list[str]) -> int:
    from hypernix.interfaces.wiki_cli import cli_main as wiki_main
    return wiki_main(raw)

def _run_vera(raw: list[str]) -> int:
    from hypernix.evaluation.vera import cli_main as vera_main
    return vera_main(raw)

def _run_scavenger(raw: list[str]) -> int:
    from hypernix.data.scavenger import cli_main as scavenger_main
    return scavenger_main(raw)

def _run_config(raw: list[str]) -> int:
    from hypernix.system.config import cli_main as config_main
    return config_main(raw)

def _run_map(raw: list[str]) -> int:
    """`hypernix map` — steampunk schematic TUI (dials/pipes/steam)."""
    from hypernix.monitoring.map import cli_main as map_main
    return map_main(raw)

def _run_gkey(raw: list[str]) -> int:
    """`hypernix gkey` — Gatekeeper + Keymaster unified CLI."""
    from hypernix.security.gkey_cli import main as gkey_main
    return gkey_main(raw)

def _run_fizzle(raw: list[str]) -> int:
    """`hypernix fizzle` / `fiz` — Fuzed Architecture module."""
    from hypernix.training.fizzle import main as fizzle_main
    return fizzle_main(raw)


def _run_tvtop(raw: list[str]) -> int:
    """`hypernix tvtop` — live training dashboard."""
    from hypernix.monitoring.tv import cli_main as tvtop_main
    return tvtop_main(raw)


def _run_cctvtop(raw: list[str]) -> int:
    """`hypernix cctvtop` — Python training dashboard with hardware metrics and VNC."""
    from hypernix.monitoring.cctvtop import cli_main as cctvtop_main
    return cctvtop_main(raw)


def _run_brew(raw: list[str]) -> int:
    """`hypernix brew` — custom architecture builder and model training suite.
    
    When invoked with a JSON recipe file path, runs instant_pot.brew().
    When invoked with subcommands (new, list, train, export), uses the brewer CLI.
    """
    # Check if first argument is a JSON file path (for instant_pot.brew)
    if raw and len(raw) >= 1:
        first_arg = raw[0]
        # If it looks like a file path (not a subcommand), treat as recipe
        if not first_arg.startswith('-') and first_arg.endswith('.json'):
            import json

            from hypernix.training.instant_pot import brew

            recipe_path = Path(first_arg)
            if not recipe_path.exists():
                print(f"Error: Recipe file not found: {recipe_path}", file=sys.stderr)
                return 1
            
            try:
                recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"Error reading recipe: {e}", file=sys.stderr)
                return 1
            
            # Parse optional --set overrides
            parser = argparse.ArgumentParser(prog="hypernix brew")
            parser.add_argument("recipe", help="Path to a JSON recipe file.")
            parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                              help="Override a recipe key (repeatable).")
            ns = parser.parse_args(raw)
            
            # Apply --set overrides
            for override in ns.set:
                if "=" not in override:
                    print(f"Error: Invalid --set format: {override} (expected KEY=VALUE)", file=sys.stderr)
                    return 1
                key, value = override.split("=", 1)
                # Try to parse as JSON for proper type conversion
                try:
                    recipe[key] = json.loads(value)
                except json.JSONDecodeError:
                    recipe[key] = value
            
            try:
                brew(recipe)
                return 0
            except Exception as e:
                print(f"Error running brew: {e}", file=sys.stderr)
                return 1
    
    # Otherwise, delegate to brewer CLI for subcommands
    from hypernix.training.brewer import cli_main as brewer_main
    try:
        brewer_main(raw)
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1







def _run_cli(raw: list[str]) -> int:
    """`hypernix cli` — Interactive TUI/CLI menu for all HyperNix operations."""
    from hypernix.chat.countertop import interactive_cli

    p = argparse.ArgumentParser(
        prog="hypernix cli",
        description="Interactive TUI/CLI menu for all HyperNix operations.",
    )
    p.add_argument("--simple", action="store_true", help="Use simple text-based menu instead of rich TUI")
    ns = p.parse_args(raw)

    print("[hypernix cli] Launching interactive menu...")
    return interactive_cli(use_rich=not ns.simple)


def _run_pipeline(raw: list[str]) -> int:
    """`hypernix pipeline` — Run ASR → LLM → TTS pipeline with interactive prompts."""
    from hypernix.models.workshop import ASRConfig, ASREngine, ASRToLLMToTTS, TTSConfig, TTSEngine
    
    p = argparse.ArgumentParser(
        prog="hypernix pipeline",
        description="Run full ASR → LLM → TTS pipeline interactively.",
    )
    p.add_argument("--audio", "-a", help="Path to input audio file")
    p.add_argument("--active", action="store_true", help="Record from microphone in real-time (press Ctrl+C to stop and process)")
    p.add_argument("--asr", default="nano-whisper", help="ASR engine to use")
    p.add_argument("--llm", default="qwen3.5-1b", help="LLM model to use (default: Qwen3.5 1B)")
    p.add_argument("--tts", default="nano-tacotron", help="TTS engine to use")
    p.add_argument("--prompt", "-p", default="", help="System prompt for LLM")
    p.add_argument("--output", "-o", help="Output audio file path")
    ns = p.parse_args(raw)
    
    # Initialize engines
    print(f"Initializing ASR engine: {ns.asr}")
    asr_config = ASRConfig(sample_rate=16000)
    asr_engine = ASREngine(asr_config)
    asr_engine.initialize()
    
    print(f"Initializing TTS engine: {ns.tts}")
    tts_config = TTSConfig(sample_rate=22050)
    tts_engine = TTSEngine(tts_config)
    tts_engine.initialize()
    
    # Create simple LLM wrapper
    class SimpleLLM:
        def generate(self, prompt, max_new_tokens=256, temperature=0.7):
            # Extract last user message
            if "User:" in prompt:
                last_msg = prompt.split("User:")[-1].strip().split("\n")[0]
                return f"I heard: '{last_msg[:50]}...' - this is a simulated response."
            return "Hello! How can I assist you today?"
    
    llm = SimpleLLM()
    
    # Create pipeline
    print("Creating ASR → LLM → TTS pipeline...")
    pipeline = ASRToLLMToTTS(
        asr_engine=asr_engine,
        llm=llm,
        tts_engine=tts_engine,
        system_prompt=ns.prompt or "You are a helpful assistant."
    )
    
    # Get audio file
    audio_path = ns.audio
    if not audio_path:
        if ns.active:
            # Record from microphone
            print("Recording from microphone... Press Ctrl+C to stop and process")
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            audio_path = temp_file.name
            try:
                import sounddevice as sd
                import soundfile as sf
                
                RATE = 16000
                CHANNELS = 1
                
                print("Recording... (Using sounddevice for better compatibility)")
                # Start a recording stream
                recording = []
                def callback(indata, frames, time, status):
                    if status:
                        print(status)
                    recording.append(indata.copy())
                
                with sd.InputStream(samplerate=RATE, channels=CHANNELS, callback=callback):
                    try:
                        while True:
                            sd.sleep(100)
                    except KeyboardInterrupt:
                        pass
                
                import numpy as np
                audio_data = np.concatenate(recording, axis=0)
                sf.write(audio_path, audio_data, RATE)
                print("\nStopped recording. Processing...")
            except ImportError:
                print("\nMissing sounddevice or soundfile. Install with: pip install sounddevice soundfile")
                return 1
            except KeyboardInterrupt:
                print("\nStopped recording. Processing...")
            
            print(f"Recorded audio saved to: {audio_path}")
        else:
            audio_path = input("Enter path to audio file: ").strip()
    
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1
    
    # Run pipeline
    print(f"Processing audio: {audio_path}")
    try:
        response_text, audio_bytes = pipeline.process(audio_path)
        print("\n✓ Transcription complete!")
        print(f"Response: {response_text}")
        
        # Save output
        output_path = ns.output or "pipeline_output.wav"
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        print(f"✓ Audio saved to: {output_path}")
        
        return 0
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return 1


def _run_assistant(raw: list[str]) -> int:
    """`hypernix assistant` — interactive local AI assistant (chat + voice)."""
    from hypernix.interfaces.assistant import InteractiveCLI

    p = argparse.ArgumentParser(
        prog="hypernix assistant",
        description=(
            "Interactive assistant. Chat turns are dispatched through the same "
            "provider registry as `hyped` (local snapshots, local GGUF, or a "
            "cloud API), so --model selects a real model."
        ),
    )
    p.add_argument(
        "--voice", "-v", action="store_true",
        help="Speak replies with the TTS engine (also toggled in-session with /voice)",
    )
    p.add_argument(
        "--model", "-m", default=None,
        help="Model short name; defaults to the one set by `hypernix config`",
    )
    ns = p.parse_args(raw)

    InteractiveCLI(model=ns.model, voice=ns.voice).cmd_assistant()
    return 0


def _run_camo(raw: list[str]) -> int:
    """`hypernix camo` - RLHF/RLAF alignment module."""
    from hypernix.training.camouflage import main as camo_main
    return camo_main(raw)


def _run_stml(raw: list[str]) -> int:
    """`hypernix stml` - CLI command for the short term memory loss context calculator."""
    from hypernix.models.stml import calculate_vram_context

    p = argparse.ArgumentParser(
        prog="hypernix stml",
        description="Calculate trained context sequence length based on available GPU VRAM and model parameters."
    )
    p.add_argument("--vram", type=float, required=True, help="Available VRAM in GB (e.g. 8.0, 16.0, 24.0)")
    p.add_argument("--params", type=float, default=4.0, help="Model parameters in billions (e.g. 1.0, 4.0, 7.0)")
    p.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    p.add_argument("--precision", choices=["fp32", "fp16", "int8", "int4"], default="fp16", help="Precision (fp32, fp16, int8, int4)")
    p.add_argument("--num-layers", type=int, default=32, help="Number of model layers")
    p.add_argument("--num-heads", type=int, default=32, help="Number of attention heads")
    p.add_argument("--head-dim", type=int, default=128, help="Dimension per head")

    ns = p.parse_args(raw)
    
    ctx = calculate_vram_context(
        vram_gb=ns.vram,
        model_size_params=ns.params,
        batch_size=ns.batch_size,
        precision=ns.precision,
        num_layers=ns.num_layers,
        num_heads=ns.num_heads,
        head_dim=ns.head_dim
    )
    print(f"Calculated max trained context length: {ctx} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
