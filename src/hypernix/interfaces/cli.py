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
    fetch-llama-quantize  pre-seed the llama-quantize cache (fllq)
    train               training utilities (scratch / expand / loop)
    auto                download, convert if needed, quantise (0.72.8a3)
    extra               the commands kept out of the main menu

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
    "neuron",
    "dilute",
    "errors",
    "elements",
    "scavenger",
    "config",
    "gather",
    "fusebox",
    "fuse-box",
    # 0.72.8: explode, measure, reap and sow a model
    "c4",
    "runtime",
    # v0.71.0
    "gkey",
    # v0.71.1
    "map",
    "websearch",
    "cctvtop",
    "fiz",
    # v0.71.5rc2
    "path",
    # 0.72.8a3
    "auto",
    "extra",
    "fllq",
    "tvtopmax",
    # 0.72.8rc1: the hnx-scriptgen entry point, as a subcommand too
    "scriptgen",
    "hypedpro",
    "image",
    "claude",
    # 0.72.8a4
    "list-all",
}


#: Spellings that are not subcommands but are obviously meant as one.
#:
#: Separate from difflib's fuzzy matching because these are not typos --
#: they are the other correct spelling, or the name of the tool rather
#: than the name of the subcommand. "quantise" is British English and
#: this package's own docstrings use it throughout; "tvtoppro" is a
#: console script people reasonably expect to work as a subcommand too.
_COMMAND_ALIASES: dict[str, str] = {
    "quantise": "quantize",
    "quant": "quantize",
    "q": "quantize",
    "dl": "download",
    "get": "download",
    "pull": "download",
    "conv": "convert",
    "gguf": "convert",
    "ls": "info",
    "list": "info",
    "version": "--version",
    "help": "--help",
    "tvtoppro": "tvtop",
    "tvtop-pro": "tvtop",
    "hyprslug": "quantize",
    "doomslug": "quantize",
    "keys": "gkey",
    "key": "gkey",
    "docs": "wiki",
    "diagnose": "doctor",
    "check": "doctor",
    "dashboard": "cctvtop",
    "monitor": "cctvtop",
    "tvtop-max": "tvtopmax",
    "hyped-pro": "hypedpro",
    "fetch-llama-quantise": "fetch-llama-quantize",
    "fetch-llama-quantisize": "fetch-llama-quantize",
    "claude-code": "claude",
}


def _suggest(typed: str, limit: int = 3) -> list[str]:
    """Subcommands *typed* was probably meant to be.

    Prefix matches first and exactly: somebody who typed ``down`` meant
    ``download`` and difflib's ratio does not reliably rank a prefix
    above a similarly-sized edit. Fuzzy matches fill the rest.
    """
    import difflib

    lowered = typed.lower()
    known = sorted(_SUBCOMMANDS)
    prefix = [name for name in known if name.startswith(lowered)]
    contains = [
        name for name in known
        if lowered in name and name not in prefix and len(lowered) >= 3
    ]
    fuzzy = [
        name for name in difflib.get_close_matches(lowered, known, n=limit, cutoff=0.6)
        if name not in prefix and name not in contains
    ]
    ordered: list[str] = []
    for name in (*prefix, *contains, *fuzzy):
        if name not in ordered:
            ordered.append(name)
    return ordered[:limit]


def _unknown_command(typed: str) -> int:
    """Say what was wrong with it, and what to type instead."""
    print(f"hypernix: unknown command {typed!r}", file=sys.stderr)

    alias = _COMMAND_ALIASES.get(typed.lower())
    if alias:
        print(f"\n  You want:  hypernix {alias}", file=sys.stderr)
    else:
        suggestions = _suggest(typed)
        if len(suggestions) == 1:
            print(f"\n  Did you mean:  hypernix {suggestions[0]}", file=sys.stderr)
        elif suggestions:
            print("\n  Did you mean one of these?", file=sys.stderr)
            for name in suggestions:
                print(f"      hypernix {name}", file=sys.stderr)

    print("\n  hypernix --help   every subcommand", file=sys.stderr)
    return 2


#: The main menu, in the order `hypernix --help` lists it.
_MAIN_MENU: tuple[tuple[str, str], ...] = (
    ("auto", "download a repo, convert it if it is not a GGUF, quantise it: hnx auto REPO -q Q4_K_M"),
    ("all", "download -> convert -> \\[quantize] (classic pipeline)"),
    ("download", "fetch a HuggingFace model snapshot, one file, or owner/repo:QUANT"),
    ("convert", "produce fp32 / fp16 GGUF from a local snapshot; -P -Q TARGET quantises too"),
    ("quantize", "run llama-quantize on an fp16/fp32 GGUF"),
    ("verify", "read-check a GGUF and print its headers"),
    ("info", "show package + GGUF header info"),
    ("upload", "push files to a HuggingFace repo"),
    ("doctor", "environment diagnostic (pass --fix to install missing deps)"),
    ("fetch-llama-quantize", "pre-seed the llama-quantize cache (shortcut: fllq)"),
    ("train", "init / expand / run training utilities"),
    ("scriptgen", "build a training script that runs: a window, a terminal wizard, or flags"),
    ("chat", "interactive chat REPL with any HyperNix-family model"),
    ("claude", "run Claude Code on a model this machine's T1 server is serving"),
    ("brew", "custom architecture builder & model training suite (brewer)"),
    ("image", "hnx0xv2image: an image-to-text model fused onto LFM2.5, Qwen3.5 or HyperNix"),
    ("cli", "Interactive TUI/CLI menu for all operations"),
    ("stml", "VRAM trained context length calculator"),
    ("fizzle", "Fuzed Architecture module: fuse models and LoRAs (CLI: fiz)"),
    ("tvtopmax", "tvtop-max: the run's dashboard on OpenTUI, with its modules, model and logs"),
    ("hypedpro", "hyped-pro: the OpenTUI chat client with tools"),
    ("net", "DEPRECATED: Tailscale and peers are now in the T1 API (/net)"),
    ("vera", "Vera assistant CLI"),
    ("dilute", "Best-of-n sampling: many answers per prompt, keep the good ones"),
    ("errors", "Look up an error code (L#-NNNNN.kS)"),
    ("elements", "Addons: hydrogen, carbon, magnesium, and your own"),
    ("gather", "Crawl a site into a corpus (robots-aware)"),
    ("fusebox", "GPU thermal governor: pace a run, trip on heat, underclock"),
    ("c4", "Explode a model; measure, reap and sow its layers and weights"),
    ("runtime", "Use the HyperNix llama.cpp from LM Studio and other apps"),
    ("config", "Configuration management"),
    ("gkey", "API key & access management (Gatekeeper + Keymaster)"),
    ("devices", "list the GPUs/accelerators this machine has, and what they support"),
    ("wakeup", "wake-word detection: train a trigger phrase, then listen for it"),
    ("hyprslug-headers", "self-describing GGUF headers + the OpenAI-compatible sub-bit server"),
    ("extra", "more commands: neuron, pipeline, assistant, generate, wiki, prot, oven, ..."),
    ("list-all", "every command: hnx, hnx extra, hnx-t1, and every other HyperNix command"),
)

#: `hypernix extra`: commands that still work, kept out of the main menu.
_EXTRA_MENU: tuple[tuple[str, str], ...] = (
    ("neuron", "Train small networks that act: RL, imitation, supervised"),
    ("pipeline", "Speech in, a model's answer, speech out (ASR -> LLM -> TTS)"),
    ("assistant", "Local AI assistant with voice commands"),
    ("generate", "sample text from a local HyperNix snapshot"),
    ("wiki", "HyperNix documentation wiki CLI (every module and package)"),
    ("prot", "Blank the screen until a wake word, without locking it (protect)"),
    ("oven", "code-generation wrapper (preheat + complete/fill)"),
    ("path", "put hypernix's console scripts on your PATH"),
    ("map", "Steampunk schematic TUI: dials/pipes/steam for model + training state"),
    # `hypernix tvtop` is the classic dashboard, not an alias of cctvtop
    # (the standalone `tvtop` console script is cctvtop; see wiki/Dashboards.md).
    ("tvtop", "the classic training dashboard (the `tvtop` command runs cctvtop)"),
    ("cctvtop", "Live training dashboard TUI with hardware metrics"),
    ("camo", "RLHF/RLAF Camouflage alignment"),
    ("websearch", "web search without an API (Bing/DuckDuckGo), real result links"),
    ("scavenger", "search + size Hugging Face datasets under storage/quality budgets"),
)

_ALSO_ACCEPTED = "fllq (fetch-llama-quantize) · fiz (fizzle) · camouflage (camo) · protect (prot) · fuse-box (fusebox)"


def _print_menu(title_text: str, rows: tuple[tuple[str, str], ...], footer: str) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print(f"{title_text}\n\nusage: hypernix <subcommand> [options]\n\nSubcommands:")
        for name, what in rows:
            print(f"  {name:<22} {what.replace(chr(92), '')}")
        print(f"\n{footer}")
        return
    console = Console()
    title = Text("HyperNix", style="bold cyan")
    title.append(f" — {title_text}", style="dim")
    table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
    table.add_column("Command")
    table.add_column("Description")
    for name, what in rows:
        table.add_row(f"[green]{name}[/]", what)
    console.print(Panel.fit(title))
    console.print(table)
    console.print(Text(footer, style="dim"))


def _print_usage() -> None:
    _print_menu(
        "download, convert, quantize, train HyperNix models",
        _MAIN_MENU,
        "Shortcuts:\n"
        "  --auto-oven            download the default snapshot and run code completion\n"
        "                         (equivalent to `hypernix oven --auto ...`).\n"
        f"\nAlso accepted: {_ALSO_ACCEPTED}\n"
        "\nRun `hypernix extra` for more commands (neuron, pipeline, assistant, generate, wiki,\n"
        "prot, oven, path, map, tvtop, cctvtop, camo, websearch, scavenger).\n"
        "Run `hypernix <subcommand> --help` for per-command flags.\n"
        "Run `hypernix all --help` for the classic pipeline flags.",
    )


def _print_extra() -> None:
    _print_menu(
        "more commands",
        _EXTRA_MENU,
        "These work exactly as before; they are listed here to keep `hypernix --help` short.\n"
        "Run `hypernix <subcommand> --help` for per-command flags.",
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

def _download_latest(ns: argparse.Namespace) -> int:
    """``hnx download --latest``: the newest HyperNix model, ready to serve."""
    from hypernix.hyperlink.brewed import BrewedError, download
    from hypernix.hyperlink.catalogue import DEFAULT_LOCAL_DIR
    from hypernix.hyperlink.latest import latest

    release = latest(online=not ns.offline)
    target = Path(ns.output or ns.local_dir or (DEFAULT_LOCAL_DIR / release.folder)).expanduser()
    if not ns.quiet:
        print(f"Newest HyperNix model: {release.folder} ({release.repo})")
        if release.note:
            print(f"  {release.note}")
    if ns.token:
        os.environ.setdefault("HF_TOKEN", ns.token)
    try:
        folder = download(release.repo, target)
    except BrewedError as exc:
        print(f"hypernix download: {exc}", file=sys.stderr)
        return 1
    if not ns.quiet:
        print(f"Saved to {folder}")
        print(f"  Serve it:  hypernix-t1 runner start {release.model_id}")
        print(f"  Relay it:  POST /inference/relay with \"model\": \"{release.model_id}\" "
              f"(a document longer than its context, read in hops)")
    return 0


def _run_download(raw: list[str]) -> int:
    """``hnx download [SOURCE] [-o OUT]``.

    SOURCE is a repo id, a short name, a link to a repo, or a link to one
    file in it (0.72.8); ``--repo-id`` still works and is now optional.
    """
    from hypernix.models.hflinks import LinkError, parse_source, pick_gguf
    from hypernix.timing.spinner import Spinner

    p = argparse.ArgumentParser(
        prog="hypernix download",
        description="Download a model: a repo, a link to one, or a link to a single file.",
        epilog="examples:\n"
               "  hnx download ray0rf1re/hyper-nix.1\n"
               "  hnx download https://huggingface.co/owner/repo -o ./repo\n"
               "  hnx download https://huggingface.co/owner/repo/blob/main/m-Q4_K_M.gguf -o m.gguf\n"
               "  hnx download owner/repo-GGUF -f m-Q8_0.gguf\n"
               "  hnx download --latest                     the newest HyperNix model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", nargs="?", default=None,
                   help="Repo id, short name, https://huggingface.co/... link, or a link to a .gguf")
    p.add_argument("--repo-id", default=None, help="The same as SOURCE (kept for scripts)")
    p.add_argument("-f", "--file", default=None,
                   help="One file from the repo instead of the whole model, e.g. a .gguf")
    p.add_argument("-o", "--output", default=None,
                   help="Where it goes: a folder, or for one file a file name")
    p.add_argument("--revision", default=None)
    p.add_argument("--local-dir", default=None, help="Same as -o for a whole repo")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--token", default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--no-verify", dest="verify", action="store_false", default=True,
        help="Skip the post-download sanity check.",
    )
    p.add_argument("--latest", action="store_true",
                   help="the newest HyperNix model (HyperNix.3.1-mini at the time of writing), "
                        "into the models folder the runner reads")
    p.add_argument("--offline", action="store_true",
                   help="with --latest: the newest this HyperNix knows, without asking the Hub")
    ns = p.parse_args(raw)
    if ns.latest:
        if ns.source or ns.repo_id or ns.file:
            print("hypernix download: --latest names the model itself; give no SOURCE or -f.",
                  file=sys.stderr)
            return 2
        return _download_latest(ns)
    text = ns.source or ns.repo_id or "ray0rf1re/hyper-nix.1"
    try:
        source = parse_source(text)
    except LinkError as exc:
        print(f"hypernix download: {exc}", file=sys.stderr)
        return 2
    revision = ns.revision or source.revision
    filename = ns.file or source.filename

    if source.quant and not filename:
        from hypernix.models.hflinks import pick_gguf_files

        files = _repo_files(source.repo_id, revision, ns.token)
        chosen = pick_gguf_files(files, source.quant)
        if not chosen:
            ggufs = sorted(f for f in files if f.lower().endswith(".gguf"))
            from hypernix.system.errorcatalogue import REPO_NO_SUCH_QUANT

            print(f"hypernix download: {REPO_NO_SUCH_QUANT.code}: {source.repo_id} has no single {source.quant} GGUF"
                  + ("; it has:" if ggufs else " (or the Hub could not be asked)."), file=sys.stderr)
            for name in ggufs:
                print(f"  {name}", file=sys.stderr)
            return 2
        paths = []
        for name in chosen:
            with Spinner(f"Downloading {name} from {source.repo_id}", style="arc"):
                try:
                    paths.append(_download_file(source.repo_id, name, revision=revision,
                                                output=ns.output or ns.local_dir, token=ns.token,
                                                cache_dir=ns.cache_dir))
                except Exception as exc:  # noqa: BLE001 - shown, not a traceback
                    from hypernix.models.download import explain_hub_error

                    print(f"hypernix download: {explain_hub_error(exc, source.repo_id) or exc}",
                          file=sys.stderr)
                    return 1
        print(paths[0])
        return 0

    if not filename and "/" in source.repo_id:
        # A repo of GGUFs has no config.json: fetching it as a model
        # finds "nothing to download". Take its one GGUF instead.
        files = _repo_files(source.repo_id, revision, ns.token)
        if files and not any(f.endswith((".safetensors", ".bin", ".pt")) for f in files):
            filename = pick_gguf(files) or ""
            if not filename and any(f.lower().endswith(".gguf") for f in files):
                print(f"hypernix download: {source.repo_id} has several GGUF files; pick one with -f:",
                      file=sys.stderr)
                for name in sorted(f for f in files if f.lower().endswith(".gguf")):
                    print(f"  {name}", file=sys.stderr)
                return 2

    if filename:
        with Spinner(f"Downloading {filename} from {source.repo_id}", style="arc"):
            try:
                path = _download_file(source.repo_id, filename, revision=revision,
                                      output=ns.output or ns.local_dir, token=ns.token,
                                      cache_dir=ns.cache_dir)
            except Exception as exc:  # noqa: BLE001 - shown, not a traceback
                from hypernix.models.download import explain_hub_error

                print(f"hypernix download: {explain_hub_error(exc, source.repo_id) or exc}", file=sys.stderr)
                return 1
        print(path)
        return 0

    from hypernix.models.download import DownloadError, download_model, explain_hub_error

    try:
        with Spinner(f"Downloading {source.repo_id}", style="arc"):
            path = download_model(
                repo_id=source.repo_id, revision=revision,
                local_dir=ns.output or ns.local_dir, cache_dir=ns.cache_dir, token=ns.token,
                quiet=ns.quiet, verify=ns.verify,
            )
    except (DownloadError, FileNotFoundError) as exc:
        print(f"hypernix download: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


def _repo_files(repo_id: str, revision: str | None, token: str | None) -> list[str]:
    """The repo's file names, or [] when the Hub cannot be asked."""
    try:
        from huggingface_hub import HfApi

        return list(HfApi().list_repo_files(repo_id, revision=revision, token=token))
    except Exception:  # noqa: BLE001 - offline: fall back to a model download
        return []


def _download_file(repo_id: str, filename: str, *, revision: str | None, output: str | None,
                   token: str | None, cache_dir: str | None) -> Path:
    """One file from a repo, to *output* (a folder, or a file name)."""
    import shutil

    from huggingface_hub import hf_hub_download

    if output and not output.endswith(("/", "\\")) and Path(output).suffix and not Path(output).is_dir():
        target = Path(output).expanduser().resolve()
        folder = target.parent
    else:
        if output:
            folder = Path(output).expanduser().resolve()
        else:
            try:
                from hypernix.system.config import get_models_dir

                folder = get_models_dir() / repo_id.split("/")[-1]
            except Exception:  # noqa: BLE001
                folder = Path.home() / ".hypernix" / "models" / repo_id.split("/")[-1]
        target = folder / Path(filename).name
    folder.mkdir(parents=True, exist_ok=True)
    fetched = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision,
                                   local_dir=str(folder), token=token, cache_dir=cache_dir))
    if fetched.resolve() != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(fetched), str(target))
        # hf_hub_download mirrors the repo's folders; drop the empty ones.
        parent = fetched.parent
        while parent != folder and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    return target


def _run_convert(raw: list[str]) -> int:
    from hypernix.quant.convert import convert_to_gguf
    from hypernix.timing.spinner import Spinner

    p = argparse.ArgumentParser(
        prog="hypernix convert",
        description="Produce an F32/F16 GGUF from a snapshot, or with -P, "
                    "convert and quantise in one step.",
        epilog="examples:\n"
               "  hypernix convert --model-dir ./snap --output snap.f16.gguf\n"
               "  hypernix convert ./snap -P -Q Q4_K_M\n"
               "  hnx convert ./snap -q -p              (Q4_K_M)\n"
               "  hnx convert ./snap/model.safetensors -P -Q q6h4 -o snap.q6h4.gguf\n"
               "  hnx convert ./brewer_models/mine -P -Q q6h",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", nargs="?", default=None,
                   help="Model folder, .safetensors, Brewer folder/.pt, or .gguf "
                        "(same as --model-dir)")
    p.add_argument("--model-dir", default=None)
    p.add_argument("-o", "--output", default=None)
    p.add_argument("-P", "-p", "--pipeline", action="store_true",
                   help="If the source is safetensors, convert it, then quantise "
                        "the result with hyprslug to the -Q target. A Brewer "
                        "model or a GGUF is quantised as it is.")
    p.add_argument("-Q", "-q", "--quant", nargs="?", const="Q4_K_M", default=None, metavar="TARGET",
                   help="hyprslug target for -P: any llama.cpp type or mix "
                        "(Q4_K_M, Q8_0...), a HyperNix tier (INT3, FP8, "
                        "IQ0.5...), or a hybrid (q6h, q6h4, q6h2, hnx_q6h1, "
                        "hnx_q6h1m). Implies -P. Alone, Q4_K_M.")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="With -P, keep the F16 GGUF beside the output")
    p.add_argument("--imatrix", default=None, help="With -P, an importance matrix")
    p.add_argument("--dtype", default="fp16", choices=["fp32", "f32", "fp16", "f16"])
    p.add_argument("--arch", default="hypernix")
    p.add_argument("--name", default="HyperNix")
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--context-length", type=int, default=None)
    ns = p.parse_args(raw)
    if ns.quant and not (ns.source or ns.model_dir) and Path(ns.quant).expanduser().exists():
        # `hnx convert -q ./model -p`: -q took the model as its target.
        ns.source, ns.quant = ns.quant, "Q4_K_M"
    source = ns.source or ns.model_dir
    if not source:
        p.error("give a model: hypernix convert MODEL [-P -Q TARGET], or --model-dir")
    if ns.pipeline or ns.quant:
        return _run_convert_pipeline(ns, source)
    if not ns.output:
        p.error("--output is required without -P")
    ns.model_dir = source
    with Spinner(f"Converting to {ns.dtype.upper()} GGUF", style="grow"):
        out = convert_to_gguf(
            model_dir=ns.model_dir, output=ns.output, dtype=ns.dtype,
            arch_name=ns.arch, name=ns.name,
            n_head_hint=ns.n_head, context_length=ns.context_length,
        )
    _note_architecture(out)
    print(out)
    return 0


def _note_architecture(path) -> None:
    """Say so when a GGUF was written under a name llama.cpp does not know."""
    from hypernix.quant.gguf import GGUFFile

    try:
        arch = GGUFFile.read(path).metadata.get("general.architecture", "")
    except Exception:  # noqa: BLE001 - a note, never a failure
        return
    if arch == "hypernix":
        print("note: written as general.architecture=hypernix. HnxRun runs it; llama.cpp, "
              "patched or not, does not know that architecture. A hyperNix0x-v2 (Brewer) "
              "model is written as llama instead -- this one was not recognised as one.",
              file=sys.stderr)


def _run_convert_pipeline(ns: argparse.Namespace, source: str) -> int:
    """``convert -P``: safetensors -> GGUF -> hyprslug."""
    from hypernix.models.brewer_gguf import GGUFExportError
    from hypernix.quant.convertq import ConvertQuantizeError, convert_and_quantize, source_kind
    from hypernix.quant.hyprslug import HyprslugError
    from hypernix.timing.spinner import Spinner

    if not ns.quant:
        print("hypernix convert: -P needs a target, e.g. -Q Q4_K_M, -Q q6h4 or -Q FP8",
              file=sys.stderr)
        return 2
    kind = source_kind(source)
    steps = 1 if kind == "gguf" else 2

    def on_stage(name: str, what: str) -> None:
        number = 1 if name == "convert" or steps == 1 else 2
        verb = "converting" if name == "convert" else "quantising to"
        print(f"[{number}/{steps}] {verb} {what}", file=sys.stderr, flush=True)

    try:
        with Spinner(f"{'Quantising' if steps == 1 else 'Converting, then quantising'} to {ns.quant}",
                     style="grow"):
            result = convert_and_quantize(
                source, ns.quant, ns.output, dtype=ns.dtype, arch_name=ns.arch,
                name=None if ns.name == "HyperNix" else ns.name,
                keep_intermediate=ns.keep_intermediate, imatrix=ns.imatrix,
                on_stage=on_stage,
            )
    except (ConvertQuantizeError, HyprslugError, GGUFExportError, FileNotFoundError) as exc:
        print(f"hypernix convert: {exc}", file=sys.stderr)
        return 1
    print(result.report.describe())
    if result.intermediate:
        print(f"kept {result.intermediate}")
    print(result.output)
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
             "needed. Covers what llama-quantize cannot produce at all: the "
             "HyperNix sub-bit and codebook tiers (IQ0.5_XXXL, INT3, FP8...), "
             "Q8_K, and the hybrids q6h, q6h4 and q6h2.",
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
    p_run.add_argument(
        "--thermal-target", type=float, default=None, metavar="C",
        help="Keep the GPU at or below this temperature by pausing briefly "
             "between steps. This costs throughput -- it buys a cooler "
             "card, not a faster run -- and prints what it cost when the "
             "run ends. Changes no card settings; `hnx fusebox` can.",
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
            thermal_target_c=ns.thermal_target,
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

    # Python 3.15+ gets torch from the shim, not from pip's dependency
    # resolution (see hypernix.system.torch_shim): the first real command
    # that finds it missing installs it.
    if raw[0] not in ("doctor", "list-all", "errors", "wiki", "path"):
        from hypernix.system.torch_shim import auto as _torch_shim

        _torch_shim()

    # Top-level --auto-oven shortcut: translate to `oven --auto ...` so users
    # can run a one-liner `hypernix --auto-oven --prompt "def fib(n):"` and
    # get a working PyTorch model + a completion with zero extra ceremony.
    if raw[0] == "--auto-oven":
        return _run_oven(["--auto", *raw[1:]])

    # First arg isn't a subcommand.
    #
    # This used to print the whole forty-row menu and exit 1, without ever
    # mentioning what was typed. `hypernix downlod` produced fifty lines
    # of table, none of which said "downlod", so the failure looked
    # exactly like asking for help on purpose. With this many subcommands
    # -- and pairs like quantize/quantise and fizzle/fiz -- a suggestion
    # is worth more than the whole menu.
    if raw[0] not in _SUBCOMMANDS:
        return _unknown_command(raw[0])

    cmd, rest = raw[0], raw[1:]
    try:
        return _dispatch(cmd, rest)
    except ModuleNotFoundError as exc:
        # A subcommand that trains, converts or runs a PyTorch model on a
        # Python PyTorch has no wheel for (3.15, until it ships one): say
        # so, rather than a traceback ending in an import line.
        if (exc.name or "").split(".")[0] != "torch":
            raise
        print(_needs_torch(cmd), file=sys.stderr)
        return 2


def _needs_torch(cmd: str) -> str:
    version = ".".join(str(part) for part in sys.version_info[:2])
    return (f"hypernix {cmd} needs PyTorch, which is not installed for Python {version}.\n"
            f"  Install it (pip install torch), or use Python 3.12-3.14: PyTorch has no\n"
            f"  wheel for every new Python on its first day. Commands that do not run a\n"
            f"  model -- download, info, verify, hyprslug-headers, doctor, the T1 API and\n"
            f"  HyperLink -- work without it.")


def _dispatch(cmd: str, rest: list[str]) -> int:
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
    if cmd == "scriptgen":
        from hypernix.scriptgen.cli import main as _scriptgen_main

        return _scriptgen_main(rest)
    if cmd == "gather":
        from hypernix.data.gather_cli import main as _gather_main

        return _gather_main(rest)
    if cmd == "c4":
        from hypernix.c4.cli import cli_main as _c4_main

        return _c4_main(rest)
    if cmd in ("fusebox", "fuse-box"):
        from hypernix.system.fusebox_cli import main as _fusebox_main

        return _fusebox_main(rest)
    if cmd == "runtime":
        from hypernix.quant.runtime_bridge_cli import main as _runtime_main

        return _runtime_main(rest)
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
    if cmd in ("fetch-llama-quantize", "fllq"):
        return _run_fetch_llama_quantize(rest)
    if cmd == "auto":
        from hypernix.quant.auto import cli_main as _auto_main

        return _auto_main(rest)
    if cmd == "extra":
        _print_extra()
        return 0
    if cmd == "list-all":
        from hypernix.interfaces.list_all import cli_main as _list_all_main

        return _list_all_main(rest)
    if cmd == "tvtopmax":
        from hypernix.monitoring.tvtop_max import cli_main as _tvtop_max_main

        return _tvtop_max_main(rest)
    if cmd == "hypedpro":
        from hypernix.interfaces.hyped_pro_otui import cli_main as _hyped_pro_main

        return _hyped_pro_main(rest)
    if cmd == "image":
        from hypernix.models.hnx0xv2image_cli import cli_main as _image_main

        return _image_main(rest)
    if cmd == "claude":
        from hypernix.t1api.claudecode import cli_main as _claude_main

        return _claude_main(rest)
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
    if cmd == "neuron":
        return _run_neuron(rest)
    if cmd == "dilute":
        return _run_dilute(rest)
    if cmd == "elements":
        from hypernix.elements.elements_cli import cli_main as elements_main
        return elements_main(rest)
    if cmd == "errors":
        from hypernix.system.errors_cli import cli_main as errors_main
        return errors_main(rest)
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

def _run_neuron(raw: list[str]) -> int:
    """`hypernix neuron` — train small networks that act: RL, imitation."""
    from hypernix.neuron.cli import cli_main as neuron_main
    return neuron_main(raw)


def _run_dilute(raw: list[str]) -> int:
    """`hypernix dilute` — many answers per prompt, keep the best."""
    from hypernix.dilute.cli import cli_main as dilute_main
    return dilute_main(raw)

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


#: Short names for the pipeline's speech models; anything else is a Hub id.
_PIPELINE_ASR = {"nano-whisper": "openai/whisper-tiny", "whisper-tiny": "openai/whisper-tiny",
                 "whisper-base": "openai/whisper-base", "whisper-small": "openai/whisper-small"}
_PIPELINE_TTS = {"nano-tacotron": "facebook/mms-tts-eng", "mms": "facebook/mms-tts-eng",
                 "mms-tts-eng": "facebook/mms-tts-eng"}


def _run_pipeline(raw: list[str]) -> int:
    """`hypernix pipeline` — speech in, a real model's answer, speech out.

    It used to answer every recording with "this is a simulated
    response" and ignore --llm, --asr and --tts. The model is now the
    one --llm names, run through the same dispatch as `hyped` and
    `hypernix assistant`.
    """
    from hypernix.interfaces import hyped_pro_core as core
    from hypernix.interfaces.assistant import ChatLLM, _default_model_short
    from hypernix.models.workshop import ASRConfig, ASREngine, ASRToLLMToTTS, TTSConfig, TTSEngine

    p = argparse.ArgumentParser(
        prog="hypernix pipeline",
        description="Speech in, a model's answer, speech out: ASR -> LLM -> TTS.",
    )
    p.add_argument("--audio", "-a", help="Path to input audio file")
    p.add_argument("--active", action="store_true",
                   help="Record from the microphone (Ctrl+C stops and processes)")
    p.add_argument("--asr", default="nano-whisper",
                   help="Speech-to-text model: nano-whisper, whisper-base, whisper-small or a Hub id")
    p.add_argument("--llm", default=None,
                   help="Chat model short name (see `hyped --list`); default: the one "
                        "`hypernix config` set, else a cloud model you have a key for")
    p.add_argument("--tts", default="nano-tacotron", help="Text-to-speech model: mms or a Hub id")
    p.add_argument("--prompt", "-p", default="", help="System prompt for the LLM")
    p.add_argument("--output", "-o", help="Output audio file path (default pipeline_output.wav)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    ns = p.parse_args(raw)

    known = {m.short for m in core.MODELS}
    llm_name = ns.llm or _default_model_short()
    if not llm_name:
        print("hypernix pipeline: no chat model is set up. Pick one with --llm, e.g. "
              "--llm qwen3-4b-gguf, or set a default with `hypernix config`.", file=sys.stderr)
        return 2
    if llm_name not in known:
        print(f"hypernix pipeline: unknown model {llm_name!r}. Known: {', '.join(sorted(known))}",
              file=sys.stderr)
        return 2

    audio_path = ns.audio
    if not audio_path and ns.active:
        import tempfile

        from hypernix.interfaces.assistant import _record_microphone

        recorded = _record_microphone(Path(tempfile.mkstemp(suffix=".wav")[1]))
        if recorded is None:
            return 1
        audio_path = str(recorded)
    if not audio_path:
        audio_path = input("Path to an audio file: ").strip()
    if not Path(audio_path).is_file():
        print(f"hypernix pipeline: no audio file at {audio_path}", file=sys.stderr)
        return 1

    asr = ASREngine(ASRConfig(model_name=_PIPELINE_ASR.get(ns.asr, ns.asr)))
    tts = TTSEngine(TTSConfig(model_name=_PIPELINE_TTS.get(ns.tts, ns.tts)))
    system = ns.prompt or "You are a helpful assistant. Answer in a sentence or two: it will be spoken."
    pipeline = ASRToLLMToTTS(asr_engine=asr, llm=ChatLLM(llm_name, system=system),
                             tts_engine=tts, system_prompt=system)
    print(f"ASR {asr.config.model_name} -> {llm_name} -> TTS {tts.config.model_name}")
    try:
        response_text, audio_bytes = pipeline.process(audio_path, max_response_length=ns.max_new_tokens)
    except Exception as exc:  # noqa: BLE001 - reported, with its type
        print(f"hypernix pipeline: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    heard = pipeline.conversation_history[0]["content"] if pipeline.conversation_history else ""
    print(f"Heard:    {heard}")
    print(f"Response: {response_text}")
    output_path = ns.output or "pipeline_output.wav"
    Path(output_path).write_bytes(audio_bytes)
    print(f"Audio:    {output_path} ({tts.config.sample_rate} Hz)")
    return 0


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
