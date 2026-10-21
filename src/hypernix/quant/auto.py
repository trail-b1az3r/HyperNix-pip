"""``hnx auto`` — a Hugging Face repo to the quantised GGUF you asked for.

::

    hnx auto HuggingFaceTB/SmolLM2-135M-Instruct              # Q4_K_M
    hnx auto Qwen/Qwen3-4B -q Q4_K_M -q Q8_0 -o ./qwen3
    hnx auto unsloth/Qwen3-4B-GGUF -q Q5_K_M                   # a GGUF repo
    hnx auto ./my-hf-folder -q q6h4                              # a HyperNix hybrid

What happens depends on the repo:

* **a repo of GGUFs that already has the quant** — that file is
  downloaded and nothing else is done (all parts of a split file);
* **a repo of GGUFs without it** — its highest-precision GGUF (F32,
  BF16, F16, else Q8_0) is downloaded and quantised to the quant, with a
  warning when that means quantising an already-quantised file;
* **anything else** (safetensors, ``pytorch_model.bin``, a Brewer
  model) — the snapshot is downloaded, converted to an F16 GGUF and
  quantised to each quant. A model llama.cpp knows is converted with
  llama.cpp's own converter, so the result runs in llama.cpp and LM
  Studio; a hyperNix0x-v2 model with HyperNix's exporter (written as
  ``llama``); anything else with HyperNix's converter, which only HnxRun
  runs, and says so.

Quantising is hyprslug's, so the llama.cpp k-quants and every HyperNix
tier and hybrid work as ``-q``; the llama.cpp types hyprslug does not
write (the IQ types, TQ, MXFP4) go to ``llama-quantize``, fetched the
way ``hnx fetch-llama-quantize`` fetches it.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["AutoError", "AutoResult", "auto", "cli_main"]

#: Highest precision first: what to requantise from when the quant is missing.
_BASES = ("F32", "BF16", "F16", "Q8_0")
_WEIGHTS = (".safetensors", ".bin", ".pt", ".pth")


class AutoError(RuntimeError):
    """Something ``hnx auto`` cannot do, and why."""


@dataclass
class AutoResult:
    source: str
    outputs: list[Path] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _repo_files(repo_id: str, revision: str | None, token: str | None) -> list[str]:
    from huggingface_hub import HfApi

    from ..models.download import explain_hub_error

    try:
        return list(HfApi().list_repo_files(repo_id, revision=revision, token=token))
    except Exception as exc:  # noqa: BLE001
        raise AutoError(explain_hub_error(exc, repo_id) or f"could not list {repo_id}: {exc}") from exc


def _fetch_file(repo_id: str, name: str, folder: Path, revision: str | None, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    from ..models.download import explain_hub_error

    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=name, revision=revision,
                                    local_dir=str(folder), token=token))
    except Exception as exc:  # noqa: BLE001
        raise AutoError(explain_hub_error(exc, repo_id) or f"could not download {name}: {exc}") from exc


def _label(source: str) -> str:
    return Path(source.rstrip("/")).name.split(":")[0] or "model"


def resolve(quant: str) -> tuple[str, str]:
    """``(engine, canonical)``: hyprslug's targets, else llama-quantize's types."""
    from .hyprslug import HyprslugError, resolve_target

    try:
        return "hyprslug", resolve_target(quant)[1]
    except HyprslugError:
        import gguf

        names = {n.removeprefix("MOSTLY_") for n in gguf.LlamaFileType.__members__} - {"GUESSED", "ALL_F32"}
        if quant.upper() in names:
            return "llama-quantize", quant.upper()
        raise


def _output(folder: Path, label: str, quant: str) -> Path:
    return folder / f"{label}.{resolve(quant)[1]}.gguf"


def _quantise(src: Path, quant: str, out: Path, say: Callable[[str], None]) -> Path:
    engine, canonical = resolve(quant)
    say(f"quantising {src.name} to {canonical} ({engine})")
    if engine == "hyprslug":
        from .hyprslug import quantize_gguf

        quantize_gguf(src, out, canonical)
    else:
        from .quantize import quantize_gguf as llama_quantize

        llama_quantize(src, out, canonical, backend="llama")
    return out


def _convert(model_dir: Path, staged: Path, label: str, converter: str,
             result: AutoResult, say: Callable[[str], None]) -> None:
    """An F16 GGUF from a downloaded snapshot, with the right converter."""
    from .hyprslug import is_brewer_source

    if is_brewer_source(model_dir):
        from ..models.brewer_gguf import export_gguf

        say(f"converting {label} (hyperNix0x-v2) to F16 GGUF")
        export_gguf(model_dir, staged, outtype="f16")
        result.steps.append("converted with HyperNix's Brewer exporter (llama)")
        return
    if converter in ("auto", "llama.cpp"):
        from .llamaconvert import LlamaConvertError, convert_hf

        try:
            say(f"converting {label} to F16 GGUF with llama.cpp's converter")
            convert_hf(model_dir, staged, outtype="f16", log=say)
            result.steps.append("converted with llama.cpp's convert_hf_to_gguf.py")
            return
        except LlamaConvertError as exc:
            if converter == "llama.cpp":
                raise AutoError(str(exc)) from exc
            result.notes.append(f"llama.cpp's converter could not convert it ({str(exc).splitlines()[-1]}); "
                                "HyperNix's converter was used, and only HnxRun runs the result.")
    from .convert import convert_to_gguf

    say(f"converting {label} to F16 GGUF with HyperNix's converter")
    convert_to_gguf(model_dir, staged, dtype="fp16", name=label)
    result.steps.append("converted with HyperNix's converter (general.architecture=hypernix)")


def auto(source: str, quants: list[str] | None = None, *, output: str | Path | None = None,
         revision: str | None = None, token: str | None = None, keep_f16: bool = False,
         keep_download: bool = False, converter: str = "auto",
         log: Callable[[str], None] | None = None) -> AutoResult:
    """Download *source*, convert it if it is not a GGUF, and quantise it to each of *quants*."""
    from ..models.hflinks import LinkError, parse_source, pick_gguf_files

    say = log or (lambda msg: print(f"[auto] {msg}", file=sys.stderr, flush=True))
    quants = list(quants or [])
    result = AutoResult(source)
    local = Path(source).expanduser()
    label = _label(source)

    if local.exists():
        repo_id, files = "", []
    else:
        try:
            parsed = parse_source(source)
        except LinkError as exc:
            raise AutoError(str(exc)) from exc
        if "/" not in parsed.repo_id:
            from ..models.download import resolve_repo_id

            parsed = type(parsed)(resolve_repo_id(parsed.repo_id), parsed.filename,
                                  parsed.revision, parsed.quant)
        repo_id = parsed.repo_id
        revision = revision or parsed.revision
        if parsed.quant:
            quants.append(parsed.quant)
        label = repo_id.split("/")[-1].removesuffix("-GGUF").removesuffix("-gguf")
        files = _repo_files(repo_id, revision, token)
    quants = quants or ["Q4_K_M"]
    for quant in quants:
        resolve(quant)                 # refuse a bad quant before downloading gigabytes

    folder = Path(output).expanduser() if output else _default_folder(label)
    folder.mkdir(parents=True, exist_ok=True)
    scratch = folder / ".hnx-auto"

    try:
        ggufs = [f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]
        has_weights = any(f.endswith(_WEIGHTS) for f in files)
        if local.exists():
            if local.suffix.lower() == ".gguf":
                for quant in quants:
                    result.outputs.append(_quantise(local, quant, _output(folder, label, quant), say))
                result.steps.append(f"quantised {local.name}")
                return result
            _from_snapshot(local, folder, scratch, label, quants, converter, keep_f16, result, say)
            return result

        if ggufs and not has_weights:
            base: Path | None = None
            for quant in quants:
                _, canonical = resolve(quant)
                ready = pick_gguf_files(ggufs, canonical) or pick_gguf_files(ggufs, quant)
                if ready:
                    say(f"{repo_id} already has {canonical}: downloading it")
                    paths = [_fetch_file(repo_id, name, folder, revision, token) for name in ready]
                    result.outputs.append(paths[0])
                    result.steps.append(f"downloaded {', '.join(ready)}")
                    continue
                if base is None:
                    base_name = next((pick_gguf_files(ggufs, b) for b in _BASES if pick_gguf_files(ggufs, b)), None)
                    if not base_name:
                        raise AutoError(f"{repo_id} has no {canonical} and no F32/BF16/F16/Q8_0 GGUF "
                                        f"to make one from. It has: {', '.join(sorted(ggufs))}")
                    if len(base_name) > 1:
                        raise AutoError(f"the source GGUF {base_name[0]} is split into parts; "
                                        "merge it first (llama-gguf-split --merge)")
                    say(f"{repo_id} has no {canonical}; downloading {base_name[0]} to quantise")
                    base = _fetch_file(repo_id, base_name[0], scratch, revision, token)
                    if "q8_0" in base.name.lower():
                        result.notes.append(f"{canonical} was made from Q8_0, not from full precision: "
                                            "it is a little worse than one quantised from F16.")
                result.outputs.append(_quantise(base, quant, _output(folder, label, quant), say))
                result.steps.append(f"quantised {base.name} to {canonical}")
            return result

        if not has_weights:
            raise AutoError(f"{repo_id} has neither weights nor GGUF files to work from")
        from ..models.download import DownloadError, download_model

        say(f"downloading {repo_id}")
        try:
            snapshot = download_model(repo_id, revision=revision, local_dir=str(scratch / "snapshot"),
                                      token=token, quiet=True)
        except (DownloadError, FileNotFoundError) as exc:
            raise AutoError(str(exc)) from exc
        result.steps.append(f"downloaded {repo_id}")
        _from_snapshot(snapshot, folder, scratch, label, quants, converter, keep_f16, result, say)
        return result
    finally:
        if scratch.exists() and not keep_download:
            shutil.rmtree(scratch, ignore_errors=True)


def _from_snapshot(model_dir: Path, folder: Path, scratch: Path, label: str, quants: list[str],
                   converter: str, keep_f16: bool, result: AutoResult, say: Callable[[str], None]) -> None:
    staged = (folder if keep_f16 else scratch) / f"{label}.f16.gguf"
    staged.parent.mkdir(parents=True, exist_ok=True)
    _convert(model_dir, staged, label, converter, result, say)
    for quant in quants:
        if resolve(quant)[1].upper() in ("F16", "FP16"):
            target = _output(folder, label, quant)
            shutil.copyfile(staged, target)
            result.outputs.append(target)
            continue
        result.outputs.append(_quantise(staged, quant, _output(folder, label, quant), say))
    if keep_f16:
        result.notes.append(f"kept {staged}")


def _default_folder(label: str) -> Path:
    try:
        from ..system.config import get_models_dir

        return get_models_dir() / f"{label}-GGUF"
    except Exception:  # noqa: BLE001
        return Path.home() / ".hypernix" / "models" / f"{label}-GGUF"


def cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hnx auto", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="repo id, Hub link, owner/repo:QUANT, short name, or a local folder/.gguf")
    p.add_argument("-q", "--quant", action="append", default=None,
                   help="target quant (repeatable): Q4_K_M, Q8_0, IQ4_XS, a HyperNix tier or hybrid. "
                        "Default Q4_K_M")
    p.add_argument("-o", "--output", default=None, help="folder for the GGUFs (default ~/.hypernix/models/NAME-GGUF)")
    p.add_argument("--revision", default=None)
    p.add_argument("--token", default=None)
    p.add_argument("--keep-f16", action="store_true", help="keep the F16 GGUF converted on the way")
    p.add_argument("--keep-download", action="store_true", help="keep the downloaded snapshot")
    p.add_argument("--converter", default="auto", choices=["auto", "llama.cpp", "hypernix"],
                   help="auto: llama.cpp's for models it knows, else HyperNix's")
    ns = p.parse_args(argv)
    try:
        result = auto(ns.source, ns.quant, output=ns.output, revision=ns.revision, token=ns.token,
                      keep_f16=ns.keep_f16, keep_download=ns.keep_download, converter=ns.converter)
    except Exception as exc:  # noqa: BLE001 - one line, not a traceback
        from .hyprslug import HyprslugError

        if not isinstance(exc, AutoError | HyprslugError | FileNotFoundError | ValueError):
            raise
        print(f"hnx auto: {exc}", file=sys.stderr)
        return 1
    for note in result.notes:
        print(f"note: {note}", file=sys.stderr)
    for path in result.outputs:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
