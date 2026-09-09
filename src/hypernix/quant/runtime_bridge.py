"""runtime_bridge — using the HyperNix llama.cpp from other applications.

    hnx runtime status
    hnx runtime serve  model-IQ0.5_XXXL.gguf
    hnx runtime path
    hnx runtime install --yes
    hnx runtime restore --yes

``native/ggml-hnx`` produces a llama.cpp that reads the sub-bit types.
This is how anything *else* gets to use it, and there are two routes
with very different risk.

Serving is the one to reach for
------------------------------
``serve`` starts the patched ``llama-server``, which speaks the
OpenAI-compatible API that LM Studio, Jan, Open WebUI, Continue, Cursor,
Zed and most of the rest already know how to talk to. Nothing on the
machine is modified: it is a process listening on a port, and closing it
puts everything back. It works with applications this has never heard
of, and it does not care what version of them you have.

Installing is the one that can break things
-------------------------------------------
``install`` copies the patched shared libraries over the ones LM Studio
bundles, so LM Studio loads a sub-bit model natively rather than through
a proxy. That is genuinely nicer when it works, and it is surgery on
somebody else's application:

* It is refused without ``--yes``.
* Everything it replaces is copied into
  ``~/.hypernix/runtime-bridge/backup`` first, with a manifest, and
  ``restore`` puts it back.
* It checks the target looks like an LM Studio runtime before writing,
  and refuses rather than guessing when it does not.
* LM Studio updates will overwrite it, and the ABI it expects can change
  between versions. When that happens the symptom is LM Studio failing
  to start a model, and ``restore`` is the fix.

LM Studio does not support this and will not help if it goes wrong. The
layout below is what LM Studio used at the time of writing; it is
detected rather than assumed, and this refuses a tree it does not
recognise instead of scattering libraries into it.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "BridgeError",
    "Build",
    "find_build",
    "candidate_build_dirs",
    "home",
    "backup_dir",
    "manifest_path",
    "lmstudio_runtime_dirs",
    "detect_targets",
    "serve_argv",
    "install",
    "restore",
    "status",
]


class BridgeError(RuntimeError):
    """Something is missing or does not look like what it claims."""


def home() -> Path:
    """Where this keeps backups and its manifest."""
    override = os.environ.get("HNX_RUNTIME_BRIDGE_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".hypernix" / "runtime-bridge"


def backup_dir() -> Path:
    return home() / "backup"


def manifest_path() -> Path:
    return home() / "installed.json"


# ---------------------------------------------------------------------------
# Finding the build
# ---------------------------------------------------------------------------

#: The libraries a patched build produces that another loader needs.
#: llama.cpp splits ggml into a base and a per-backend piece, and the
#: sub-bit decode lands in both -- the format half in ggml-base, the
#: arithmetic in ggml-cpu -- so replacing one and not the other gives a
#: loader that knows the type exists and cannot compute with it.
CORE_LIBRARIES = ("libggml-base", "libggml-cpu", "libggml", "libllama")


def _library_suffix() -> str:
    system = platform.system()
    if system == "Darwin":
        return ".dylib"
    if system == "Windows":
        return ".dll"
    return ".so"


@dataclass
class Build:
    """A built llama.cpp, and whether it is one of ours."""

    root: Path
    bin_dir: Path
    libraries: dict[str, Path] = field(default_factory=dict)
    server: Path | None = None
    cli: Path | None = None
    #: True when the build registers the HyperNix types. Checked by
    #: looking for the decoder's symbols, not by trusting the path.
    patched: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["root"] = str(self.root)
        payload["bin_dir"] = str(self.bin_dir)
        payload["libraries"] = {k: str(v) for k, v in self.libraries.items()}
        payload["server"] = str(self.server) if self.server else None
        payload["cli"] = str(self.cli) if self.cli else None
        return payload


def candidate_build_dirs() -> list[Path]:
    """Where a build might be, most specific first."""
    paths: list[Path] = []
    override = os.environ.get("HNX_LLAMA_BUILD")
    if override:
        paths.append(Path(override).expanduser())
    # The one build.sh makes.
    here = Path(__file__).resolve().parents[3]
    paths.append(here / "native" / "ggml-hnx" / "llama.cpp" / "build")
    paths.append(Path.home() / "llama.cpp" / "build")
    paths.append(Path.home() / ".hypernix" / "llama.cpp" / "build")
    return paths


def _looks_patched(library: Path) -> bool:
    """Whether a built library carries the HyperNix decoder.

    Read out of the binary rather than inferred from where it sits: a
    directory called llama.cpp next to this repository is not evidence
    that anyone ran the patcher on it, and installing an unpatched
    runtime over LM Studio's would swap one that cannot read sub-bit
    models for another that cannot, while looking like a fix.
    """
    try:
        blob = library.read_bytes()
    except OSError:
        return False
    # The shim's exported names. Present in the symbol table of any
    # build that registered the types, absent from a stock one.
    return b"hnx_ggml_to_float_iq0_5" in blob or b"IQ0.5_XXXL" in blob


def find_build(explicit: str | Path | None = None) -> Build:
    """Locate a built llama.cpp. Raises if there is not one."""
    roots = [Path(explicit).expanduser()] if explicit else candidate_build_dirs()

    tried: list[str] = []
    for root in roots:
        # Accept either the build directory or the checkout above it.
        for build_root in (root, root / "build"):
            bin_dir = build_root / "bin"
            if not bin_dir.is_dir():
                tried.append(str(bin_dir))
                continue
            suffix = _library_suffix()
            libraries = {}
            for stem in CORE_LIBRARIES:
                found = bin_dir / f"{stem}{suffix}"
                if found.is_file():
                    libraries[stem] = found
            if not libraries:
                tried.append(f"{bin_dir} (no libraries)")
                continue
            server = bin_dir / ("llama-server.exe" if suffix == ".dll"
                                else "llama-server")
            cli = bin_dir / ("llama-cli.exe" if suffix == ".dll" else "llama-cli")
            base = libraries.get("libggml-base") or next(iter(libraries.values()))
            build = Build(
                root=build_root.parent,
                bin_dir=bin_dir,
                libraries=libraries,
                server=server if server.is_file() else None,
                cli=cli if cli.is_file() else None,
                patched=_looks_patched(base),
            )
            if not build.patched:
                build.note = (
                    "This build does not carry the HyperNix decoder. It will "
                    "run ordinary GGUFs and refuse the sub-bit types. Run "
                    "native/ggml-hnx/build.sh to make a patched one."
                )
            return build

    raise BridgeError(
        "No built llama.cpp found. Looked in:\n  "
        + "\n  ".join(tried)
        + "\n\nBuild one with:\n  ./native/ggml-hnx/build.sh\n"
        "or point at an existing build with HNX_LLAMA_BUILD=/path/to/build"
    )


# ---------------------------------------------------------------------------
# Serving — the route that changes nothing
# ---------------------------------------------------------------------------

def serve_argv(build: Build, model: str | Path, *, host: str = "127.0.0.1",
               port: int = 8080, gpu_layers: int = 0,
               context: int = 0, alias: str = "") -> list[str]:
    """The command line that starts the patched server.

    Returned rather than run, so the caller can print it, and so the one
    place that builds it is the one place a test can check.
    """
    if build.server is None:
        raise BridgeError(
            f"{build.bin_dir} has no llama-server. Build it with "
            f"-DLLAMA_BUILD_TOOLS=ON, or use `hnx runtime path` and wire "
            f"the libraries in yourself."
        )
    model_path = Path(model).expanduser()
    if not model_path.is_file():
        raise BridgeError(f"No such model: {model_path}")

    argv = [
        str(build.server),
        "-m", str(model_path),
        "--host", host,
        "--port", str(int(port)),
    ]
    if gpu_layers:
        argv += ["-ngl", str(int(gpu_layers))]
    if context:
        argv += ["-c", str(int(context))]
    if alias:
        argv += ["--alias", alias]
    return argv


#: Applications that can be pointed at an OpenAI-compatible base URL,
#: and where the setting lives. Names and menu paths as they were at the
#: time of writing -- they move, so these are a hint rather than a
#: promise, and the base URL is the part that matters.
OPENAI_CLIENTS = (
    ("LM Studio", "Developer ▸ add a Remote/OpenAI-compatible provider"),
    ("Jan", "Settings ▸ Model Providers ▸ add an OpenAI-compatible provider"),
    ("Open WebUI", "Settings ▸ Connections ▸ OpenAI API"),
    ("Continue (VS Code)", 'config.json: "provider": "openai", "apiBase"'),
    ("Zed", "assistant settings ▸ openai ▸ api_url"),
    ("Cursor", "Settings ▸ Models ▸ Override OpenAI Base URL"),
    ("AnythingLLM", "LLM Preference ▸ Local AI / Generic OpenAI"),
)


# ---------------------------------------------------------------------------
# Installing — the route that modifies another application
# ---------------------------------------------------------------------------

def lmstudio_runtime_dirs() -> list[Path]:
    """Directories where LM Studio keeps a llama.cpp runtime.

    LMSTUDIO_HOME wins, so a non-standard install is not a dead end.
    Everything else is the layout LM Studio used at the time of writing,
    and is *searched* rather than assumed -- a directory only counts if
    it actually contains llama.cpp libraries.
    """
    override = os.environ.get("LMSTUDIO_HOME")
    bases: list[Path] = []
    if override:
        bases.append(Path(override).expanduser())
    else:
        home_dir = Path.home()
        bases += [
            home_dir / ".lmstudio",
            home_dir / ".cache" / "lm-studio",
            home_dir / "Library" / "Application Support" / "LM Studio",
            home_dir / "AppData" / "Roaming" / "LM Studio",
            home_dir / "AppData" / "Local" / "LM Studio",
        ]

    suffix = _library_suffix()
    found: list[Path] = []
    for base in bases:
        for sub in ("extensions/backends", "extensions", "runtimes"):
            root = base / sub
            if not root.is_dir():
                continue
            # Bounded walk: LM Studio nests runtimes one or two deep and
            # the model tree lives elsewhere, so there is no reason to
            # descend into a directory of gigabytes.
            for depth_root in (root, *[p for p in root.iterdir() if p.is_dir()]):
                try:
                    entries = list(depth_root.iterdir())
                except OSError:
                    continue
                names = {e.name for e in entries if e.is_file()}
                if any(f"libllama{suffix}" == n or n.startswith("libggml")
                       for n in names):
                    if depth_root not in found:
                        found.append(depth_root)
    return found


def detect_targets() -> list[dict[str, Any]]:
    """Applications on this machine that could take the runtime."""
    targets: list[dict[str, Any]] = []
    for directory in lmstudio_runtime_dirs():
        suffix = _library_suffix()
        present = sorted(
            p.name for p in directory.iterdir()
            if p.is_file() and p.name.endswith(suffix)
        )
        targets.append({
            "application": "LM Studio",
            "runtime_dir": str(directory),
            "libraries": present,
            "replaceable": [
                n for n in present
                if any(n.startswith(stem) for stem in CORE_LIBRARIES)
            ],
        })
    return targets


def _load_manifest() -> dict[str, Any]:
    try:
        return json.loads(manifest_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_manifest(data: dict[str, Any]) -> None:
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(manifest_path())


def install(build: Build, target_dir: str | Path, *, confirmed: bool = False,
            dry_run: bool = False) -> dict[str, Any]:
    """Put the patched libraries into *target_dir*, reversibly.

    Refuses without ``confirmed``. Everything replaced is copied to
    :func:`backup_dir` first and recorded, so :func:`restore` can undo it
    even after this process is gone.
    """
    target = Path(target_dir).expanduser()
    if not target.is_dir():
        raise BridgeError(f"{target} is not a directory")

    suffix = _library_suffix()
    existing = {
        p.name: p for p in target.iterdir()
        if p.is_file() and p.name.endswith(suffix)
    }
    # A directory with no llama.cpp libraries in it is not a runtime
    # directory, whatever its name says. Refusing beats scattering four
    # shared objects into somebody's Documents folder.
    if not any(name.startswith(stem)
               for name in existing for stem in CORE_LIBRARIES):
        raise BridgeError(
            f"{target} does not look like a llama.cpp runtime directory: "
            f"nothing in it is named like libllama or libggml. Refusing to "
            f"write into it. `hnx runtime status` lists what was detected."
        )

    if not build.patched:
        raise BridgeError(
            "Refusing to install a build that does not carry the HyperNix "
            "decoder -- it would replace a runtime that cannot read sub-bit "
            "models with another that cannot, and look like a fix. "
            + (build.note or "")
        )

    planned: list[dict[str, str]] = []
    for stem, source in sorted(build.libraries.items()):
        destination = target / f"{stem}{suffix}"
        planned.append({
            "library": f"{stem}{suffix}",
            "from": str(source),
            "to": str(destination),
            "replaces_existing": destination.is_file(),
        })

    if dry_run or not confirmed:
        return {
            "applied": False,
            "reason": ("dry run" if dry_run else
                       "not confirmed -- pass --yes to write"),
            "target": str(target),
            "planned": planned,
        }

    # A fresh directory, never an existing one. strftime has one-second
    # resolution, so two installs in the same second landed in the same
    # place and the second overwrote the first's copies -- with *our*
    # libraries, since that is what the target held by then. Restore
    # then put ours back and the originals were gone for good. Found by
    # a test that installed twice; the window is small and the loss is
    # total, which is the worst combination.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    saved_to = backup_dir() / stamp
    attempt = 1
    while saved_to.exists():
        attempt += 1
        saved_to = backup_dir() / f"{stamp}-{attempt}"
    saved_to.mkdir(parents=True)

    backed_up: list[dict[str, str]] = []
    written: list[str] = []
    for entry in planned:
        destination = Path(entry["to"])
        if destination.is_file():
            keep = saved_to / destination.name
            shutil.copy2(destination, keep)
            backed_up.append({"original": str(destination), "saved": str(keep)})
        # Copy to a temporary name and rename, so a runtime directory is
        # never left holding half a shared object -- LM Studio may be
        # running while this happens.
        scratch = destination.with_suffix(destination.suffix + ".hnx-new")
        shutil.copy2(entry["from"], scratch)
        scratch.replace(destination)
        written.append(str(destination))

    record = _load_manifest()
    installs = record.setdefault("installs", [])
    installs.append({
        "at": stamp,
        "target": str(target),
        "backup": str(saved_to),
        "written": written,
        "backed_up": backed_up,
        "from_build": str(build.bin_dir),
    })
    _save_manifest(record)

    return {
        "applied": True,
        "target": str(target),
        "written": written,
        "backup": str(saved_to),
        "backed_up": len(backed_up),
        "restore_with": "hnx runtime restore --yes",
    }


def restore(*, confirmed: bool = False, which: str = "") -> dict[str, Any]:
    """Put back what :func:`install` replaced.

    Reads the manifest rather than guessing, and restores newest first
    so a directory installed into twice ends up at its original
    contents rather than at the older copy.
    """
    record = _load_manifest()
    installs = list(record.get("installs") or [])
    if not installs:
        return {"restored": 0, "note": "nothing is recorded as installed"}

    chosen = [i for i in installs if not which or i.get("target") == which]
    if not chosen:
        return {
            "restored": 0,
            "note": f"nothing recorded for {which}",
            "known": sorted({i.get("target", "") for i in installs}),
        }

    if not confirmed:
        return {
            "restored": 0,
            "reason": "not confirmed -- pass --yes to write",
            "would_restore": [
                {"target": i.get("target"), "files": len(i.get("backed_up", []))}
                for i in reversed(chosen)
            ],
        }

    restored: list[str] = []
    removed: list[str] = []
    problems: list[str] = []
    for entry in reversed(chosen):                 # newest first
        for saved in entry.get("backed_up", []):
            source = Path(saved["saved"])
            destination = Path(saved["original"])
            if not source.is_file():
                problems.append(f"backup missing: {source}")
                continue
            try:
                shutil.copy2(source, destination)
                restored.append(str(destination))
            except OSError as exc:
                problems.append(f"{destination}: {exc}")
        # Files we added where nothing had been: putting the directory
        # back means removing them, not leaving ours behind.
        backed = {s["original"] for s in entry.get("backed_up", [])}
        for written in entry.get("written", []):
            if written in backed:
                continue
            path = Path(written)
            if path.is_file():
                try:
                    path.unlink()
                    removed.append(written)
                except OSError as exc:
                    problems.append(f"{path}: {exc}")

    remaining = [i for i in installs if i not in chosen]
    record["installs"] = remaining
    _save_manifest(record)

    return {
        "restored": len(restored),
        "removed": len(removed),
        "files": restored,
        "problems": problems,
    }


def status(explicit_build: str | Path | None = None) -> dict[str, Any]:
    """What is built, what is detected, and what is installed."""
    payload: dict[str, Any] = {}
    try:
        build = find_build(explicit_build)
        payload["build"] = build.to_dict()
    except BridgeError as exc:
        payload["build"] = None
        payload["build_error"] = str(exc)

    payload["targets"] = detect_targets()
    record = _load_manifest()
    payload["installs"] = record.get("installs") or []
    payload["bridge_home"] = str(home())
    return payload
