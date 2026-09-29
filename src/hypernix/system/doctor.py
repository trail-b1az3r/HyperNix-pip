"""Environment diagnostic for the hypernix package.

``hypernix doctor`` answers "will this machine run what I am about to
ask of it": the interpreter and the packages ``pyproject.toml``
requires, the GPU and the training preset it suits, the llama.cpp
builds (the stock ``llama-quantize`` and the patched one that runs the
HNX types), the models folder, and which optional extras are in.

A check is *mandatory* when a missing piece breaks the package itself;
those decide the exit code. Everything else is reported and never fails
the run -- a machine without a GPU, or without the ``t1api`` extra, is a
fine machine for the things it is used for.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path

from hypernix.quant.fetcher import cache_dir, cached_binary
from hypernix.quant.quantize import _detect_distro_id, _find_llama_quantize  # noqa: PLC2701
from hypernix.system import deps

#: The Python versions ``requires-python`` accepts, inclusive.
PYTHON_RANGE: tuple[tuple[int, int], tuple[int, int]] = ((3, 10), (3, 14))

#: ``pyproject.toml``'s ``dependencies`` without torch -- what
#: ``doctor --fix`` may install. ``torch`` is intentionally absent: see
#: ``deps.PROTECTED``. ``tests/system/test_doctor.py`` keeps the two
#: lists the same.
_RUNTIME_DEPS: tuple[str, ...] = (
    "numpy>=1.26,<3",
    "safetensors>=0.4.3",
    "huggingface-hub>=0.24",
    "gguf>=0.10.0",
    "tqdm>=4.66",
    "rich>14,<=15.0.0",
    "sentencepiece>=0.2.1",
)
#: The ``train`` extra, plus the tokenizers it pulls in: needed for HF
#: tokenizers and training, and installed by ``--fix`` as well.
_OPTIONAL_DEPS: tuple[str, ...] = (
    "tokenizers>=0.20",
    "transformers>=4.44",
    "accelerate>=0.33",
)

#: The modules behind each required dependency, by import name.
_REQUIRED_IMPORTS: tuple[str, ...] = (
    "numpy", "safetensors", "huggingface_hub", "gguf", "tqdm", "rich", "sentencepiece",
)

#: ``[project.optional-dependencies]`` extra -> the modules it provides.
#: Looked up with ``find_spec``, so nothing heavy is imported.
EXTRAS: dict[str, tuple[str, ...]] = {
    "llama-cpp": ("llama_cpp",),
    "train": ("transformers", "accelerate"),
    "t1api": ("fastapi", "uvicorn", "pydantic", "dotenv", "multipart", "cryptography"),
    "t1api-pg": ("psycopg",),
    "security": ("cryptography",),
    "elements": ("psutil",),
    "gui": ("PySide6",),
}


def _check_python() -> tuple[bool, str]:
    v = sys.version_info
    low, high = PYTHON_RANGE
    ok = low <= (v.major, v.minor) <= high
    # 3.12 is the main CI target, but every version in the range is
    # tested and there's no quality difference for users.
    span = f"{low[0]}.{low[1]}–{high[0]}.{high[1]}"
    return ok, f"python {v.major}.{v.minor}.{v.micro} ({'ok' if ok else 'expected ' + span})"


def _check_os() -> tuple[bool, str]:
    """OS check is informational — hypernix runs on Linux, macOS, and Windows."""
    uname = platform.uname()
    supported = {"Linux", "Darwin", "Windows"}
    ok = uname.system in supported
    extra = ""
    if uname.system == "Linux":
        distro = _detect_distro_id() or "unknown"
        extra = f" distro={distro}"
    return ok, f"{uname.system} {uname.release} ({uname.machine}){extra}"


def _check_import(mod: str, minver: str | None = None) -> tuple[bool, str]:
    try:
        m = importlib.import_module(mod)
    except Exception as exc:
        return False, f"{mod} import failed: {exc}"
    ver = getattr(m, "__version__", None)
    if ver is None:
        # Some libs (gguf) don't expose __version__; fall back to dist metadata.
        try:
            from importlib.metadata import PackageNotFoundError, version
            ver = version(mod)
        except (PackageNotFoundError, Exception):  # noqa: BLE001
            ver = "?"
    return True, f"{mod} {ver}"


def _check_torch_version() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:
        return False, f"torch import failed: {exc}"
    # Floor is 1.13 (the last 1.x release, for old-Intel-Mac support
    # via hypernix.torch_compat shims).  2.7+ is recommended because it
    # provides native nn.RMSNorm and the fused SDPA kernels; between
    # 1.13 and 2.7 the shim fills in the missing pieces, but
    # torch.compile is unavailable and FlashAttention is off.
    base = torch.__version__.split("+", 1)[0]
    parts = base.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return False, f"torch {torch.__version__} (unparseable version)"
    ok = (major, minor) >= (1, 13) and major < 3
    cuda = getattr(torch.version, "cuda", None)
    tag = f"cuda={cuda}" if cuda else "cpu"
    status = "ok" if ok else "expected >=1.13,<3"
    if ok and (major, minor) < (2, 7):
        status = (
            "ok (legacy; torch_compat shim active — recommend torch>=2.7 "
            "for native RMSNorm / fused SDPA)"
        )
    return ok, f"torch {torch.__version__} ({tag}) ({status})"


def _check_llama_quantize() -> tuple[bool, str]:
    try:
        # Don't trigger an auto-fetch inside `doctor`; just report what's
        # already resolvable.
        path = _find_llama_quantize(auto_fetch=False, quiet=True)
        return True, f"llama-quantize: {path}"
    except Exception:  # noqa: BLE001 - not finding it is the answer
        # Not a failure: `hypernix quantize` downloads a prebuilt binary
        # the first time it needs one.
        return False, ("not found -- `hypernix quantize` fetches one on first use, "
                       "or run `hypernix fetch-llama-quantize` now")


def _check_fetch_cache() -> tuple[bool, str]:
    cached = cached_binary()
    cdir = cache_dir()
    if cached is not None:
        return True, f"cache: {cached}"
    return True, f"cache: (empty) -> {cdir}"


def _check_tool(name: str) -> tuple[bool, str]:
    path = shutil.which(name)
    return (bool(path), f"{name}: {path or 'missing (optional)'}")


def _check_scripts_on_path() -> tuple[bool, str]:
    """Whether the console scripts directory is on PATH.

    Reported as optional rather than mandatory: everything still works via
    ``python -m hypernix``, so this is a usability gap, not a broken install.
    """
    try:
        from hypernix.system import pathfix
        scripts = pathfix.scripts_dir()
    except Exception as exc:  # noqa: BLE001 - never let a check crash doctor
        return False, f"could not determine scripts directory: {exc}"
    if pathfix.is_on_path(scripts):
        return True, str(scripts)
    if pathfix.in_isolated_env():
        return False, f"{scripts} not on PATH (virtualenv — activate it)"
    return False, f"{scripts} not on PATH — run `hypernix path --apply`"


def _check_gpu() -> tuple[bool, str]:
    """The GPU, its memory, and the brewer preset and optimizer that fit.

    The same table as the Model Training Guide. Always ``ok``: a CPU is
    a supported place to train, with the ``cpu-*`` presets.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return True, "unknown (torch not importable)"
    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            gib = props.total_memory / 2**30
            preset, optimizer = _preset_for(gib)
            many = f"{count}x " if count > 1 else ""
            tip = "; several GPUs: lazy_suzan" if count > 1 else ""
            return True, (f"{many}{props.name}, {gib:.0f} GB -> preset {preset}, "
                          f"optimizer {optimizer}{tip}")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return True, "Apple GPU (MPS, shares system memory) -> size the preset to free RAM"
    except Exception as exc:  # noqa: BLE001 - a driver fault is a report, not a crash
        return True, f"GPU query failed: {exc}"
    preset, optimizer = _preset_for(None)
    return True, f"none, CPU only -> preset {preset}, optimizer {optimizer}"


def _preset_for(gib: float | None) -> tuple[str, str]:
    """``(brewer preset, optimizer)`` for *gib* of VRAM; ``None`` is no GPU."""
    if gib is None:
        return "cpu-nano … cpu-small", "pressure_cooker_v6"
    if gib < 6:
        return "33m at a small batch (or cpu-*)", "pressure_cooker_v5s"
    if gib < 10:
        return "33m / micro", "pressure_cooker_v5s"
    if gib < 16:
        return "small", "pressure_cooker_v5"
    if gib < 40:
        return "medium", "pressure_cooker_v6"
    return "large", "pressure_cooker_v6"


def _check_patched_llamacpp() -> tuple[bool, str]:
    """The llama.cpp build that runs the HNX types, if there is one.

    Only the hyprslug HNX types need it; ordinary GGUFs run anywhere.
    """
    try:
        from hypernix.quant.runtime_bridge import find_build
        build = find_build()
    except Exception:  # noqa: BLE001
        return False, ("none -- needed only for the HNX types; "
                       "build one with native/ggml-hnx/build.sh")
    if build.patched:
        from hypernix.quant.ggufcheck import _type_name  # noqa: PLC2701
        from hypernix.quant.runtime_bridge import HNX_TYPE_SYMBOLS
        missing = build.missing(set(HNX_TYPE_SYMBOLS))
        if missing:
            names = ", ".join(_type_name(t) for t in missing)
            return False, (f"{build.bin_dir} was patched before {names} -- "
                           "run native/ggml-hnx/build.sh again")
        return True, f"patched, all {len(build.hnx_types)} HNX types: {build.bin_dir}"
    return False, (f"{build.bin_dir} is a stock build (no HNX types) -- "
                   "native/ggml-hnx/build.sh makes a patched one")


def _models_dir() -> Path:
    """Where models live, as ``config.get_models_dir`` resolves it, without
    creating the folder: a diagnostic should not change the machine."""
    env_dir = os.getenv("HYPERNIX_MODELS_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    from hypernix.system.config import _load_config  # noqa: PLC2701
    configured = _load_config().get("download_dir")
    return Path(configured).expanduser() if configured else Path.home() / ".hypernix" / "models"


def _check_models_dir() -> tuple[bool, str]:
    """How many models the folder holds, following symlinks, and any
    link whose target has gone (a moved model, an unmounted disk)."""
    from hypernix.system.linkwalk import broken_links, walk_files
    root = _models_dir()
    if not root.is_dir():
        return True, f"{root} (not created yet)"
    gguf = safetensors = 0
    for path in walk_files(root):
        name = path.name.lower()
        if name.endswith(".gguf"):
            gguf += 1
        elif name.endswith(".safetensors"):
            safetensors += 1
    summary = f"{root}: {gguf} GGUF, {safetensors} safetensors files"
    broken = broken_links(root)
    if broken:
        first, target = broken[0]
        more = f" (+{len(broken) - 1} more)" if len(broken) > 1 else ""
        return False, f"{summary}; broken link {first.name} -> {target}{more}"
    return True, summary


def _check_extras() -> tuple[bool, str]:
    """Which ``pip install 'hypernix[...]'`` extras are fully present."""
    have, missing = [], []
    for extra, modules in EXTRAS.items():
        try:
            present = all(importlib.util.find_spec(m) is not None for m in modules)
        except (ImportError, ValueError):
            present = False
        (have if present else missing).append(extra)
    return True, (f"installed: {', '.join(have) or 'none'}; "
                  f"not installed: {', '.join(missing) or 'none'}")


def run(*, fix: bool = False) -> int:
    """Run the environment check. If ``fix`` is True, pip-install missing deps.

    ``fix`` will NOT install or reinstall torch (see ``deps.PROTECTED``) —
    users pick their CUDA / CPU flavour manually.
    """
    if fix:
        print("[hypernix] doctor --fix: installing / upgrading runtime deps")
        deps.ensure(list(_RUNTIME_DEPS), upgrade=True)
        deps.ensure(list(_OPTIONAL_DEPS), upgrade=True)

    # Optional tool checks vary by OS — nice/ionice are POSIX-only.
    optional_tools: list[tuple[str, tuple[bool, str]]] = []
    if sys.platform != "win32":
        optional_tools = [
            ("nice (optional)", _check_tool("nice")),
            ("ionice (optional)", _check_tool("ionice")),
        ]

    checks: list[tuple[str, tuple[bool, str]]] = [
        ("OS", _check_os()),
        ("Python", _check_python()),
        ("torch", _check_torch_version()),
        *((mod, _check_import(mod)) for mod in _REQUIRED_IMPORTS),
        ("GPU", _check_gpu()),
        ("llama-quantize", _check_llama_quantize()),
        ("auto-fetch cache", _check_fetch_cache()),
        ("patched llama.cpp", _check_patched_llamacpp()),
        ("models folder", _check_models_dir()),
        ("extras", _check_extras()),
        ("console scripts on PATH", _check_scripts_on_path()),
        *optional_tools,
    ]

    mandatory = {"OS", "Python", "torch", *_REQUIRED_IMPORTS}
    all_ok = True
    for label, (ok, msg) in checks:
        icon = "[ok]" if ok else ("[--]" if label not in mandatory else "[!!]")
        print(f"  {icon} {label:<24} {msg}")
        if not ok and label in mandatory:
            all_ok = False

    print()
    print(f"hypernix executable: {shutil.which('hypernix') or 'not on PATH'}")
    print(f"working dir: {Path.cwd()}")

    # A PATH gap is not a *dependency* problem, so it never fails the check
    # — but --fix is exactly the "make my install work" request, so it is
    # repaired there rather than only described.
    if fix:
        from hypernix.system import pathfix
        scripts = pathfix.scripts_dir()
        if pathfix.is_on_path(scripts):
            pass
        elif pathfix.in_isolated_env():
            print()
            print("[hypernix] scripts are off PATH, but this is a virtualenv/conda env —")
            print("           not editing a shell profile. Activate the environment, or run:")
            print(f"             {pathfix.session_hint(scripts)}")
        else:
            print()
            print(f"[hypernix] doctor --fix: {pathfix.ensure_on_path(apply=True).message}")

    if not all_ok and not fix:
        print()
        print("tip: run `hypernix doctor --fix` to pip-install missing runtime deps")
        print("     (torch is never auto-installed; pick your own CUDA/CPU flavour)")
    return 0 if all_ok else 1
