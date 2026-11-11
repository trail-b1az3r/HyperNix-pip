"""strife.kernels — Strife's native kernels: built here, verified, then used.

``native/strife_kernels.c`` decodes quantised blocks and multiplies by a
weight that is still quantised (``x @ W.T`` a row at a time), in C, with
SIMD and threads. It is compiled on the machine that will run it::

    hnx strife kernels build      # compile for this CPU, verify, record
    hnx strife kernels status
    hnx strife kernels verify     # check the built library again
    hnx strife kernels remove

Nothing is compiled behind the user's back: until ``build`` has been run
Strife computes with numpy, exactly as before. (``HNX_STRIFE_KERNELS=build``
opts in to building on first use; ``off`` ignores a built library.)

A kernel is used only for a type it was **verified** on. ``build`` and
``verify`` decode generated blocks of every type with the library and with
the Python decoders (:mod:`hypernix.quant.llamaquants`), and compare the
quantised matmul against the decoded one; the types that agree are written
to the build's manifest, and :func:`load` offers those and no others. A
compiler that miscompiles one type costs that type its kernel, not the
model its answers.

The library lives under ``~/.hypernix/strife/kernels/<key>/`` where the key
covers the source, the compiler, its flags and the CPU, so a new CPU or a
new HyperNix gets a new build instead of a stale one. The directory and
library are refused if anyone but their owner can write them: it is code
this process will run.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .errors import StrifeError

__all__ = ["Kernels", "load", "build", "verify", "status", "remove", "reset",
           "source_path", "kernels_dir", "ABI", "ENV"]

ABI = 1
ENV = "HNX_STRIFE_KERNELS"
_LIB_NAME = {"win32": "strife_kernels.dll", "darwin": "libstrife_kernels.dylib"}.get(
    sys.platform, "libstrife_kernels.so")
_MANIFEST = "manifest.json"
_COMPILE_TIMEOUT = 300.0

#: GGML type id -> (name, block elements, block bytes, byte ranges holding
#: float16 fields). The same table the C file has; :func:`verify` proves
#: the two agree.
_TYPES: dict[int, tuple[str, int, int, tuple[tuple[int, int], ...]]] = {
    0: ("F32", 32, 128, ()),
    1: ("F16", 32, 64, ()),
    30: ("BF16", 32, 64, ()),
    2: ("Q4_0", 32, 18, ((0, 2),)),
    3: ("Q4_1", 32, 20, ((0, 4),)),
    6: ("Q5_0", 32, 22, ((0, 2),)),
    7: ("Q5_1", 32, 24, ((0, 4),)),
    8: ("Q8_0", 32, 34, ((0, 2),)),
    10: ("Q2_K", 256, 84, ((80, 84),)),
    11: ("Q3_K", 256, 110, ((108, 110),)),
    12: ("Q4_K", 256, 144, ((0, 4),)),
    13: ("Q5_K", 256, 176, ((0, 4),)),
    14: ("Q6_K", 256, 210, ((208, 210),)),
}


def source_path() -> Path:
    return Path(__file__).with_name("native") / "strife_kernels.c"


def kernels_dir() -> Path:
    base = os.environ.get("HYPERNIX_HOME") or str(Path.home() / ".hypernix")
    return Path(base) / "strife" / "kernels"


# ---------------------------------------------------------------------------
# The loaded library
# ---------------------------------------------------------------------------

class Kernels:
    """A loaded, verified kernel library. ``types`` are the ones it may be used for."""

    def __init__(self, lib: ctypes.CDLL, path: Path, types: frozenset[int],
                 manifest: dict[str, Any]) -> None:
        self._lib = lib
        self.path = path
        self.types = types
        self.manifest = manifest
        lib.strife_abi.restype = ctypes.c_int
        lib.strife_openmp.restype = ctypes.c_int
        lib.strife_simd.restype = ctypes.c_char_p
        lib.strife_supported.argtypes = [ctypes.c_int]
        lib.strife_supported.restype = ctypes.c_int
        lib.strife_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64,
                                      ctypes.c_void_p, ctypes.c_int]
        lib.strife_decode.restype = ctypes.c_int
        lib.strife_matmul.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int64,
                                      ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
                                      ctypes.c_void_p, ctypes.c_int]
        lib.strife_matmul.restype = ctypes.c_int
        self.abi = int(lib.strife_abi())
        self.openmp_threads = int(lib.strife_openmp())
        self.simd = (lib.strife_simd() or b"").decode("ascii", "replace")

    def has(self, ggml_type: int) -> bool:
        return int(ggml_type) in self.types

    @staticmethod
    def _bytes_for(ggml_type: int, elements: int) -> int:
        if int(ggml_type) not in _TYPES:
            raise StrifeError("strife.kernels.type", f"no native kernel for type {ggml_type}",
                              details={"ggml_type": int(ggml_type)})
        _name, elems, size, _f16 = _TYPES[int(ggml_type)]
        if elems == 32 and int(ggml_type) in (0, 1, 30):
            return elements * (4 if int(ggml_type) == 0 else 2)
        return elements // elems * size

    def decode(self, raw: Any, ggml_type: int, elements: int, threads: int = 0) -> np.ndarray:
        """*elements* values of *ggml_type* from the buffer *raw*, as float32."""
        src = np.frombuffer(raw, dtype=np.uint8)
        if src.size < self._bytes_for(ggml_type, elements):
            raise StrifeError("strife.kernels.short", "the buffer is shorter than the tensor")
        out = np.empty(elements, dtype=np.float32)
        code = self._lib.strife_decode(src.ctypes.data, int(ggml_type), int(elements),
                                       out.ctypes.data, int(threads))
        if code:
            raise StrifeError("strife.kernels.failed", f"native decode failed (code {code})",
                              details={"ggml_type": int(ggml_type)})
        return out

    def matmul(self, raw: Any, ggml_type: int, rows: int, cols: int, x: np.ndarray,
               threads: int = 0) -> np.ndarray:
        """``x @ W.T`` for the quantised ``(rows, cols)`` weight in *raw*; ``x`` is ``(n, cols)``."""
        src = np.frombuffer(raw, dtype=np.uint8)
        if src.size < rows * self._bytes_for(ggml_type, cols):
            raise StrifeError("strife.kernels.short", "the buffer is shorter than the tensor")
        xs = np.ascontiguousarray(x, dtype=np.float32)
        if xs.ndim != 2 or xs.shape[1] != cols:
            raise StrifeError("strife.kernels.shape", f"x is {xs.shape}, the weight has {cols} columns")
        out = np.empty((xs.shape[0], rows), dtype=np.float32)
        code = self._lib.strife_matmul(src.ctypes.data, int(ggml_type), int(rows), int(cols),
                                       xs.ctypes.data, int(xs.shape[0]), out.ctypes.data,
                                       int(threads))
        if code:
            raise StrifeError("strife.kernels.failed", f"native matmul failed (code {code})",
                              details={"ggml_type": int(ggml_type)})
        return out

    def describe(self) -> dict[str, Any]:
        return {"path": str(self.path), "abi": self.abi, "simd": self.simd,
                "openmp_threads": self.openmp_threads,
                "types": [_TYPES[t][0] for t in sorted(self.types)],
                "compiler": self.manifest.get("compiler", ""),
                "built_at": self.manifest.get("built_at", 0)}


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def _compiler() -> str | None:
    wanted = os.environ.get("CC", "").strip()
    for name in ([wanted] if wanted else []) + ["cc", "gcc", "clang"]:
        found = shutil.which(name)
        if found:
            return found
    return None


def _compiler_version(cc: str) -> str:
    try:
        out = subprocess.run([cc, "--version"], capture_output=True, text=True, timeout=30,
                             check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (out.stdout or out.stderr).splitlines()[0].strip() if (out.stdout or out.stderr) else ""


def _cpu_id() -> str:
    """What ``-march=native`` depends on: the CPU's own description of itself."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith(("flags", "Features")):
                    return " ".join(sorted(line.split(":", 1)[1].split()))
    except OSError:
        pass
    return f"{platform.machine()} {platform.processor()}"


def _flag_sets() -> list[list[str]]:
    """Most capable first; :func:`build` takes the first set that compiles."""
    base = ["-O3", "-fPIC", "-shared", "-fvisibility=hidden"]
    native = ["-march=native"]
    return [base + native + ["-fopenmp"], base + native, base + ["-fopenmp"], base]


def _key(cc: str) -> str:
    digest = hashlib.sha256()
    digest.update(source_path().read_bytes())
    digest.update(_compiler_version(cc).encode())
    digest.update(_cpu_id().encode())
    digest.update(f"{sys.platform} {platform.machine()} abi{ABI}".encode())
    return digest.hexdigest()[:16]


def _check_owner(path: Path) -> None:
    if os.name != "posix":
        return
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise StrifeError("strife.kernels.unsafe",
                          f"{path} is writable by someone other than its owner, so it will "
                          "not be loaded as code",
                          remedy="`hnx strife kernels remove`, then `hnx strife kernels build`.")


def _current_dir() -> Path | None:
    """The build directory for this source, compiler and CPU, when there is one."""
    root = kernels_dir()
    if not root.is_dir():
        return None
    cc = _compiler()
    if cc is not None:
        candidate = root / _key(cc)
        if (candidate / _MANIFEST).is_file():
            return candidate
    # No compiler now (or a different one): a build whose manifest names this
    # exact source and CPU is still this machine's build.
    source = hashlib.sha256(source_path().read_bytes()).hexdigest()
    cpu = hashlib.sha256(_cpu_id().encode()).hexdigest()
    for manifest in sorted(root.glob(f"*/{_MANIFEST}")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("custom"):
            continue
        if data.get("source_sha256") == source and data.get("cpu_sha256") == cpu \
                and data.get("abi") == ABI:
            return manifest.parent
    return None


def _open(directory: Path) -> Kernels:
    manifest_path = directory / _MANIFEST
    library = directory / _LIB_NAME
    for path in (directory, library, manifest_path):
        _check_owner(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lib = ctypes.CDLL(str(library))
    except (OSError, ValueError) as exc:
        raise StrifeError("strife.kernels.unloadable", f"{library} could not be loaded: {exc}",
                          remedy="`hnx strife kernels build --force` rebuilds it.") from exc
    kernels = Kernels(lib, library, frozenset(), manifest)
    if kernels.abi != ABI:
        raise StrifeError("strife.kernels.abi", f"{library} is ABI {kernels.abi}, this Strife "
                          f"needs {ABI}", remedy="`hnx strife kernels build --force`.")
    verified = {int(t) for t in manifest.get("verified_types", []) if int(t) in _TYPES}
    kernels.types = frozenset(t for t in verified if lib.strife_supported(t))
    return kernels


def build(*, force: bool = False, compiler: str | None = None,
          flags: list[str] | None = None) -> Kernels:
    """Compile, verify and record the kernels for this machine.

    *flags* replaces the machine flags (``-march=native``, ``-fopenmp``) with
    the ones given, for building a specific code path -- plain C, AVX2 --
    to test it. Such a build is kept apart and is never the one
    :func:`load` picks.
    """
    cc = compiler or _compiler()
    if cc is None:
        raise StrifeError("strife.kernels.no_compiler", "no C compiler was found (cc, gcc, clang)",
                          remedy="Install one (e.g. `apt install gcc`), or set CC. Strife "
                                 "works without the kernels, on numpy.")
    source = source_path()
    if not source.is_file():
        raise StrifeError("strife.kernels.no_source", f"{source} is missing from this install")
    directory = kernels_dir() / _key(cc)
    if flags is not None:
        tag = hashlib.sha256(" ".join(flags).encode()).hexdigest()[:8]
        directory = kernels_dir() / f"custom-{_key(cc)}-{tag}"
    if not force and (directory / _MANIFEST).is_file() and (directory / _LIB_NAME).is_file():
        return _open(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(directory, 0o700)
    library = directory / _LIB_NAME
    scratch = directory / (_LIB_NAME + ".tmp")
    used: list[str] = []
    errors: list[str] = []
    base = ["-O3", "-fPIC", "-shared", "-fvisibility=hidden"]
    attempts = [base + list(flags)] if flags is not None else _flag_sets()
    for attempt in attempts:
        argv = [cc, *attempt, "-o", str(scratch), str(source)]
        try:
            done = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=_COMPILE_TIMEOUT, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"{' '.join(attempt)}: {exc}")
            continue
        if done.returncode == 0 and scratch.is_file():
            used = attempt
            break
        errors.append(f"{' '.join(attempt)}: {done.stderr.strip()[-300:]}")
    if not used:
        raise StrifeError("strife.kernels.compile_failed",
                          f"{cc} could not compile the kernels:\n" + "\n".join(errors[-2:]),
                          remedy="Strife keeps working on numpy. Try another compiler with CC=.")
    os.replace(scratch, library)
    if os.name == "posix":
        os.chmod(library, 0o700)
    manifest = {
        "abi": ABI, "compiler": _compiler_version(cc) or cc, "compiler_path": cc,
        "flags": used, "openmp": "-fopenmp" in used,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cpu_sha256": hashlib.sha256(_cpu_id().encode()).hexdigest(),
        "built_at": time.time(), "verified_types": [], "verification": {},
        "custom": directory.name.startswith("custom-"),
    }
    _write_manifest(directory, manifest)
    kernels = _open(directory)
    report = _verify(kernels)
    manifest["verified_types"] = sorted(t for t, r in report.items() if r["ok"])
    manifest["verification"] = {_TYPES[t][0]: r for t, r in report.items()}
    _write_manifest(directory, manifest)
    reset()
    return _open(directory)


def _write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    tmp = directory / (_MANIFEST + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(tmp, directory / _MANIFEST)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _blocks(ggml_type: int, count: int, rng: np.random.Generator) -> bytes:
    """*count* blocks with every payload bit pattern and finite float16 fields."""
    _name, _elems, size, f16 = _TYPES[ggml_type]
    if ggml_type == 0:
        return rng.standard_normal(count * 32).astype("<f4").tobytes()
    if ggml_type == 1:
        return rng.standard_normal(count * 32).astype("<f2").tobytes()
    if ggml_type == 30:
        values = rng.standard_normal(count * 32).astype("<f4")
        return (values.view("<u4") >> 16).astype("<u2").tobytes()
    raw = rng.integers(0, 256, size=(count, size), dtype=np.uint8)
    for start, stop in f16:
        fields = (stop - start) // 2
        scale = (rng.standard_normal((count, fields)) * 0.05).astype("<f2")
        scale[0] = 0.0                              # a zero scale is a real case
        raw[:, start:stop] = scale.view(np.uint8).reshape(count, stop - start)
    return raw.tobytes()


def _verify(kernels: Kernels, *, seed: int = 20261111) -> dict[int, dict[str, Any]]:
    """Each type: native decode against the Python decoder, native matmul against both."""
    from .quants import decode as reference_decode

    rng = np.random.default_rng(seed)
    report: dict[int, dict[str, Any]] = {}
    for ggml_type, (_name, elems, _size, _f16) in _TYPES.items():
        entry: dict[str, Any] = {"ok": False}
        report[ggml_type] = entry
        try:
            if not kernels._lib.strife_supported(ggml_type):
                entry["error"] = "the library has no kernel for it"
                continue
            rows, blocks_per_row = 301, 512 // elems + (3 if elems == 32 else 1)
            cols = blocks_per_row * elems
            raw = _blocks(ggml_type, rows * blocks_per_row, rng)
            want = reference_decode(raw, ggml_type)
            got = kernels.decode(raw, ggml_type, rows * cols)
            scale = float(np.max(np.abs(want))) or 1.0
            decode_err = float(np.max(np.abs(got - want))) / scale
            entry["decode_max_rel_error"] = decode_err
            if not np.isfinite(got).all() or decode_err > 2e-6:
                entry["error"] = "decoded values differ from the Python decoder"
                continue
            weight = want.reshape(rows, cols)
            worst = 0.0
            for n in (1, 3):
                x = rng.standard_normal((n, cols)).astype(np.float32)
                ref = x.astype(np.float64) @ weight.astype(np.float64).T
                out = kernels.matmul(raw, ggml_type, rows, cols, x)
                bound = float(np.max(np.abs(x).astype(np.float64) @ np.abs(weight).T)) or 1.0
                worst = max(worst, float(np.max(np.abs(out - ref))) / bound)
            entry["matmul_max_rel_error"] = worst
            if worst > 1e-5:
                entry["error"] = "the quantised matmul differs from the decoded one"
                continue
            entry["ok"] = True
        except Exception as exc:  # noqa: BLE001 - a failing kernel is reported, not raised
            entry["error"] = f"{type(exc).__name__}: {exc}"
    return report


def verify() -> dict[str, Any]:
    """Check the built library again, and update what it may be used for."""
    directory = _current_dir()
    if directory is None:
        raise StrifeError("strife.kernels.not_built", "the kernels have not been built here",
                          remedy="`hnx strife kernels build`.")
    kernels = _open(directory)
    report = _verify(kernels)
    manifest = dict(kernels.manifest)
    manifest["verified_types"] = sorted(t for t, r in report.items() if r["ok"])
    manifest["verification"] = {_TYPES[t][0]: r for t, r in report.items()}
    manifest["verified_at"] = time.time()
    _write_manifest(directory, manifest)
    reset()
    return {"ok": all(r["ok"] for r in report.values()), "types": manifest["verification"]}


# ---------------------------------------------------------------------------
# Finding the built library
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_loaded: Kernels | None = None
_load_tried = False
_load_note = ""


def reset() -> None:
    """Forget the loaded library (after a build, a removal, or in tests)."""
    global _loaded, _load_tried, _load_note
    with _lock:
        _loaded, _load_tried, _load_note = None, False, ""


def load() -> Kernels | None:
    """The verified kernels for this machine, or ``None`` (numpy is used).

    Never raises: a library that cannot be trusted or loaded is a reason in
    :func:`status`, and Strife carries on without it.
    """
    global _loaded, _load_tried, _load_note
    with _lock:
        if _load_tried:
            return _loaded
        _load_tried = True
        mode = os.environ.get(ENV, "").strip().lower()
        if mode in ("off", "0", "no", "numpy"):
            _load_note = f"{ENV}={mode}"
            return None
        try:
            directory = _current_dir()
            if directory is None and mode == "build":
                _loaded = build()
                return _loaded
            if directory is None:
                _load_note = "not built (hnx strife kernels build)"
                return None
            _loaded = _open(directory)
            if not _loaded.types:
                _load_note = "built, but no type passed verification"
                _loaded = None
        except StrifeError as exc:
            _load_note = f"{exc} [{exc.code}]"
            _loaded = None
        except Exception as exc:  # noqa: BLE001 - never let the fast path break the slow one
            _load_note = f"{type(exc).__name__}: {exc}"
            _loaded = None
        return _loaded


def status() -> dict[str, Any]:
    kernels = load()
    out: dict[str, Any] = {"active": kernels is not None, "directory": str(kernels_dir()),
                           "compiler": _compiler() or "", "source": str(source_path())}
    if kernels is not None:
        out.update(kernels.describe())
    else:
        out["reason"] = _load_note
    return out


def remove() -> int:
    """Delete every built kernel library. Returns how many builds were removed."""
    root = kernels_dir()
    count = 0
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and (child / _MANIFEST).exists():
                shutil.rmtree(child, ignore_errors=True)
                count += 1
    reset()
    return count
