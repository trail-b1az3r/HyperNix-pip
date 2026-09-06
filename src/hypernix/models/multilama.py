"""hypernix.multilama — one interface over several llama.cpp variants.

The llama.cpp ecosystem has forked several times, each fork trading
upstream compatibility for something specific: ``ik_llama.cpp`` for
newer/faster quant types and MoE throughput, ``PrismML-Eng/llama.cpp``
for custom sub-2-bit kernels some GGUFs need and no one else ships,
``koboldcpp`` for a more feature-complete server. A GGUF that won't load
on one of them often loads fine on another — this module's job is to
stop making that the caller's problem.

Every backend converges on one call::

    from hypernix.multilama import load
    llm = load(repo_id="Qwen/Qwen3-4B-GGUF", filename="Qwen3-4B-Q4_K_M.gguf")
    llm.chat([{"role": "user", "content": "hi"}])

which fork actually answered is an implementation detail, not something
the caller ever branches on — same shape whether it ran in-process
(``vanilla``) or against a local server subprocess (everything else).

Backends
--------
  vanilla   ggml-org/llama.cpp, via the ``llama-cpp-python`` PyPI package
            (in-process bindings — ``llama_cpp.Llama``). What
            ``pip install hypernix[llama-cpp]`` already gets you; also
            what ``hypernix.quantize`` reaches for as a last-resort
            quantizer. Standard GGUF quant types (Q2..Q8, IQ*).
  ik        ikawrakow/ik_llama.cpp — newer SOTA quant types (IQK,
            Trellis), fused MoE ops, generally better CPU/GPU hybrid
            offload throughput than vanilla for MoE models. No Python
            bindings; runs as its own ``llama-server`` binary, fetched
            and cached the same way ``hypernix.fetcher`` already fetches
            ``llama-quantize`` — then talked to over the OpenAI-compatible
            HTTP API every llama.cpp-derived server exposes. Diverged
            from upstream in August 2024, so newer upstream-only flags
            (e.g. ``-hf`` convenience downloading) aren't assumed to work
            here — this module always resolves the GGUF file itself via
            huggingface_hub first and hands every backend a plain ``-m
            <path>``, which has been stable across every fork and vintage.
  prismml   PrismML-Eng/llama.cpp — custom Q1_0_g128 1-bit hybrid-
            attention kernels needed for prism-ml/Bonsai-* GGUFs, absent
            from every other backend here (including vanilla — that's the
            whole reason this module exists rather than just calling
            ``llama_cpp.Llama`` directly everywhere). No known prebuilt
            releases as of this writing; ``ensure_binary()`` raises with
            the exact build-from-source command rather than guessing at a
            release asset that may not exist.
  kobold    LostRuins/koboldcpp — a different binary/release layout, but
            still an OpenAI-compatible ``/v1/chat/completions`` endpoint
            once running.

Server-binary backends are cached under
``~/.cache/hypernix/multilama/<backend>/`` (override with
``HYPERNIX_CACHE_DIR``), independent of ``hypernix.fetcher``'s own
``llama-quantize`` cache — different binary, different purpose, no reason
for a stale one to shadow the other.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HYPERNIX_VERSION = "0.71.4b6"
_USER_AGENT = f"hypernix/multilama {_HYPERNIX_VERSION} (+https://github.com/trail-b1az3r/hypernix-pip)"


class MultiLlamaError(RuntimeError):
    """A real, coded failure — see ``.code``. Never swallowed into a fake reply."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LlamaBackend:
    name: str
    label: str
    kind: str                       # "python-binding" | "server-binary"
    repo: str | None = None          # GitHub org/repo, for server-binary kinds
    branch: str | None = None        # specific branch needed, when not the default
    binary_names: tuple[str, ...] = ()
    docs_url: str = ""
    notes: str = ""
    has_prebuilt_releases: bool = True
    build_hint: str = ""


BACKENDS: dict[str, LlamaBackend] = {
    "vanilla": LlamaBackend(
        name="vanilla", label="llama.cpp (upstream, via llama-cpp-python)",
        kind="python-binding", repo="ggml-org/llama.cpp",
        docs_url="https://github.com/ggml-org/llama.cpp",
        notes="Standard GGUF quants (Q2_K..Q8_0, IQ*). "
              "Install: pip install 'hypernix[llama-cpp]'.",
    ),
    "ik": LlamaBackend(
        name="ik", label="ik_llama.cpp (ikawrakow) — SOTA quants, faster MoE offload",
        kind="server-binary", repo="ikawrakow/ik_llama.cpp",
        binary_names=("llama-server",),
        docs_url="https://github.com/ikawrakow/ik_llama.cpp",
        notes="IQK/Trellis quant types, fused MoE ops, generally better CPU/GPU "
              "hybrid offload than vanilla for MoE models. Diverged from upstream "
              "in Aug 2024 — no -hf flag assumed; this module downloads first.",
    ),
    "prismml": LlamaBackend(
        name="prismml", label="PrismML llama.cpp fork — custom 1-bit hybrid-attention kernels",
        kind="server-binary", repo="PrismML-Eng/llama.cpp",
        binary_names=("llama-server",),
        docs_url="https://github.com/PrismML-Eng/llama.cpp",
        notes="Needed for prism-ml/Bonsai-* Q1_0_g128 GGUFs — no other backend "
              "here (including vanilla) understands that quant type.",
        has_prebuilt_releases=False,
        build_hint=(
            "git clone https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp && "
            "cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-server"
        ),
    ),
    "kobold": LlamaBackend(
        name="kobold", label="KoboldCpp (LostRuins) — feature-rich llama.cpp-based server",
        kind="server-binary", repo="LostRuins/koboldcpp",
        binary_names=("koboldcpp",),
        docs_url="https://github.com/LostRuins/koboldcpp",
        notes="OpenAI-compatible endpoint at /v1 once running.",
    ),
    "nanbeige": LlamaBackend(
        name="nanbeige", label="Nanbeige llama.cpp fork — looped-transformer (general.architecture=nanbeige)",
        kind="server-binary", repo="Nanbeige/llama.cpp", branch="nanbeige42",
        binary_names=("llama-server",),
        docs_url="https://github.com/Nanbeige/llama.cpp/tree/nanbeige42",
        notes="Nanbeige4.x GGUFs (owao/Nanbeige4.2-3B-GGUF etc.) use a looped-transformer "
              "architecture not in upstream llama.cpp — 'general.architecture=nanbeige' isn't "
              "understood by vanilla, ik, or kobold, only this fork's nanbeige42 branch.",
        has_prebuilt_releases=False,
        build_hint=(
            "git clone -b nanbeige42 https://github.com/Nanbeige/llama.cpp.git && cd llama.cpp && "
            "cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j --target llama-server"
        ),
    ),
}


def get_backend(name: str) -> LlamaBackend:
    b = BACKENDS.get(name)
    if b is None:
        raise MultiLlamaError("ML-CFG-001", f"unknown backend {name!r}. Options: {', '.join(BACKENDS)}")
    return b


# ---------------------------------------------------------------------------
# Locating / fetching server-binary backends.
#
# Generalizes hypernix.fetcher's proven llama-quantize fetch (same asset
# scoring, same zip extraction shape) to an arbitrary repo and an arbitrary
# target binary name — fetcher.py itself stays scoped to llama-quantize,
# since hypernix.quantize's CLI surface (--auto, LLAMA_QUANTIZE, etc.)
# already documents that contract and other callers depend on it.
# ---------------------------------------------------------------------------


def _cache_root() -> Path:
    override = os.environ.get("HYPERNIX_CACHE_DIR")
    if override:
        base = Path(override).expanduser()
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "hypernix"
    return base / "multilama"


def _cache_dir_for(backend: LlamaBackend) -> Path:
    return _cache_root() / backend.name


def _cached_binary(backend: LlamaBackend) -> Path | None:
    names = list(backend.binary_names)
    if sys.platform == "win32":
        names = [*names, *(f"{n}.exe" for n in names)]
    d = _cache_dir_for(backend)
    for name in names:
        candidate = d / name
        if not candidate.exists():
            continue
        if sys.platform == "win32" or os.access(candidate, os.X_OK):
            return candidate
    return None


def _which_on_path(backend: LlamaBackend) -> Path | None:
    env_override = os.environ.get(f"HYPERNIX_MULTILAMA_{backend.name.upper()}_BIN")
    if env_override and Path(env_override).exists():
        return Path(env_override)
    for name in backend.binary_names:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def locate_binary(backend: LlamaBackend) -> Path | None:
    """An already-available binary for ``backend`` — env override, then
    $PATH, then our own fetch cache. Never triggers a download."""
    return _which_on_path(backend) or _cached_binary(backend)


def _detect_asset_tokens() -> tuple[str, list[str]]:
    import platform
    system = platform.system().lower()
    os_tag = {"darwin": "macos", "windows": "win"}.get(system, "ubuntu" if system == "linux" else system)
    m = platform.machine().lower()
    if m in {"x86_64", "amd64"}:
        arch_tokens = ["x64", "x86_64", "amd64"]
    elif m in {"aarch64", "arm64"}:
        arch_tokens = ["arm64", "aarch64"]
    else:
        arch_tokens = [m]
    return os_tag, arch_tokens


def _http_get(url: str, accept: str = "application/json") -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": accept})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _pick_asset(assets: list[dict], binary_names: tuple[str, ...]) -> dict | None:
    os_tag, arch_tokens = _detect_asset_tokens()
    exclude_common = ("cuda", "hip", "rocm", "vulkan", "sycl", "musa", "kompute", "cann")
    exclude = (*exclude_common, "macos", "arm64-apple") if os_tag != "macos" else exclude_common

    def score(name: str) -> int:
        lower = name.lower()
        if not lower.endswith(".zip"):
            return -1
        if os_tag not in lower or not any(t in lower for t in arch_tokens):
            return -1
        if any(t in lower for t in exclude):
            return -1
        return 15 if "bin" in lower else 10

    best: tuple[int, dict] | None = None
    for a in assets:
        sc = score(a.get("name") or "")
        if sc > 0 and (best is None or sc > best[0]):
            best = (sc, a)
    return best[1] if best else None


def _download_to_temp(url: str) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/octet-stream"})
    fh, path = tempfile.mkstemp(prefix="hypernix-multilama-", suffix=".zip")
    os.close(fh)
    dest = Path(path)
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    return dest


def _extract_binary(zip_path: Path, target_dir: Path, binary_names: tuple[str, ...]) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    libs_pattern = re.compile(
        r"(?:^|/)(lib(?:llama|ggml)[^/]*\.(?:so|dylib)(?:\.[0-9]+)*"
        r"|ggml[^/]*\.so(?:\.[0-9]+)*|(?:llama|ggml)[^/]*\.dll"
        r"|msvcp[0-9]+\.dll|vcruntime[0-9]+\.dll)$",
        re.IGNORECASE,
    )
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        match: list[str] = []
        for bn in binary_names:
            match = [n for n in names if re.search(rf"(?:^|/){re.escape(bn)}(?:\.exe)?$", n)]
            if match:
                break
        if not match:
            raise FileNotFoundError(f"none of {binary_names} found inside {zip_path.name}")
        bin_name = match[0]

        with zf.open(bin_name) as src:
            dest_bin = target_dir / Path(bin_name).name
            with dest_bin.open("wb") as out:
                shutil.copyfileobj(src, out)
            if sys.platform != "win32":
                dest_bin.chmod(dest_bin.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        for n in names:
            if libs_pattern.search(n):
                with zf.open(n) as src, (target_dir / Path(n).name).open("wb") as out:
                    shutil.copyfileobj(src, out)
    return dest_bin


def fetch_binary(backend: LlamaBackend, *, quiet: bool = False, search_releases: int = 10) -> Path:
    """Download+cache a prebuilt server binary for ``backend``.

    Raises MultiLlamaError (not a bare exception) with the exact
    build-from-source command when ``backend`` has no prebuilt releases
    (currently: prismml) rather than probing GitHub for an asset that's
    known not to exist.
    """
    if backend.kind != "server-binary":
        raise MultiLlamaError("ML-CFG-001", f"{backend.name} is a python-binding backend, nothing to fetch")
    cached = locate_binary(backend)
    if cached is not None:
        return cached
    if not backend.has_prebuilt_releases:
        raise MultiLlamaError(
            "ML-FETCH-001",
            f"{backend.label} has no known prebuilt release binaries. Build from source:\n"
            f"  {backend.build_hint}\n"
            f"then put the resulting binary on $PATH, or set "
            f"HYPERNIX_MULTILAMA_{backend.name.upper()}_BIN=/path/to/{backend.binary_names[0]}.",
        )

    def log(msg: str) -> None:
        if not quiet:
            print(f"[hypernix.multilama] {msg}", file=sys.stderr)

    api_base = f"https://api.github.com/repos/{backend.repo}"
    try:
        raw = _http_get(f"{api_base}/releases?per_page={max(1, search_releases)}")
        candidates = json.loads(raw)
    except urllib.error.URLError as exc:
        raise MultiLlamaError(
            "ML-FETCH-002",
            f"could not reach the GitHub API for {backend.repo}: {exc}. "
            f"Set HYPERNIX_MULTILAMA_{backend.name.upper()}_BIN to an existing binary instead.",
        ) from exc

    os_tag, arch_tokens = _detect_asset_tokens()
    tried: list[str] = []
    for release in candidates:
        tag = release.get("tag_name", "?")
        asset = _pick_asset(release.get("assets") or [], backend.binary_names)
        if asset is None:
            tried.append(tag)
            log(f"{backend.repo} release {tag}: no {os_tag}/{arch_tokens[0]} asset, trying older release")
            continue
        url = asset["browser_download_url"]
        size_mb = (asset.get("size") or 0) / (1024 * 1024)
        log(f"downloading {backend.repo} {tag} asset: {asset['name']} ({size_mb:.1f} MB)")
        zip_path: Path | None = None
        try:
            zip_path = _download_to_temp(url)
            extracted = _extract_binary(zip_path, _cache_dir_for(backend), backend.binary_names)
        except (FileNotFoundError, zipfile.BadZipFile) as exc:
            tried.append(tag)
            log(f"{backend.repo} release {tag}: asset didn't contain a usable binary ({exc}), trying older release")
            continue
        finally:
            if zip_path is not None:
                zip_path.unlink(missing_ok=True)
        log(f"cached binary at {extracted}")
        return extracted

    raise MultiLlamaError(
        "ML-FETCH-001",
        f"no {os_tag}/{arch_tokens!r} asset found in the latest {len(candidates)} "
        f"{backend.repo} release(s) (checked: {', '.join(tried)}). "
        f"Build from source or set HYPERNIX_MULTILAMA_{backend.name.upper()}_BIN.",
    )


def ensure_binary(backend: LlamaBackend, *, quiet: bool = False) -> Path:
    """locate_binary(), falling back to fetch_binary() if nothing's found."""
    found = locate_binary(backend)
    return found if found is not None else fetch_binary(backend, quiet=quiet)


# ---------------------------------------------------------------------------
# GGUF file resolution — shared by every backend. Always resolves to a real
# local path via huggingface_hub before any backend is launched, rather than
# relying on each fork's own (version-dependent) -hf convenience flag.
# ---------------------------------------------------------------------------


def resolve_gguf(repo_id: str, filename: str, *, quiet: bool = False) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise MultiLlamaError(
            "ML-DEPS-001",
            f"'huggingface_hub' is not installed for this interpreter ({sys.executable}). "
            f"Install with: {sys.executable} -m pip install huggingface_hub --break-system-packages.",
        ) from exc
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
        return Path(path)
    except Exception:  # noqa: BLE001 — not cached, fall through to a real download
        pass
    if not quiet:
        print(f"[hypernix.multilama] fetching {filename} from {repo_id} ...", file=sys.stderr)
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except Exception as exc:  # noqa: BLE001
        raise MultiLlamaError("ML-FETCH-003", f"could not download {filename!r} from {repo_id!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Server process management (ik / prismml / kobold) — one subprocess per
# (backend, gguf path, n_gpu_layers, n_ctx) combination, reused across
# chat() calls instead of relaunching per turn.
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class _ServerHandle:
    proc: subprocess.Popen
    port: int


_RUNNING_SERVERS: dict[tuple, _ServerHandle] = {}


def _wait_for_health(port: int, proc: subprocess.Popen, timeout: float = 120.0) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise MultiLlamaError(
                "ML-SERVER-001",
                f"server process exited (code={proc.returncode}) before becoming healthy — "
                f"see its output above on this terminal.",
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    raise MultiLlamaError("ML-SERVER-002", f"server did not become healthy within {timeout:.0f}s")


def _ensure_server(
    backend: LlamaBackend,
    gguf_path: Path,
    n_gpu_layers: int,
    n_ctx: int,
    quiet: bool = False,
) -> _ServerHandle:
    key = (backend.name, str(gguf_path), n_gpu_layers, n_ctx)
    existing = _RUNNING_SERVERS.get(key)
    if existing is not None and existing.proc.poll() is None:
        return existing

    binary = ensure_binary(backend, quiet=quiet)
    port = _free_port()
    cmd = [
        str(binary), "-m", str(gguf_path),
        "--host", "127.0.0.1", "--port", str(port),
        "-ngl", str(n_gpu_layers), "-c", str(n_ctx),
    ]
    if not quiet:
        print(f"[hypernix.multilama] launching {backend.label}: {' '.join(cmd)}", file=sys.stderr)
    try:
        # stderr/stdout inherited: the server's own real logs (including any
        # load errors) print straight to this terminal, matching every other
        # real-backend dispatch in this codebase — never swallowed.
        proc = subprocess.Popen(cmd, stdout=sys.stderr, stderr=sys.stderr)
    except FileNotFoundError as exc:
        raise MultiLlamaError("ML-SERVER-001", f"could not launch {binary}: {exc}") from exc
    _wait_for_health(port, proc)
    handle = _ServerHandle(proc=proc, port=port)
    _RUNNING_SERVERS[key] = handle
    return handle


def shutdown_all_servers() -> None:
    """Terminate every server subprocess this module has launched."""
    for handle in _RUNNING_SERVERS.values():
        if handle.proc.poll() is None:
            handle.proc.terminate()
    _RUNNING_SERVERS.clear()


atexit.register(shutdown_all_servers)


def _post_chat_completion(
    port: int, messages: list[dict[str, str]], max_tokens: int, temperature: float,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise MultiLlamaError("ML-SERVER-003", f"HTTP {exc.code} from local server: {exc.read().decode(errors='replace')[:400]}") from exc
    except urllib.error.URLError as exc:
        raise MultiLlamaError("ML-SERVER-003", f"could not reach local server on port {port}: {exc.reason}") from exc
    try:
        return data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MultiLlamaError("ML-SERVER-004", f"unexpected response shape from local server: {data!r}"[:400]) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MultiLlama:
    """A loaded GGUF model, backed by whichever llama.cpp variant answered
    ``load()``. Construct via :func:`load`, not directly."""

    def __init__(self, backend: LlamaBackend, repo_id: str, filename: str,
                 gguf_path: Path, n_gpu_layers: int, n_ctx: int,
                 _python_llm: Any = None):
        self.backend = backend
        self.repo_id = repo_id
        self.filename = filename
        self.gguf_path = gguf_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._python_llm = _python_llm  # only set for the vanilla/python-binding backend

    def chat_message(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Like :meth:`chat`, but returns the raw OpenAI-style message dict
        (``{"role": ..., "content": ..., "tool_calls": [...]}``) instead of
        just the text — needed by callers running their own tool-call loop.
        Tool-calling support depends on the GGUF's chat template actually
        having one; a model/template without it just never populates
        ``tool_calls``, same as any OpenAI-compatible server.
        """
        full_messages = list(messages)
        if system:
            full_messages = [{"role": "system", "content": system}] + full_messages

        if self.backend.kind == "python-binding":
            try:
                kwargs: dict[str, Any] = {"messages": full_messages, "max_tokens": max_tokens, "temperature": temperature}
                if tools:
                    kwargs["tools"] = tools
                resp = self._python_llm.create_chat_completion(**kwargs)
                return resp["choices"][0]["message"]
            except MultiLlamaError:
                raise
            except (KeyError, IndexError, TypeError) as exc:
                raise MultiLlamaError("ML-CHAT-001", f"unexpected response shape: {exc}") from exc
            except Exception as exc:  # noqa: BLE001
                raise MultiLlamaError("ML-CHAT-002", f"inference failed ({self.backend.name}): {exc}") from exc

        handle = _ensure_server(self.backend, self.gguf_path, self.n_gpu_layers, self.n_ctx)
        return _post_chat_completion(handle.port, full_messages, max_tokens, temperature, tools=tools)

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        msg = self.chat_message(messages, system=system, max_tokens=max_tokens, temperature=temperature)
        return msg.get("content") or ""

    def close(self) -> None:
        for key, handle in list(_RUNNING_SERVERS.items()):
            if key[0] == self.backend.name and key[1] == str(self.gguf_path):
                if handle.proc.poll() is None:
                    handle.proc.terminate()
                del _RUNNING_SERVERS[key]


def load(
    repo_id: str,
    filename: str,
    backend: str = "vanilla",
    n_gpu_layers: int = -1,
    n_ctx: int = 8192,
    quiet: bool = False,
) -> MultiLlama:
    """Load a GGUF model through ``backend`` (see :data:`BACKENDS`).

    Always resolves the GGUF file to a real local path first (see
    :func:`resolve_gguf`) — every backend, including the vanilla
    in-process one, is handed a path rather than a bare repo id, so
    behavior doesn't depend on any one fork's HF-downloading flag.
    """
    be = get_backend(backend)
    gguf_path = resolve_gguf(repo_id, filename, quiet=quiet)

    if be.kind == "python-binding":
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise MultiLlamaError(
                "ML-DEPS-001",
                f"'llama_cpp' is not installed for this interpreter ({sys.executable}). "
                f"Install with: {sys.executable} -m pip install llama-cpp-python --break-system-packages.",
            ) from exc
        try:
            llm = Llama(model_path=str(gguf_path), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)
        except Exception as exc:  # noqa: BLE001
            raise MultiLlamaError("ML-LOAD-001", f"vanilla llama.cpp failed to load {gguf_path}: {exc}") from exc
        return MultiLlama(be, repo_id, filename, gguf_path, n_gpu_layers, n_ctx, _python_llm=llm)

    # server-binary backends: fetch/build the binary now so load() fails
    # fast with a clear error rather than lazily on the first chat() call.
    ensure_binary(be, quiet=quiet)
    return MultiLlama(be, repo_id, filename, gguf_path, n_gpu_layers, n_ctx)


def auto_select_backend(repo_id: str, filename: str) -> str:
    """Best-effort default backend for a given GGUF, based on naming
    conventions already in use across the ecosystem. Callers with better
    information (a known quant type, a specific fork requirement) should
    just pass ``backend=`` to :func:`load` directly instead of relying on
    this — it's a starting guess, not a resolver.
    """
    hint = f"{repo_id} {filename}".lower()
    if "bonsai" in hint or "prism-ml" in hint or "q1_0_g128" in hint:
        return "prismml"
    if "nanbeige" in hint:
        return "nanbeige"
    if "iqk" in hint or "trellis" in hint:
        return "ik"
    return "vanilla"


def status_json() -> dict[str, Any]:
    """Serializable availability report for every registered backend —
    what a `doctor`/`list`-style command shows."""
    out = {}
    for name, be in BACKENDS.items():
        if be.kind == "python-binding":
            try:
                import llama_cpp  # noqa: F401
                available = True
            except ImportError:
                available = False
            location = "in-process (llama_cpp)" if available else None
        else:
            found = locate_binary(be)
            available = found is not None
            location = str(found) if found else None
        out[name] = {
            "label": be.label, "kind": be.kind, "available": available,
            "location": location, "has_prebuilt_releases": be.has_prebuilt_releases,
            "notes": be.notes, "docs_url": be.docs_url,
        }
    return out


# ---------------------------------------------------------------------------
# CLI: python -m hypernix.multilama {list,chat} ...
# ---------------------------------------------------------------------------


def _cli_list() -> int:
    for name, info in status_json().items():
        mark = "✓" if info["available"] else "✗"
        loc = f" -> {info['location']}" if info["location"] else ""
        print(f"  [{mark}] {name:10s} {info['label']}{loc}")
        if not info["available"]:
            print(f"            {info['notes']}")
    return 0


def _cli_chat(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m hypernix.multilama chat")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--filename", required=True)
    p.add_argument("--backend", default="vanilla", choices=list(BACKENDS))
    p.add_argument("--message", required=True)
    p.add_argument("--n-gpu-layers", type=int, default=-1)
    p.add_argument("--n-ctx", type=int, default=8192)
    ns = p.parse_args(argv)
    try:
        llm = load(ns.repo_id, ns.filename, backend=ns.backend, n_gpu_layers=ns.n_gpu_layers, n_ctx=ns.n_ctx)
        reply = llm.chat([{"role": "user", "content": ns.message}])
        print(reply)
        return 0
    except MultiLlamaError as exc:
        print(f"[{exc.code}] {exc.message}", file=sys.stderr)
        return 1


def cli_main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: python -m hypernix.multilama {list|chat} ...")
        return 0
    cmd, *rest = argv
    if cmd == "list":
        return _cli_list()
    if cmd == "chat":
        return _cli_chat(rest)
    print(f"unknown command {cmd!r}. Use: list, chat", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(cli_main())
