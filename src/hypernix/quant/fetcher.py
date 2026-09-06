"""Auto-fetch a prebuilt ``llama-quantize`` binary from the official
``ggml-org/llama.cpp`` GitHub releases so users who skipped the
``[llama-cpp]`` extra still get working k-quant support.

Downloaded binaries are cached under ``~/.cache/hypernix/bin/`` (or
``$HYPERNIX_CACHE_DIR``). The resolver in ``quantize.py`` automatically
adds this directory to its search path, so the next invocation finds
the binary without any further intervention.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
_LATEST_API = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases/latest"
_LIST_API = f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases"
_USER_AGENT = "hypernix/fetcher (+https://github.com/trail-b1az3r/hypernix-pip)"


def cache_dir() -> Path:
    """Directory where fetched binaries are stored. Respects XDG / override."""
    override = os.environ.get("HYPERNIX_CACHE_DIR")
    if override:
        return Path(override).expanduser() / "bin"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "hypernix" / "bin"


def cached_binary() -> Path | None:
    """Return the path to a cached ``llama-quantize`` if one exists."""
    names = ["llama-quantize", "quantize"]
    if sys.platform == "win32":
        names = [*names, *(f"{n}.exe" for n in names)]
    for name in names:
        candidate = cache_dir() / name
        if not candidate.exists():
            continue
        # On Windows the executable bit doesn't exist; existence is enough.
        if sys.platform == "win32" or os.access(candidate, os.X_OK):
            return candidate
    return None


def _detect_asset_tokens() -> tuple[str, list[str]]:
    """Return (os_tag, arch_tokens) used to match release asset filenames.

    Examples:
        linux x86_64 -> ("ubuntu", ["x64", "x86_64"])
        linux aarch64 -> ("ubuntu", ["arm64", "aarch64"])
    """
    system = platform.system().lower()
    if system != "linux":
        # We still try, but upstream only publishes ubuntu + macos + windows.
        os_tag = {"darwin": "macos", "windows": "win"}.get(system, system)
    else:
        os_tag = "ubuntu"
    m = platform.machine().lower()
    if m in {"x86_64", "amd64"}:
        arch_tokens = ["x64", "x86_64", "amd64"]
    elif m in {"aarch64", "arm64"}:
        arch_tokens = ["arm64", "aarch64"]
    else:
        arch_tokens = [m]
    return os_tag, arch_tokens


def _pick_asset(assets: Iterable[dict]) -> dict | None:
    """Pick the best CPU-only asset from a release payload for the current OS/arch."""
    os_tag, arch_tokens = _detect_asset_tokens()
    # On macOS we want the macos asset; everywhere else skip macOS-specific builds.
    exclude_common = ("cuda", "hip", "rocm", "vulkan", "sycl", "musa", "kompute", "cann")
    if os_tag == "macos":
        exclude = exclude_common
    elif os_tag == "win":
        exclude = (*exclude_common, "macos", "arm64-apple", "ubuntu")
    else:
        exclude = (*exclude_common, "macos", "arm64-apple")

    def score(name: str) -> int:
        lower = name.lower()
        if not lower.endswith(".zip"):
            return -1
        if os_tag not in lower:
            return -1
        if not any(tok in lower for tok in arch_tokens):
            return -1
        if any(tok in lower for tok in exclude):
            return -1
        # Prefer "bin-<os>-<arch>" flavour.
        s = 10
        if "bin" in lower:
            s += 5
        return s

    best: tuple[int, dict] | None = None
    for a in assets:
        name = a.get("name") or ""
        sc = score(name)
        if sc <= 0:
            continue
        if best is None or sc > best[0]:
            best = (sc, a)
    return best[1] if best else None


def _http_get(url: str, accept: str = "application/json") -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": accept})
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _latest_release() -> dict:
    raw = _http_get(_LATEST_API)
    return json.loads(raw)


def _recent_releases(limit: int = 10) -> list[dict]:
    """Return up to ``limit`` recent releases (newest first).

    Used as a fallback when the *latest* release doesn't ship a matching
    CPU-only asset for the current OS/arch — upstream occasionally skips a
    given tag's ubuntu binary, and walking back a handful of releases
    almost always finds one that does.
    """
    raw = _http_get(f"{_LIST_API}?per_page={limit}")
    return json.loads(raw)


def _download_to_temp(url: str) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/octet-stream"})
    fh, path = tempfile.mkstemp(prefix="hypernix-llama-", suffix=".zip")
    os.close(fh)
    dest = Path(path)
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    return dest


def _extract_binary(zip_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    # Capture Linux .so, macOS .dylib, AND Windows .dll co-shipped with
    # the binary so the cache is self-contained on every platform.
    libs_pattern = re.compile(
        r"(?:^|/)("
        r"lib(?:llama|ggml)[^/]*\.(?:so|dylib)(?:\.[0-9]+)*"
        r"|ggml[^/]*\.so(?:\.[0-9]+)*"
        r"|(?:llama|ggml)[^/]*\.dll"
        r"|msvcp[0-9]+\.dll|vcruntime[0-9]+\.dll"
        r")$",
        re.IGNORECASE,
    )
    extracted: Path | None = None

    with zipfile.ZipFile(zip_path) as zf:
        # Prefer llama-quantize over quantize if both exist.
        names = zf.namelist()
        match = [n for n in names if re.search(r"(?:^|/)llama-quantize(?:\.exe)?$", n)]
        if not match:
            match = [n for n in names if re.search(r"(?:^|/)quantize(?:\.exe)?$", n)]
        if not match:
            raise FileNotFoundError(
                f"No llama-quantize / quantize binary found inside {zip_path.name}"
            )
        bin_name = match[0]

        # Extract the binary + any co-shipped shared libs that it may need.
        with zf.open(bin_name) as src:
            dest_bin = target_dir / Path(bin_name).name
            with dest_bin.open("wb") as out:
                shutil.copyfileobj(src, out)
            if sys.platform != "win32":
                # chmod is a no-op on Windows; calling it only on POSIX
                # keeps the intent clear and avoids surprises in tests.
                dest_bin.chmod(
                    dest_bin.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                )
            extracted = dest_bin

        for n in names:
            if libs_pattern.search(n):
                with zf.open(n) as src:
                    dest = target_dir / Path(n).name
                    with dest.open("wb") as out:
                        shutil.copyfileobj(src, out)

    if extracted is None:
        raise RuntimeError("extraction did not produce a binary (unreachable)")
    return extracted


def fetch_llama_quantize(
    *,
    force: bool = False,
    quiet: bool = False,
    prefer_cached: bool = True,
    search_releases: int = 10,
) -> Path:
    """Download a prebuilt ``llama-quantize`` binary into the user cache.

    Tries the latest llama.cpp release first; if that release doesn't ship a
    CPU-only asset matching the current OS/arch, walks back through up to
    ``search_releases`` prior releases looking for one that does. This makes
    the fetch resilient to upstream occasionally skipping a binary on a given
    tag.

    Args:
        force: Ignore any cached binary and redownload.
        quiet: Suppress progress prints.
        prefer_cached: If a cached binary already exists, return it without
            hitting the network (unless ``force=True``).
        search_releases: How many recent releases to probe (newest first)
            when the latest tag has no matching asset. Set to 1 to disable.

    Returns:
        Path to the executable binary.
    """
    if prefer_cached and not force:
        cached = cached_binary()
        if cached is not None:
            return cached

    def log(msg: str) -> None:
        if not quiet:
            print(f"[hypernix] {msg}", file=sys.stderr)

    # Pull the list of candidate releases newest-first. Use /releases?per_page
    # rather than /releases/latest so a single API call covers both the
    # happy path and the walk-back fallback.
    try:
        candidates = _recent_releases(limit=max(1, search_releases))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach the GitHub API to find a llama.cpp release: {exc}. "
            "Provide --llama-quantize /path/to/llama-quantize, set LLAMA_QUANTIZE, "
            "or `pip install 'hypernix[llama-cpp]'`."
        ) from exc

    os_tag, arch_tokens = _detect_asset_tokens()
    tried: list[str] = []
    for release in candidates:
        tag = release.get("tag_name", "?")
        asset = _pick_asset(release.get("assets") or [])
        if asset is None:
            tried.append(tag)
            log(f"release {tag}: no CPU-only {os_tag}/{arch_tokens[0]} asset, trying older release")
            continue

        url = asset["browser_download_url"]
        size_mb = (asset.get("size") or 0) / (1024 * 1024)
        log(f"downloading llama.cpp {tag} asset: {asset['name']} ({size_mb:.1f} MB)")

        zip_path: Path | None = None
        try:
            zip_path = _download_to_temp(url)
            extracted = _extract_binary(zip_path, cache_dir())
        finally:
            if zip_path is not None:
                try:
                    zip_path.unlink()
                except OSError:
                    pass

        log(f"cached binary at {extracted}")
        return extracted

    raise RuntimeError(
        f"No CPU-only asset for os={os_tag} arch={arch_tokens!r} found in the "
        f"latest {len(candidates)} llama.cpp release(s) (checked: {', '.join(tried)}). "
        "Use --auto to try the PyPI fallback, install llama.cpp from your distro "
        "(e.g. `pacman -S llama.cpp`), or `pip install 'hypernix[llama-cpp]'`."
    )
