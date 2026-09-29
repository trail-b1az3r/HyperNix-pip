"""release — what the current public release of ``hypernix`` is.

``hyped-pro`` shows the installed version next to the latest published one
so a person can see at a glance whether they're behind, without leaving the
TUI to check PyPI. Everything else that wants the same answer (``hypernix
info``, ``hypernix doctor``) can use this too rather than each growing its
own copy of the fetch-and-cache logic.

Three properties matter more than freshness here:

* **It never blocks the thing that called it.** A short timeout, and any
  failure returns "unknown" rather than raising. A banner is not worth a
  hung TUI on a machine with no network.
* **It never phones home on its own.** The only request is to PyPI's public
  JSON API for this one package, it carries no identifying information
  beyond a User-Agent, and ``HYPERNIX_NO_VERSION_CHECK`` (already used by
  the launcher) turns it off entirely.
* **It caches.** One lookup per ``CACHE_TTL_SECONDS`` per machine, in
  ``~/.hypernix/release-cache.json``, so opening the TUI ten times in an
  afternoon makes one request.

Pre-releases are reported separately from stable ones. ``0.71.5rc2`` is
newer than ``0.71.5b3`` but is not "the current public release" for someone
running a stable version, and telling them to upgrade to a release
candidate they didn't ask for would be wrong.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hypernix.security.safeurl import urlopen as safe_urlopen

__all__ = [
    "CACHE_TTL_SECONDS",
    "ReleaseInfo",
    "cache_path",
    "clear_cache",
    "compare",
    "current_release",
    "installed_version",
]

PYPI_JSON_URL = "https://pypi.org/pypi/hypernix/json"
CACHE_TTL_SECONDS = 6 * 3600
DEFAULT_TIMEOUT = 4.0

# Same variable the multi-version launcher already honours, so one opt-out
# covers both rather than two that people have to discover separately.
OPT_OUT_ENV_VARS = ("HYPERNIX_NO_VERSION_CHECK", "HYPERNIX_OFFLINE")


@dataclass(frozen=True)
class ReleaseInfo:
    """What we know about the installed vs. published versions.

    ``status`` is one of:

    ``current``      installed == latest stable
    ``outdated``     a newer stable release exists
    ``prerelease``   running something newer than the latest stable
    ``unknown``      no answer (offline, opted out, PyPI unreachable)
    """

    installed: str
    latest: str | None = None
    latest_prerelease: str | None = None
    status: str = "unknown"
    checked_at: float = 0.0
    from_cache: bool = False
    error: str | None = None
    releases: list[str] = field(default_factory=list)

    @property
    def update_available(self) -> bool:
        return self.status == "outdated"

    def summary(self) -> str:
        """One line, safe to put in a banner."""
        if self.status == "current":
            return f"v{self.installed} (latest)"
        if self.status == "outdated":
            return f"v{self.installed} — v{self.latest} available"
        if self.status == "prerelease":
            return f"v{self.installed} (pre-release; latest stable v{self.latest})"
        return f"v{self.installed}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "installed": self.installed,
            "latest": self.latest,
            "latest_prerelease": self.latest_prerelease,
            "status": self.status,
            "update_available": self.update_available,
            "summary": self.summary(),
            "checked_at": self.checked_at,
            "from_cache": self.from_cache,
            "error": self.error,
        }


def installed_version() -> str:
    from hypernix import __version__

    return __version__


def cache_path() -> Path:
    return Path.home() / ".hypernix" / "release-cache.json"


def clear_cache() -> None:
    try:
        cache_path().unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------


def _parse(version: str) -> tuple[Any, bool] | None:
    """(comparable, is_prerelease), or ``None`` if it can't be parsed.

    ``packaging`` is used when present — it's the only thing that gets PEP
    440 exactly right — and a numeric fallback covers a stripped-down
    install. The fallback treats anything with a non-numeric segment as a
    pre-release, which is the conservative reading: better to under-report
    an update than to tell someone a release candidate is a stable release.
    """
    try:
        from packaging.version import InvalidVersion, Version

        try:
            parsed = Version(version)
        except InvalidVersion:
            return None
        return parsed, parsed.is_prerelease
    except ImportError:
        pass

    parts = version.replace("-", ".").split(".")
    numbers: list[int] = []
    prerelease = False
    for part in parts:
        if part.isdigit():
            numbers.append(int(part))
        else:
            prerelease = True
            leading = ""
            for char in part:
                if char.isdigit():
                    leading += char
                else:
                    break
            numbers.append(int(leading) if leading else 0)
            break
    if not numbers:
        return None
    return tuple(numbers), prerelease


def compare(installed: str, latest_stable: str | None) -> str:
    """The ``status`` field: ``current`` / ``outdated`` / ``prerelease`` / ``unknown``."""
    if not latest_stable:
        return "unknown"
    here = _parse(installed)
    there = _parse(latest_stable)
    if here is None or there is None:
        return "unknown"
    here_v, here_pre = here
    there_v, _ = there
    try:
        if here_v > there_v:
            # Ahead of the latest stable: either a pre-release, or a local
            # build. Both mean "don't tell them to upgrade".
            return "prerelease" if here_pre else "current"
        if here_v == there_v:
            return "prerelease" if here_pre else "current"
        return "outdated"
    except TypeError:  # pragma: no cover - mixed fallback/packaging types
        return "unknown"


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------


def _opted_out() -> str | None:
    for var in OPT_OUT_ENV_VARS:
        if os.environ.get(var):
            return var
    return None


def _read_cache(ttl: float) -> dict[str, Any] | None:
    try:
        raw = json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    checked_at = raw.get("checked_at")
    if not isinstance(checked_at, (int, float)):
        return None
    if time.time() - checked_at > ttl:
        return None
    return raw


def _write_cache(payload: dict[str, Any]) -> None:
    path = cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass  # A read-only home costs a cache, not the answer.


def _fetch(timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        PYPI_JSON_URL,
        headers={"Accept": "application/json",
                 "User-Agent": f"hypernix/{installed_version()}"},
    )
    with safe_urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _split_versions(payload: dict[str, Any]) -> tuple[str | None, str | None, list[str]]:
    """(latest_stable, latest_prerelease, all_versions).

    ``info.version`` is PyPI's own "latest", which excludes pre-releases —
    but only when at least one stable release exists, so the ``releases``
    map is scanned too rather than trusting one field for both answers.
    """
    all_versions = [v for v in (payload.get("releases") or {}) if isinstance(v, str)]

    stable: list[tuple[Any, str]] = []
    prereleases: list[tuple[Any, str]] = []
    for version in all_versions:
        parsed = _parse(version)
        if parsed is None:
            continue
        comparable, is_pre = parsed
        (prereleases if is_pre else stable).append((comparable, version))

    def newest(items: list[tuple[Any, str]]) -> str | None:
        if not items:
            return None
        try:
            return max(items, key=lambda pair: pair[0])[1]
        except TypeError:  # pragma: no cover - unorderable mix
            return None

    latest_stable = newest(stable)
    if latest_stable is None:
        info_version = (payload.get("info") or {}).get("version")
        if isinstance(info_version, str) and _parse(info_version) and not _parse(info_version)[1]:
            latest_stable = info_version

    return latest_stable, newest(prereleases), sorted(all_versions)


def current_release(
    *,
    timeout: float = DEFAULT_TIMEOUT,
    ttl: float = CACHE_TTL_SECONDS,
    force: bool = False,
) -> ReleaseInfo:
    """The installed and latest-published versions. Never raises.

    ``force=True`` bypasses the cache but still honours the opt-out: a
    "check now" button should not become a way around ``--offline``.
    """
    installed = installed_version()

    opted_out = _opted_out()
    if opted_out:
        return ReleaseInfo(installed=installed, status="unknown",
                           error=f"version check disabled ({opted_out})")

    if not force:
        cached = _read_cache(ttl)
        if cached is not None:
            latest = cached.get("latest")
            return ReleaseInfo(
                installed=installed,
                latest=latest,
                latest_prerelease=cached.get("latest_prerelease"),
                status=compare(installed, latest),
                checked_at=float(cached.get("checked_at") or 0.0),
                from_cache=True,
                releases=list(cached.get("releases") or []),
            )

    try:
        payload = _fetch(timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return ReleaseInfo(installed=installed, status="unknown",
                           error=f"{type(exc).__name__}: {exc}")

    latest, latest_pre, all_versions = _split_versions(payload)
    now = time.time()
    _write_cache({
        "latest": latest,
        "latest_prerelease": latest_pre,
        "releases": all_versions[-40:],
        "checked_at": now,
    })
    return ReleaseInfo(
        installed=installed,
        latest=latest,
        latest_prerelease=latest_pre,
        status=compare(installed, latest),
        checked_at=now,
        releases=all_versions,
    )
