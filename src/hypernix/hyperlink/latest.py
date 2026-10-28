"""The newest HyperNix model, for ``hnx download --latest`` (0.72.8rc1).

HyperNix's own models are a short line of releases, published by two
accounts: ``ray0rf1re`` (HyperNix.3-mini, trained from scratch) and
``Compactbot``, which trains continuations for the model-requests board
(HyperNix.3.1-mini, a pretraining continuation of 3-mini). The newest is
the one with the highest version.

:data:`RELEASES` is the line as this version of HyperNix knows it.
:func:`discover` asks the Hub for later ones -- from those two accounts
only, and only names shaped like a release (``HyperNix.4-mini``,
``hypernix-4.2-mini``): anybody can publish a repo called
"hypernix-9-mini", and ``--latest`` must not download a stranger's
weights because of a name. Offline, the known list is the answer.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.parse
from dataclasses import asdict, dataclass

from hypernix.security.safeurl import urlopen as safe_urlopen

logger = logging.getLogger(__name__)

__all__ = ["RELEASES", "TRUSTED_OWNERS", "Release", "discover", "find", "latest"]

#: The accounts HyperNix models are published from.
TRUSTED_OWNERS = ("ray0rf1re", "Compactbot")
_NAME = re.compile(r"^hyper-?nix[.\-_](\d+(?:\.\d+)*)-(mini|nano|small|base|medium|large)$", re.IGNORECASE)


@dataclass(frozen=True)
class Release:
    version: str
    repo: str
    #: The id the runner and the catalogue know it by.
    model_id: str
    #: Its folder under the models directory.
    folder: str
    note: str = ""

    @property
    def key(self) -> tuple[int, ...]:
        return tuple(int(p) for p in self.version.split("."))

    def to_dict(self) -> dict:
        return {**asdict(self), "url": f"https://huggingface.co/{self.repo}"}


#: The line as of this HyperNix. A later release is found by discover().
RELEASES: tuple[Release, ...] = (
    Release("3", "ray0rf1re/HyperNix.3-mini", "hypernix.3-mini", "HyperNix.3-mini",
            "48.7M, trained from scratch on one GTX 1080; 512 tokens of context"),
    Release("3.1", "Compactbot/hypernix-3.1-mini", "hypernix.3.1-mini", "HyperNix.3.1-mini",
            "3-mini with 20,000 more steps of pretraining; same 48.7M architecture, 512 tokens"),
)


def _from_repo(repo: str) -> Release | None:
    owner, _, name = repo.partition("/")
    if owner not in TRUSTED_OWNERS:
        return None
    match = _NAME.match(name)
    if not match:
        return None
    version, size = match.group(1), match.group(2).lower()
    return Release(version, repo, f"hypernix.{version}-{size}", f"HyperNix.{version}-{size}")


def discover(timeout: float = 10.0) -> list[Release]:
    """Releases on the Hub from the trusted accounts, newest first. Empty when
    the Hub cannot be asked."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    if not endpoint.startswith("https://"):
        return []
    found: list[Release] = []
    for owner in TRUSTED_OWNERS:
        query = urllib.parse.urlencode({"author": owner, "search": "hypernix", "limit": 100})
        try:
            with safe_urlopen(f"{endpoint}/api/models?{query}", timeout=timeout) as response:
                listing = json.load(response)
        except (OSError, ValueError) as exc:
            logger.info("latest: could not list %s's models: %s", owner, exc)
            continue
        for entry in listing if isinstance(listing, list) else []:
            release = _from_repo(str(entry.get("id") or entry.get("modelId") or ""))
            if release is not None:
                found.append(release)
    return sorted(found, key=lambda r: r.key, reverse=True)


def latest(*, online: bool = True, timeout: float = 10.0) -> Release:
    """The newest release: the Hub's when it can be asked and has a later
    one, else the newest this HyperNix knows."""
    known = {r.repo.lower(): r for r in RELEASES}
    candidates = list(RELEASES)
    if online:
        for release in discover(timeout=timeout):
            # A known release keeps its own description and folder name.
            candidates.append(known.get(release.repo.lower(), release))
    return max(candidates, key=lambda r: r.key)


def find(model_id: str) -> Release | None:
    """The release *model_id* names: its id, repo or folder, any case; or
    ``latest`` / ``hypernix-latest`` for the newest one known here."""
    wanted = (model_id or "").strip().lower()
    if wanted in ("latest", "hypernix-latest", "hypernix:latest"):
        return max(RELEASES, key=lambda r: r.key)
    for release in RELEASES:
        if wanted in (release.model_id, release.repo.lower(), release.folder.lower()):
            return release
    return None
