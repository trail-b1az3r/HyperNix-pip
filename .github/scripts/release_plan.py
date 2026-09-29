#!/usr/bin/env python3
"""Decide what one run of public-release.yml is: a release, a nightly, or nothing.

public-release.yml runs three ways:

* **dispatched with a version** -- a normal release, exactly as before;
* **dispatched with ``nightly``** -- a nightly build, by hand;
* **on its nightly schedule** -- a nightly build, but only when the
  repository variable ``PUBLIC_RELEASE_NIGHTLY`` is ``true`` (the
  setting), and only when something landed since the last one.

"Something" means a change a person made. The stat bots commit to the
default branch around the clock -- the JSON stats, the generated docs
data, the README header -- and each of those is a new commit, so a
nightly keyed on "HEAD moved" was rebuilt and republished every night
with no code in it. :data:`STAT_UPDATES` names those commits by their
subject *and* the files they may touch; a commit counts as a stat
update only when both match, so a person's commit that happens to share
a subject, or a bot commit that strays outside its files, still counts.

A nightly is not a release. It never bumps the version in the tree,
never commits, and never goes to PyPI: its version is the tree's with a
PEP 440 *local* label, ``0.72.6.post3+nightly.20260930.1a2b3c4``, which
pip installs from a URL and which no package index accepts -- so it
cannot be mistaken for, or collide with, a real release. It goes to one
rolling GitHub prerelease tagged ``nightly``, replaced each night.

    release_plan.py --event schedule --enabled true --head SHA \\
        --last-nightly SHA --date 20260930 [--version V] [--nightly true]

Prints ``key=value`` lines for ``$GITHUB_OUTPUT``: run, nightly,
version, tag, reason. Exits 1 on a dispatch with neither a version nor
``nightly``, which is a mistake rather than a no-op.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SETTING = "PUBLIC_RELEASE_NIGHTLY"
NIGHTLY_TAG = "nightly"

#: ``(subject pattern, paths it may touch)`` for each bot commit that is
#: bookkeeping rather than a change. A path ending in "/" is a folder.
#: The workflows that write these: update-json-stats.yml,
#: update-docs-data.yml, update-readme.yml and arch-map.yml.
STAT_UPDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"^chore: update JSON stats\b", ("docs/public/v1/json",)),
    (r"^chore: refresh generated docs data\b", ("docs/public/v1/",)),
    (r"^Auto-update README header\b", ("README.md",)),
    (r"^docs: architecture chart\b", ("wiki/Architecture.md",)),
)


def _under(path: str, allowed: tuple[str, ...]) -> bool:
    return any(
        path == a.rstrip("/") or path.startswith(a if a.endswith("/") else a + "/")
        for a in allowed
    )


def is_stat_update(subject: str, files: list[str]) -> bool:
    """True when this commit is a stat bot's, touching only its own files."""
    if not files:
        return False
    for pattern, allowed in STAT_UPDATES:
        if re.search(pattern, subject):
            return all(_under(f, allowed) for f in files)
    return False


def commits_since(last: str, head: str) -> list[tuple[str, list[str]]] | None:
    """``[(subject, files)]`` for the non-merge commits in last..head.

    ``None`` when *last* is not an ancestor of *head* -- the nightly tag
    was built from history the branch no longer has, so whatever is
    there now is new.
    """
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout

    try:
        git("merge-base", "--is-ancestor", last, head)
    except subprocess.CalledProcessError:
        return None
    commits: list[tuple[str, list[str]]] = []
    for line in git("log", "--no-merges", "--format=%H%x1f%s", f"{last}..{head}").splitlines():
        sha, _, subject = line.partition("\x1f")
        files = git("diff-tree", "--no-commit-id", "--name-only", "-r", sha).split()
        commits.append((subject, files))
    return commits


def tree_version(pyproject: Path) -> str:
    match = re.search(r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
    if not match:
        raise ValueError(f"{pyproject} has no version line")
    return match.group(1)


def nightly_version(base: str, date: str, head: str) -> str:
    """``<tree version>+nightly.<date>.<short sha>`` -- a PEP 440 local label."""
    base = base.split("+", 1)[0]
    return f"{base}+nightly.{date}.{head[:7]}"


def plan(
    *,
    event: str,
    version: str,
    nightly: bool,
    enabled: bool,
    head: str,
    last_nightly: str,
    date: str,
    base: str,
    since: list[tuple[str, list[str]]] | None = None,
) -> dict[str, str]:
    scheduled = event == "schedule"
    if scheduled and not enabled:
        return {"run": "false", "nightly": "true", "version": "", "tag": "",
                "reason": f"nightly builds are off: set the repository variable {SETTING}=true"}
    if scheduled or nightly:
        if scheduled and last_nightly and last_nightly == head:
            return {"run": "false", "nightly": "true", "version": "", "tag": "",
                    "reason": f"nothing new since the last nightly ({head[:7]})"}
        if scheduled and since is not None:
            real = [subject for subject, files in since if not is_stat_update(subject, files)]
            if not real:
                return {"run": "false", "nightly": "true", "version": "", "tag": "",
                        "reason": f"only stat updates since the last nightly "
                                  f"({len(since)} commit{'s' if len(since) != 1 else ''})"}
        reason = "scheduled nightly" if scheduled else "nightly, dispatched by hand"
        if version and not scheduled:
            reason += f"; the version {version!r} was ignored -- a nightly takes the tree's"
        return {"run": "true", "nightly": "true",
                "version": nightly_version(base, date, head), "tag": NIGHTLY_TAG,
                "reason": reason}
    if not version:
        raise ValueError("give a version to release, or tick nightly for a nightly build")
    return {"run": "true", "nightly": "false", "version": version,
            "tag": f"v{version}", "reason": "release"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event", required=True)
    parser.add_argument("--version", default="")
    parser.add_argument("--nightly", default="false")
    parser.add_argument("--enabled", default="")
    parser.add_argument("--head", required=True)
    parser.add_argument("--last-nightly", default="")
    parser.add_argument("--date", required=True)
    parser.add_argument("--pyproject", default="pyproject.toml")
    args = parser.parse_args(argv)
    last = args.last_nightly.strip()
    head = args.head.strip()
    since = commits_since(last, head) if (args.event == "schedule" and last and last != head) else None
    try:
        decided = plan(
            event=args.event,
            version=args.version.strip(),
            nightly=args.nightly.strip().lower() == "true",
            enabled=args.enabled.strip().lower() == "true",
            head=args.head.strip(),
            last_nightly=args.last_nightly.strip(),
            date=args.date.strip(),
            base=tree_version(Path(args.pyproject)),
            since=since,
        )
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    for key, value in decided.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
