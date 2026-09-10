#!/usr/bin/env python3
"""Decide whether the version a release was dispatched with may be cut.

Three questions, and they are not the same question.

**Does it go backwards?** This is what the guard was written for.
`public-release` writes whatever version it is handed, so a dispatch
naming an older one silently downgrades main -- v0.72.3.post2 was once
dispatched against a tree already at 0.72.3.post4 and rewrote all three
version strings to the older number, leaving main claiming a release
that predates the code in it. Nothing downstream can detect that: the
built wheel is internally consistent, it is just wrong about which
release it is.

**Is it the version the tree was prepared with?** The guard used to
refuse this outright -- "version X is already what the tree says, use a
.postN suffix" -- and that was wrong. Preparing a release *is* writing
the version into `pyproject.toml`, `setup.cfg` and `__init__.py`, adding
the changelog heading under that number, and then dispatching it. Both
downstream steps already expect it: "Commit version bump" notices there
is nothing to commit, and "Tag and push" skips a tag that exists. Only
the guard disagreed, and the cost was paid in the changelog -- 0.72.4
`post1` and `post3` were cut under numbers invented at dispatch time to
get past it, so neither has a heading and neither says what shipped.

**Is this number already published?** That is the real hazard the old
message was reaching for, and a version string cannot answer it. A tag
can: `v<version>` pointing at a *different* commit means the number is
spoken for by other code, and cutting it again publishes two different
trees under one release. Pointing at the commit being released is not
that -- it is the same code, and re-running a release that failed after
the tag push is a thing people legitimately do.

Exit status is 0 to proceed and 1 to stop; every refusal prints a
GitHub-Actions `::error::` line naming what to do instead.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

# `0.70.6-2` means "rebuild of an already-released version as build 2".
# PyPI rejects a raw hyphen-number suffix; `.post2` is PEP 440's own
# spelling of exactly that, so the two are the same request.
_REBUILD = re.compile(r"^(\d+\.\d+\.\d+)-(\d+)$")


def normalise(requested: str) -> str:
    """The dispatch input as PEP 440 spells it."""
    value = requested.strip().lstrip("v").replace("postr", ".post")
    value = re.sub(r"\.{2,}post", ".post", value)
    match = _REBUILD.match(value)
    if match:
        return f"{match.group(1)}.post{match.group(2)}"
    return value


def tree_version(pyproject: Path) -> str:
    found = re.search(r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
    if found is None:
        fail(f"no version line in {pyproject}")
    return found.group(1)


def fail(message: str) -> None:
    print(f"::error::{message}")
    raise SystemExit(1)


def note(message: str) -> None:
    print(f"::notice::{message}")


def warn(message: str) -> None:
    print(f"::warning::{message}")


def _git(*args: str, repo: Path) -> str | None:
    """A git command's stdout, or None if git could not answer.

    A shallow checkout, a missing tag and a machine with no git all end
    up here, and none of them is a reason to stop a release -- the guard
    just has less to go on and says so.
    """
    try:
        done = subprocess.run(  # noqa: S603
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def tag_points_at(tag: str, repo: Path) -> str | None:
    """The commit `tag` names, or None when there is no such tag."""
    return _git("rev-list", "-n", "1", tag, repo=repo)


def changelog_mentions(version: str, changelog: Path) -> bool:
    if not changelog.is_file():
        return False
    heading = re.compile(rf"^#+\s*{re.escape(version)}(?![\w.])", re.M)
    return bool(heading.search(changelog.read_text(encoding="utf-8", errors="replace")))


def decide(
    requested: str,
    *,
    repo: Path,
    allow_downgrade: bool = False,
    head: str | None = None,
) -> None:
    pep440 = normalise(requested)
    current = tree_version(repo / "pyproject.toml")

    try:
        new, old = Version(pep440), Version(current)
    except InvalidVersion as exc:
        fail(str(exc))

    if new < old:
        if not allow_downgrade:
            fail(
                f"version {new} is OLDER than the tree's {old}. Publishing it "
                f"would make main claim a release that predates its own code. "
                f"If you really mean to roll back, re-run with allow_downgrade."
            )
        warn(f"{old} -> {new} is a downgrade, allowed by allow_downgrade")
        return

    if new > old:
        note(f"{old} -> {new}")
        return

    # Equal: the tree was prepared with this number. The only thing that
    # makes that unsafe is the number already naming other code.
    tag = f"v{requested.strip().lstrip('v')}"
    tagged = tag_points_at(tag, repo)
    if tagged is None and tag != f"v{pep440}":
        tag = f"v{pep440}"
        tagged = tag_points_at(tag, repo)

    if tagged is not None:
        # An empty --head is not a commit. Treat it as "unknown" so the
        # refusal says so, rather than reporting a mismatch against a
        # blank sha nobody can act on.
        current_head = head or _git("rev-parse", "HEAD", repo=repo) or None
        if current_head is None:
            fail(
                f"{tag} already exists and this checkout cannot say which commit "
                f"is being released, so whether it is the same code is unknown. "
                f"Cut a .postN instead, or delete the tag if it was a mistake."
            )
        if tagged != current_head:
            fail(
                f"{tag} already exists and points at {tagged[:12]}, not the "
                f"{current_head[:12]} being released. Cutting {new} again would "
                f"publish two different trees under one release. Use a .postN "
                f"suffix, or delete the tag if it was pushed by mistake."
            )
        note(f"{tag} already points at {current_head[:12]} -- re-running the same release")

    if not changelog_mentions(pep440, repo / "wiki" / "Changelog.md"):
        warn(
            f"the tree is at {new} but wiki/Changelog.md has no heading for it, "
            f"so this release will ship without notes. That is how 0.72.4.post1 "
            f"and .post3 went out unexplained."
        )

    note(f"releasing {new}, the version the tree is already prepared with")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="the dispatch input, with or without a leading v")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--allow-downgrade", action="store_true")
    parser.add_argument("--head", default=None, help="commit being released (default: HEAD)")
    args = parser.parse_args(argv)

    decide(
        args.version,
        repo=args.repo.resolve(),
        allow_downgrade=args.allow_downgrade,
        head=args.head,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
