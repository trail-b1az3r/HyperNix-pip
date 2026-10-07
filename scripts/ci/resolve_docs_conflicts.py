#!/usr/bin/env python3
"""Merge the base branch into a PR branch, settling generated-data conflicts.

``docs/public/v1`` holds generated files (the API reference, code stats,
changelog summaries, download stats) that a scheduled job on ``main``
rewrites every hour. A pull request that also regenerated them conflicts
on every one, every time, though no human edited a line.

This merges the base branch into the checked-out PR branch. When the only
conflicts are under the given paths, each file takes the version from
whichever branch changed it most recently -- the newer commit touching
that file since the two branches diverged -- and the merge is committed.
A conflict anywhere else is left for a person: the merge is undone, the
branch is exactly as it was, and the conflicted files are listed.

    resolve_docs_conflicts.py --base origin/main [--paths docs/public/v1]
                              [--summary FILE] [--no-commit]

Exit status: 0 nothing to do, or resolved (and committed unless
--no-commit); 2 conflicts outside --paths, nothing changed; 1 error.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_PATHS = ("docs/public/v1",)

CLEAN, RESOLVED, BLOCKED = "clean", "resolved", "blocked"


@dataclass(frozen=True)
class Choice:
    path: str
    side: str           # "base" or "pr"
    deleted: bool       # the chosen side deleted the file
    pr_time: int | None
    base_time: int | None


class GitError(RuntimeError):
    pass


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    done = subprocess.run(["git", *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=False)
    if check and done.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed:\n{done.stdout}{done.stderr}")
    return done


def _under(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(root.rstrip("/") + "/") for root in roots)


def _last_change(range_or_ref: str, path: str) -> int | None:
    out = git("log", "-1", "--format=%ct", range_or_ref, "--", path).stdout.strip()
    return int(out) if out else None


def _side_time(merge_base: str, tip: str, path: str) -> int | None:
    """When *tip*'s side last changed *path* after the branches diverged."""
    return _last_change(f"{merge_base}..{tip}", path) or _last_change(tip, path)


def _stages(path: str) -> set[int]:
    """Which index stages hold *path*: 2 is the PR side, 3 the base side."""
    stages = set()
    for line in git("ls-files", "-u", "--", path).stdout.splitlines():
        stages.add(int(line.split()[2]))
    return stages


def choose(path: str, merge_base: str, base: str) -> Choice:
    pr_time = _side_time(merge_base, "HEAD", path)
    base_time = _side_time(merge_base, base, path)
    # Newer wins; on a tie the base branch, whose copy the hourly job keeps current.
    side = "pr" if (pr_time or 0) > (base_time or 0) else "base"
    present = 2 if side == "pr" else 3
    return Choice(path, side, present not in _stages(path), pr_time, base_time)


def apply(choice: Choice) -> None:
    if choice.deleted:
        git("rm", "-q", "--", choice.path)
        return
    git("checkout", "--ours" if choice.side == "pr" else "--theirs", "--", choice.path)
    git("add", "--", choice.path)


def resolve(base: str, paths: tuple[str, ...], *, commit: bool = True,
            message: str | None = None) -> tuple[str, list[Choice], list[str]]:
    """Merge *base* into HEAD; returns (outcome, choices, unresolvable files)."""
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise GitError("the working tree has uncommitted changes; refusing to merge into it")
    merge_base = git("merge-base", "HEAD", base).stdout.strip()

    merged = git("merge", "--no-ff", "--no-commit", base, check=False)
    conflicted = git("diff", "--name-only", "--diff-filter=U").stdout.split()
    if merged.returncode == 0 and not conflicted:
        # Merges cleanly: nothing to fix, so leave the branch untouched.
        git("merge", "--abort", check=False)
        return CLEAN, [], []
    if not conflicted:
        git("merge", "--abort", check=False)
        raise GitError(f"git merge {base} failed without conflicts:\n{merged.stdout}{merged.stderr}")

    others = sorted(p for p in conflicted if not _under(p, paths))
    if others:
        git("merge", "--abort")
        return BLOCKED, [], others

    choices = [choose(path, merge_base, base) for path in sorted(conflicted)]
    for choice in choices:
        apply(choice)
    left = git("diff", "--name-only", "--diff-filter=U").stdout.split()
    if left:
        git("merge", "--abort")
        raise GitError(f"still conflicted after resolving: {left}")
    if commit:
        git("commit", "--no-edit", "-m", message or default_message(base, choices))
    return RESOLVED, choices, []


def default_message(base: str, choices: list[Choice]) -> str:
    name = base.removeprefix("origin/")
    lines = [f"Merge {name}: resolve generated docs data conflicts", "",
             "Each conflicted file takes the version from the branch that changed it last:"]
    lines += [f"- {c.path}: {'deleted, from' if c.deleted else 'from'} {_side_label(c, name)}" for c in choices]
    return "\n".join(lines) + "\n"


def _side_label(choice: Choice, base_name: str) -> str:
    return base_name if choice.side == "base" else "this branch"


def _when(timestamp: int | None) -> str:
    if timestamp is None:
        return "—"
    return datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def summary(outcome: str, base: str, choices: list[Choice], others: list[str],
            paths: tuple[str, ...]) -> str:
    name = base.removeprefix("origin/")
    where = ", ".join(f"`{p}`" for p in paths)
    if outcome == CLEAN:
        return f"No merge conflicts with `{name}`; nothing to do.\n"
    if outcome == BLOCKED:
        listed = "\n".join(f"- `{p}`" for p in others)
        return (f"**Not merged automatically.** Merging `{name}` conflicts outside {where}, "
                f"which needs a person:\n\n{listed}\n\nThe branch was left as it was.\n")
    rows = "\n".join(
        f"| `{c.path}` | {'deleted on ' if c.deleted else ''}{'`' + name + '`' if c.side == 'base' else 'this PR'} "
        f"| {_when(c.pr_time)} | {_when(c.base_time)} |"
        for c in choices)
    return (f"Merged `{name}` into this branch. The conflicts were all in {where}, so each file "
            f"took the version from the branch that changed it most recently:\n\n"
            f"| File | Taken from | Changed on this PR | Changed on `{name}` |\n"
            f"| --- | --- | --- | --- |\n{rows}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="the branch to merge in, e.g. origin/main")
    parser.add_argument("--paths", nargs="+", default=list(DEFAULT_PATHS),
                        help="directories whose conflicts may be settled automatically")
    parser.add_argument("--summary", type=Path, help="write a Markdown summary here")
    parser.add_argument("--no-commit", action="store_true", help="resolve but do not commit")
    args = parser.parse_args(argv)
    paths = tuple(p.strip("/") for p in args.paths)

    try:
        outcome, choices, others = resolve(args.base, paths, commit=not args.no_commit)
    except GitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    text = summary(outcome, args.base, choices, others, paths)
    print(f"outcome: {outcome}")
    print(text)
    if args.summary:
        args.summary.write_text(text, encoding="utf-8")
    return 2 if outcome == BLOCKED else 0


if __name__ == "__main__":
    sys.exit(main())
