"""hyped_pro_git: git for hyped-pro — for the person and for the model.

Every call runs ``git`` directly (never through a shell) in the workspace
hyped-pro was started in, the same root :mod:`hyped_pro_tools` scopes file
access to. Two kinds of caller use it:

* the person, through ``/git`` in the TUI (via the bridge), who can do
  anything they could do in a terminal, including push and pull;
* the model, through the tools at the bottom of this file. Reading
  (status, diff, log, show, branches) is free. Anything that changes the
  repository — commit, switch, restore — is in :data:`GATED_TOOLS` and
  runs only after the person says yes. The model gets no push, pull,
  reset or rebase at all: those reach other machines or throw work
  away, and a person who wants them types ``/git push``.

Arguments a model supplies are never allowed to become options: a branch
called ``--upload-pack=...`` is refused, and paths go after ``--``.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hypernix.interfaces.hyped_pro_tools import ToolError, _resolve_safe_path, workspace_root

#: Output a model is shown is cut here; a diff of a vendored directory
#: would otherwise fill its whole context.
MAX_OUTPUT = 60_000

#: Seconds before a local git command is given up on. Push and pull get
#: longer — they wait on a network.
LOCAL_TIMEOUT = 30
REMOTE_TIMEOUT = 180

# A ref name as git itself allows it, minus anything that could be read
# as an option. See git-check-ref-format(1); this is stricter.
_REF = re.compile(r"^(?!-)(?!.*\.\.)(?!.*//)(?!.*@\{)[A-Za-z0-9._/@+-]{1,200}(?<![./])$")
_REV = re.compile(r"^(?!-)[A-Za-z0-9._/@^~{}:+-]{1,200}$")


def _git_binary() -> str:
    found = shutil.which("git")
    if not found:
        raise ToolError("TOOL-GIT-001", "git is not installed or not on PATH.")
    return found


def run(args: list[str], *, timeout: int = LOCAL_TIMEOUT, cwd: Path | None = None, check: bool = True) -> str:
    """Run ``git <args>`` in the workspace and return stdout.

    Prompts are switched off: a push that wants a password fails with a
    message instead of hanging the bridge waiting on a terminal nobody
    can see.
    """
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_PAGER="cat", PAGER="cat", LC_ALL="C")
    try:
        result = subprocess.run(
            [_git_binary(), *args],
            cwd=str(cwd or workspace_root()),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError("TOOL-GIT-002", f"git {args[0]} took longer than {timeout}s and was stopped.") from exc
    if check and result.returncode != 0:
        message = (result.stderr or result.stdout).strip() or f"exit {result.returncode}"
        if "not a git repository" in message:
            raise ToolError("TOOL-GIT-003", f"{workspace_root()} is not inside a git repository.")
        raise ToolError("TOOL-GIT-004", f"git {args[0]} failed: {message}")
    return result.stdout


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    return text[:MAX_OUTPUT] + f"\n… cut at {MAX_OUTPUT} characters of {len(text)}; ask for a single path to see the rest."


def check_ref(name: str, what: str = "branch") -> str:
    name = (name or "").strip()
    if not _REF.match(name):
        raise ToolError("TOOL-GIT-005", f"{name!r} is not a {what} name hyped-pro will pass to git.")
    return name


def check_rev(rev: str) -> str:
    rev = (rev or "").strip()
    if not _REV.match(rev):
        raise ToolError("TOOL-GIT-005", f"{rev!r} is not a revision hyped-pro will pass to git.")
    return rev


def _paths(paths: list[str] | str | None) -> list[str]:
    """Workspace-relative paths, each checked to stay inside the workspace."""
    if paths is None:
        return []
    if isinstance(paths, str):
        paths = [paths]
    root = workspace_root()
    out = []
    for p in paths:
        resolved = _resolve_safe_path(p)
        out.append(str(resolved.relative_to(root)) or ".")
    return out


# ---------------------------------------------------------------------------
# Structured status, for the TUI's header and /git status
# ---------------------------------------------------------------------------


@dataclass
class FileStatus:
    path: str
    index: str  # X of git's XY: staged change
    worktree: str  # Y: unstaged change
    original: str | None = None  # for renames

    @property
    def label(self) -> str:
        if self.index == "?":
            return "untracked"
        if "U" in (self.index, self.worktree) or (self.index, self.worktree) in (("A", "A"), ("D", "D")):
            return "conflict"
        names = {"M": "modified", "A": "added", "D": "deleted", "R": "renamed", "C": "copied", "T": "type changed"}
        return names.get(self.index if self.index != "." else self.worktree, "changed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "index": self.index,
            "worktree": self.worktree,
            "original": self.original,
            "staged": self.index not in (".", "?"),
            "unstaged": self.worktree != "." or self.index == "?",
            "label": self.label,
        }


@dataclass
class RepoStatus:
    root: str
    branch: str | None
    upstream: str | None = None
    ahead: int = 0
    behind: int = 0
    detached: bool = False
    files: list[FileStatus] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.files

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "branch": self.branch,
            "upstream": self.upstream,
            "ahead": self.ahead,
            "behind": self.behind,
            "detached": self.detached,
            "clean": self.clean,
            "files": [f.to_dict() for f in self.files],
        }


def parse_status(porcelain: str, root: str = "") -> RepoStatus:
    """Parse ``git status --porcelain=v2 --branch -z``."""
    status = RepoStatus(root=root, branch=None)
    entries = porcelain.split("\0")
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if not entry:
            continue
        if entry.startswith("# branch.head "):
            head = entry[len("# branch.head ") :]
            status.detached = head == "(detached)"
            status.branch = None if status.detached else head
        elif entry.startswith("# branch.upstream "):
            status.upstream = entry[len("# branch.upstream ") :]
        elif entry.startswith("# branch.ab "):
            ahead, behind = entry[len("# branch.ab ") :].split()
            status.ahead, status.behind = int(ahead), -int(behind)
        elif entry.startswith("1 "):
            parts = entry.split(" ", 8)
            status.files.append(FileStatus(parts[8], parts[1][0], parts[1][1]))
        elif entry.startswith("2 "):
            parts = entry.split(" ", 9)
            # A rename's original path is the next NUL-separated field.
            original = entries[i] if i < len(entries) else None
            i += 1
            status.files.append(FileStatus(parts[9], parts[1][0], parts[1][1], original))
        elif entry.startswith("u "):
            parts = entry.split(" ", 10)
            status.files.append(FileStatus(parts[10], "U", "U"))
        elif entry.startswith("? "):
            status.files.append(FileStatus(entry[2:], "?", "?"))
    return status


def status() -> RepoStatus:
    top = run(["rev-parse", "--show-toplevel"]).strip()
    return parse_status(run(["status", "--porcelain=v2", "--branch", "-z"]), root=top)


def is_repository() -> bool:
    try:
        run(["rev-parse", "--git-dir"])
    except ToolError:
        return False
    return True


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def diff(path: str | None = None, staged: bool = False, rev: str | None = None) -> str:
    args = ["diff", "--no-color", "--no-ext-diff"]
    if staged:
        args.append("--cached")
    if rev:
        args.append(check_rev(rev))
    args.append("--")
    args.extend(_paths(path))
    return _clip(run(args))


def log(count: int = 20, path: str | None = None) -> list[dict[str, str]]:
    count = max(1, min(int(count), 200))
    fmt = "%H%x1f%h%x1f%an%x1f%ar%x1f%s%x1e"
    out = run(["log", f"-n{count}", f"--format={fmt}", "--", *_paths(path)])
    commits = []
    for record in out.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        full, short, author, when, subject = record.split("\x1f")
        commits.append({"hash": full, "short": short, "author": author, "when": when, "subject": subject})
    return commits


def show(rev: str = "HEAD", path: str | None = None) -> str:
    args = ["show", "--no-color", "--stat", "--patch", check_rev(rev), "--", *_paths(path)]
    return _clip(run(args))


def branches() -> dict[str, Any]:
    fmt = "%(HEAD)%1f%(refname:short)%1f%(upstream:short)%1f%(subject)"
    out = run(["branch", f"--format={fmt}"])
    found = []
    for line in out.splitlines():
        head, name, upstream, subject = (line.split("\x1f") + ["", "", "", ""])[:4]
        found.append({"name": name, "current": head == "*", "upstream": upstream, "subject": subject})
    return {"branches": found, "current": next((b["name"] for b in found if b["current"]), None)}


# ---------------------------------------------------------------------------
# Changing
# ---------------------------------------------------------------------------


def add(paths: list[str] | str) -> str:
    targets = _paths(paths)
    if not targets:
        raise ToolError("TOOL-GIT-006", "name at least one path to stage ('.' for everything).")
    run(["add", "--", *targets])
    return f"Staged {', '.join(targets)}."


def unstage(paths: list[str] | str) -> str:
    targets = _paths(paths)
    if not targets:
        raise ToolError("TOOL-GIT-006", "name at least one path to unstage.")
    run(["restore", "--staged", "--", *targets])
    return f"Unstaged {', '.join(targets)}."


def commit(message: str, all_tracked: bool = False) -> str:
    message = (message or "").strip()
    if not message:
        raise ToolError("TOOL-GIT-007", "a commit needs a message.")
    if all_tracked:
        run(["add", "--update"])
    staged = run(["diff", "--cached", "--name-only"]).split()
    if not staged:
        raise ToolError("TOOL-GIT-008", "nothing is staged — stage files with git_add first.")
    # -F - : the message goes through stdin, so no part of it is ever an
    # option, and --no-verify is never passed: the repository's own hooks
    # still run.
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", LC_ALL="C")
    result = subprocess.run(
        [_git_binary(), "commit", "-F", "-"],
        cwd=str(workspace_root()),
        input=message + "\n",
        capture_output=True,
        text=True,
        timeout=LOCAL_TIMEOUT,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise ToolError("TOOL-GIT-004", f"git commit failed: {(result.stderr or result.stdout).strip()}")
    head = run(["log", "-1", "--format=%h %s"]).strip()
    return f"Committed {len(staged)} file(s): {head}"


def switch(branch: str, create: bool = False) -> str:
    name = check_ref(branch)
    run(["switch", *(["-c"] if create else []), name])
    return f"{'Created and switched to' if create else 'Switched to'} {name}."


def restore(paths: list[str] | str) -> str:
    """Throw away unstaged changes to ``paths``. Cannot be undone."""
    targets = _paths(paths)
    if not targets:
        raise ToolError("TOOL-GIT-006", "name the paths whose changes to discard.")
    run(["restore", "--worktree", "--", *targets])
    return f"Discarded changes to {', '.join(targets)}."


def push(remote: str | None = None, branch: str | None = None, set_upstream: bool = False) -> str:
    args = ["push"]
    if set_upstream:
        args.append("--set-upstream")
    if remote:
        args.append(check_ref(remote, "remote"))
        if branch:
            args.append(check_ref(branch))
    result = subprocess.run(
        [_git_binary(), *args],
        cwd=str(workspace_root()),
        capture_output=True,
        text=True,
        timeout=REMOTE_TIMEOUT,
        env=dict(os.environ, GIT_TERMINAL_PROMPT="0", LC_ALL="C"),
        check=False,
    )
    # git push reports progress and results on stderr, success or not.
    output = (result.stderr + result.stdout).strip()
    if result.returncode != 0:
        raise ToolError("TOOL-GIT-004", f"git push failed: {output}")
    return output or "Pushed."


def pull(rebase: bool = False) -> str:
    return run(["pull", "--rebase" if rebase else "--ff-only"], timeout=REMOTE_TIMEOUT).strip() or "Up to date."


# ---------------------------------------------------------------------------
# The model's git tools
# ---------------------------------------------------------------------------


def _status_text() -> str:
    s = status()
    lines = [f"On {'a detached HEAD' if s.detached else 'branch ' + str(s.branch)}"]
    if s.upstream:
        lines.append(f"Tracking {s.upstream} (ahead {s.ahead}, behind {s.behind})")
    if s.clean:
        lines.append("Working tree clean.")
    for f in s.files:
        where = "staged" if f.index not in (".", "?") else "unstaged"
        if f.index not in (".", "?") and f.worktree != ".":
            where = "staged + unstaged"
        rename = f" (from {f.original})" if f.original else ""
        lines.append(f"  {f.label:<12} {where:<18} {f.path}{rename}")
    return "\n".join(lines)


def _log_text(count: int = 20, path: str | None = None) -> str:
    return "\n".join(f"{c['short']}  {c['when']:<16} {c['author']}: {c['subject']}" for c in log(count, path)) or "(no commits)"


def _branches_text() -> str:
    found = branches()["branches"]
    return "\n".join(f"{'*' if b['current'] else ' '} {b['name']}" + (f" -> {b['upstream']}" if b["upstream"] else "") for b in found)


def _diff_text(path: str | None = None, staged: bool = False, rev: str | None = None) -> str:
    return diff(path, staged, rev) or "(no differences)"


#: Tools that change the repository in a way the person should agree to.
GATED_TOOLS: frozenset[str] = frozenset({"git_commit", "git_switch", "git_restore"})


def tool_definitions() -> list[tuple[str, str, dict[str, Any], Any]]:
    """(name, description, JSON schema, function) for hyped_pro_tools."""
    path_prop = {"type": "string", "description": "Path relative to the workspace root. Omit for the whole repository."}
    paths_prop = {
        "type": "array",
        "items": {"type": "string"},
        "description": "Paths relative to the workspace root; '.' for everything.",
    }
    return [
        ("git_status", "Show the current branch, how it compares to its upstream, and every changed, staged and untracked file.",
         {"type": "object", "properties": {}, "required": []}, _status_text),
        ("git_diff", "Show a unified diff: unstaged changes by default, staged ones with staged=true, or against a revision with rev.",
         {"type": "object", "properties": {
             "path": path_prop,
             "staged": {"type": "boolean", "description": "Diff what is staged for the next commit."},
             "rev": {"type": "string", "description": "Compare the working tree with this revision, e.g. HEAD~1 or main."},
         }, "required": []}, _diff_text),
        ("git_log", "List recent commits, newest first, optionally only those touching a path.",
         {"type": "object", "properties": {
             "count": {"type": "integer", "description": "How many commits (1-200, default 20)."},
             "path": path_prop,
         }, "required": []}, _log_text),
        ("git_show", "Show one commit: its message, the files it changed and its diff.",
         {"type": "object", "properties": {
             "rev": {"type": "string", "description": "Commit to show. Defaults to HEAD."},
             "path": path_prop,
         }, "required": []}, show),
        ("git_branches", "List local branches and which one is checked out.",
         {"type": "object", "properties": {}, "required": []}, _branches_text),
        ("git_add", "Stage files for the next commit.",
         {"type": "object", "properties": {"paths": paths_prop}, "required": ["paths"]}, add),
        ("git_commit", "Commit what is staged. The person is asked to approve the message first. Set all_tracked to stage every modified tracked file first.",
         {"type": "object", "properties": {
             "message": {"type": "string", "description": "The commit message: a short summary line, a blank line, then detail if needed."},
             "all_tracked": {"type": "boolean", "description": "Stage all modified tracked files before committing."},
         }, "required": ["message"]}, commit),
        ("git_switch", "Switch to a branch, or create it and switch with create=true. The person is asked first.",
         {"type": "object", "properties": {
             "branch": {"type": "string", "description": "Branch name."},
             "create": {"type": "boolean", "description": "Create the branch from the current commit."},
         }, "required": ["branch"]}, switch),
        ("git_restore", "Discard unstaged changes to files, putting them back as they are in the index. Cannot be undone; the person is asked first.",
         {"type": "object", "properties": {"paths": paths_prop}, "required": ["paths"]}, restore),
    ]
