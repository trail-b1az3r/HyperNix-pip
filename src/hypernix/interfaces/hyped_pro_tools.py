"""hypernix.hyped_pro_tools — real file creation/editing/search tools for
hyped-pro's agentic chat loop.

Every tool here does exactly what it says — there's no simulated or
stubbed path. All operations are scoped to a workspace root (the
directory hyped-pro was launched from, or ``HYPED_PRO_WORKSPACE`` if set)
and every resolved path is checked to stay inside it before anything
touches disk; a path that would escape the workspace (``../../etc/passwd``,
an absolute path elsewhere, a symlink pointing out) is refused rather than
followed.

Definitions here are vendor-neutral (name, description, a JSON-schema
``parameters`` block) and converted to whichever wire format a given
backend needs — Anthropic's ``tools``/``input_schema`` shape, or the
OpenAI-compatible ``tools``/``function``/``parameters`` shape used by
OpenAI, DashScope (Qwen), Moonshot (Kimi), and llama-cpp-python's own
``create_chat_completion(tools=...)`` for GGUF models whose chat template
supports function calling.

Every tool call is logged to stderr as it executes — real-time, visible
on the same terminal the rest of this codebase already prints backend
logs to — so file operations a model makes are never silent.
"""
from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import json
import os
import re
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ToolError(RuntimeError):
    """A real, reportable tool failure — fed back to the model as the
    tool result (so it can react/retry), never swallowed."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


def _log(msg: str) -> None:
    print(f"[hypernix.tools] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Workspace scoping — every tool call resolves through this first.
# ---------------------------------------------------------------------------


def workspace_root() -> Path:
    override = os.environ.get("HYPED_PRO_WORKSPACE")
    return Path(override).expanduser().resolve() if override else Path.cwd().resolve()


def _resolve_safe_path(rel_path: str) -> Path:
    root = workspace_root()
    candidate = (root / rel_path).resolve() if not Path(rel_path).is_absolute() else Path(rel_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ToolError(
            "TOOL-PATH-001",
            f"{rel_path!r} resolves outside the workspace ({root}) — refusing. "
            f"Set HYPED_PRO_WORKSPACE to widen it if this is intentional.",
        ) from exc
    return candidate


def _resolve_writable_path(rel_path: str) -> Path:
    """A path the model may write to: inside the workspace and not in .git.

    Writing under .git is how a file edit becomes code execution — a hook
    runs on the next commit — so no tool that writes goes there, whatever
    the workspace.
    """
    dest = _resolve_safe_path(rel_path)
    rel = dest.relative_to(workspace_root())
    if rel.parts and rel.parts[0] == ".git":
        raise ToolError("TOOL-PATH-002", f"{rel_path!r} is inside .git — tools never write there.")
    return dest


# ---------------------------------------------------------------------------
# Real implementations
# ---------------------------------------------------------------------------

_MAX_READ_BYTES = 512_000    # ~512KB — enough for any real source file, not a whole log dump
_MAX_SEARCH_MATCHES = 200
_MAX_SEARCH_FILES = 5000


def create_file(path: str, content: str) -> str:
    dest = _resolve_writable_path(path)
    if dest.exists():
        raise ToolError("TOOL-CREATE-001", f"{path!r} already exists — use edit_file to modify it, not create_file.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    _log(f"create_file: wrote {len(content)} bytes -> {dest}")
    return f"Created {path} ({len(content)} bytes)."


def edit_file(path: str, old_str: str, new_str: str) -> str:
    dest = _resolve_writable_path(path)
    if not dest.exists():
        raise ToolError("TOOL-EDIT-001", f"{path!r} does not exist — use create_file for a new file.")
    if dest.is_dir():
        raise ToolError("TOOL-EDIT-002", f"{path!r} is a directory, not a file.")
    text = dest.read_text(encoding="utf-8")
    count = text.count(old_str)
    if count == 0:
        raise ToolError("TOOL-EDIT-003", f"old_str not found in {path!r} — it must match the file's current content exactly, including whitespace.")
    if count > 1:
        raise ToolError("TOOL-EDIT-004", f"old_str matches {count} places in {path!r} — it must be unique. Include more surrounding context.")
    new_text = text.replace(old_str, new_str, 1)
    dest.write_text(new_text, encoding="utf-8")
    delta = len(new_str) - len(old_str)
    _log(f"edit_file: replaced {len(old_str)} chars with {len(new_str)} chars in {dest} ({delta:+d})")
    return f"Edited {path} ({delta:+d} chars)."


def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    src = _resolve_safe_path(path)
    if not src.exists():
        raise ToolError("TOOL-READ-001", f"{path!r} does not exist.")
    if src.is_dir():
        raise ToolError("TOOL-READ-002", f"{path!r} is a directory — use list_directory instead.")
    size = src.stat().st_size
    if size > _MAX_READ_BYTES and start_line is None and end_line is None:
        raise ToolError(
            "TOOL-READ-003",
            f"{path!r} is {size} bytes, over the {_MAX_READ_BYTES}-byte read limit — "
            f"pass start_line/end_line to read a slice instead of the whole file.",
        )
    try:
        text = src.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError("TOOL-READ-004", f"{path!r} isn't valid UTF-8 text: {exc}") from exc
    if start_line is None and end_line is None:
        _log(f"read_file: {dest_repr(src)} ({len(text)} chars)")
        return text
    lines = text.splitlines()
    s = max(1, start_line or 1)
    e = min(len(lines), end_line or len(lines))
    _log(f"read_file: {dest_repr(src)} lines {s}-{e}")
    numbered = "\n".join(f"{i}\t{lines[i - 1]}" for i in range(s, e + 1))
    return numbered


def content_hash(text: str) -> str:
    """What the TUI's editor remembers about a file it opened, so a save
    can tell whether someone else changed it in the meantime."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_file(path: str, content: str, expected_hash: str | None = None) -> str:
    """Create or overwrite a whole file.

    ``expected_hash`` is the :func:`content_hash` of the file as the caller
    last saw it. When given, a file that has changed since is left alone —
    the model (or the person's editor) is working from a stale copy.
    """
    dest = _resolve_writable_path(path)
    if dest.is_dir():
        raise ToolError("TOOL-WRITE-001", f"{path!r} is a directory, not a file.")
    if expected_hash is not None:
        # Bytes, as file_read sent them: a text-mode read turns CRLF into
        # LF on Windows, and the hash of that never matches what the
        # editor was given, so every save of a CRLF file looked stale.
        current = dest.read_bytes().decode("utf-8", errors="replace") if dest.exists() else ""
        if content_hash(current) != expected_hash:
            raise ToolError(
                "TOOL-WRITE-002",
                f"{path!r} changed on disk since it was read — read it again before writing.",
            )
    existed = dest.exists()
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(f".{dest.name}.hyped-pro.tmp")
    # Bytes too: write_text would turn each "\n" into "\r\n" on Windows,
    # so a CRLF file came back with "\r\r\n".
    temporary.write_bytes(content.encode("utf-8"))
    if existed:
        with contextlib.suppress(OSError):
            os.chmod(temporary, dest.stat().st_mode)
    os.replace(temporary, dest)
    _log(f"write_file: {'overwrote' if existed else 'created'} {dest_repr(dest)} ({len(content)} chars)")
    return f"{'Wrote' if existed else 'Created'} {path} ({len(content)} chars)."


def delete_file(path: str) -> str:
    dest = _resolve_writable_path(path)
    if not dest.exists():
        raise ToolError("TOOL-DELETE-001", f"{path!r} does not exist.")
    if dest.is_dir():
        raise ToolError("TOOL-DELETE-002", f"{path!r} is a directory — delete_file removes single files only.")
    dest.unlink()
    _log(f"delete_file: removed {dest_repr(dest)}")
    return f"Deleted {path}."


def move_file(source: str, destination: str) -> str:
    src = _resolve_writable_path(source)
    dst = _resolve_writable_path(destination)
    if not src.exists():
        raise ToolError("TOOL-MOVE-001", f"{source!r} does not exist.")
    if dst.exists():
        raise ToolError("TOOL-MOVE-002", f"{destination!r} already exists — moving would overwrite it.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)
    _log(f"move_file: {dest_repr(src)} -> {dest_repr(dst)}")
    return f"Moved {source} to {destination}."


def dest_repr(p: Path) -> str:
    try:
        return str(p.relative_to(workspace_root()))
    except ValueError:
        return str(p)


def list_directory(path: str = ".") -> str:
    d = _resolve_safe_path(path)
    if not d.exists():
        raise ToolError("TOOL-LIST-001", f"{path!r} does not exist.")
    if not d.is_dir():
        raise ToolError("TOOL-LIST-002", f"{path!r} is a file, not a directory.")
    entries = sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines = []
    for e in entries:
        if e.name.startswith(".git") or e.name == "__pycache__" or e.name == "node_modules":
            continue
        marker = "/" if e.is_dir() else ""
        size = "" if e.is_dir() else f"  ({e.stat().st_size}B)"
        lines.append(f"{e.name}{marker}{size}")
    _log(f"list_directory: {dest_repr(d)} ({len(lines)} entries)")
    return "\n".join(lines) if lines else "(empty directory)"


def search_files(query: str, path: str = ".", mode: str = "content", glob: str | None = None) -> str:
    root = _resolve_safe_path(path)
    if not root.exists():
        raise ToolError("TOOL-SEARCH-001", f"{path!r} does not exist.")

    skip_dirs = {".git", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache", "dist", "build"}

    def walk() -> list[Path]:
        out: list[Path] = []
        stack = [root] if root.is_dir() else [root.parent]
        start = root if root.is_file() else None
        if start is not None:
            return [start]
        while stack and len(out) < _MAX_SEARCH_FILES:
            cur = stack.pop()
            try:
                children = list(cur.iterdir())
            except OSError:
                continue
            for c in children:
                if c.name in skip_dirs:
                    continue
                if c.is_dir():
                    stack.append(c)
                else:
                    if glob and not fnmatch.fnmatch(c.name, glob):
                        continue
                    out.append(c)
        return out

    files = walk()

    if mode == "filename":
        matches = [f for f in files if fnmatch.fnmatch(f.name, query)]
        _log(f"search_files(filename): {query!r} under {dest_repr(root)} -> {len(matches)} matches")
        return "\n".join(dest_repr(f) for f in matches[:_MAX_SEARCH_MATCHES]) or "(no matches)"

    if mode != "content":
        raise ToolError("TOOL-SEARCH-002", f"unknown mode {mode!r} — use 'content' or 'filename'.")

    try:
        pattern = re.compile(query)
    except re.error as exc:
        raise ToolError("TOOL-SEARCH-003", f"invalid regex {query!r}: {exc}") from exc

    results: list[str] = []
    for f in files:
        if len(results) >= _MAX_SEARCH_MATCHES:
            break
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                results.append(f"{dest_repr(f)}:{i}: {line.strip()[:200]}")
                if len(results) >= _MAX_SEARCH_MATCHES:
                    break
    _log(f"search_files(content): {query!r} under {dest_repr(root)} -> {len(results)} matches ({len(files)} files scanned)")
    return "\n".join(results) or "(no matches)"


# ---------------------------------------------------------------------------
# Tool registry — one source of truth, converted to each vendor's wire format.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema
    fn: Callable[..., str]


TOOLS: list[ToolDef] = [
    ToolDef(
        name="create_file",
        description="Create a new file with the given content. Fails if the file already exists — use edit_file to modify an existing one.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root."},
                "content": {"type": "string", "description": "Full file content to write."},
            },
            "required": ["path", "content"],
        },
        fn=create_file,
    ),
    ToolDef(
        name="edit_file",
        description="Replace one exact occurrence of old_str with new_str in an existing file. old_str must match the file's current content exactly (including whitespace) and appear exactly once — read_file first if unsure of the exact text.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root."},
                "old_str": {"type": "string", "description": "Exact text to replace; must be unique in the file."},
                "new_str": {"type": "string", "description": "Replacement text. Empty string deletes old_str."},
            },
            "required": ["path", "old_str", "new_str"],
        },
        fn=edit_file,
    ),
    ToolDef(
        name="read_file",
        description="Read a file's contents, optionally a specific line range (1-indexed, inclusive). Reading a range returns line-numbered output.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root."},
                "start_line": {"type": "integer", "description": "First line to read (1-indexed). Omit to read from the start."},
                "end_line": {"type": "integer", "description": "Last line to read (inclusive). Omit to read to the end."},
            },
            "required": ["path"],
        },
        fn=read_file,
    ),
    ToolDef(
        name="list_directory",
        description="List the contents of a directory (non-recursive). Directories are marked with a trailing /.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path relative to the workspace root. Defaults to the workspace root itself."}},
            "required": [],
        },
        fn=list_directory,
    ),
    ToolDef(
        name="search_files",
        description="Search files under a directory. mode='content' (default) treats query as a regex searched line-by-line across file contents. mode='filename' treats query as a glob pattern (e.g. '*.py') matched against filenames.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Regex (content mode) or glob pattern (filename mode)."},
                "path": {"type": "string", "description": "Directory to search under. Defaults to the workspace root."},
                "mode": {"type": "string", "enum": ["content", "filename"], "description": "Search mode. Defaults to 'content'."},
                "glob": {"type": "string", "description": "Optional filename glob to restrict which files are scanned in content mode, e.g. '*.py'."},
            },
            "required": ["query"],
        },
        fn=search_files,
    ),
]

TOOLS.extend([
    ToolDef(
        name="write_file",
        description="Write a whole file, creating it or replacing what is there. Prefer edit_file for small changes to a large file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the workspace root."},
                "content": {"type": "string", "description": "The file's complete new content."},
            },
            "required": ["path", "content"],
        },
        fn=write_file,
    ),
    ToolDef(
        name="move_file",
        description="Move or rename a file. Refuses to overwrite an existing destination.",
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Current path, relative to the workspace root."},
                "destination": {"type": "string", "description": "New path, relative to the workspace root."},
            },
            "required": ["source", "destination"],
        },
        fn=move_file,
    ),
    ToolDef(
        name="delete_file",
        description="Delete one file. The person is asked to approve it first.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path relative to the workspace root."}},
            "required": ["path"],
        },
        fn=delete_file,
    ),
])

# Git, for the model. Imported here rather than at the top because
# hyped_pro_git builds on this module's path checks.
from hypernix.interfaces import hyped_pro_git as _git  # noqa: E402

TOOLS.extend(ToolDef(name=n, description=d, parameters=p, fn=f) for n, d, p, f in _git.tool_definitions())

_TOOLS_BY_NAME: dict[str, ToolDef] = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# Consent: which tools need a yes from the person, and how it is asked
# ---------------------------------------------------------------------------

#: Tools that change things in a way that cannot simply be edited back:
#: a deleted file, a commit, a switched branch, discarded changes.
#: create/edit/write/move are not here — they are the point of a coding
#: model and every one of them shows up in `git diff`.
GATED_TOOLS: frozenset[str] = frozenset({"delete_file"}) | _git.GATED_TOOLS

#: (tool name, arguments) -> (allowed, reason when refused)
ConsentFn = Callable[[str, dict[str, Any]], tuple[bool, str]]
#: (tool name, arguments, result or None, error or None)
ProgressFn = Callable[[str, dict[str, Any], str | None, str | None], None]

_hooks = threading.local()


@contextlib.contextmanager
def hooks(consent: ConsentFn | None = None, progress: ProgressFn | None = None) -> Iterator[None]:
    """Route consent questions and progress for tool calls made on this thread.

    The bridge wraps each chat in this, so a tool the model calls during
    that chat asks the person in the TUI rather than a terminal nobody is
    reading.
    """
    previous = (getattr(_hooks, "consent", None), getattr(_hooks, "progress", None))
    _hooks.consent, _hooks.progress = consent, progress
    try:
        yield
    finally:
        _hooks.consent, _hooks.progress = previous


def consent_policy() -> str:
    """HYPERNIX_TOOL_POLICY, as hyped uses it: ask (default), deny, allow."""
    policy = os.environ.get("HYPERNIX_TOOL_POLICY", "ask").strip().lower()
    return policy if policy in ("ask", "deny", "allow") else "ask"


def describe_call(name: str, arguments: dict[str, Any]) -> str:
    """The call as the person has to see it to decide: the real arguments."""
    if name == "git_commit":
        return str(arguments.get("message", ""))
    if name in ("delete_file",):
        return str(arguments.get("path", ""))
    if name == "git_restore":
        paths = arguments.get("paths", [])
        return "discard changes to: " + (", ".join(paths) if isinstance(paths, list) else str(paths))
    if name == "git_switch":
        return ("create and switch to " if arguments.get("create") else "switch to ") + str(arguments.get("branch", ""))
    return json.dumps(arguments, default=str)


def _ask(name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
    policy = consent_policy()
    if policy == "allow":
        return True, ""
    if policy == "deny":
        return False, "refused: HYPERNIX_TOOL_POLICY=deny. Tell the person what you would have done instead."
    hook = getattr(_hooks, "consent", None)
    if hook is not None:
        return hook(name, arguments)
    if sys.stdin is not None and sys.stdin.isatty():
        sys.stderr.write(f"\n  the model wants to run {name}:\n    {describe_call(name, arguments)}\n  allow? [y/N] ")
        sys.stderr.flush()
        answer = sys.stdin.readline().strip().lower()
        if answer in ("y", "yes"):
            return True, ""
        return False, "refused by the person."
    # Nobody to ask. "Don't" is the only safe answer.
    return False, "refused: nobody is present to approve this. Tell the person what you would have done instead."


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Run a tool by name with the given arguments, returning its result as
    a string (always — this is what gets fed back to the model as the tool
    result message). Raises ToolError on failure; callers should catch it
    and feed .message back to the model as an error result rather than
    aborting the whole turn, so the model can see what went wrong and
    adjust — same as how a real coding agent would surface a failed edit.
    """
    tool = _TOOLS_BY_NAME.get(name)
    if tool is None:
        raise ToolError("TOOL-CFG-001", f"unknown tool {name!r}. Available: {', '.join(_TOOLS_BY_NAME)}")
    if not isinstance(arguments, dict):
        raise ToolError("TOOL-CFG-002", f"arguments for {name} must be an object, not {type(arguments).__name__}.")
    progress = getattr(_hooks, "progress", None)
    try:
        if name in GATED_TOOLS:
            allowed, reason = _ask(name, arguments)
            if not allowed:
                raise ToolError("TOOL-CONSENT-001", reason or "refused by the person.")
        result = tool.fn(**arguments)
    except ToolError as exc:
        if progress is not None:
            progress(name, arguments, None, exc.message)
        raise
    except TypeError as exc:
        error = ToolError("TOOL-CFG-002", f"bad arguments for {name}: {exc}")
        if progress is not None:
            progress(name, arguments, None, error.message)
        raise error from exc
    except Exception as exc:  # noqa: BLE001
        error = ToolError("TOOL-EXEC-001", f"{name} failed: {exc}")
        if progress is not None:
            progress(name, arguments, None, error.message)
        raise error from exc
    if progress is not None:
        progress(name, arguments, result, None)
    return result


def anthropic_tools_schema() -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in TOOLS]


def openai_tools_schema() -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}} for t in TOOLS]
