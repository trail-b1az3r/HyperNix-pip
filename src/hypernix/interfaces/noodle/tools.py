"""noodle.tools — the tools every model in the swarm can call.

Nine tools, and they are the ones an agent actually needs to do work
rather than describe it::

    create_file      edit_file       execute_file
    web_search       update_memory   read_memory
    compact_context  create_todo     update_todo

Every one is defined once, as a :class:`Tool`, and rendered into whatever
schema a provider wants by :func:`tool_schemas`. Writing the same
function three times for three wire formats is how one of them ends up
with a subtly different description and the model behaves differently
depending on which backend it landed on.

The sandbox is the interesting part
-----------------------------------
An autonomous swarm writing and executing files is exactly as dangerous
as it sounds. :class:`ToolContext` is the boundary:

* **Every path is resolved against a root** and rejected if it escapes,
  including via symlink. ``..`` is the obvious case; a symlink planted
  by an earlier tool call is the one that gets missed.
* **Execution is opt-in, time-boxed, and never uses a shell.** ``sh -c``
  with model-generated text is a remote code execution primitive with a
  friendly name. Commands are argv lists.
* **Writes are budgeted.** A loop that writes a million files is a real
  failure mode for an agent that has lost the plot, and a disk that
  fills takes the host down with it.
* **Memory is off unless the server enabled it**, which is the release's
  own requirement, and is checked here rather than trusted to callers.

None of this makes an autonomous agent safe. It makes the blast radius
a directory instead of a machine, and it makes what happened auditable
afterwards.
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "Tool",
    "ToolContext",
    "ToolResult",
    "ToolError",
    "TOOLS",
    "tool_schemas",
    "run_tool",
    "TodoItem",
]


class ToolError(RuntimeError):
    """A tool refused or failed. Always reported back to the model.

    Returned rather than raised out of the loop: a model that gets "that
    path is outside the workspace" can correct itself, and one that gets
    a stack trace cannot.
    """

    def __init__(self, message: str, *, code: str = "error"):
        super().__init__(message)
        self.code = code


@dataclass
class ToolResult:
    ok: bool
    content: str
    tool: str = ""
    code: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "tool": self.tool, "code": self.code,
            "content": self.content, "data": dict(self.data),
        }


@dataclass
class TodoItem:
    todo_id: str
    text: str
    status: str = "pending"          # pending | in_progress | done | removed
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.todo_id, "text": self.text,
            "status": self.status, "created_at": self.created_at,
        }


@dataclass
class ToolContext:
    """Where a tool may act, and how much it may do.

    Constructed once per agent run. Everything here is a limit rather
    than a capability: the defaults are the restrictive ones, and a
    caller opts in to more.
    """

    root: Path
    #: Executing model-written files is off unless explicitly enabled.
    allow_execute: bool = False
    #: Memory is off unless the *server* enabled it — the release's own
    #: requirement, checked here rather than trusted to each call site.
    memory_enabled: bool = False
    allow_web_search: bool = True
    max_file_bytes: int = 2 * 1024 * 1024
    max_writes: int = 200
    execute_timeout: float = 60.0
    memory_path: Path | None = None

    writes_done: int = field(default=0, init=False)
    todos: dict[str, TodoItem] = field(default_factory=dict, init=False)
    audit: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.memory_path is None:
            self.memory_path = self.root / ".noodle-memory.json"

    # -- the boundary -------------------------------------------------

    def resolve(self, relative: str) -> Path:
        """Resolve a model-supplied path inside the workspace, or refuse.

        ``resolve()`` follows symlinks *before* the containment check, so
        a link planted by an earlier tool call cannot be used to step
        outside. Checking the textual path first and resolving after is
        the version of this that looks correct and is not.
        """
        if not relative or not str(relative).strip():
            raise ToolError("A path is required", code="bad_path")
        # expanduser() *after* the join, deliberately. By then any `~` the
        # model supplied sits in the middle of the path, where it expands
        # to nothing and stays a literal directory name under the root.
        # Expanding first would turn `~/.ssh/id_rsa` into an absolute
        # path to the real home directory -- and `Path(root) / "/abs"` is
        # `/abs`, because pathlib's `/` discards the left side when the
        # right is absolute. The containment check below still catches
        # that, but it would be the only thing catching it. Do not
        # "fix" this into expanding first.
        candidate = (self.root / str(relative)).expanduser()
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise ToolError(f"Could not resolve {relative!r}: {exc}", code="bad_path") from exc
        if resolved != self.root and self.root not in resolved.parents:
            raise ToolError(
                f"{relative!r} is outside the workspace ({self.root}). "
                "Tools may only touch files under the workspace root.",
                code="outside_workspace",
            )
        return resolved

    def note_write(self, path: Path) -> None:
        if self.writes_done >= self.max_writes:
            raise ToolError(
                f"Write budget exhausted ({self.max_writes} files). An agent that needs more "
                "than this has usually lost track of what it is doing; raise max_writes "
                "deliberately if that is wrong.",
                code="write_budget",
            )
        self.writes_done += 1

    def record(self, tool: str, detail: dict[str, Any]) -> None:
        self.audit.append({"tool": tool, "at": time.time(), **detail})


@dataclass(frozen=True)
class Tool:
    """One tool: a name, a description, a schema, and an implementation."""

    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[ToolContext, dict[str, Any]], ToolResult]
    #: Tools that change something. The swarm surfaces these differently
    #: from reads when it reports what an agent did.
    mutating: bool = False


def _p(**properties: Any) -> dict[str, Any]:
    required = [k for k, v in properties.items() if v.pop("_required", False)]
    return {"type": "object", "properties": properties, "required": required}


def _str(desc: str, *, required: bool = False) -> dict[str, Any]:
    return {"type": "string", "description": desc, "_required": required}


def _int(desc: str, *, required: bool = False) -> dict[str, Any]:
    return {"type": "integer", "description": desc, "_required": required}


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


def _create_file(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    path = ctx.resolve(str(args.get("path", "")))
    content = str(args.get("content", ""))
    if len(content.encode("utf-8")) > ctx.max_file_bytes:
        raise ToolError(
            f"Refusing to write {len(content)} characters; the limit is "
            f"{ctx.max_file_bytes} bytes.",
            code="too_large",
        )
    if path.exists() and not args.get("overwrite"):
        raise ToolError(
            f"{args.get('path')} already exists. Pass overwrite=true, or use edit_file to "
            "change part of it.",
            code="exists",
        )
    ctx.note_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    ctx.record("create_file", {"path": str(path), "bytes": len(content)})
    return ToolResult(True, f"Wrote {len(content)} characters to {args.get('path')}",
                      tool="create_file", data={"path": str(path)})


def _edit_file(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    path = ctx.resolve(str(args.get("path", "")))
    if not path.exists():
        raise ToolError(f"{args.get('path')} does not exist", code="not_found")
    old = str(args.get("old_text", ""))
    new = str(args.get("new_text", ""))
    if not old:
        raise ToolError(
            "edit_file needs old_text. To replace a whole file use create_file with "
            "overwrite=true.",
            code="bad_args",
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    occurrences = text.count(old)
    if occurrences == 0:
        raise ToolError(
            f"old_text was not found in {args.get('path')}. Read the file first — it may "
            "have changed since you last saw it.",
            code="no_match",
        )
    if occurrences > 1 and not args.get("replace_all"):
        # Ambiguity is the failure mode here: replacing "the first one"
        # silently is how an agent edits the wrong line and reports
        # success.
        raise ToolError(
            f"old_text appears {occurrences} times in {args.get('path')}. Include more "
            "surrounding context to make it unique, or pass replace_all=true.",
            code="ambiguous",
        )
    ctx.note_write(path)
    path.write_text(text.replace(old, new), encoding="utf-8")
    ctx.record("edit_file", {"path": str(path), "occurrences": occurrences})
    return ToolResult(True, f"Replaced {occurrences} occurrence(s) in {args.get('path')}",
                      tool="edit_file", data={"occurrences": occurrences})


def _read_file(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    path = ctx.resolve(str(args.get("path", "")))
    if not path.exists():
        raise ToolError(f"{args.get('path')} does not exist", code="not_found")
    text = path.read_text(encoding="utf-8", errors="replace")
    limit = int(args.get("max_chars") or 40000)
    truncated = len(text) > limit
    body = text[:limit] + (f"\n…[truncated, {len(text) - limit} more characters]" if truncated else "")
    return ToolResult(True, body, tool="read_file", data={"truncated": truncated})


def _execute_file(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not ctx.allow_execute:
        raise ToolError(
            "Execution is disabled for this agent. Enable it deliberately "
            "(ToolContext(allow_execute=True)) — running model-written code is not a "
            "default anyone should get by accident.",
            code="execute_disabled",
        )
    path = ctx.resolve(str(args.get("path", "")))
    if not path.exists():
        raise ToolError(f"{args.get('path')} does not exist", code="not_found")

    interpreter = str(args.get("interpreter") or "").strip()
    if interpreter:
        # Split rather than shell out. `sh -c` with model-generated text
        # is a remote code execution primitive with a friendly name.
        argv = shlex.split(interpreter) + [str(path)]
    elif path.suffix == ".py":
        argv = ["python3", str(path)]
    elif path.suffix in (".sh", ".bash"):
        argv = ["bash", str(path)]
    else:
        raise ToolError(
            f"Do not know how to run {path.suffix or 'a file with no extension'}. "
            "Pass an interpreter.",
            code="unknown_type",
        )

    extra = args.get("args") or []
    if isinstance(extra, str):
        extra = shlex.split(extra)
    argv += [str(a) for a in extra]

    started = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603 - argv list, never a shell
            argv,
            cwd=str(ctx.root),
            capture_output=True,
            text=True,
            timeout=ctx.execute_timeout,
            check=False,
            # A minimal environment: the agent's process may hold API
            # keys for every provider in the roster, and there is no
            # reason for a script it wrote to inherit them.
            env={"PATH": os.environ.get("PATH", ""), "HOME": str(ctx.root),
                 "LANG": os.environ.get("LANG", "C.UTF-8")},
        )
    except subprocess.TimeoutExpired:
        raise ToolError(
            f"Killed after {ctx.execute_timeout:.0f}s. If it needs longer, say so and raise "
            "the limit deliberately.",
            code="timeout",
        ) from None
    except OSError as exc:
        raise ToolError(f"Could not run {argv[0]}: {exc}", code="spawn_failed") from exc

    elapsed = time.monotonic() - started
    ctx.record("execute_file", {"argv": argv, "returncode": proc.returncode})
    body = (
        f"exit {proc.returncode} in {elapsed:.1f}s\n"
        f"--- stdout ---\n{proc.stdout[-8000:]}\n"
        f"--- stderr ---\n{proc.stderr[-8000:]}"
    )
    return ToolResult(
        proc.returncode == 0, body, tool="execute_file",
        code="" if proc.returncode == 0 else "nonzero_exit",
        data={"returncode": proc.returncode, "seconds": elapsed},
    )


def _web_search(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not ctx.allow_web_search:
        raise ToolError("Web search is disabled for this agent.", code="search_disabled")
    query = str(args.get("query", "")).strip()
    if not query:
        raise ToolError("web_search needs a query", code="bad_args")

    # DuckDuckGo's instant-answer endpoint: no key, no tracking, and a
    # documented JSON shape. It is a weak search engine and the result
    # says so rather than presenting thin results as though they were
    # comprehensive.
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    )
    req = urllib.request.Request(url, headers={"User-Agent": "hypernix-noodle/0.72.1"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8", "replace") or "{}")
    except Exception as exc:  # noqa: BLE001 - reported to the model, not raised
        raise ToolError(f"Search failed: {exc}", code="search_failed") from exc

    lines: list[str] = []
    if payload.get("AbstractText"):
        lines.append(f"{payload['AbstractText']}  ({payload.get('AbstractURL', '')})")
    for topic in (payload.get("RelatedTopics") or [])[:8]:
        if isinstance(topic, dict) and topic.get("Text"):
            lines.append(f"- {topic['Text']}  ({topic.get('FirstURL', '')})")
    if not lines:
        return ToolResult(
            True,
            f"No instant answer for {query!r}. This backend only returns instant answers, "
            "not a ranked web index — treat a blank result as 'unknown', not 'nothing exists'.",
            tool="web_search",
        )
    return ToolResult(True, "\n".join(lines), tool="web_search", data={"results": len(lines)})


def _load_memory(ctx: ToolContext) -> dict[str, Any]:
    if not ctx.memory_path or not ctx.memory_path.exists():
        return {}
    try:
        return json.loads(ctx.memory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _update_memory(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not ctx.memory_enabled:
        raise ToolError(
            "Memory is not enabled on this server. Ask the operator to turn it on; an agent "
            "cannot enable its own persistence.",
            code="memory_disabled",
        )
    key = str(args.get("key", "")).strip()
    if not key:
        raise ToolError("update_memory needs a key", code="bad_args")
    memory = _load_memory(ctx)
    if args.get("delete"):
        existed = memory.pop(key, None) is not None
        message = f"Forgot {key!r}" if existed else f"Nothing stored under {key!r}"
    else:
        memory[key] = {"value": args.get("value"), "updated_at": time.time()}
        message = f"Remembered {key!r}"
    assert ctx.memory_path is not None
    ctx.memory_path.write_text(json.dumps(memory, indent=2, default=str), encoding="utf-8")
    ctx.record("update_memory", {"key": key, "deleted": bool(args.get("delete"))})
    return ToolResult(True, message, tool="update_memory", data={"keys": len(memory)})


def _read_memory(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    if not ctx.memory_enabled:
        raise ToolError("Memory is not enabled on this server.", code="memory_disabled")
    memory = _load_memory(ctx)
    key = str(args.get("key", "")).strip()
    if key:
        if key not in memory:
            return ToolResult(True, f"Nothing stored under {key!r}", tool="read_memory")
        return ToolResult(
            True, json.dumps(memory[key], indent=2, default=str), tool="read_memory"
        )
    if not memory:
        return ToolResult(True, "Memory is empty.", tool="read_memory")
    return ToolResult(
        True,
        "\n".join(f"- {k}: {json.dumps(v.get('value'), default=str)[:200]}" for k, v in memory.items()),
        tool="read_memory",
        data={"keys": len(memory)},
    )


def _compact_context(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Signal that the conversation should be summarised.

    Deliberately does not do the summarising. The agent loop owns the
    transcript and is the only thing that can replace it; a tool that
    returned a summary would be handing the model a string and hoping it
    used it. :class:`~hypernix.interfaces.noodle.agent.Agent` watches for
    this result and performs the compaction itself.
    """
    keep = int(args.get("keep_last") or 6)
    reason = str(args.get("reason") or "context is getting long")
    ctx.record("compact_context", {"keep_last": keep, "reason": reason})
    return ToolResult(
        True,
        f"Compaction requested ({reason}); the agent loop will summarise everything except "
        f"the last {keep} messages before your next turn.",
        tool="compact_context",
        code="compact_requested",
        data={"keep_last": keep, "reason": reason},
    )


def _create_todo(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    text = str(args.get("text", "")).strip()
    if not text:
        raise ToolError("create_todo needs text", code="bad_args")
    todo_id = f"t{len(ctx.todos) + 1}"
    ctx.todos[todo_id] = TodoItem(todo_id=todo_id, text=text)
    ctx.record("create_todo", {"id": todo_id})
    return ToolResult(
        True, f"[{todo_id}] {text}", tool="create_todo", data={"id": todo_id}
    )


def _update_todo(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    todo_id = str(args.get("id", "")).strip()
    if todo_id not in ctx.todos:
        raise ToolError(
            f"No todo {todo_id!r}. Current: "
            + (", ".join(ctx.todos) or "none"),
            code="not_found",
        )
    item = ctx.todos[todo_id]
    status = str(args.get("status") or "").strip().lower()
    if status:
        if status not in ("pending", "in_progress", "done", "removed"):
            raise ToolError(
                f"status must be pending, in_progress, done or removed; got {status!r}",
                code="bad_args",
            )
        item.status = status
    if args.get("text"):
        item.text = str(args["text"])
    ctx.record("update_todo", {"id": todo_id, "status": item.status})
    remaining = sum(1 for t in ctx.todos.values() if t.status in ("pending", "in_progress"))
    return ToolResult(
        True, f"[{todo_id}] {item.status}: {item.text}  ({remaining} still open)",
        tool="update_todo", data=item.to_dict(),
    )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

TOOLS: dict[str, Tool] = {
    t.name: t
    for t in (
        Tool("create_file",
             "Create a file in the workspace. Fails if it exists unless overwrite is true.",
             _p(path=_str("Path relative to the workspace root", required=True),
                content=_str("The file's full contents", required=True),
                overwrite={"type": "boolean", "description": "Replace an existing file"}),
             _create_file, mutating=True),
        Tool("edit_file",
             "Replace exact text in an existing file. old_text must be unique unless "
             "replace_all is true.",
             _p(path=_str("Path relative to the workspace root", required=True),
                old_text=_str("Exact text to find", required=True),
                new_text=_str("Replacement text", required=True),
                replace_all={"type": "boolean", "description": "Replace every occurrence"}),
             _edit_file, mutating=True),
        Tool("read_file",
             "Read a file from the workspace.",
             _p(path=_str("Path relative to the workspace root", required=True),
                max_chars=_int("Truncate beyond this many characters (default 40000)")),
             _read_file),
        Tool("execute_file",
             "Run a file in the workspace and return its output. Disabled unless the "
             "operator enabled execution.",
             _p(path=_str("Path relative to the workspace root", required=True),
                interpreter=_str("Command to run it with, e.g. 'python3'"),
                args=_str("Extra arguments")),
             _execute_file, mutating=True),
        Tool("web_search",
             "Search the web for current information.",
             _p(query=_str("What to search for", required=True)),
             _web_search),
        Tool("update_memory",
             "Store or delete a durable note. Only works if the server enabled memory.",
             _p(key=_str("What to file it under", required=True),
                value=_str("What to remember"),
                delete={"type": "boolean", "description": "Forget this key instead"}),
             _update_memory, mutating=True),
        Tool("read_memory",
             "Read durable notes. Omit key to list everything.",
             _p(key=_str("A specific key, or omit for all")),
             _read_memory),
        Tool("compact_context",
             "Ask the agent loop to summarise the conversation so far, keeping the most "
             "recent messages. Use when the transcript is getting long.",
             _p(reason=_str("Why compaction is needed"),
                keep_last=_int("How many recent messages to keep verbatim (default 6)")),
             _compact_context),
        Tool("create_todo",
             "Add an item to this run's todo list.",
             _p(text=_str("What needs doing", required=True)),
             _create_todo, mutating=True),
        Tool("update_todo",
             "Change a todo's status or text. Use status=removed to drop it.",
             _p(id=_str("The todo id, e.g. t1", required=True),
                status=_str("pending, in_progress, done or removed"),
                text=_str("New text")),
             _update_todo, mutating=True),
    )
}


def tool_schemas(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Render the tools as provider-neutral schemas.

    One definition, rendered per provider by
    :class:`~hypernix.interfaces.noodle.providers.ModelClient`. Writing
    the same tool three times for three wire formats is how one of them
    ends up with a different description and the model behaves
    differently depending on which backend it landed on.
    """
    chosen = [TOOLS[n] for n in (names or list(TOOLS)) if n in TOOLS]
    return [
        {"name": t.name, "description": t.description, "parameters": t.parameters}
        for t in chosen
    ]


def run_tool(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> ToolResult:
    """Run one tool, turning every failure into a result the model can read.

    Nothing escapes as an exception. A model that receives "that path is
    outside the workspace" can correct itself; one that receives a
    traceback, or nothing, cannot.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(
            False,
            f"No tool named {name!r}. Available: {', '.join(sorted(TOOLS))}",
            tool=name, code="unknown_tool",
        )
    try:
        return tool.run(ctx, arguments or {})
    except ToolError as exc:
        return ToolResult(False, str(exc), tool=name, code=exc.code)
    except Exception as exc:  # noqa: BLE001 - the model gets to see this and retry
        logger.exception("noodle.tools: %s raised", name)
        return ToolResult(False, f"{type(exc).__name__}: {exc}", tool=name, code="exception")
