"""hyperlink.codetools — what a model can do in a code session's sandbox (0.72.8a3).

HyperLink's code section ties a conversation to a ``/code`` sandbox. The
model then gets tools that work on exactly that sandbox — list, read
(by line range), write, edit in place, search, run, and git — through the
same :class:`~hypernix.t1api.codebox.SandboxStore` the ``/code`` routes
use, so the sandbox's permissions (``s1`` run, ``s2`` network, disk and
time budgets) bind the model exactly as they bind a person.

A caller who could only read the sandbox by hand gets only the tools
that read: listing, reading, searching, and git's status, diff, log and
show.
"""
from __future__ import annotations

import json
from typing import Any

__all__ = ["CodeTools"]

MAX_TEXT = 40_000


def _fn(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


class CodeTools:
    """The tools for one sandbox."""

    def __init__(self, store: Any, box: Any, perms: Any, *, may_write: bool) -> None:
        self.store = store
        self.box = box
        self.perms = perms
        self.may_write = may_write

    def openai_tools(self) -> list[dict[str, Any]]:
        where = f"the code sandbox {self.box.name or self.box.id}"
        tools = [
            _fn("code_list_files", f"List the files in {where}.", {}, []),
            _fn("code_read_file", f"Read a file in {where}, optionally only some lines (1-based, inclusive).",
                {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
                ["path"]),
            _fn("code_search", f"Find lines containing text in {where}'s files.",
                {"query": {"type": "string"}, "glob": {"type": "string", "description": "e.g. *.py"}}, ["query"]),
            _fn("code_git", f"git in {where}. Reading: status, diff, log, show"
                + ("; changing: init, add, commit, branch, checkout, restore, clone." if self.may_write else "."),
                {"op": {"type": "string"}, "message": {"type": "string"},
                 "paths": {"type": "array", "items": {"type": "string"}}, "ref": {"type": "string"},
                 "name": {"type": "string"}, "url": {"type": "string"}, "create": {"type": "boolean"},
                 "staged": {"type": "boolean"}, "repo": {"type": "string"}}, ["op"]),
        ]
        if self.may_write:
            tools += [
                _fn("code_write_file", f"Create or replace a file in {where} with this content.",
                    {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
                _fn("code_edit_file", f"Replace exactly one occurrence of old_text with new_text in a file in {where}.",
                    {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
                    ["path", "old_text", "new_text"]),
            ]
            if getattr(self.perms, "execute", False):
                tools.append(_fn("code_run", f"Run a file in {where} (path), or new code (code, language).",
                                 {"path": {"type": "string"}, "code": {"type": "string"},
                                  "language": {"type": "string", "description": "python or bash"},
                                  "args": {"type": "array", "items": {"type": "string"}}}, []))
        return tools

    @property
    def names(self) -> set[str]:
        return {t["function"]["name"] for t in self.openai_tools()}

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """``(ok, text)``; a refusal is text the model can read."""
        from ..t1api import codegit
        from ..t1api.codebox import SandboxError

        try:
            if name == "code_list_files":
                files = self.box.files()
                return True, "\n".join(f"{f['path']}  ({f['bytes']} bytes)" for f in files) or "(no files yet)"
            if name == "code_read_file":
                text, truncated = self.store.read(self.box, str(arguments.get("path", "")))
                lines = text.splitlines()
                start = max(1, int(arguments.get("start_line") or 1))
                end = min(len(lines), int(arguments.get("end_line") or len(lines)))
                body = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start, end + 1))
                if len(body) > MAX_TEXT:
                    body = body[:MAX_TEXT] + "\n[cut: read fewer lines with start_line/end_line]"
                return True, (body or "(empty file)") + ("\n[the file is longer than can be read at once]" if truncated else "")
            if name == "code_search":
                found = self.store.search(self.box, str(arguments.get("query", "")), glob=str(arguments.get("glob") or ""))
                return True, "\n".join(f"{m['path']}:{m['line']}: {m['text']}" for m in found) or "no matches"
            if name == "code_git":
                op = str(arguments.get("op", ""))
                if op not in ("status", "diff", "log", "show") and not (op == "branch" and not arguments.get("name")) \
                        and not self.may_write:
                    return False, "This credential may only read the sandbox: status, diff, log and show."
                result = codegit.git(self.box, self.perms, op, **{k: v for k, v in arguments.items() if k != "op"})
                text = (result.get("output") or "").strip() or ("done" if result["ok"] else "")
                if not result["ok"]:
                    return False, f"git {op} failed: {result.get('error', '').strip() or text}"
                return True, text[:MAX_TEXT]
            if not self.may_write:
                return False, "This credential may only read the sandbox."
            if name == "code_write_file":
                path = str(arguments.get("path", ""))
                written = self.store.write(self.box, path, str(arguments.get("content", "")), self.perms)
                return True, f"wrote {path} ({written.stat().st_size} bytes)"
            if name == "code_edit_file":
                path = str(arguments.get("path", ""))
                text, truncated = self.store.read(self.box, path)
                if truncated:
                    return False, f"{path} is too large to edit through this tool"
                old, new = str(arguments.get("old_text", "")), str(arguments.get("new_text", ""))
                count = text.count(old) if old else 0
                if count != 1:
                    return False, (f"old_text occurs {count} times in {path}; it has to occur exactly once "
                                   "(include more surrounding lines)")
                self.store.write(self.box, path, text.replace(old, new, 1), self.perms)
                return True, f"edited {path}"
            if name == "code_run":
                result = self.store.run(self.box, self.perms, language=str(arguments.get("language") or ""),
                                        code=arguments.get("code") if isinstance(arguments.get("code"), str) else None,
                                        path=str(arguments.get("path") or ""),
                                        args=[str(a) for a in arguments.get("args") or []])
                data = result.to_dict()
                return data.get("exit_code", 1) == 0, json.dumps(
                    {k: data.get(k) for k in ("exit_code", "stdout", "stderr", "timed_out", "elapsed_seconds")})[:MAX_TEXT]
        except SandboxError as exc:
            return False, str(exc)
        except (OSError, ValueError, TypeError) as exc:
            return False, f"{name} failed: {exc}"
        return False, f"there is no tool called {name}"
