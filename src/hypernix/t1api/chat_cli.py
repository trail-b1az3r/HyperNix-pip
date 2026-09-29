"""``hypernix-t1 chat`` — talk to the served model from the terminal.

    hypernix-t1 chat "What can you do here?"
    hypernix-t1 chat -s "You are terse. Answer in one line." "Which model are you?"
    hypernix-t1 chat "and what about GPUs?"          # carries on the same chat
    hypernix-t1 chat --new "a fresh conversation"
    echo "summarise this" | hypernix-t1 chat

The same HyperLink route the phone uses, so the reply comes from whatever
the server would answer with: the built-in runner when a model is loaded
there, with HyperLink's default system prompt and tools, else LM Studio.

It exists because the alternative was curl, a key pasted into an
environment variable, and three `VAR=$(...)` lines that fish rejects
outright ("Unsupported use of '='"). This works the same in any shell.

The chat is remembered in ``~/.hypernix/t1api/cli_session`` so a second
``chat`` carries on the first; ``--new`` starts again. ``--system`` sets
the chat's own system prompt, which is added after HyperLink's default
one rather than replacing it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from hypernix.security.safeurl import urlopen as safe_urlopen

from . import runner_cli
from .localserver import config_dir

__all__ = ["main", "cli_main"]


def _session_file():
    return config_dir() / "cli_session"


def _remembered_session() -> str:
    try:
        return _session_file().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _remember_session(session_id: str) -> None:
    try:
        path = _session_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(session_id + "\n", encoding="utf-8")
    except OSError:
        pass


def _new_session(url: str, key: str, *, system: str, model: str, title: str) -> str:
    status, body = runner_cli._request(
        "POST", f"{url}/hyperlink/sessions", key,
        {"title": title, "system_prompt": system, "model_id": model},
    )
    if status >= 400:
        raise SystemExit(runner_cli._fail(status, body))
    return str(((body or {}).get("session") or {}).get("session_id") or "")


def _session_exists(url: str, key: str, session_id: str) -> bool:
    status, _body = runner_cli._request("GET", f"{url}/hyperlink/sessions/{session_id}", key)
    return status < 400


def _set_system(url: str, key: str, session_id: str, system: str) -> None:
    status, body = runner_cli._request(
        "PATCH", f"{url}/hyperlink/sessions/{session_id}", key, {"system_prompt": system},
    )
    if status >= 400:
        raise SystemExit(runner_cli._fail(status, body))


def _frames(url: str, key: str, payload: dict[str, Any]):
    """The stream's frames, as dicts, in order."""
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "text/event-stream")
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    try:
        response = safe_urlopen(request, timeout=900)
    except urllib.error.HTTPError as error:
        body = error.read() or b"{}"
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = {"error": {"message": body.decode(errors="replace")}}
        raise SystemExit(runner_cli._fail(error.code, parsed)) from None
    except urllib.error.URLError as error:
        raise runner_cli.Unreachable(url, error.reason) from error
    with response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                yield json.loads(line[6:])
            except ValueError:
                continue


def _chat(url: str, key: str, args: argparse.Namespace, message: str) -> int:
    session_id = "" if args.new else (args.session or _remembered_session())
    if session_id and not _session_exists(url, key, session_id):
        session_id = ""  # deleted, or remembered from another server
    if session_id:
        if args.system is not None:
            _set_system(url, key, session_id, args.system)
    else:
        session_id = _new_session(url, key, system=args.system or "", model=args.model,
                                  title=message[:60])
    if not session_id:
        print("The server did not return a chat to send to.", file=sys.stderr)
        return 1
    _remember_session(session_id)

    payload: dict[str, Any] = {"content": message}
    if args.model:
        payload["model_id"] = args.model
    done: dict[str, Any] = {}
    wrote = False
    for frame in _frames(f"{url}/hyperlink/sessions/{session_id}/chat/stream", key, payload):
        kind = frame.get("type")
        if kind == "delta":
            sys.stdout.write(frame.get("text", ""))
            sys.stdout.flush()
            wrote = True
        elif kind == "tool":
            mark = "✓" if frame.get("ok") else "✗"
            print(f"\n[{frame.get('tool', 'tool')} {mark}]", file=sys.stderr)
        elif kind == "error":
            error = frame.get("error") or {}
            print(f"\n{error.get('message') or error}", file=sys.stderr)
            return 1
        elif kind == "done":
            done = frame
    if wrote:
        sys.stdout.write("\n")
    if not args.quiet:
        model = done.get("model_id") or "?"
        finish = done.get("finish_reason") or ""
        print(f"— {model}{f' ({finish})' if finish and finish != 'stop' else ''} · chat {session_id}",
              file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypernix-t1 chat",
        description="Send a message to the model this server serves, and print the reply.",
    )
    parser.add_argument("message", nargs="*", help="the message (default: read standard input)")
    parser.add_argument("-s", "--system", default=None,
                        help="this chat's system prompt, added after HyperLink's default one")
    parser.add_argument("--new", action="store_true", help="start a new chat")
    parser.add_argument("--session", default="", help="carry on this chat id")
    parser.add_argument("-m", "--model", default="", help="ask for this model")
    parser.add_argument("-q", "--quiet", action="store_true", help="print only the reply")
    parser.add_argument("--url", default="", help="server base URL (found the way `runner` finds it)")
    parser.add_argument("--key", default="", help="key (default: T1_ADMIN_KEY, then the server's .env)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    message = " ".join(args.message).strip()
    if not message and not sys.stdin.isatty():
        message = sys.stdin.read().strip()
    if not message:
        print("Nothing to send. Give a message, or pipe one in.", file=sys.stderr)
        return 2
    url = runner_cli._base_url(args.url)
    key = runner_cli._resolve_key(args.key)
    try:
        return _chat(url, key, args, message)
    except runner_cli.Unreachable:
        if args.url or os.environ.get("T1_URL"):
            raise
        from .localserver import discover

        found = discover(exclude=url)
        if not found:
            raise
        print(f"Nothing answered at {url}; using the T1 API found at {found}.\n", file=sys.stderr)
        return _chat(found, key, args, message)


def cli_main() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli_main()
