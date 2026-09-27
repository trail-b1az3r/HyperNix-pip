"""Serve a native HyperNix model over the OpenAI API, as llama-server does.

    python -m hypernix.hyperlink.brewed_server --model ~/.hypernix/models/HyperNix.3-mini

The built-in runner starts this instead of llama-server when the model is
a ``hyperNix0x-v2`` folder (see :mod:`hypernix.hyperlink.brewed`), and
talks to it the same way: ``/health``, ``/v1/models``,
``/v1/chat/completions`` (streamed or not) and ``/v1/completions``.

A chat is written out as a transcript for the model to continue::

    <system text>

    User: ...
    Assistant: ...
    User: ...
    Assistant:

and generation stops at the next ``User:``. That is the only chat format
a base model can follow, since it has never seen a template. The oldest
text falls off the front when the prompt is longer than the model's
context, so the newest turns always survive. The ``tools`` field is
accepted and ignored: a base model cannot call tools.

Standard library HTTP, loopback by default, one generation at a time.
The runner is its only intended client.
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["BrewedModel", "render_chat", "build_server", "main"]

MAX_BODY_BYTES = 8 << 20
#: Where a continued transcript has moved on to someone else's turn.
CHAT_STOPS = ("\nUser:", "\nAssistant:", "\nSystem:", "\nTool:", "<|endoftext|>")
_LABELS = {"user": "User", "assistant": "Assistant", "tool": "Tool", "function": "Tool"}


def _text(content: Any) -> str:
    """A message's text, from a string or OpenAI content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content
                         if isinstance(part, dict) and part.get("type") == "text")
    return "" if content is None else str(content)


def render_chat(messages: list[dict[str, Any]]) -> str:
    """The transcript a base model continues, ending with ``Assistant:``."""
    system = "\n\n".join(_text(m.get("content")).strip() for m in messages
                         if m.get("role") == "system" and _text(m.get("content")).strip())
    lines = [system, ""] if system else []
    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        text = _text(message.get("content")).strip()
        if text:
            lines.append(f"{_LABELS.get(str(role), 'User')}: {text}")
    lines.append("Assistant:")
    return "\n".join(lines)


class BrewedModel:
    """One loaded checkpoint, and the one thing a request asks of it."""

    def __init__(self, path: str | Path, *, alias: str = "", device: str = "auto") -> None:
        from ..models.neo_oven import preheat_brewed

        self.path = Path(path)
        chosen = None if device in ("", "auto") else device
        self.oven = preheat_brewed(self.path, tokenizer_source=self.path, device=chosen)
        config = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        self.alias = alias or str(config.get("name") or self.path.name).lower()
        self.context_length = int(config.get("max_seq_len") or 512)
        self._lock = threading.Lock()

    def stream(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float,
               stop: tuple[str, ...]):
        """Text pieces as they are generated; one generation at a time."""
        budget = max(1, min(int(max_tokens), self.context_length))
        with self._lock:
            yield from self.oven.stream(
                prompt, max_new_tokens=budget, temperature=max(0.0, float(temperature)),
                top_p=float(top_p), stop=stop,
            )


def build_server(model: Any, *, host: str = "127.0.0.1", port: int = 8781) -> ThreadingHTTPServer:
    """An HTTP server answering for *model*. Not started."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("brewed_server: " + fmt, *args)

        def _send(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _fail(self, code: int, message: str) -> None:
            self._send(code, {"error": {"message": message, "type": "invalid_request_error"}})

        def _body(self) -> dict[str, Any] | None:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._fail(413, "request body too large")
                return None
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                self._fail(400, "the body is not JSON")
                return None
            if not isinstance(data, dict):
                self._fail(400, "the body must be a JSON object")
                return None
            return data

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler
            if self.path == "/health":
                return self._send(200, {"status": "ok"})
            if self.path in ("/v1/models", "/models"):
                return self._send(200, {"object": "list", "data": [{
                    "id": model.alias, "object": "model", "owned_by": "hypernix",
                    "context_length": model.context_length,
                }]})
            return self._fail(404, f"no route {self.path}")

        def do_POST(self) -> None:  # noqa: N802
            chat = self.path in ("/v1/chat/completions", "/chat/completions")
            if not chat and self.path not in ("/v1/completions", "/completions"):
                return self._fail(404, f"no route {self.path}")
            body = self._body()
            if body is None:
                return None
            if chat:
                messages = body.get("messages")
                if not isinstance(messages, list) or not messages:
                    return self._fail(400, "messages must be a non-empty list")
                prompt = render_chat([m for m in messages if isinstance(m, dict)])
            else:
                prompt = _text(body.get("prompt"))
            stops = list(CHAT_STOPS if chat else ("<|endoftext|>",))
            extra = body.get("stop")
            if isinstance(extra, str):
                stops.append(extra)
            elif isinstance(extra, list):
                stops += [s for s in extra if isinstance(s, str) and s]
            try:
                max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 256)
                temperature = float(body.get("temperature", 0.7) if body.get("temperature") is not None else 0.7)
                top_p = float(body.get("top_p", 0.95) if body.get("top_p") is not None else 0.95)
            except (TypeError, ValueError):
                return self._fail(400, "max_tokens, temperature and top_p must be numbers")
            pieces = model.stream(prompt, max_tokens=max_tokens, temperature=temperature,
                                  top_p=top_p, stop=tuple(stops))
            ident = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            if body.get("stream"):
                return self._stream(pieces, ident, created, chat, max_tokens)
            text = "".join(pieces).lstrip()
            finish = "stop"
            if chat:
                choice = {"index": 0, "message": {"role": "assistant", "content": text},
                          "finish_reason": finish}
            else:
                choice = {"index": 0, "text": text, "finish_reason": finish}
            return self._send(200, {
                "id": ident, "object": "chat.completion" if chat else "text_completion",
                "created": created, "model": model.alias, "choices": [choice],
            })

        def _stream(self, pieces, ident: str, created: int, chat: bool, max_tokens: int) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            kind = "chat.completion.chunk" if chat else "text_completion"

            def frame(delta: dict[str, Any] | None, text: str = "", finish: str | None = None) -> bytes:
                choice: dict[str, Any] = {"index": 0, "finish_reason": finish}
                if chat:
                    choice["delta"] = delta or {}
                else:
                    choice["text"] = text
                return f"data: {json.dumps({'id': ident, 'object': kind, 'created': created, 'model': model.alias, 'choices': [choice]})}\n\n".encode()

            started = False
            try:
                if chat:
                    self.wfile.write(frame({"role": "assistant"}))
                for piece in pieces:
                    if not started:
                        piece = piece.lstrip()
                        if not piece:
                            continue
                        started = True
                    self.wfile.write(frame({"content": piece}, piece))
                    self.wfile.flush()
                self.wfile.write(frame({}, "", "stop"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The phone went away. The generator is dropped here, which
                # stops it at its next token.
                logger.debug("brewed_server: client went away mid-stream")

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="a hyperNix0x-v2 folder")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument("--alias", default="", help="the model id to answer as")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model = BrewedModel(args.model, alias=args.alias, device=args.device)
    server = build_server(model, host=args.host, port=args.port)
    print(f"serving {model.alias} on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
