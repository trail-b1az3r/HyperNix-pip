"""Serve a GGUF on Strife over the OpenAI API, as llama-server does.

    python -m hypernix.strife.server --model model.gguf --port 8781

The built-in runner (``hnx-t1 runner start --backend strife``, the
"Strife" choice in HyperLink) starts this instead of llama-server, and
T1 and HyperLink talk to it the same way: ``GET /health``,
``GET /v1/models``, ``POST /v1/chat/completions`` and
``POST /v1/completions``, each streamed (server-sent events, ending
``data: [DONE]``) or not.

A chat is rendered through the model's own chat template, so a Qwen3.5,
LFM3 or GLM-4.5 conversation is the conversation the model was trained
on. Fields honoured: ``max_tokens`` (or ``max_completion_tokens``),
``temperature``, ``top_p``, ``top_k``, ``min_p``, ``seed``, ``stop``,
``repeat_penalty`` / ``frequency_penalty``, and
``chat_template_kwargs`` (``{"enable_thinking": false}``). ``tools`` is
passed to the template; Strife does not parse tool calls out of the reply
yet.

Each request is matched to the cached session that shares the longest
prefix with it (:meth:`~.engine.Engine.session_for`), so a conversation
continued from HyperLink only runs its newest message. A client that
disconnects mid-stream stops the generation at its next token.

Standard library HTTP, loopback by default. The runner is its intended
client; on another address put T1 in front of it.
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

from .engine import Engine
from .errors import StrifeError
from .sampler import SamplerConfig

logger = logging.getLogger(__name__)

__all__ = ["build_server", "main"]

MAX_BODY_BYTES = 16 << 20


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content
                       if isinstance(part, dict) and part.get("type") in ("text", "input_text"))
    return "" if content is None else str(content)


def _messages(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise StrifeError("strife.request.bad", "messages must be a non-empty list")
    out = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        msg: dict[str, Any] = {"role": str(m.get("role") or "user"), "content": _text(m.get("content"))}
        for key in ("name", "tool_calls", "tool_call_id", "reasoning_content"):
            if key in m:
                msg[key] = m[key]
        out.append(msg)
    return out


def _sampling(body: dict[str, Any]) -> SamplerConfig:
    def num(key: str, default: float) -> float:
        value = body.get(key)
        return default if value is None else float(value)

    penalty = body.get("repeat_penalty")
    if penalty is None and body.get("frequency_penalty"):
        penalty = 1.0 + float(body["frequency_penalty"]) / 2.0
    seed = body.get("seed")
    return SamplerConfig(temperature=num("temperature", 0.7), top_p=num("top_p", 0.95),
                         top_k=int(num("top_k", 40)), min_p=num("min_p", 0.05),
                         repeat_penalty=float(penalty) if penalty is not None else 1.0,
                         seed=int(seed) if seed is not None else None)


def build_server(engine: Engine, *, host: str = "127.0.0.1", port: int = 8781,
                 alias: str = "") -> ThreadingHTTPServer:
    model_id = alias or engine.path.stem

    class Handler(BaseHTTPRequestHandler):
        server_version = "strife"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            logger.debug("strife.server: " + fmt, *args)

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _fail(self, status: int, message: str, code: str = "") -> None:
            self._send(status, {"error": {"message": message, "type": "invalid_request_error"
                                          if status < 500 else "server_error", "code": code or None}})

        def _body(self) -> dict[str, Any] | None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self._fail(413 if length > 0 else 400, "bad or oversized body")
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
                return self._send(200, {"status": "ok", "engine": "strife"})
            if self.path in ("/v1/models", "/models"):
                return self._send(200, {"object": "list", "data": [{
                    "id": model_id, "object": "model", "owned_by": "hypernix-strife",
                    "context_length": engine.n_ctx, "architecture": engine.report.architecture,
                }]})
            if self.path == "/strife/status":
                return self._send(200, engine.status())
            return self._fail(404, f"no route {self.path}")

        def do_POST(self) -> None:  # noqa: N802
            chat = self.path in ("/v1/chat/completions", "/chat/completions")
            if not chat and self.path not in ("/v1/completions", "/completions"):
                return self._fail(404, f"no route {self.path}")
            body = self._body()
            if body is None:
                return None
            try:
                stops = body.get("stop") or []
                stops = [stops] if isinstance(stops, str) else [s for s in stops if isinstance(s, str)]
                max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 512)
                sampling = _sampling(body)
                template_vars: dict[str, Any] = dict(body.get("chat_template_kwargs") or {})
                if body.get("tools"):
                    template_vars["tools"] = body["tools"]
                if chat:
                    ids = engine.prompt_tokens(messages=_messages(body.get("messages")),
                                               template_vars=template_vars)
                else:
                    ids = engine.prompt_tokens(_text(body.get("prompt")))
            except (TypeError, ValueError) as exc:
                return self._fail(400, f"bad request: {exc}")
            except StrifeError as exc:
                return self._fail(400, str(exc), exc.code)
            cancel = threading.Event()
            session = engine.session_for(ids)
            events = engine.stream(prompt_ids=ids, session=session, max_tokens=max_tokens,
                                   stop=stops, sampling=sampling, cancel=cancel)
            ident = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            try:
                if body.get("stream"):
                    return self._stream(events, ident, created, chat, cancel)
                final: dict[str, Any] = {}
                for event in events:
                    if event["type"] == "done":
                        final = event
            except StrifeError as exc:
                status = 503 if exc.code == "strife.engine.busy" else 400
                return self._fail(status, str(exc), exc.code)
            finish = _finish(final.get("stop_reason", "length"))
            choice: dict[str, Any] = (
                {"index": 0, "message": {"role": "assistant", "content": final.get("text", "")},
                 "finish_reason": finish} if chat else
                {"index": 0, "text": final.get("text", ""), "finish_reason": finish})
            return self._send(200, {
                "id": ident, "object": "chat.completion" if chat else "text_completion",
                "created": created, "model": model_id, "choices": [choice],
                "usage": _usage(final.get("stats", {})), "strife": final.get("stats", {})})

        def _stream(self, events: Any, ident: str, created: int, chat: bool,
                    cancel: threading.Event) -> None:
            kind = "chat.completion.chunk" if chat else "text_completion"

            def frame(delta: dict[str, Any] | None, text: str = "", finish: str | None = None,
                      extra: dict[str, Any] | None = None) -> bytes:
                choice: dict[str, Any] = {"index": 0, "finish_reason": finish}
                if chat:
                    choice["delta"] = delta or {}
                else:
                    choice["text"] = text
                payload = {"id": ident, "object": kind, "created": created, "model": model_id,
                           "choices": [choice], **(extra or {})}
                return f"data: {json.dumps(payload)}\n\n".encode()

            started = False
            try:
                for event in events:
                    if not started:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.close_connection = True
                        started = True
                        if chat:
                            self.wfile.write(frame({"role": "assistant"}))
                    if event["type"] == "token":
                        self.wfile.write(frame({"content": event["text"]}, event["text"]))
                        self.wfile.flush()
                    else:
                        self.wfile.write(frame({}, "", _finish(event["stop_reason"]),
                                               {"usage": _usage(event["stats"])}))
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                cancel.set()  # the client went away: stop at the next token
                for _ in events:
                    pass
                logger.debug("strife.server: client went away mid-stream")
            except StrifeError as exc:
                if not started:
                    self._fail(503 if exc.code == "strife.engine.busy" else 400, str(exc), exc.code)

    return ThreadingHTTPServer((host, port), Handler)


def _finish(reason: str) -> str:
    return {"length": "length", "context_full": "length"}.get(reason, "stop")


def _usage(stats: dict[str, Any]) -> dict[str, int]:
    prompt = int(stats.get("prompt_tokens") or 0)
    done = int(stats.get("generated_tokens") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": done, "total_tokens": prompt + done}


def _selftest() -> int:
    """Load, prefill and decode a generated model with *this* Strife.

    ``hnx strife build`` runs it in the build's own interpreter, so what it
    proves is that build: its import path, its kernels, its server module.
    """
    import platform
    import tempfile

    from .. import __version__
    from .fixtures import make_model

    report: dict[str, Any] = {"ok": False, "version": __version__,
                              "python": platform.python_version(),
                              "module": str(Path(__file__).resolve())}
    try:
        with tempfile.TemporaryDirectory(prefix="strife-selftest-") as tmp:
            made = make_model(Path(tmp) / "selftest.gguf", arch="qwen2", seed=11)
            with Engine(made["path"], n_ctx=64) as engine:
                result = engine.generate("selftest", max_tokens=4)
            report.update(ok=len(result.tokens) > 0, tokens=len(result.tokens),
                          architecture=engine.report.architecture)
    except Exception as exc:  # noqa: BLE001 - the report is the point
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report), flush=True)
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", help="a .gguf file")
    parser.add_argument("--selftest", action="store_true",
                        help="check this Strife on a generated model, print one JSON line, exit")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8781)
    parser.add_argument("--alias", default="", help="the model id to answer as")
    parser.add_argument("--ctx", type=int, default=0, help="context length")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--cache-mb", type=int, default=-1)
    parser.add_argument("--sessions", type=int, default=8, help="conversations kept warm")
    parser.add_argument("--kernels", choices=("auto", "off"), default="auto",
                        help="native kernels: used when built, or off")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.model:
        parser.error("--model is required")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        engine = Engine(args.model, n_ctx=args.ctx or None, threads=args.threads or None,
                        cache_bytes=None if args.cache_mb < 0 else args.cache_mb * 1024 * 1024,
                        max_sessions=args.sessions,
                        kernels=None if args.kernels == "auto" else "off")
    except StrifeError as exc:
        print(f"strife: {exc} [{exc.code}]", flush=True)
        if exc.remedy:
            print(f"  {exc.remedy}", flush=True)
        return 2
    server = build_server(engine, host=args.host, port=args.port, alias=args.alias)
    print(f"strife: serving {args.alias or engine.path.stem} ({engine.report.architecture}) "
          f"on http://{args.host}:{args.port}; kernels: {engine.kernels_note}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
