"""t1api.mcptools — the T1 capabilities, described for a model.

:mod:`hypernix.t1api.mcp` is the protocol. This is what the protocol
serves: the tools an assistant gets when it connects to a HyperNix
server, built from the same registry, runner and catalogue the REST
surface uses.

The descriptions are the interface
----------------------------------
A REST endpoint is documented for a person who will read the docs once
and then write a client. An MCP tool is read by a model on every call,
with no docs and no second chance, so the description *is* the
contract: what it does, what it costs, and when not to use it. The ones
here say when not to — `runner_load` evicts whatever is currently
answering, and a model that does not know that will switch models
mid-conversation to answer a question about model names.

Scopes are declared, not checked here
-------------------------------------
Each tool names the scope it needs and dispatch enforces it before the
handler runs. A handler that checks its own permissions is a handler
that forgets to, and the failure is silent in exactly the direction
that matters.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .mcp import MCPResource, MCPServer, MCPTool, ToolError

__all__ = ["build_server", "ServerContext"]


class ServerContext:
    """What the tools need, gathered once.

    A small object rather than a pile of closures over `request.app`,
    so the tool handlers can be tested by handing them a fake with the
    same three attributes.
    """

    def __init__(self, *, config=None, registry=None, runner=None, version: str = ""):
        self.config = config
        self.registry = registry
        self.runner = runner
        self.version = version


def _catalogue(context: ServerContext):
    from ..hyperlink.catalogue import collect

    bridge = None
    if getattr(context.config, "lmstudio_enabled", False):
        try:
            from .routers.bridge import _bridge_for

            bridge = _bridge_for(context.config)
        except Exception:  # noqa: BLE001
            # A bridge that cannot be built is "LM Studio is not
            # reachable", which the catalogue reports as a source that
            # did not answer. It is not a reason to fail the call.
            bridge = None
    return collect(
        registry=context.registry,
        bridge=bridge,
        local_dir=getattr(context.config, "hf_download_dir", "") or None,
    )


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------


def _list_models(context: ServerContext) -> Callable[..., Any]:
    def handler(runnable_only: bool = False) -> dict[str, Any]:
        catalogue = _catalogue(context)
        models = [m.to_dict() for m in catalogue.models]
        if runnable_only:
            models = [m for m in models if m.get("path")]
        return {
            "count": len(models),
            "models": [
                {
                    "model_id": m.get("model_id"),
                    "source": m.get("source"),
                    "parameters_b": m.get("parameters_b"),
                    "quant": m.get("quant"),
                    "context_limit": m.get("context_limit"),
                    "loadable": bool(m.get("path")),
                }
                for m in models[:200]
            ],
            # Sources, not just the list: "no models" and "LM Studio is
            # not running" produce the same empty list and want
            # completely different things doing about them.
            "sources": [s.to_dict() for s in catalogue.sources],
        }

    return handler


def _runner_status(context: ServerContext) -> Callable[..., Any]:
    def handler() -> dict[str, Any]:
        runner = context.runner
        if runner is None:
            return {"loaded": False, "detail": "this server has no built-in runner"}
        current = getattr(runner, "current", None)
        return {
            "loaded": current is not None,
            "model": current.to_dict() if current is not None else {},
            "base_url": getattr(runner, "base_url", ""),
        }

    return handler


def _runner_load(context: ServerContext) -> Callable[..., Any]:
    def handler(model_id: str, gpu_layers: int | None = None,
                context_length: int | None = None) -> dict[str, Any]:
        runner = context.runner
        if runner is None:
            raise ToolError("this server has no built-in runner to load into")

        catalogue = _catalogue(context)
        match = next(
            (m for m in catalogue.models if m.model_id == model_id and m.path), None
        )
        if match is None:
            loadable = [m.model_id for m in catalogue.models if m.path][:20]
            raise ToolError(
                f"no loadable model called {model_id!r} on this server. "
                f"Loadable: {', '.join(loadable) or '(none)'}"
            )
        try:
            current = runner.load(
                match.path,
                model_id=model_id,
                gpu_layers=gpu_layers,
                context_length=context_length or 0,
            )
        except Exception as exc:  # noqa: BLE001 - the reason is for the model
            raise ToolError(f"could not load {model_id}: {exc}") from exc
        return {"loaded": True, "model": current.to_dict()}

    return handler


def _hardware(context: ServerContext) -> Callable[..., Any]:
    def handler() -> dict[str, Any]:
        from ..system.hardware import snapshot

        return snapshot().to_dict()

    return handler


def _server_version(context: ServerContext) -> Callable[..., Any]:
    def handler() -> dict[str, Any]:
        return {"version": context.version or "unknown"}

    return handler


def build_server(context: ServerContext, *, instructions: str = "") -> MCPServer:
    """Every built-in tool, on a fresh server."""
    server = MCPServer(
        server_name="hypernix-t1",
        server_version=context.version,
        instructions=instructions or (
            "This is a HyperNix T1 server. It holds models, can run one "
            "of them itself, and knows what hardware it is on. Loading a "
            "model evicts whatever is currently answering, so check "
            "runner_status before runner_load."
        ),
    )

    server.add_tool(MCPTool(
        name="list_models",
        description=(
            "Every model this server can offer, from its registry, its "
            "models directory, and LM Studio if that is enabled. Returns "
            "the sources as well as the list, because an empty list means "
            "either 'no models' or 'LM Studio is not running' and those "
            "need different things doing about them. Set runnable_only to "
            "see just the ones with a file on disk that the built-in "
            "runner can load."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "runnable_only": {
                    "type": "boolean",
                    "description": "Only models with a local file.",
                }
            },
        },
        handler=_list_models(context),
        scope="read",
    ))

    server.add_tool(MCPTool(
        name="runner_status",
        description=(
            "What the built-in runner is currently serving, where its "
            "layers are, and the URL it answers on. Ask this before "
            "runner_load — loading evicts whatever is answering now."
        ),
        input_schema={"type": "object", "properties": {}},
        handler=_runner_status(context),
        scope="read",
    ))

    server.add_tool(MCPTool(
        name="runner_load",
        description=(
            "Load a model into this server's own llama.cpp, replacing "
            "whatever is running. This EVICTS the model currently "
            "answering and takes anywhere from seconds to minutes. Do not "
            "call it to answer a question about which models exist — "
            "list_models does that without changing anything."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string",
                             "description": "As returned by list_models."},
                "gpu_layers": {"type": "integer",
                               "description": "Layers on the GPU; omit to work it out."},
                "context_length": {"type": "integer",
                                   "description": "Context window; omit for the model's own."},
            },
            "required": ["model_id"],
        },
        handler=_runner_load(context),
        scope="write",
        mutating=True,
    ))

    server.add_tool(MCPTool(
        name="hardware",
        description=(
            "GPUs, CPU and memory on this machine, with what is free. "
            "Use it to work out whether a model will fit before loading "
            "it."
        ),
        input_schema={"type": "object", "properties": {}},
        handler=_hardware(context),
        scope="read",
    ))

    server.add_tool(MCPTool(
        name="server_version",
        description="The HyperNix version this server is running.",
        input_schema={"type": "object", "properties": {}},
        handler=_server_version(context),
        scope="",
    ))

    server.add_resource(MCPResource(
        uri="hypernix://models",
        name="Model catalogue",
        description="Every model this server can offer, as JSON.",
        mime_type="application/json",
        reader=lambda: json.dumps(_list_models(context)(), indent=2, default=str),
        scope="read",
    ))

    return server
